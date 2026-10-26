#!/usr/bin/env python3
"""Run the prespecified SSGMM theory study; resume completed repeats safely.

python -m experiments.study --groups calibration tracking quadratic-linear quadratic-relu exp-tanh
python -m experiments.study --plot-only
"""

from __future__ import annotations
import os

for key in [
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
]:
    os.environ[key] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import numpy as np
from . import config as cfg
from .kernels import (
    aggregate_paths,
    basis_code,
    basis_rows,
    feature_dim,
    multivariate_data,
    pilot_jacobian,
    quadratic_population_g,
    train_path,
)
from ssgmm.models import QuadraticModel

ROOT = Path(__file__).resolve().parents[1]
GROUP_IDS = {
    "calibration": 1,
    "tracking": 2,
    "quadratic-linear": 3,
    "quadratic-relu": 4,
    "exp-tanh": 5,
}


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def seed_for(group, repeat, role=0):
    return [cfg.SEED, GROUP_IDS[group], role, repeat]


def cases_and_parameters(group, n_pilot=None, pilot_batch=None):
    n_pilot = n_pilot or cfg.N_PILOT
    pilot_batch = pilot_batch or cfg.PILOT_BATCH
    if group in ["calibration", "tracking"]:
        specs = [
            dict(x)
            for x in (cfg.CALIBRATION_CONFIGS if group == "calibration" else cfg.TRACKING_CONFIGS)
        ]
        for s in specs:
            s.update(model=0, first_stage=0, offset=100.0, ridge=1.0, radius=1.0, center=[0.0])
        return specs, {"truth": np.array([0.0]), "gamma": np.ones((1, 1))}, {}
    rng = np.random.default_rng(cfg.PARAMETER_SEED)
    gamma = rng.normal(size=(8, 4)) * (0.1 if group == "exp-tanh" else 1.0)
    model = 2 if group == "exp-tanh" else 1
    first = 2 if model == 2 else (1 if group == "quadratic-relu" else 0)
    pilot = {}
    if model == 1:
        truth = np.r_[2 * rng.normal(size=4), np.eye(4)[np.triu_indices(4)]]
    else:
        raw = rng.normal(size=4)
        prng = np.random.default_rng(seed_for(group, 0, 11))
        hs = np.zeros(4)
        hss = np.zeros((4, 4))
        n = 0
        for start in range(0, n_pilot, pilot_batch):
            h = np.tanh(prng.standard_normal((min(pilot_batch, n_pilot - start), 8)) @ gamma)
            hs += h.sum(0)
            hss += h.T @ h
            n += len(h)
        covariance = hss / n - np.outer(hs / n, hs / n) + np.eye(4)
        truth = raw * (0.5 / np.sqrt(raw @ covariance @ raw))
        pilot.update(
            scale_covariance=covariance,
            scale_raw_theta=raw,
            scale_variance=np.array(truth @ covariance @ truth),
        )
    if first == 0:
        fullg = quadratic_population_g(gamma, (0, 1, 2, 3))
        pilot["population_g"] = fullg
        pilot["g_mcse"] = np.zeros_like(fullg)
    else:
        batches = [
            pilot_jacobian(gamma, first, model, truth, seed_for(group, r, 12), n_pilot, pilot_batch)
            for r in range(2)
        ]
        pilot["batch_g"] = np.stack(batches)
        fullg = np.concatenate(batches).mean(0)
        pilot["population_g"] = fullg
        pilot["g_mcse"] = np.concatenate(batches).std(0, ddof=1) / np.sqrt(
            sum(len(b) for b in batches)
        )
        pilot["pilot_difference_frobenius"] = np.array(
            np.linalg.norm(batches[0].mean(0) - batches[1].mean(0))
        )
    hs = []
    for basis in cfg.BASIS_SETS[2:]:
        g = fullg[basis_rows(8, basis)]
        mu = np.linalg.svd(g, compute_uv=False)[-1] ** 2
        hs.append(mu / (mu + 0.01))
    assert np.linalg.norm(truth) < (cfg.EXP_RADIUS if model == 2 else 20.0)
    harmonic_a = 2 / min(hs)
    specs = []
    if model == 1:
        for schedule in cfg.MULTIVARIATE_SCHEDULES:
            for basis in cfg.BASIS_SETS:
                specs.append(
                    dict(
                        label="h" + "".join(map(str, basis)) + "_" + schedule,
                        basis=basis,
                        model=model,
                        first_stage=first,
                        a=0.1 if schedule == "power09" else harmonic_a,
                        eta=0.9 if schedule == "power09" else 1.0,
                        kappa=0.6 if schedule == "power09" else 2 / 3,
                        offset=max(100.0, np.ceil(harmonic_a)),
                        ridge=0.01,
                        radius=20.0,
                        center=np.zeros(len(truth)).tolist(),
                        curvature=0.0,
                        oracle=False,
                    )
                )
    else:
        for initialization in cfg.EXP_INITIALIZATIONS:
            for basis in cfg.BASIS_SETS:
                specs.append(
                    dict(
                        label="h" + "".join(map(str, basis)) + "_" + initialization,
                        basis=basis,
                        model=2,
                        first_stage=2,
                        a=0.1,
                        eta=0.9,
                        kappa=0.6,
                        offset=100.0,
                        ridge=0.01,
                        radius=cfg.EXP_LOCAL_RADIUS
                        if initialization == "local"
                        else cfg.EXP_RADIUS,
                        center=(truth if initialization == "local" else np.zeros(4)).tolist(),
                        curvature=0.0,
                        oracle=False,
                        initialization=initialization,
                    )
                )
    return specs, {"truth": truth, "gamma": gamma}, pilot


def run_repeat(task):
    group, repeat, steps, checkpoints, specs, parameters, pilot, outdir = task
    folder = Path(outdir) / "repeats"
    target = folder / f"repeat_{repeat:04d}.npz"
    if target.exists():
        return repeat, "cached"
    rng = np.random.default_rng(seed_for(group, repeat))
    width = 3 if group in ["calibration", "tracking"] else 17
    draws = rng.standard_normal((steps, width))
    initial_rng = np.random.default_rng(seed_for(group, repeat, 1))
    truth = parameters["truth"]
    d = len(truth)
    if d == 14:
        base = np.r_[initial_rng.normal(0, 0.1, 4), np.zeros(10)]
    else:
        base = initial_rng.normal(0, 0.1, d)
    near = truth + initial_rng.normal(0, cfg.EXP_NEAR_SD, d)
    raw = []
    averages = []
    diags = []
    states = {}
    initials = []
    failures = []
    elapsed = []
    for i, s in enumerate(specs):
        gamma = np.array([[s["gamma"]]]) if s["model"] == 0 else parameters["gamma"]
        code = basis_code(s["basis"])
        p = feature_dim(len(gamma), code)
        initial = (
            np.array([0.5])
            if s["model"] == 0
            else (near.copy() if s.get("initialization") in ["near", "local"] else base.copy())
        )
        center = np.array(s["center"])
        distance = np.linalg.norm(initial - center)
        if distance > s["radius"]:
            initial = center + (initial - center) * (s["radius"] / distance)
        popg = (
            pilot["population_g"][basis_rows(len(gamma), s["basis"])].copy()
            if s["model"] != 0
            else np.zeros((p, d))
        )
        args = (
            gamma,
            truth,
            initial,
            center,
            s["radius"],
            s["model"],
            s["first_stage"],
            code,
            s["a"],
            s["eta"],
            s["kappa"],
            s["offset"],
            s["ridge"],
            s["curvature"],
            s["oracle"],
        )
        # Compile/cache load before timing; discarded warm-up does not update the formal stream.
        train_path(draws[:2], *args, np.array([2]), popg)
        start = time.perf_counter()
        result = train_path(draws, *args, checkpoints, popg)
        elapsed.append(time.perf_counter() - start)
        r, av, diag, state, failure = result
        raw.append(r)
        averages.append(av)
        diags.append(diag)
        states[f"final_S_{i}"] = state
        initials.append(initial)
        failures.append(failure)
    tmp = target.with_suffix(".partial.npz")
    np.savez_compressed(
        tmp,
        theta=np.stack(raw),
        average=np.stack(averages),
        diagnostics=np.stack(diags),
        initial=np.stack(initials),
        failure_step=np.array(failures),
        seconds=np.array(elapsed),
        **states,
    )
    tmp.replace(target)
    return repeat, "completed"


def risk_metrics(group, specs, parameters, theta, average):
    rng = np.random.default_rng(seed_for(group, 0, 20))
    truth = parameters["truth"]
    if group in ["calibration", "tracking"]:
        draws = rng.standard_normal((cfg.N_TEST, 3))
        xbase = draws[:, 0]
        u = draws[:, 1]
        eps = 0.5 * u + 0.5 * (u * u - 1) + draws[:, 2]
        result = []
        for i, s in enumerate(specs):
            x = s["gamma"] * xbase + u
            q = x * x
            mu4 = np.mean(q * q)
            cross = np.mean(q * eps)
            noise = np.mean(eps * eps)
            pair = []
            for values in [theta[i], average[i]]:
                f = values[:, :, 0] + s["curvature"] * values[:, :, 0] ** 2
                pair.append((f * f * mu4, f * f * mu4 - 2 * f * cross + noise))
            result.append(pair)
        return result, {"test_z_u_nu": draws}
    draws = rng.standard_normal((cfg.N_TEST, 17))
    x, y, h = multivariate_data(
        draws, parameters["gamma"], specs[0]["first_stage"], specs[0]["model"], truth
    )
    results = []
    if specs[0]["model"] == 1:
        q = QuadraticModel().gradient(truth, x)
        moment = q.T @ q / len(q)
        eps = y - q @ truth
        cross = q.T @ eps / len(q)
        noise = eps @ eps / len(q)
        for i, s in enumerate(specs):
            pair = []
            for values in [theta[i], average[i]]:
                error = values - truth
                structural = np.einsum("rci,ij,rcj->rc", error, moment, error)
                pair.append(
                    (structural, structural - 2 * np.einsum("rci,i->rc", error, cross) + noise)
                )
            results.append(pair)
    else:
        true = np.exp(x @ truth)
        for i, s in enumerate(specs):
            pair = []
            for values in [theta[i], average[i]]:
                flat = values.reshape(-1, len(truth))
                risk = np.empty(len(flat))
                mse = risk.copy()
                for start in range(0, len(flat), 64):
                    pred = np.exp(flat[start : start + 64] @ x.T)
                    risk[start : start + 64] = np.mean((pred - true) ** 2, axis=1)
                    mse[start : start + 64] = np.mean((pred - y) ** 2, axis=1)
                pair.append((risk.reshape(values.shape[:2]), mse.reshape(values.shape[:2])))
            results.append(pair)
    return results, {"test_x": x, "test_y": y}


def aggregate_group(folder, group, specs, parameters, steps, repeats, checkpoints, metadata):
    records = []
    for r in range(repeats):
        with np.load(folder / "repeats" / f"repeat_{r:04d}.npz") as a:
            records.append(
                {
                    k: a[k]
                    for k in [
                        "theta",
                        "average",
                        "diagnostics",
                        "failure_step",
                        "seconds",
                        "initial",
                    ]
                }
            )
    theta = np.stack([r["theta"] for r in records], axis=1)
    avg = np.stack([r["average"] for r in records], axis=1)
    diagnostics = np.stack([r["diagnostics"] for r in records], axis=1)
    failures = np.stack([r["failure_step"] for r in records], axis=1)
    elapsed = np.stack([r["seconds"] for r in records], axis=1)
    initials = np.stack([r["initial"] for r in records], axis=1)
    risks, test_data = risk_metrics(group, specs, parameters, theta, avg)
    np.savez_compressed(folder / "test_data.npz", **test_data)
    np.savez_compressed(
        folder / "trajectories.npz",
        steps=checkpoints,
        theta=theta,
        average=avg,
        diagnostics=diagnostics,
        failure_step=failures,
        initial=initials,
        seconds=elapsed,
    )
    aggregated = {}
    rows = []
    fit = checkpoints >= steps / 4
    for i, s in enumerate(specs):
        values = aggregate_paths(theta[i], avg[i], parameters["truth"], checkpoints)
        track = diagnostics[i, :, :, 0]
        values["tracking_mse"] = track.mean(0)
        values["tracking_fourth"] = (track**2).mean(0)
        values["projection_hits"] = diagnostics[i, :, :, 1].mean(0)
        for est_idx, prefix in [(0, ""), (1, "average_")]:
            structural, outcome = risks[i][est_idx]
            values[prefix + "structural_risk"] = structural.mean(0)
            values[prefix + "structural_risk_mcse"] = structural.std(0, ddof=1) / np.sqrt(repeats)
            values[prefix + "pred_mse_mean"] = outcome.mean(0)
            values[prefix + "pred_mse_std"] = outcome.std(0)
            values[prefix + "pred_mse_median"] = np.median(outcome, axis=0)
            mse = values[prefix + "param_mse"]
            slope = np.polyfit(np.log(checkpoints[fit]), np.log(mse[fit]), 1)[0]
            rows.append(
                dict(
                    configuration=s["label"],
                    estimator="last" if est_idx == 0 else "average",
                    observations=steps,
                    repeats=repeats,
                    param_mse=mse[-1],
                    mcse=values[prefix + "param_mse_mcse"][-1],
                    rmse=np.sqrt(mse[-1]),
                    median_error=values[prefix + "param_error_median"][-1],
                    q90_error=values[prefix + "param_error_q90"][-1],
                    structural_risk=structural.mean(0)[-1],
                    outcome_mse=outcome.mean(0)[-1],
                    log_log_slope=slope,
                    projection_rate=diagnostics[i, :, -1, 1].mean() / steps,
                    max_distance=diagnostics[i, :, -1, 2].max(),
                    max_index=diagnostics[i, :, -1, 3].max(),
                    max_gram_relative_error=diagnostics[i, :, -1, 4].max(),
                    numerical_failures=np.count_nonzero(failures[i]),
                    seconds_per_repeat=elapsed[i].mean(),
                )
            )
        for k, v in values.items():
            aggregated[s["label"] + "_" + k] = v
    np.savez_compressed(folder / "results.npz", **aggregated)
    with (folder / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "=" * 88,
        "SSGMM Theory Study: " + group,
        "=" * 88,
        f"Training observations: {steps} per configuration per repeat; repeats: {repeats}",
        "MSE and MCSE use squared errors; median is Euclidean error. No failed repeats excluded.",
        f"{'Configuration':<29} {'Estimator':<8} {'Param MSE':>12} {'MCSE':>11} {'slope':>8} {'fail':>5}",
    ]
    for row in rows:
        lines.append(
            f"{row['configuration']:<29} {row['estimator']:<8} {row['param_mse']:12.5g} {row['mcse']:11.4g} {row['log_log_slope']:8.3f} {row['numerical_failures']:5d}"
        )
    (folder / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    metadata.update(
        status="complete", finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    )
    json_write(folder / "metadata.json", metadata)


def provenance(outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    snap = outdir / "source"
    snap.mkdir(exist_ok=True)
    files = ["pyproject.toml", "requirements-study.txt", "docs/reproducibility.md"]
    files += [
        str(p.relative_to(ROOT))
        for folder in ["src/ssgmm", "experiments", "tests"]
        for p in sorted((ROOT / folder).glob("*.py"))
    ]
    checks = {}
    for name in files:
        p = ROOT / name
        target = snap / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = p.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if target.exists() and target.read_bytes() != content:
            raise RuntimeError(
                f"Source changed since this study started: {name}; use a new output directory."
            )
        target.write_bytes(content)
        checks[name] = digest

    def git_output(*args):
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    state = {
        "git_head": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "source_sha256": checks,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timezone": "Asia/Shanghai",
        "numpy": np.__version__,
    }
    if not (outdir / "provenance.json").exists():
        json_write(outdir / "provenance.json", state)
    if not (outdir / "requirements-frozen.txt").exists():
        (outdir / "requirements-frozen.txt").write_text(
            subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=list(GROUP_IDS), default=list(GROUP_IDS))
    parser.add_argument("--output", type=Path, default=ROOT / cfg.OUTDIR)
    parser.add_argument("--jobs", type=int, default=cfg.N_JOBS)
    parser.add_argument(
        "--steps", type=int, help="Explicit alternative budget, recorded in metadata."
    )
    parser.add_argument(
        "--repeats",
        type=int,
        help="Explicit alternative repetition count; use a separate output folder.",
    )
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="512 steps, 2 repeats, small pilots; validates execution only.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.steps = 512 if args.steps is None else args.steps
        args.repeats = 2 if args.repeats is None else args.repeats
        if args.output == ROOT / cfg.OUTDIR:
            args.output = ROOT / "results/smoke"
    if (
        args.jobs < 1
        or (args.steps is not None and args.steps < 8)
        or (args.repeats is not None and args.repeats < 2)
    ):
        parser.error("jobs must be >= 1, steps >= 8, and repeats >= 2 (for Monte Carlo SEs)")
    pilot_samples = 2**12 if args.smoke else cfg.N_PILOT
    pilot_batch = 2**9 if args.smoke else cfg.PILOT_BATCH
    outdir = args.output.resolve()
    if args.plot_only:
        from .plot import plot_study

        plot_study(outdir)
        return
    provenance(outdir)
    for group in args.groups:
        scalar = group in ["calibration", "tracking"]
        steps = args.steps or (cfg.CALIBRATION_STEPS if scalar else cfg.MULTIVARIATE_STEPS)
        repeats = args.repeats or (cfg.CALIBRATION_REPEATS if scalar else cfg.MULTIVARIATE_REPEATS)
        checkpoints = np.unique(np.rint(np.geomspace(1, steps, cfg.N_CHECKPOINTS)).astype(np.int64))
        folder = outdir / group
        folder.mkdir(exist_ok=True)
        (folder / "repeats").mkdir(exist_ok=True)
        old = folder / "metadata.json"
        if old.exists():
            prior = json.loads(old.read_text())
            if (
                prior["steps"] != steps
                or prior["repeats"] != repeats
                or prior.get("smoke", False) != args.smoke
                or prior.get("configured_pilot_samples") != pilot_samples
                or prior.get("configured_pilot_batch") != pilot_batch
            ):
                raise RuntimeError("Budget mismatch; use a new output folder.")
            if prior["status"] == "complete":
                print(group, "already complete", flush=True)
                continue
        start = time.time()
        specs, parameters, pilot = cases_and_parameters(group, pilot_samples, pilot_batch)
        shutil.copy2(ROOT / "experiments/config.py", folder / "config.py")
        json_write(folder / "config.json", specs)
        np.savez_compressed(folder / "parameters.npz", **parameters, **pilot)
        if pilot:
            signal = []
            for basis in cfg.BASIS_SETS:
                g = pilot["population_g"][basis_rows(8, basis)]
                sv = np.linalg.svd(g, compute_uv=False)
                rank = np.linalg.matrix_rank(g)
                mu = sv[-1] ** 2 if g.shape[0] >= g.shape[1] else 0.0
                se = pilot["g_mcse"][basis_rows(8, basis)]
                signal.append(
                    {
                        "basis": list(basis),
                        "p": len(g),
                        "rank": int(rank),
                        "mu": float(mu),
                        "min_parameter_singular_value": float(np.sqrt(mu)),
                        "g_mcse_frobenius": float(np.linalg.norm(se)),
                        "g_degree3_frobenius": float(np.linalg.norm(pilot["population_g"][45:])),
                        "signal_type": "exact"
                        if group == "quadratic-linear"
                        else "independent Rao-Blackwell Monte Carlo pilot",
                    }
                )
            json_write(folder / "signal.json", signal)
        meta = {
            "status": "running",
            "smoke": args.smoke,
            "configured_pilot_samples": pilot_samples,
            "configured_pilot_batch": pilot_batch,
            "group": group,
            "steps": steps,
            "repeats": repeats,
            "seed": cfg.SEED,
            "seed_lineage": "SeedSequence([SEED, group_id, role, repeat]); PCG64",
            "group_id": GROUP_IDS[group],
            "test_samples": cfg.N_TEST,
            "pilot_samples_per_stream": pilot_samples
            if group in ["quadratic-relu", "exp-tanh"]
            else 0,
            "pilot_streams": 2 if group in ["quadratic-relu", "exp-tanh"] else 0,
            "scale_pilot_samples": pilot_samples if group == "exp-tanh" else 0,
            "started_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "recursion": "One observation per iteration; old theta and old Jacobian gain; projection then same-observation Jacobian update.",
            "coupling": "Common Gaussian draws and base initializations across configurations within each repeat; independent repeats.",
            "stabilization": "Only declared parameter projection and fixed gain ridge; no direction clipping, exp clipping, restarts or early stopping.",
            "metrics": "Arithmetic mean of squared parameter errors, Monte Carlo SE, Euclidean quantiles, independent-test structural risk.",
            "slope_window": "Prespecified T/4<=t<=T; descriptive OLS log-log slope.",
            "tracking": "Exact population target for calibration/linear; pilot target for ReLU; not evaluated for exp.",
            "timing": "Compiled kernel wall seconds per repeat, excluding JIT/cache warm-up, pilot, test evaluation and plotting.",
        }
        json_write(old, meta)
        tasks = [
            (group, r, steps, checkpoints, specs, parameters, pilot, str(folder))
            for r in range(repeats)
        ]
        print(
            f"\nSTART {group}: {len(specs)} configurations x {repeats} repeats x {steps} observations",
            flush=True,
        )
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_repeat, t) for t in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                r, status = future.result()
                if count % max(1, repeats // 16) == 0 or count == repeats:
                    print(
                        f"{group}: {count}/{repeats} repeats; elapsed {time.time() - start:.1f}s",
                        flush=True,
                    )
        aggregate_group(folder, group, specs, parameters, steps, repeats, checkpoints, meta)
    from .plot import plot_study

    plot_study(outdir)


if __name__ == "__main__":
    main()
