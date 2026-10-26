#!/usr/bin/env python3
"""Replot saved SSGMM theory-study results without running experiments."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2471a3", "#c45c26", "#278565", "#9457a4", "#74787c"]
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "figure.constrained_layout.use": True,
    }
)


def save(fig, folder, name):
    fig.savefig(folder / (name + ".png"), dpi=180)
    fig.savefig(folder / (name + ".svg"))
    plt.close(fig)


def curve(ax, steps, mse, se, label, color, style="-"):
    ax.loglog(steps, mse, label=label, color=color, ls=style, lw=1.6)
    ax.fill_between(
        steps, np.maximum(mse - 1.96 * se, 1e-16), mse + 1.96 * se, color=color, alpha=0.10, lw=0
    )
    ax.set(xlabel="Training observations per repeat", ylabel="Mean squared parameter error")
    ax.grid(alpha=0.16)


def reference(ax, steps, anchor, power):
    xs = steps[steps >= steps[-1] / 16]
    ax.loglog(
        xs, anchor * (steps[-1] / xs) ** power, "k:", lw=1.1, label=f"T^(-{power:g}) reference"
    )


def plot_group(folder):
    specs = json.loads((folder / "config.json").read_text())
    meta = json.loads((folder / "metadata.json").read_text())
    group = meta["group"]
    with np.load(folder / "results.npz") as z:
        data = {k: z[k] for k in z.files}
    with np.load(folder / "trajectories.npz") as z:
        paths = {k: z[k] for k in z.files}
    steps = paths["steps"]
    figdir = folder / "figures"
    figdir.mkdir(exist_ok=True)

    def draw(ax, i, prefix="", color=None, style="-"):
        name = specs[i]["label"]
        curve(
            ax,
            steps,
            data[name + "_" + prefix + "param_mse"],
            data[name + "_" + prefix + "param_mse_mcse"],
            name,
            color or COLORS[i % len(COLORS)],
            style,
        )

    if group == "calibration":
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for i in range(len(specs)):
            draw(axes[0, 0], i)
            draw(axes[0, 1], i, "average_")
            name = specs[i]["label"]
            axes[1, 0].loglog(steps, steps * data[name + "_param_mse"], label=name, color=COLORS[i])
            err = np.abs(paths["theta"][i, :, -1, 0])
            ys = np.arange(1, len(err) + 1) / len(err)
            axes[1, 1].semilogx(np.sort(err), ys, label=name, color=COLORS[i])
        axes[0, 0].set_title("Last iterate")
        axes[0, 1].set_title("Averaged iterate")
        reference(axes[0, 0], steps, data[specs[1]["label"] + "_param_mse"][-1] * 2, 1)
        axes[1, 0].set(
            xlabel="Training observations",
            ylabel="T x parameter MSE",
            title="Scaled last-iterate error",
        )
        axes[1, 1].set(
            xlabel="Terminal Euclidean parameter error",
            ylabel="Empirical CDF",
            title="All repeated streams",
        )
        axes[0, 0].legend(fontsize=8, frameon=False)
        save(fig, figdir, "rates")
        fig, ax = plt.subplots(figsize=(7, 4))
        grid = np.linspace(-1, 1, 301)
        ax.plot(grid, np.zeros_like(grid), label="degree 1: no identifying signal", color=COLORS[4])
        for gamma, color in [(1.0, COLORS[0]), (0.5, COLORS[1])]:
            ax.plot(
                grid,
                np.sqrt(2) * gamma**2 * (grid + 0.01 * grid**2),
                label=f"degree 2, gamma={gamma}",
                color=color,
            )
        ax.axvline(0, color="k", ls=":", lw=0.8)
        ax.set(
            xlabel="Parameter theta",
            ylabel="Population moment",
            title="Analytic identification contrast",
        )
        ax.legend(frameon=False)
        save(fig, figdir, "identification")
    elif group == "tracking":
        for kind in ["nonlinear", "linear"]:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            for i, s in enumerate(specs):
                if not s["label"].startswith(kind + "_"):
                    continue
                color = COLORS[[0.7, 0.9, 1.0].index(s["eta"])]
                style = "--" if s["oracle"] else "-"
                draw(axes[0], i, color=color, style=style)
                draw(axes[1], i, "average_", color=color, style=style)
                if not s["oracle"]:
                    axes[2].loglog(
                        steps, data[s["label"] + "_tracking_fourth"], label=s["label"], color=color
                    )
            axes[0].set_title("Last iterate")
            axes[1].set_title("Averaged iterate")
            axes[2].set(
                xlabel="Training observations",
                ylabel="Fourth moment of tracking error",
                title="Exact population tracking target",
            )
            axes[0].legend(fontsize=7, frameon=False)
            axes[2].legend(fontsize=7, frameon=False)
            save(fig, figdir, kind + "-tracking")
    elif group.startswith("quadratic"):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig2, riskaxes = plt.subplots(1, 2, figsize=(12, 4.5))
        for i, s in enumerate(specs):
            row = 0 if s["eta"] < 1 else 1
            color = COLORS[i % 4]
            draw(axes[row, 0], i, color=color)
            draw(axes[row, 1], i, "average_", color=color)
            label = s["label"]
            riskaxes[row].loglog(
                steps, data[label + "_average_structural_risk"], color=color, label=label
            )
        for row, title in enumerate(
            ["Power steps: eta=0.9, a=0.1", "Harmonic steps: coefficient from signal diagnostic"]
        ):
            axes[row, 0].set_title(title + ", last")
            axes[row, 1].set_title(title + ", average")
            axes[row, 0].legend(fontsize=8, frameon=False)
            riskaxes[row].set(
                title=title,
                xlabel="Training observations",
                ylabel="Mean structural risk (averaged iterate)",
            )
            riskaxes[row].legend(fontsize=8, frameon=False)
        save(fig, figdir, "basis-rates")
        save(fig2, figdir, "structural-risk")
        signals = json.loads((folder / "signal.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = ["H" + "".join(map(str, s["basis"])) for s in signals]
        axes[0].bar(labels, [s["rank"] for s in signals], color=COLORS[:4])
        axes[0].axhline(14, color="k", ls=":", label="parameter dimension")
        axes[0].set(
            ylabel="Rank",
            title="Exact rank" if group == "quadratic-linear" else "Numerical pilot rank",
        )
        axes[0].legend(frameon=False, fontsize=8)
        axes[1].bar(labels, [s["mu"] for s in signals], color=COLORS[:4])
        axes[1].set(ylabel="mu = minimum eigenvalue of G.T G", title="Identification strength")
        save(fig, figdir, "signal")
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for i, s in enumerate(specs):
            col = ["zero", "near", "local"].index(s["initialization"])
            color = COLORS[i % 4]
            draw(axes[0, col], i, color=color)
            draw(axes[1, col], i, "average_", color=color)
        for col, name in enumerate(
            [
                "Near zero; radius 2 about zero",
                "Near truth; radius 2 about zero",
                "Near truth; radius 0.1 about truth",
            ]
        ):
            axes[0, col].set_title(name + "\nlast iterate")
            axes[1, col].set_title(name + "\naveraged iterate")
            axes[0, col].legend(fontsize=7, frameon=False)
        save(fig, figdir, "initialization-rates")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        with np.load(folder / "parameters.npz") as p:
            truth = p["truth"]
        for i, s in enumerate(specs):
            col = ["zero", "near", "local"].index(s["initialization"])
            err = np.linalg.norm(paths["theta"][i, :, -1] - truth, axis=1)
            axes[col].semilogx(
                np.sort(err),
                np.arange(1, len(err) + 1) / len(err),
                label=s["label"],
                color=COLORS[i % 4],
            )
        for col, name in enumerate(["zero", "near", "local"]):
            axes[col].set(
                title=name, xlabel="Terminal Euclidean parameter error", ylabel="Empirical CDF"
            )
            axes[col].legend(fontsize=7, frameon=False)
        save(fig, figdir, "terminal-distributions")
    caption = (
        f"# {group}: figure definitions\n\n{meta['repeats']} independent repeated streams; "
        f"{meta['steps']} training observations per configuration and repetition. "
        "Line plots report arithmetic mean squared parameter error. Shading, where shown, is "
        "mean +/- 1.96 Monte Carlo standard errors from repeated squared errors; these are "
        "pointwise normal approximations, not simultaneous confidence bands. "
        "All configurations within a repeat share Gaussian draws. Last and equally averaged "
        "iterates are shown separately. Reference slopes show scale only. "
        "Population-Jacobian curves use additional known population information. "
        "ReLU signal matrices are independent pilot estimates; numerical rank is not a population-rank proof. "
        "Exponential local projection uses knowledge of the simulated truth and is not claimed to verify the local-radius inequality.\n"
    )
    (folder / "figure_captions.md").write_text(caption)


def plot_study(outdir):
    for folder in sorted(Path(outdir).iterdir()):
        if (folder / "metadata.json").exists() and json.loads(
            (folder / "metadata.json").read_text()
        )["status"] == "complete":
            plot_group(folder)
            print("Plotted", folder.name, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outdir", type=Path)
    plot_study(parser.parse_args().outdir)
