# Reproducibility protocol

The source implementation is described in the [accompanying methods post](https://ziye5731.github.io/posts/2026/10/sieve-sgmm/). Study settings and numerical recursions were extracted from the original research checkout; source file hashes are recorded in [upstream-source.json](upstream-source.json). Seeds, data-generating processes, full-run budgets, per-observation updates, metrics, and plotting definitions are preserved. The new `--smoke` mode changes only the declared execution-check budgets and pilot sizes.

## Run and reproduce

From a source checkout, install `python -m pip install -e '.[study]'` and run `python -m unittest discover -s tests -v`. Use `python -m experiments.study --smoke --jobs 1` to validate all five groups with 512 observations, two repetitions and small independent pilots (4096 observations per stream, batches of 512). Smoke results are not scientific rate evidence.

For the full commands and the 2^18, 2^19, 2^20 budgets, see the [README](../README.md#reproduce-the-experiments). `--groups` selects calibration, tracking, quadratic-linear, quadratic-relu and exp-tanh. `--jobs` controls independent worker processes; each worker uses one BLAS/Numba thread. Overrides are recorded. `--steps` must be at least 8 and `--repeats` at least 2 so that the terminal slope and Monte Carlo SE can be computed.

Each new budget or changed source requires a new output directory. Completed repetitions resume under the identical source and protocol. The runner refuses a source-hash or budget mismatch. Replot with `python -m experiments.plot results/scalar`, or select an existing study root using `python -m experiments.study --plot-only --output results/scalar`.

The default scalar budget is 512 repetitions of 2^18 observations; the default multivariate budget is 64 repetitions of 2^18 observations. Full independent pilots use 2^20 instruments per stream and batches of 2^14. Fixed extensions of the multivariate groups use the separate 2^19 and 2^20 commands. These are computationally substantial runs. Parallel workers also require memory for the training draws and path aggregation.

The public core requires only NumPy. The experiment extra adds Numba, SciPy and Matplotlib. Source hashes and the exact installed environment are saved per run. Dependency ranges are supported installation bounds, not a lock of the historical October 23 environment. Historical raw outputs are not bundled, and cross-version floating-point bit identity is not guaranteed.

## Settings

All settings live in `experiments/config.py` and are copied into each output folder as `config.py`; the expanded per-case settings are `config.json`.

- **A/B, calibration:** `x=gamma*z+u`, `epsilon=0.5*u+0.5*(u^2-1)+nu`, `g=(theta+0.01*theta^2)*x^2`, true theta zero, `Theta=[-1,1]`, initial theta 0.5, initial S zero, ridge 1, offset 100. Independent standard Gaussian z,u,nu. Degree 1 is unidentifying. Degree 2 has `G=sqrt(2)*gamma^2`, `mu=2*gamma^4`. Compare gamma 1/a 3, gamma 0.5/a 3, gamma 0.5/a 12, and the optional boundary coefficient 9. Harmonic parameter steps; Jacobian exponent 2/3. Both local-region inequalities are analytically verified in the experiment plan. The quadratic error term makes `E[x^2*epsilon]=1`, so ordinary structural least squares has a nonzero population score at the truth.
- **C, tracking:** both curvature 0.01 and 0, exponent pairs (0.7,0.6), (0.9,0.6), and (1,2/3), estimated versus exact population-Jacobian gain. Exact moving-target tracking errors are recorded. The population gain uses information unavailable to the estimator and is a diagnostic, not a competitive method under equal information.
- **D, quadratic-linear/ReLU:** d_x=4, d_z=8; true symmetric quadratic A=I, b~N(0,4I), Gamma~N(0,1) from fixed parameter seed 10, with actual vectors saved. First stage h(Gamma'z)+c+v, c,v independent N(0,0.5I), structural error `0.5*sum(c)+e`. Full degree blocks H1,H01,H012,H0123 have 8,9,45,165 moments. Projection radius 20 about zero, true parameter inside; b initialized N(0,0.01I), A initialized zero. Power schedule a=0.1, eta=0.9, kappa=0.6, plus a prespecified harmonic diagnostic: `a=2/min(h_H012,h_H0123)`, kappa=2/3, shared across bases, with offset at least ceil(a). Linear-stage signal is exact; ReLU uses two independent Rao–Blackwell pilots, each with 2^20 instruments, batch size 2^14. Pilot-based coefficients do not establish the exact population threshold. The parameterization includes the factor 2 on off-diagonal quadratic gradients.
- **E, exp-tanh:** Gamma~N(0,0.1^2), `epsilon=0.25*sum(c)+0.1*e`; a separate 2^20-instrument pilot sets the true index variance to 0.25 using Var(x)=Var(tanh(Gamma'z))+I. Use exact exp without index clipping, a=0.1, eta=0.9, kappa=0.6, ridge 0.01, offset 100. Near-zero initialization is N(0,0.01I); near-truth initialization has coordinate SD 0.025. Compare radius 2 about zero with radius 0.1 about the truth, projecting initial values when necessary. Local projection is an explicitly idealized diagnostic and does not certify the radius inequality. Numerical failures are recorded and never silently excluded.

The polynomial and quadratic groups meet the stated finite fourth-moment conditions. This is not a claim that all exploratory exponential configurations satisfy the nonlinear local-region condition.

## Numerical equivalence

`experiments/kernels.py` uses the same single-observation recursion as `ssgmm.SSGMM` (itself checked against the original research implementation during extraction). It evaluates residual and gradient at the pre-update parameter, takes the parameter step with the previous Jacobian, projects, then updates the Jacobian with the same observation, and averages all updated parameters.

The compiled implementation caches `S.T@S` using its exact rank-one update, rebuilding from S every 256 observations and if a Cholesky factorization fails. This changes arithmetic cost, not the mathematical gain or update frequency. There is no extra ridge, direction clipping, exponential clipping, burn-in selection, restart, or early stopping. Gram reconstruction discrepancies, projection hits, maximum parameter distance, maximum exponential index, and failed-step indices are retained. Tests compare 513 sequential iterates, their averages and final Jacobians against the standalone NumPy implementation, crossing two Gram rebuilds. Hermite ordering, positive-definite solves and analytic Gaussian quadratic moments are checked independently.

## Seeds and evaluation

PCG64 streams derive from `SeedSequence([20261023, group_id, role, repeat])`. Group IDs are calibration=1, tracking=2, quadratic-linear=3, quadratic-relu=4, exp-tanh=5. Role 0 is training; role 1 initialization; role 11 index-scale pilot; role 12 Jacobian pilots; role 20 independent test. Comparisons share Gaussian draws within each repeat. Repeats and roles are independent.

Every group uses an independent fixed test sample of 2^15 observations. Primary metrics are arithmetic parameter MSE and structural risk against the true structural function. Euclidean error quantiles and outcome prediction MSE are also retained. Outcome MSE is auxiliary because endogeneity leaves a nonzero residual cross term. Monte Carlo SEs are computed from squared parameter errors, not from Euclidean errors. Reported slopes fit the prespecified terminal factor-of-four window; they are descriptive, not tests of an exact exponent or a proof of an asymptotic rate.

## Outputs and repository hygiene

`results/ssgmm-study/` follows the existing ignored-results convention. At the root: source snapshots with SHA256 hashes, Git state, Python/platform information and exact installed requirements. Within each group:

- `config.py`, `config.json`, `metadata.json`: snapshot, expanded protocol, run status and budgets.
- `parameters.npz`, `signal.json`: actual fixed truth, first-stage matrix, pilot batches/SEs, and signal diagnostics. ReLU/exp pilot ranks are numerical evidence only.
- `repeats/repeat_NNNN.npz`: per-repetition raw/averaged parameter paths, diagnostics, final Jacobian state, effective initialization, failure indices and elapsed times. These files permit repeat-level audits and bootstrap.
- `trajectories.npz`: combined arrays in (configuration, repeat, checkpoint, parameter) order; diagnostics columns are tracking squared error, cumulative projection hits, maximum distance from truth, maximum exp index, and maximum relative Gram-cache discrepancy.
- `results.npz`: flat algorithm-prefixed aggregates compatible with the repository's saved-result convention. Includes last and averaged parameter MSE/MCSE, Euclidean means/SD/quantiles, structural risk and outcome MSE.
- `test_data.npz`: independent test data sufficient to recompute evaluation.
- `summary.csv`, `summary.txt`: terminal statistics, Monte Carlo SE, descriptive slopes, failures and timings.
- `figures/`, `figure_captions.md`: plots regenerated solely from saved data, PNG and SVG versions, and statistical definitions.

No output from this study modifies or overwrites historical result folders. Source files and result snapshots include the protocol, while `.venv` and compiled caches stay out of Git.
