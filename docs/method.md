# Method and scope

This implementation accompanies [Streaming Sieve GMM for Nonlinear Instrumental Variable Regression](https://ziye5731.github.io/posts/2026/10/sieve-sgmm/), published October 23, 2026. That post is the primary theory reference. Earlier research drafts discuss additional results and exploratory variants; they are not substituted for the post's assumptions or conclusions here.

## Observation model and moments

Each update observes one ordinary tuple `(z, x, y)`. The structural model is `y = g(theta_star; x) + epsilon`, with `E[epsilon | z] = 0`. Endogeneity between `x` and `epsilon` is permitted. Choose a fixed finite vector of instrument functions `psi(z)`. The sample moment uses residual `g(theta; x) - y`, and its Jacobian is `psi(z) gradient(theta; x)^T`.

Selecting normalized Hermite degree blocks 0–3 includes all interactions within each selected total degree. The feature ordering is preserved from the research implementation: constant; all linear terms; all normalized squares followed by pairs; normalized cubics followed by ordered H2/H1 pairs and then triples. Normalization is orthonormal for independent standard Gaussian instruments. Non-Gaussian inputs can still be evaluated, but the Gaussian orthogonality formulas no longer follow automatically.

## Ordered update

At observation t, both the sample residual and structural gradient are evaluated at the parameter from t-1. The parameter direction uses the Jacobian state from t-1. The parameter is updated and projected onto its declared Euclidean ball; only then is the same observation absorbed into the Jacobian state. The average contains updated parameters 1 through T. There is no burn-in or tail selection.

This historical-state ordering makes the gain predictable. Using the newly updated Jacobian for the current parameter direction changes the estimator and its conditional-mean calculation. With the default zero initial Jacobian, the first parameter direction is zero.

The gain solves `(S.T @ S + ridge * I) direction = S.T @ moment`. The fixed positive ridge damps the gain but does not change the population moment root. The core implementation recomputes the Gram matrix each step. The optional compiled study kernels use the algebraically equivalent rank-one Gram update, rebuilding every 256 observations and on a failed positive-definite solve. Sequential equivalence tests span two rebuild boundaries.

## Schedules and projection

The schedules are `alpha_t = a / (t + offset)^eta` and `beta_t = b / (t + offset)^kappa`, with `0.5 < kappa < eta <= 1`. The public estimator requires a finite positive ridge and projection radius, and a first beta no greater than one. All subsequent beta values are smaller. Projection also applies to the initial parameter.

The package defaults reproduce the general reference implementation's schedules (`a=0.05`, `eta=0.8`, `b=1`, `kappa=0.6`, `offset=10`, `ridge=0.01`, radius 20). They are starting values, not universal theorem-certified settings. Experiment configurations set their own explicit values.

The projection center defaults to zero. A small ball about the simulated truth is used only in a clearly marked experimental diagnostic; an applied estimator cannot assume that unknown center. Exponential predictions are exact, with no index clipping. The public estimator raises on nonfinite evaluations, without committing that observation. Study kernels retain failure indices and NaN paths; aggregation does not silently discard failed repetitions.

## Identification and rate claims

The selected population Jacobian `G = E[psi(z) gradient(theta_star; x)^T]` must have full column rank for the post's regular local identification result. More moments than parameters is necessary for that rank condition, but not sufficient. A rank condition is local and does not by itself prove a unique root over a large parameter region.

The finite-time rate result additionally requires i.i.d. data, correct finite-dimensional specification, a fixed basis, uniform fourth moments, population smoothness, and a compact convex region meeting the stated curvature-radius restriction. Write `mu = lambda_min(G.T @ G)` and `h = mu / (mu + ridge)`. The sufficient exponent is

```math
\rho = \min\{\eta,\; 2\kappa,\; 4(\eta-\kappa)\}.
```

For `eta < 1`, the post bounds last-iterate and full-history-average parameter MSE by `O(T^-rho)` under its assumptions. With harmonic parameter steps (`eta=1`), the coefficient `a*h` also matters. In particular, `kappa=2/3` and `a*h>1` give the stated `O(1/T)` MSE specialization. A chosen power schedule, numerical rank, or observed log-log slope alone does not verify these hypotheses.

The post does not establish a central limit theorem, efficient covariance, growing/adaptively chosen sieves, dependent-data validity, or global nonlinear convergence from arbitrary initialization. This repository does not calculate confidence intervals for an applied structural parameter. Repeated-stream Monte Carlo uncertainty in the study describes simulation averages rather than such an inferential procedure.
