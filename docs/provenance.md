# Source provenance

The reference method and reproducibility code were extracted from [ziye5731/iv-regression](https://github.com/ziye5731/iv-regression), commit `f063275`, under its MIT license. The original copyright notice is preserved in `LICENSE`. [upstream-source.json](upstream-source.json) records the full upstream commit and SHA256 digests of the local source files used.

The standalone estimator implements the original `iv_sim.algorithms.SSGMM` recursion without inheriting the exploratory GMM ablation classes or simulation configuration registry. It adds a direct per-observation API, explicit shapes and finite-value validation, arbitrary fixed projection centers, and array convenience methods. Defaults match the general reference estimator. It enforces the method's separated step exponents and beta <= 1 at initialization.

Hermite features retain the original normalized polynomial formulas and column ordering. Linear and symmetric-quadratic model implementations retain the original parameterization. The public exponential model evaluates the exact exponential, as in the final study; it does not import the clipped exploratory exponential model.

The fixed study configuration, Numba kernels, runner, aggregation and plotting were carried over from the October 23 study. Changes connect imports to this package, move the experiment entry points into `experiments/`, add a declared small-budget smoke mode and argument validation, and adapt source/environment snapshots to the new tree. Run provenance supports source archives without Git and preserves the first environment snapshot on resume.

The unit tests compare 513 sequential updates of scalar, multivariate quadratic and exponential models, spanning two cached-Gram rebuilds. They also check Hermite ordering, positive-definite solves and analytic Gaussian moments against independent quadrature. During extraction, the standalone estimator is additionally compared directly with the untouched original research estimator on the same streams. These are implementation equivalence checks, not proofs of the statistical assumptions.

The new repository starts with clean history. Research notebooks, exploratory comparison algorithms, obsolete theory drafts, unpublished data, generated result folders, credentials, editor settings and virtual environments are not imported. The methods post is the primary theory reference; older drafts are not presented as updated claims of this release.
