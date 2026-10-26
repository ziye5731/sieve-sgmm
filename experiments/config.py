"""Central configuration for the SSGMM identification/rate study.

experiments.study snapshots this file in every result folder.
"""

SEED = 20261023
PARAMETER_SEED = 10
CALIBRATION_STEPS = 2**18
CALIBRATION_REPEATS = 512
MULTIVARIATE_STEPS = 2**18
MULTIVARIATE_REPEATS = 64
N_CHECKPOINTS = 97
N_TEST = 2**15
N_PILOT = 2**20
PILOT_BATCH = 2**14
N_JOBS = 4
OUTDIR = "results/ssgmm-study"
BASIS_SETS = [(1,), (0, 1), (0, 1, 2), (0, 1, 2, 3)]
# Calibrated polynomial model: (theta + curvature*theta^2)*x^2.
CALIBRATION_CONFIGS = [
    dict(
        label="h1_strong_a3",
        basis=(1,),
        gamma=1.0,
        a=3.0,
        eta=1.0,
        kappa=2 / 3,
        curvature=0.01,
        oracle=False,
    ),
    dict(
        label="h2_strong_a3",
        basis=(2,),
        gamma=1.0,
        a=3.0,
        eta=1.0,
        kappa=2 / 3,
        curvature=0.01,
        oracle=False,
    ),
    dict(
        label="h2_weaker_a3",
        basis=(2,),
        gamma=0.5,
        a=3.0,
        eta=1.0,
        kappa=2 / 3,
        curvature=0.01,
        oracle=False,
    ),
    dict(
        label="h2_weaker_a12",
        basis=(2,),
        gamma=0.5,
        a=12.0,
        eta=1.0,
        kappa=2 / 3,
        curvature=0.01,
        oracle=False,
    ),
    dict(
        label="h2_weaker_a9",
        basis=(2,),
        gamma=0.5,
        a=9.0,
        eta=1.0,
        kappa=2 / 3,
        curvature=0.01,
        oracle=False,
    ),
]
TRACKING_CONFIGS = [
    dict(
        label=f"{kind}_eta{int(eta * 10):02d}_{'population' if oracle else 'estimated'}",
        basis=(2,),
        gamma=1.0,
        a=3.0,
        eta=eta,
        kappa=kappa,
        curvature=curvature,
        oracle=oracle,
    )
    for kind, curvature in [("nonlinear", 0.01), ("linear", 0.0)]
    for eta, kappa in [(0.7, 0.6), (0.9, 0.6), (1.0, 2 / 3)]
    for oracle in [False, True]
]
# A second multivariate schedule is prespecified, before examining results.
# Harmonic a=2/min(h_identifying) uses exact G for linear first stage and
# independent pilot G for ReLU; the latter remains a numerical diagnostic.
MULTIVARIATE_SCHEDULES = ["power09", "harmonic"]
EXP_INITIALIZATIONS = ["zero", "near", "local"]
EXP_RADIUS = 2.0
EXP_LOCAL_RADIUS = 0.1
EXP_NEAR_SD = 0.025
# Predetermined extensions for the non-harmonic multivariate schedule.
EXTENSION_STEPS = [2**19, 2**20]
