"""real-sim disparity of a dataset row. real = mujoco with perturbations.py draws, twin = stored distance."""

import numpy as np

import engines
import perturbations
from controller import THETA_BOUNDS
from morphology import PARAM_BOUNDS

PARAM_NAMES = list(PARAM_BOUNDS)
THETA_COLS = [f"theta_{j}" for j in range(len(THETA_BOUNDS))]
EPISODE = 3.0
N_ROLLOUTS = 2
# the same draws for every row, so rows are comparable
PERTURBATIONS = perturbations.sample_combined(np.random.default_rng(0), N_ROLLOUTS)


# structured realities: the twin's own engine plus a lag and the perturbations.py multipliers. two fixed
# draws per rung, shared by every row. ranges and lags fixed before any result was seen.
#   R0 added after R1 and R2 were seen to fell nearly every twin gait (see SIM_DIFFS.md): no lag, quarter width
#   R1 mild: lag 2 control steps (10 ms), multiplier ranges at half width
#   R2 full: lag 4 control steps (20 ms), the perturbations.PERTURBATION_RANGES
RUNGS = {"R0": (0, 0.25), "R1": (2, 0.5), "R2": (4, 1.0)}


def reality_specs(rung: str) -> list:
    delay, width = RUNGS[rung]
    rng = np.random.default_rng(0)
    specs = []
    for _ in range(N_ROLLOUTS):
        s = {k: 1.0 + width * (rng.uniform(lo, hi) - 1.0) for k, (lo, hi) in perturbations.PERTURBATION_RANGES.items()}
        specs.append({**s, "delay": delay})
    return specs


def row_params(row) -> dict:
    return {k: float(row[k]) for k in PARAM_NAMES}


def row_theta(row) -> np.ndarray:
    return np.array([float(row[c]) for c in THETA_COLS])


def real_rollout(row, n_rollouts: int = N_ROLLOUTS, twin: str = "pybullet", rung: str = "cross"):
    """-> (mean distance, share that fell). each rollout uses one fixed draw. rung "cross" = mujoco with
    perturbations.py draws (twin is pybullet). rung R1/R2 = the twin's own engine with lag and multipliers."""
    params, theta = row_params(row), row_theta(row)
    d, fell = [], []
    if rung != "cross":
        for spec in reality_specs(rung)[:n_rollouts]:
            eng = engines.apply_reality(engines.make_engine(twin, params), spec)
            res = engines.rollout(eng, theta, EPISODE)
            if twin == "pybullet":
                engines.p.disconnect(eng.c)
            d.append(res.displacement)
            fell.append(res.fell)
        return float(np.mean(d)), float(np.mean(fell))
    for s in PERTURBATIONS[:n_rollouts]:
        eng = engines.MuJoCoEngine(params)
        perturbations.apply_perturbation(eng.m, **s)
        res = engines.rollout(eng, theta, EPISODE)
        d.append(res.displacement)
        fell.append(res.fell)
    return float(np.mean(d)), float(np.mean(fell))


def features(row_or_df):
    """scaled morphology params + scaled theta, in [0, 1]. phase is treated as linear."""
    lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES] + [b[0] for b in THETA_BOUNDS])
    hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES] + [b[1] for b in THETA_BOUNDS])
    cols = PARAM_NAMES + THETA_COLS
    X = np.asarray(row_or_df[cols], float)
    return (X - lo) / (hi - lo)


def real_rollout_row(params, theta, rung="cross", twin="pybullet"):
    """same as real_rollout but from raw params/theta."""
    return real_rollout({**params, **{f"theta_{j}": v for j, v in enumerate(theta)}}, twin=twin, rung=rung)
