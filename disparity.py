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


def row_params(row) -> dict:
    return {k: float(row[k]) for k in PARAM_NAMES}


def row_theta(row) -> np.ndarray:
    return np.array([float(row[c]) for c in THETA_COLS])


def real_rollout(row, n_rollouts: int = N_ROLLOUTS):
    """-> (mean distance, share that fell). each rollout uses one fixed perturbation draw."""
    params, theta = row_params(row), row_theta(row)
    d, fell = [], []
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
