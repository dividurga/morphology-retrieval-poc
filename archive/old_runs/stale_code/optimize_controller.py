"""CMA-ES gait optimization: find theta maximizing forward displacement for one morphology."""

import argparse
import importlib
import json
from concurrent.futures import ProcessPoolExecutor

import cma
import numpy as np

import backend_pybullet
import controller
import morphology

FALL_PENALTY = 6.0
BACKWARD_PENALTY = 10.0  # -abs(d-target) alone doesn't single out "walked backward" as
                         # categorically wrong -- a bad forward overshoot scores exactly
                         # as badly as the same-magnitude backward walk, so a seed that
                         # stumbles into backward stepping has no pressure pushing it back
                         # toward forward. Applied only in the target-conditioned branch:
                         # plain maximize-distance is already monotonic in displacement,
                         # backward is already unambiguously bad there without this.
CMA_POPSIZE = 24  # was 12 (tuned for the old 7-dim theta). Doubled since theta is now
                  # 13-dim (asymmetric left/right) -- popsize=20/maxiter=80 measurably
                  # helped but didn't fully stabilize target=3.0/6.0 across seeds; going
                  # a bit higher on both since compute isn't the binding constraint here.
CMA_MAXITER = 100
DEFAULT_SEED = 1  # MUST NOT be 0: pycma treats a falsy seed as "unseeded" and reseeds
                  # numpy's global RNG from OS entropy (see cma/evolution_strategy.py,
                  # `if not opts['seed']: np.random.seed()`) -- 0 silently breaks
                  # reproducibility for every caller of optimize() with the default seed.
EPISODE_DURATION = 8.0
TRAIN_PERTURBATION_SEED_OFFSET = 100_000  # derives train_perturbation_seed from
                                           # controller seed when not given explicitly


def fitness(model, theta: np.ndarray, duration: float = EPISODE_DURATION,
            fall_penalty: float = FALL_PENALTY, target_distance: float | None = None,
            backend=backend_pybullet) -> float:
    """Scalar score CMA-ES maximizes. NOT the headline metric reported later —
    just used to steer the search away from pathological "lunge and faceplant" gaits.

    result = backend.simulate(model, theta, duration)
    target_distance=None: maximize displacement (unchanged default behavior).
    target_distance set: match it instead, score = -abs(displacement - target_distance),
    with an extra BACKWARD_PENALTY if displacement < 0 (see constant's comment).
    Either way, fall_penalty is subtracted on top if result.fell.
    """
    result = backend.simulate(model, theta, duration)
    if target_distance is None:
        score = result.displacement
    else:
        score = -abs(result.displacement - target_distance)
        if result.displacement < 0:
            score -= BACKWARD_PENALTY
    if result.fell:
        score -= fall_penalty
    return score


def fitness_dr(models: list, theta: np.ndarray, duration: float = EPISODE_DURATION,
               fall_penalty: float = FALL_PENALTY, target_distance: float | None = None,
               backend=backend_pybullet) -> float:

    """We bring perturbed models into the training loop"""

    total_score = 0.0
    for model in models:
        score = fitness(model, theta, duration, fall_penalty, target_distance, backend)
        total_score += score

    return total_score / len(models)


AMPLITUDE_INDICES = (2, 5, 8, 11)  # a_hip_left, a_knee_left, a_hip_right, a_knee_right
                                   # in controller.PARAM_NAMES -- MUST track that layout;
                                   # left-only scaling here would re-bias the init toward
                                   # an asymmetric gait regardless of target_distance
SCALE_CLIP = (0.2, 3.0)
PROBE_MIN_DISPLACEMENT = 0.1  # below this, the baseline is too degenerate (near-zero or
                              # a fall) to scale from at all
PROBE_N_SEEDS = 3  # a single-seed probe can land in a bad basin by chance (confirmed:
                   # 5 seeds of unconditioned PyBullet optimization gave 1.37, 0.82, 1.60,
                   # 7.84, 0.91 -- 4/5 landed low, one landed far better than MuJoCo's own
                   # baseline. A good gait is clearly findable; single-seed was just
                   # unlucky 80% of the time here). Best-of-3 is cheap insurance, not a
                   # full best-of-N -- this is a starting-point heuristic, not the result.
PROBE_SEED_OFFSET = 50_000  # distinct from TRAIN_PERTURBATION_SEED_OFFSET (100_000) --
                            # keeps the probe's seed stream from colliding with the DR
                            # perturbation-draw stream derived from the same base seed


def _target_aware_x0(params: dict, seed: int, target_distance: float, duration: float,
                     backend=backend_pybullet, workers: int = 1) -> np.ndarray:
    """x0 used to always be DEFAULT_THETA regardless of target_distance, so every
    low-target search had to discover "walk slower" from scratch -- which is why
    target=6.0 (close to a natural cruising pace) converged easily across seeds and
    target=3.0 (further from it) didn't.

    DEFAULT_THETA itself is useless as a reference point: run untouched (no
    optimization at all) it falls immediately (negative displacement) -- it's a generic
    starting guess for the optimizer, not a working controller. A reduced-budget probe
    was tried and dropped: at 1/4 budget (popsize 8, maxiter 15) it only found 0.096m,
    itself below PROBE_MIN_DISPLACEMENT -- single-seed CMA-ES needs close to full budget
    to reliably find a working gait at all (same noise this whole fix exists to address).
    So: run PROBE_N_SEEDS full max-distance optimize() calls and keep the best to get a
    real working gait for this morphology, then scale stride amplitude from THAT baseline
    toward target_distance -- displacement scales roughly with stride length, holding
    frequency fixed, as a first-order guess. CMA-ES still does the actual search from
    here; this only moves the starting point into the right neighborhood. Costs
    PROBE_N_SEEDS extra full optimize() calls per target-conditioned call -- accepted,
    compute isn't the constraint here, and a single-seed probe was confirmed too
    unreliable (see above) to trust as the sole anchor for the whole search.
    """
    probe_seeds = [seed + i * PROBE_SEED_OFFSET for i in range(PROBE_N_SEEDS)]
    baselines = [optimize(params, seed=s, duration=duration, target_distance=None, backend=backend,
                          workers=workers)
                 for s in probe_seeds]
    baseline = max(baselines, key=lambda b: b["fitness"])
    baseline_theta = np.array(baseline["theta"])
    baseline_distance = baseline["fitness"]  # = displacement when target_distance=None
    bounds = controller.THETA_BOUNDS
    if baseline_distance <= PROBE_MIN_DISPLACEMENT:
        return controller.DEFAULT_THETA.copy()
    scale = np.clip(target_distance / baseline_distance, *SCALE_CLIP)
    x0 = baseline_theta.copy()
    for idx in AMPLITUDE_INDICES:
        lo, hi = bounds[idx]
        x0[idx] = np.clip(x0[idx] * scale, lo, hi)
    return x0


_W = {}  # per-process state for the worker pool


def _init_worker(backend_name, params, dr_params, duration, fall_penalty, target_distance):
    """Models hold a pybullet/mujoco handle and cannot be pickled, so each worker builds its own."""
    backend = importlib.import_module(backend_name)
    models = [backend.make_model(params)] + [backend.make_model(pp) for pp in dr_params]
    _W.update(backend=backend, models=models, duration=duration, fall_penalty=fall_penalty,
              target_distance=target_distance)


def _eval_worker(x):
    w = _W
    if len(w["models"]) > 1:
        return -fitness_dr(w["models"], x, w["duration"], w["fall_penalty"], w["target_distance"],
                           w["backend"])
    return -fitness(w["models"][0], x, w["duration"], w["fall_penalty"], w["target_distance"],
                    w["backend"])


def optimize(params: dict, seed: int = DEFAULT_SEED, popsize: int = CMA_POPSIZE,
             maxiter: int = CMA_MAXITER, duration: float = EPISODE_DURATION,
             fall_penalty: float = FALL_PENALTY, m_perturbations: int = 0,
             train_perturbation_seed: int | None = None,
             target_distance: float | None = None, backend=backend_pybullet,
             workers: int = 1) -> dict:
    """backend: any module exposing make_model(params) / simulate(model, theta, duration),
    i.e. `backend_pybullet` (the training sim) or `backend_mujoco` (the real one).

    workers > 1 evaluates each CMA-ES generation in a process pool. same result, less wall time."""
    # DR training is not ported to the two-engine pipeline yet. perturbations.py only knows
    # how to perturb a mujoco MjModel, and the DR arm has to perturb the TRAINING sim (pybullet).
    if m_perturbations > 0:
        raise NotImplementedError("DR arm: port perturbations.py to the pybullet engine first")
    dr_params = []

    bounds = controller.THETA_BOUNDS
    if target_distance is not None:
        x0 = _target_aware_x0(params, seed, target_distance, duration, backend, workers)
    else:
        x0 = controller.DEFAULT_THETA.copy()
    lowers = [b[0] for b in bounds]
    uppers = [b[1] for b in bounds]
    stds0 = [(hi - lo) / 4.0 for lo, hi in bounds]

    es = cma.CMAEvolutionStrategy(
        x0, 1.0,
        {
            "bounds": [lowers, uppers],
            "CMA_stds": stds0,
            "popsize": popsize,
            "maxiter": maxiter,
            "seed": seed,
            "verbose": -9,
        },
    )

    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(
            backend.__name__, params, dr_params, duration, fall_penalty, target_distance))
    else:
        _init_worker(backend.__name__, params, dr_params, duration, fall_penalty, target_distance)
    try:
        while not es.stop():
            solutions = es.ask()
            xs = [np.array(x) for x in solutions]
            losses = list(pool.map(_eval_worker, xs)) if pool else [_eval_worker(x) for x in xs]
            es.tell(solutions, losses)
    finally:
        if pool:
            pool.shutdown()

    best_theta = np.array(es.result.xbest)
    best_fitness = -es.result.fbest
    return {
        "theta": best_theta.tolist(),
        "fitness": best_fitness,
        "seed": seed,
        "popsize": popsize,
        "maxiter": maxiter,
        "fall_penalty": fall_penalty,
        "params": params,
        "m_perturbations": m_perturbations,
        "target_distance": target_distance,
        "backend": backend.__name__,
        "training_regime": "dr" if m_perturbations > 0 else "nominal",
    }


def optimize_best_of_n(params: dict, seeds: list, **optimize_kwargs) -> dict:
    """Run optimize() once per seed in `seeds`, keep the result with the highest fitness."""
    attempts = [optimize(params, seed=s, **optimize_kwargs) for s in seeds]
    best = max(attempts, key=lambda r: r["fitness"])
    best["seed_group"] = list(seeds)
    return best


def optimize_medoid_of_n(params: dict, seeds: list, **optimize_kwargs) -> dict:
    """Run optimize() once per seed, then keep whichever theta is most 'typical' -- closest
    (in normalized theta-space) to the others -- rather than whichever has highest fitness.
    A solution several independent seeds converge near is more likely a broad, stable
    attractor; a solution only one seed finds is more likely a narrow, fragile spike."""
    attempts = [optimize(params, seed=s, **optimize_kwargs) for s in seeds]
    thetas = np.array([a["theta"] for a in attempts])

    bounds = controller.THETA_BOUNDS
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    normed = (thetas - lo) / (hi - lo)

    diffs = normed[:, None, :] - normed[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    total_dist_to_others = dists.sum(axis=1)
    medoid_idx = int(np.argmin(total_dist_to_others))

    result = attempts[medoid_idx]
    result["seed_group"] = list(seeds)
    result["selection_method"] = "medoid"
    result["all_fitnesses"] = [a["fitness"] for a in attempts]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--morphology_id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", default="results/controller_nominal.json")
    args = parser.parse_args()

    if args.morphology_id is not None:
        raise NotImplementedError("morphology lookup table doesn't exist yet -- comes with sweep.py")

    result = optimize(morphology.NOMINAL_PARAMS, seed=args.seed)
    print(f"best fitness: {result['fitness']:.3f}")
    print(f"best theta: {result['theta']}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved {args.out}")
