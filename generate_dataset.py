"""dataset by post-hoc relabeling. per morphology: short cma-es restarts on the twin (pybullet,
full setup ladder), every evaluated gait logged. a gait that finishes the episode upright and achieves
distance d is a certified optimum for target d (fitness 0). see README.

    uv run python generate_dataset.py --morphologies 12 --out data_pilot
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cma
import numpy as np
from scipy.stats import qmc

import controller
import engines
import morphology

EPISODE = 3.0  # s. arbitrary like the old 8 s, but cheap. distances scale with it.
FALL_PENALTY, BACKWARD_PENALTY = 6.0, 10.0  # as optimize_controller
TARGETS = [None, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, None]  # one restart each. None = maximise distance. denser at the long, thin end
POPSIZE, MAXITER = 12, 40
BIN = 0.05  # m
MAX_DIST = 3.0  # bins cover 0 to this


# a small body family: three free parameters, everything else at morphology.NOMINAL_PARAMS. left and right legs are tied
# (asymmetric legs tilt the torso before the gait starts, and make standing very hard). +-30% around nominal.
FREE = {"thigh_length": 0.3, "shank_length": 0.3, "torso_size": 0.3}


def sample_morphologies(n: int, seed: int) -> list:
    """n bodies, Latin hypercube over FREE. returns full params dicts (every key morphology.build_biped_xml needs)."""
    u = qmc.LatinHypercube(d=len(FREE), seed=seed).random(n)
    nom = morphology.NOMINAL_PARAMS
    out = []
    for row in u:
        f = {k: nom[f"{k}_left" if k != "torso_size" else k] * (1 + w * (2 * v - 1)) for (k, w), v in zip(FREE.items(), row)}
        p = dict(nom)
        for side in ("left", "right"):
            p[f"thigh_length_{side}"], p[f"shank_length_{side}"] = f["thigh_length"], f["shank_length"]
        p["torso_size"] = f["torso_size"]
        out.append(p)
    return out


def search_morphology(task):
    """one morphology: restarts, log every evaluation. returns arrays."""
    mid, params, seed, kind = task
    eng = engines.make_engine(kind, params)
    lo = [b[0] for b in controller.THETA_BOUNDS]
    hi = [b[1] for b in controller.THETA_BOUNDS]
    log_theta, log_d, log_fell, log_ft = [], [], [], []

    def evaluate(theta, target):
        r = engines.rollout(eng, np.asarray(theta), EPISODE)
        log_theta.append(np.asarray(theta)); log_d.append(r.displacement)
        log_fell.append(r.fell); log_ft.append(r.fell_time if r.fell else np.nan)
        if target is None:
            score = r.displacement
        else:
            score = -abs(r.displacement - target) - (BACKWARD_PENALTY if r.displacement < 0 else 0.0)
        return score - (FALL_PENALTY if r.fell else 0.0)

    t0 = time.time()
    for i, target in enumerate(TARGETS):
        es = cma.CMAEvolutionStrategy(controller.DEFAULT_THETA, 1.0, {
            "bounds": [lo, hi], "CMA_stds": [(h - l) / 4 for l, h in zip(lo, hi)],
            "popsize": POPSIZE, "maxiter": MAXITER, "seed": seed * 100 + i + 1, "verbose": -9})
        while not es.stop():
            xs = es.ask()
            es.tell(xs, [-evaluate(x, target) for x in xs])
    return mid, params, np.array(log_theta), np.array(log_d), np.array(log_fell), np.array(log_ft), time.time() - t0


def relabel(theta, d, fell):
    """per-bin representative = the non-falling gait closest to the bin centre. every non-falling gait
    stays in the archive, so another rule can be applied later."""
    ok = (~fell) & (d > 0)
    reps = {}
    for i in np.where(ok)[0]:
        b = int(round(d[i] / BIN))
        if b not in reps or abs(d[i] - b * BIN) < abs(d[reps[b]] - b * BIN):
            reps[b] = i
    return reps


def build_csv(root: str):
    """one row per (morphology, bin). empty bins are kept, with filled=False and no controller."""
    import glob
    import pandas as pd
    rows = []
    for f in sorted(glob.glob(f"{root}/archive/*.npz")):
        z = np.load(f)
        mid = int(f.split("/m")[-1][:4])
        pn = [str(k) for k in z["param_names"]]
        pv = dict(zip(pn, z["params"].tolist()))
        for b, i in enumerate(z["rep_index"]):
            row = {"morph_id": mid, **pv, "bin_center": b * BIN, "filled": bool(i >= 0), "episode": EPISODE,
                   "training_regime": "nominal"}
            if i >= 0:
                row["distance"] = float(z["distance"][i])
                row.update({f"theta_{j}": float(v) for j, v in enumerate(z["theta"][i])})
            rows.append(row)
    pd.DataFrame(rows).to_csv(f"{root}/dataset.csv", index=False)
    print(f"wrote {root}/dataset.csv: {len(rows)} (morphology, bin) rows, {sum(r['filled'] for r in rows)} filled", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--morphologies", type=int, default=12)
    ap.add_argument("--out", default="data_pilot")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", default="pybullet", choices=["pybullet", "mujoco"], help="the twin the dataset is made in")
    args = ap.parse_args()
    os.makedirs(f"{args.out}/archive", exist_ok=True)
    morphs = sample_morphologies(args.morphologies, args.seed)
    todo = [(i, m, args.seed, args.engine) for i, m in enumerate(morphs) if not os.path.exists(f"{args.out}/archive/m{i:04d}.npz")]
    print(f"{len(morphs)} morphologies, {len(todo)} to do. episode {EPISODE}s, {len(TARGETS)} restarts x {POPSIZE}x{MAXITER}", flush=True)
    names = list(morphology.PARAM_BOUNDS)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for mid, params, th, d, fell, ft, dt in pool.map(search_morphology, todo):
            reps = relabel(th, d, fell)
            rep_index = np.full(int(round(MAX_DIST / BIN)) + 1, -1)  # -1 = bin not reached by this morphology
            for b, i in reps.items():
                if b < len(rep_index):
                    rep_index[b] = i
            np.savez_compressed(f"{args.out}/archive/m{mid:04d}.npz", theta=th, distance=d, fell=fell, fell_time=ft,
                                rep_index=rep_index, params=np.array([params[k] for k in names]),
                                param_names=np.array(names))
            walk = (~fell) & (d > 0.3)
            print(f"m{mid:04d}: {len(d)} evals, {int((~fell).sum())} upright, {int(walk.sum())} > 0.3 m, best {d[~fell].max() if (~fell).any() else float('nan'):.2f} m, "
                  f"{len(reps)} bins of {BIN} m filled, {dt:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)
    build_csv(args.out)
