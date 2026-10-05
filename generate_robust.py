"""regenerated dataset. per (morphology, target T, arm): train a gait with CMA-ES on the pybullet twin, then evaluate it
once in held-out worlds. one jsonl row per (morphology, T, arm).

arms (each picks its restart by its own objective, never by an evaluation world):
  nominal  cost = |d - T| on the nominal twin
  robust   cost = mean_i c_i + LAMBDA * std_i c_i over M draws of friction/mass/motor/damping (perturbations.py ranges)
           c_i = |d_i - T| (+ BACKWARD_PENALTY if d_i < 0) (+ FALL_PENALTY if fell)
           the draws are shared by a whole CMA-ES generation (common random numbers), fresh every generation.
the nominal arm uses the same fall and backward penalties, so the arms differ only in the world(s) they train on.

evaluation (draws never used in training: training rng seeds from 10_000+, evaluation from EVAL_SEED):
  sim:  pybullet nominal, and N_EVAL held-out DR draws
  real: real_pb_mf (pybullet + ModelForm), real_mj (mujoco), real_mj_mf (mujoco + the same ModelForm)
every rollout, in training and evaluation, starts with engines.SETTLE seconds holding the spawn pose.
nothing is tuned against any real.

    uv run python generate_robust.py --bodies 50 --workers 10 --out tubey/rows.jsonl
    (writes tubey/rows.jsonl, then tubey/dataset.csv: one flat row per (morphology, target, arm).)

sim = the pybullet twin, the world the gait was trained in. reals = never trained on. a target is a column, not a
bin: a query is a nearest-neighbour lookup on the sim distance.
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import cma
import numpy as np
import pybullet as p

import controller
import engines
import generate_dataset as G
import perturbations

N_TARGETS, T_LO, T_HI = 3, 0.3, 2.0  # per body: one target drawn uniformly from each of 3 equal strata of [T_LO, T_HI]
ARMS = ("nominal", "robust")
M_DRAWS, LAMBDA = 5, 1.0  # training draws per evaluation, weight of the spread term
N_EVAL, EVAL_SEED = 10, 777
RESTARTS, POPSIZE = 2, G.POPSIZE
MAXITER = int(os.environ.get("ROBUST_MAXITER", G.MAXITER))
EPISODE = G.EPISODE
IDENTITY = {"friction": 1.0, "mass": 1.0, "motor": 1.0, "damping": 1.0, "delay": 0}


def targets_for(mid):
    rng = np.random.default_rng(5_000 + mid)
    edges = np.linspace(T_LO, T_HI, N_TARGETS + 1)
    return [round(float(rng.uniform(edges[i], edges[i + 1])), 3) for i in range(N_TARGETS)]


def draws(rng, m):
    return [{**{k: float(rng.uniform(lo, hi)) for k, (lo, hi) in perturbations.PERTURBATION_RANGES.items()}, "delay": 0}
            for _ in range(m)]


def cost(res, T):
    c = abs(res.displacement - T)
    c += G.BACKWARD_PENALTY if res.displacement < 0 else 0.0
    return c + (G.FALL_PENALTY if res.fell else 0.0)


def objective(eng, theta, T, specs, arm):
    cs = []
    for sp in specs:
        engines.apply_reality(eng, sp)
        cs.append(cost(engines.rollout(eng, theta, EPISODE), T))
    cs = np.array(cs)
    return float(cs.mean() + (LAMBDA * cs.std() if arm == "robust" else 0.0))


def train(eng, T, arm, seed):
    lo = [b[0] for b in controller.THETA_BOUNDS]
    hi = [b[1] for b in controller.THETA_BOUNDS]
    rng = np.random.default_rng(10_000 + seed)
    cands = []
    for r in range(RESTARTS):
        es = cma.CMAEvolutionStrategy(controller.DEFAULT_THETA, 1.0, {
            "bounds": [lo, hi], "CMA_stds": [(h - l) / 4 for l, h in zip(lo, hi)],
            "popsize": POPSIZE, "maxiter": MAXITER, "seed": seed * 100 + r + 1, "verbose": -9})
        while not es.stop():
            specs = [IDENTITY] if arm == "nominal" else draws(rng, M_DRAWS)
            xs = es.ask()
            es.tell(xs, [objective(eng, np.asarray(x), T, specs, arm) for x in xs])
        cands.append(np.asarray(es.result.xbest if arm == "nominal" else es.result.xfavorite))
    sel = [IDENTITY] if arm == "nominal" else draws(rng, 10)  # restart chosen on fresh draws of its own training world
    obj = [objective(eng, th, T, sel, arm) for th in cands]
    k = int(np.argmin(obj))
    return cands[k], obj[k]


def stats(ds, fs):
    return {"mean": float(np.mean(ds)), "std": float(np.std(ds)), "fell": float(np.mean(fs)), "d": [float(x) for x in ds]}


def evaluate(params, theta):
    """one gait in the sim and in three reals. all single nominal rollouts except sim_draws.
    sim_nominal / sim_draws: the pybullet twin (nominal / N_EVAL held-out DR draws).
    real_pb_mf: pybullet + ModelForm. real_mj: native mujoco. real_mj_mf: mujoco + the same ModelForm."""
    out = {}

    def one(eng, mf=None):
        r = engines.rollout(eng, theta, EPISODE, mf=mf)
        return r.displacement, float(r.fell)

    pb = engines.make_engine("pybullet", params)
    out["sim_nominal"] = stats(*zip(*[one(engines.apply_reality(pb, IDENTITY))]))
    out["sim_draws"] = stats(*zip(*[one(engines.apply_reality(pb, sp)) for sp in draws(np.random.default_rng(EVAL_SEED), N_EVAL)]))
    out["real_pb_mf"] = stats(*zip(*[one(engines.apply_reality(pb, IDENTITY), engines.ModelForm())]))
    p.disconnect(pb.c)
    out["real_mj"] = stats(*zip(*[one(engines.MuJoCoEngine(params))]))
    out["real_mj_mf"] = stats(*zip(*[one(engines.MuJoCoEngine(params), engines.ModelForm())]))
    return out


def flatten(path):
    """rows.jsonl -> dataset.csv next to it. sim_* = pybullet twin (_nom nominal, _dr mean over the N_EVAL held-out draws).
    real_pb_mf / real_mj / real_mj_mf = the three reals, each one nominal rollout. fell = share of rollouts that fell."""
    import pandas as pd
    out = []
    for l in open(path):
        r = json.loads(l)
        e = r["eval"]
        row = {"morph_id": r["morph_id"], "target": r["T"], "arm": r["arm"], "train_objective": r["train_objective"],
               **r["params"], **{f"theta_{j}": v for j, v in enumerate(r["theta"])}}
        row["sim_dist_nom"], row["sim_fell_nom"] = e["sim_nominal"]["mean"], e["sim_nominal"]["fell"]
        row["sim_dist_dr"], row["sim_dist_dr_std"], row["sim_fell_dr"] = e["sim_draws"]["mean"], e["sim_draws"]["std"], e["sim_draws"]["fell"]
        for real in ("real_pb_mf", "real_mj", "real_mj_mf"):
            row[f"{real}_dist"], row[f"{real}_fell"] = e[real]["mean"], e[real]["fell"]
        out.append(row)
    csv = os.path.join(os.path.dirname(path), "dataset.csv")
    pd.DataFrame(out).to_csv(csv, index=False)
    print(f"wrote {csv}: {len(out)} rows", flush=True)


def task(args):
    mid, params, seed = args
    t0 = time.time()
    eng = engines.make_engine("pybullet", params)
    rows = []
    for T in targets_for(mid):
        for arm in ARMS:
            theta, obj = train(eng, T, arm, seed * 1000 + mid)
            engines.apply_reality(eng, IDENTITY)
            rows.append(dict(morph_id=mid, T=T, arm=arm, train_objective=obj, theta=theta.tolist(), params=params,
                             eval=evaluate(params, theta)))
    p.disconnect(eng.c)
    return mid, rows, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=10)
    ap.add_argument("--out", default="tubey/rows.jsonl")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0, help="body sample and CMA-ES seed. another device uses another seed, ids are offset by seed*1000 so merged files do not collide")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    morphs = G.sample_morphologies(a.bodies, a.seed)
    done = {json.loads(l)["morph_id"] for l in open(a.out)} if os.path.exists(a.out) else set()
    todo = [(a.seed * 1000 + i, m, a.seed) for i, m in enumerate(morphs) if a.seed * 1000 + i not in done]
    print(f"{len(todo)} bodies to do, {len(done)} already in {a.out}. {N_TARGETS} targets per body in [{T_LO}, {T_HI}], arms {ARMS}, "
          f"M={M_DRAWS} lambda={LAMBDA}, maxiter {MAXITER}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        for mid, rows, dt in ex.map(task, todo):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"m{mid:04d} done in {dt:.0f}s, {time.time() - t0:.0f}s elapsed", flush=True)
    flatten(a.out)
