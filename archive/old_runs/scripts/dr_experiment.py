"""domain randomisation on the controller. per (morphology, target, regime): train in the pybullet twin, then evaluate the
trained gait in worlds it never trained on: the twin itself, held-out pybullet realities, mujoco.

regimes (all choose a restart by their own training fitness, never by an evaluation world):
  nominal  fitness on the nominal twin
  dr_mult  fitness = mean over m fresh draws of friction/mass/motor/damping (perturbations.PERTURBATION_RANGES)
  dr_lag   dr_mult plus an actuator lag drawn from 0..4 control steps
draws are shared by a whole CMA-ES generation (common random numbers). training seeds differ from every evaluation draw.
nothing is tuned against mujoco output.

    uv run python dr_experiment.py --bodies 200 --out results/dr/run1.jsonl
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
import disparity as D
import engines
import generate_dataset as G
import perturbations

TARGETS = (0.6, 1.0, 1.5)
REGIMES = ("nominal", "dr_mult", "dr_lag")
M_DRAWS, RESTARTS, POPSIZE, EPISODE = 3, 2, 12, G.EPISODE
MAXITER = int(os.environ.get("DR_MAXITER", 40))
IDENTITY = {"friction": 1.0, "mass": 1.0, "motor": 1.0, "damping": 1.0, "delay": 0}


def train_specs(rng, m, lag):
    out = []
    for _ in range(m):
        s = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in perturbations.PERTURBATION_RANGES.items()}
        out.append({**s, "delay": int(rng.integers(0, 5)) if lag else 0})
    return out


def score(res, T):
    s = -abs(res.displacement - T) - (G.BACKWARD_PENALTY if res.displacement < 0 else 0.0)
    return s - (G.FALL_PENALTY if res.fell else 0.0)


def fitness(eng, theta, T, specs):
    f = []
    for sp in specs:
        engines.apply_reality(eng, sp)
        f.append(score(engines.rollout(eng, theta, EPISODE), T))
    return float(np.mean(f))


def train(eng, T, regime, seed):
    lo = [b[0] for b in controller.THETA_BOUNDS]
    hi = [b[1] for b in controller.THETA_BOUNDS]
    rng = np.random.default_rng(10_000 + seed)  # training draws. evaluation draws come from D.reality_specs (seed 0)
    cands = []
    for r in range(RESTARTS):
        es = cma.CMAEvolutionStrategy(controller.DEFAULT_THETA, 1.0, {
            "bounds": [lo, hi], "CMA_stds": [(h - l) / 4 for l, h in zip(lo, hi)],
            "popsize": POPSIZE, "maxiter": MAXITER, "seed": seed * 100 + r + 1, "verbose": -9})
        while not es.stop():
            specs = [IDENTITY] if regime == "nominal" else train_specs(rng, M_DRAWS, regime == "dr_lag")
            xs = es.ask()
            es.tell(xs, [-fitness(eng, np.asarray(x), T, specs) for x in xs])
        cands.append(np.asarray(es.result.xbest if regime == "nominal" else es.result.xfavorite))
    # choose the restart by the regime's own training distribution, with fresh draws
    sel = [IDENTITY] if regime == "nominal" else train_specs(rng, 10, regime == "dr_lag")
    fit = [fitness(eng, th, T, sel) for th in cands]
    k = int(np.argmax(fit))
    return cands[k], fit[k]


def evaluate(params, theta):
    """-> {world: (distance, share fell)}. pybullet worlds reuse one engine, mujoco worlds build fresh ones."""
    out = {}
    pb = engines.make_engine("pybullet", params)
    engines.apply_reality(pb, IDENTITY)
    r = engines.rollout(pb, theta, EPISODE)
    out["pb_nominal"] = (r.displacement, float(r.fell))
    for rung in ("R0", "R1", "R2"):
        d, f = [], []
        for sp in D.reality_specs(rung):
            engines.apply_reality(pb, sp)
            r = engines.rollout(pb, theta, EPISODE)
            d.append(r.displacement); f.append(r.fell)
        out[f"pb_{rung}"] = (float(np.mean(d)), float(np.mean(f)))
    p.disconnect(pb.c)
    r = engines.rollout(engines.MuJoCoEngine(params), theta, EPISODE)
    out["mj_nominal"] = (r.displacement, float(r.fell))
    d, f = D.real_rollout_row(params, theta, rung="cross")
    out["mj_cross"] = (d, f)
    d, f = D.real_rollout_row(params, theta, rung="R2", twin="mujoco")
    out["mj_R2"] = (d, f)
    return out


def task(args):
    mid, params, seed = args
    t0 = time.time()
    eng = engines.make_engine("pybullet", params)
    rows = []
    for T in TARGETS:
        for regime in REGIMES:
            theta, fit = train(eng, T, regime, seed * 1000 + mid)
            engines.apply_reality(eng, IDENTITY)
            ev = evaluate(params, theta)
            rows.append(dict(morph_id=mid, T=T, regime=regime, train_fitness=fit, theta=theta.tolist(), params=params,
                             eval={k: list(v) for k, v in ev.items()}))
    p.disconnect(eng.c)
    return mid, rows, time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bodies", type=int, default=200)
    ap.add_argument("--out", default="results/dr/run1.jsonl")
    ap.add_argument("--workers", type=int, default=11)
    a = ap.parse_args()
    morphs = G.sample_morphologies(200, 0)[: a.bodies]  # the data_v1 morphologies
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["morph_id"] for l in open(a.out)}
    todo = [(i, m, 0) for i, m in enumerate(morphs) if i not in done]
    print(f"{len(todo)} bodies to do, {len(done)} already in {a.out}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, open(a.out, "a") as f:
        for mid, rows, dt in ex.map(task, todo):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"m{mid:04d} done in {dt:.0f}s, {time.time() - t0:.0f}s elapsed", flush=True)
