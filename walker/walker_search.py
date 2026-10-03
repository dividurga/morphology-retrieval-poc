"""cma-es open-loop gait search on the walker2d mjcf, one engine at a time, then replay each best
gait in both engines. fitness = distance in 8 s, minus 6 if it fell.

    uv run python walker/walker_search.py [--seeds 5] [--workers 12] [--popsize 24] [--maxiter 100]
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cma
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import walker_env as W

FALL_PENALTY = 6.0
_E = {}


def _init(engine):
    _E["env"] = W.ENGINES[engine]()


def _loss(theta):
    d, fell, _ = W.rollout(_E["env"], np.asarray(theta))
    return -(d - (FALL_PENALTY if fell else 0.0))


def search(engine, seed, popsize, maxiter, workers):
    lo = [b[0] for b in W.THETA_BOUNDS]
    hi = [b[1] for b in W.THETA_BOUNDS]
    es = cma.CMAEvolutionStrategy(W.DEFAULT_THETA, 1.0, {
        "bounds": [lo, hi], "CMA_stds": [(h - l) / 4 for l, h in zip(lo, hi)],
        "popsize": popsize, "maxiter": maxiter, "seed": seed, "verbose": -9})
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(engine,)) as pool:
        while not es.stop():
            xs = es.ask()
            es.tell(xs, list(pool.map(_loss, xs)))
    return np.array(es.result.xbest), -es.result.fbest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--popsize", type=int, default=24)
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--engines", default="pybullet,mujoco", help="train in these, in order")
    args = ap.parse_args()
    envs = {n: W.ENGINES[n]() for n in W.ENGINES}
    print(f"walker2d_v5.xml, same file in both engines. cma-es popsize {args.popsize} maxiter {args.maxiter}, "
          f"{args.seeds} seeds. distance m (F = fell)", flush=True)
    for train in args.engines.split(","):
        t0, rows = time.time(), []
        for seed in range(1, args.seeds + 1):
            theta, fit = search(train, seed, args.popsize, args.maxiter, args.workers)
            res = {n: W.rollout(envs[n], theta) for n in W.ENGINES}
            rows.append((fit, res))
        print(f"\ntrained in {train} ({time.time() - t0:.0f}s)   seed: fitness | " + " | ".join(f"in {n}" for n in W.ENGINES), flush=True)
        for i, (fit, res) in enumerate(rows, 1):
            print(f"  {i}: {fit:7.2f} | " + " | ".join(f"{res[n][0]:7.2f}{'F' if res[n][1] else ' '}" for n in W.ENGINES), flush=True)
