"""open-loop cma-es on the original biped. train in one engine, replay each best gait in both.
the gap is whatever the engines do with the same mjcf and the same torque law. nothing is tuned.

    uv run python cross_engine.py [--seeds 5] [--target 3.0] [--workers 12]
"""

import argparse
import time

import numpy as np

import backend_mujoco
import backend_pybullet
import morphology
import optimize_controller as oc

BACKENDS = {"pybullet": backend_pybullet, "mujoco": backend_mujoco}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--target", type=float, default=None, help="target distance m. None = maximise distance")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--train", default="pybullet,mujoco")
    args = ap.parse_args()
    P = morphology.NOMINAL_PARAMS
    models = {n: b.make_model(P) for n, b in BACKENDS.items()}
    print(f"original biped, same mjcf in both engines, target {args.target}. distance m, F = fell", flush=True)
    for train in args.train.split(","):
        t0, rows = time.time(), []
        for seed in range(1, args.seeds + 1):
            r = oc.optimize(P, seed=seed, backend=BACKENDS[train], target_distance=args.target, workers=args.workers)
            th = np.array(r["theta"])
            rows.append((r["fitness"], {n: BACKENDS[n].simulate(models[n], th, 8.0) for n in BACKENDS}))
        print(f"\ntrained in {train} ({time.time() - t0:.0f}s)  seed: fitness | " + " | ".join(f"in {n}" for n in BACKENDS), flush=True)
        for i, (fit, res) in enumerate(rows, 1):
            print(f"  {i}: {fit:7.2f} | " + " | ".join(f"{res[n].displacement:7.2f}{'F' if res[n].fell else ' '}" for n in BACKENDS), flush=True)
