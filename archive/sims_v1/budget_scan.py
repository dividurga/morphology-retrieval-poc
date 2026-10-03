"""how much search budget does a max-distance gait need? read-only measurement.

    uv run python budget_scan.py --sim mujoco|pybullet [--seeds 4]
"""

import argparse
import time

import numpy as np

import optimize_controller as oc
import sim_mujoco
import sim_pybullet
from spec import SPEC

BUDGETS = [(24, 100), (48, 200), (96, 300)]  # (popsize, maxiter). first is the default

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", choices=["mujoco", "pybullet"], required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--scale", type=float, default=SPEC.scale)
    args = ap.parse_args()
    import dataclasses
    spec = dataclasses.replace(SPEC, scale=args.scale)
    params = spec.nominal_params()
    backend = {"mujoco": sim_mujoco, "pybullet": sim_pybullet}[args.sim]
    print(f"{args.sim}, scale {spec.scale}, max distance in 8 s, best fitness per seed", flush=True)
    for popsize, maxiter in BUDGETS:
        t0, fits = time.time(), []
        for seed in range(1, args.seeds + 1):
            fits.append(oc.optimize(params, seed=seed, popsize=popsize, maxiter=maxiter,
                                    backend=backend, spec=spec, workers=args.workers)["fitness"])
        print(f"  popsize {popsize:3d} maxiter {maxiter:3d} ({popsize * maxiter:6d} evals/seed, {time.time() - t0:5.0f}s): "
              + " ".join(f"{f:6.2f}" for f in fits) + f" | best {max(fits):.2f} median {np.median(fits):.2f}", flush=True)
