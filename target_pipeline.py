"""target-conditioned open-loop gaits trained in the twin (pybullet, full setup ladder), replayed in
mujoco ("real"). disparity = what the engines do differently. nothing is tuned.

    uv run python target_pipeline.py [--targets 2,4,6] [--seeds 2]
"""

import argparse
import time

import numpy as np

import backend_mujoco
import backend_pybullet
import morphology
import optimize_controller as oc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="2,4,6")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    P = morphology.NOMINAL_PARAMS
    twin, real = backend_pybullet.make_model(P), backend_mujoco.make_model(P)
    print("nominal biped. train in pybullet (full ladder), replay in mujoco. distance m, F = fell", flush=True)
    print(f"{'target':>6} {'seed':>4} {'twin dist':>10} {'real dist':>10} {'twin err':>9} {'real err':>9} {'real - twin':>12}", flush=True)
    for target in [float(t) for t in args.targets.split(",")]:
        for seed in range(1, args.seeds + 1):
            t0 = time.time()
            th = np.array(oc.optimize(P, seed=seed, backend=backend_pybullet, target_distance=target,
                                      workers=args.workers)["theta"])
            a, b = backend_pybullet.simulate(twin, th, 8.0), backend_mujoco.simulate(real, th, 8.0)
            f = lambda r: f"{r.displacement:9.2f}{'F' if r.fell else ' '}"
            print(f"{target:6.1f} {seed:4d} {f(a):>10} {f(b):>10} {abs(a.displacement - target):9.2f} {abs(b.displacement - target):9.2f} "
                  f"{b.displacement - a.displacement:12.2f}   ({time.time() - t0:.0f}s)", flush=True)
