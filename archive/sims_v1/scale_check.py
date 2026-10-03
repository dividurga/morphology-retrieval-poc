"""re-check the robot scale. for one scale: the best gaits mujoco finds under the real servo
(peak torque, distance) and how well pybullet follows on random gaits. read-only.

    uv run python scale_check.py --scale 0.7 [--seeds 3]
"""

import argparse
import dataclasses

import mujoco
import numpy as np
import pybullet as p

import controller
import gap_budget as G
import optimize_controller as oc
import sim_mujoco as sm
from spec import Spec, RealityDeltas

ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=float, required=True)
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--agree-sets", type=int, default=4)
args = ap.parse_args()

spec = Spec(scale=args.scale)
params = spec.nominal_params()
rows = []
for seed in range(1, args.seeds + 1):
    theta = np.array(oc.optimize(params, seed=seed, backend=sm, spec=spec)["theta"])  # max distance
    m = sm.make_model(params, spec)
    d = mujoco.MjData(m.mj)
    peak, n, fell_t = np.zeros(4), 0, None
    delay = spec.delay_steps * spec.timestep
    for step in range(int(8.0 / spec.timestep)):
        tg = controller.joint_targets(theta, max(0.0, step * spec.timestep - delay))
        for _ in range(spec.mj_substeps):
            sm.apply_control(m, d, tg)
            peak = np.maximum(peak, np.abs(d.ctrl))
            mujoco.mj_step(m.mj, d)
        if sm.has_fallen(d.qpos[2], d.qpos[1], m.standing_height):
            fell_t = (step + 1) * spec.timestep
            break
    rows.append((d.qpos[0], peak.max(), fell_t))
print(f"scale {args.scale}: mass {spec.total_mass(params):.2f} kg")
for i, (x, pk, ft) in enumerate(rows, 1):
    print(f"  search seed {i}: best distance {x:6.2f} m  peak servo torque {pk:5.2f} n*m "
          f"({'fell at %.1fs' % ft if ft else 'no fall'})")
client = p.connect(p.DIRECT)
res = [G.measure(RealityDeltas(), G.cases(s, spec), client, spec) for s in range(args.agree_sets)]
print("  agreement with pybullet on random gaits (nominal real, "
      f"{args.agree_sets} sets x 45): " + "  ".join(
          f"{k} {np.mean([r[k] for r in res]):.3f}" for k in ("loss", "pitch_rms", "fall_corr", "disp_corr")))
