"""what limits the gaits? start unconstrained, add one real constraint at a time, and walk from an
old-like model to the current one. mujoco only, max-distance search, read-only.

fitness = best distance in 8 s, minus 6 if the best gait fell. negative means it fell.

    uv run python ablation.py --scale 0.7 [--seeds 5 --workers 6]
"""

import argparse
import dataclasses
import time

import numpy as np

import optimize_controller as oc
import sim_mujoco as sm
from spec import SPEC, MX64_ARMATURE

A = MX64_ARMATURE


def configs(s: float):
    base = dataclasses.replace(
        SPEC, scale=s, torque_cap=None, speed_torque=False, armature=0.0, armature_ankle=0.0,
        delay_steps=0, pad_enabled=False, legs_collide=False, contact_timeconst=None)
    r = dataclasses.replace
    out = [("BASE: everything off (no cap, armature, delay, pad; feet-only collide, stiff contact)", base)]
    out += [
        ("  + cap (constant 6 n*m)", r(base, torque_cap=6.0)),
        ("  + cap and speed-torque", r(base, torque_cap=6.0, speed_torque=True)),
        ("  + armature on hip, knee", r(base, armature=A)),
        ("  + armature on hip, knee, ankle", r(base, armature=A, armature_ankle=A)),
        ("  + delay 3", r(base, delay_steps=3)),
        ("  + soft foot pad", r(base, pad_enabled=True)),
        ("  + legs and torso collide", r(base, legs_collide=True)),
        ("  + soft default contact (0.02 s)", r(base, contact_timeconst=0.02)),
    ]
    old = r(base, armature=A, armature_ankle=A, legs_collide=True, contact_timeconst=0.02)
    out.append(("OLD-LIKE: armature all joints, legs collide, soft contact", old))
    walk = [
        ("  ankle armature -> 0", {"armature_ankle": 0.0}),
        ("  feet only collide", {"legs_collide": False}),
        ("  stiff contact (2 dt)", {"contact_timeconst": None}),
        ("  soft foot pad on", {"pad_enabled": True}),
        ("  cap 6 n*m (constant)", {"torque_cap": 6.0}),
        ("  speed-torque curve", {"speed_torque": True}),
        ("  delay 3  (= CURRENT)", {"delay_steps": 3}),
    ]
    cur = old
    for name, change in walk:
        cur = r(cur, **change)
        out.append((name, cur))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=SPEC.scale)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    print(f"scale {args.scale}, mujoco, default budget, {args.seeds} seeds. fitness = distance, -6 if fell", flush=True)
    for name, sp in configs(args.scale):
        t0 = time.time()
        fits = [oc.optimize(sp.nominal_params(), seed=sd, backend=sm, spec=sp, workers=args.workers)["fitness"]
                for sd in range(1, args.seeds + 1)]
        print(f"{name:72s} " + " ".join(f"{f:6.2f}" for f in fits)
              + f" | median {np.median(fits):6.2f} max {max(fits):6.2f} ({time.time() - t0:3.0f}s)", flush=True)
