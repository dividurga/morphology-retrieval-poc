"""ablation: what a person does when setting up the pybullet twin from the mjcf, rung by rung.

each rung adds one fix that comes from the file or the engine's documented behaviour (engines.LADDER).
none uses the real sim's output: the final product does not exist for system identification. the gap
to mujoco ("real") is MEASURED after each rung. it MUST NOT be used to choose or tune a rung.

    uv run python human_ladder.py [--seeds 10] [--random 20] [--horizon 1.0]
"""

import argparse

import numpy as np

import backend_mujoco
import controller
import engines
import morphology
import optimize_controller as oc

SCALE = np.array([0.05, 0.02, 0.1])  # x, z, pitch error scales for the trace loss


def gait_set(seeds, n_random, workers):
    P = morphology.NOMINAL_PARAMS
    thetas = [np.array(oc.optimize(P, seed=s, backend=backend_mujoco, workers=workers)["theta"]) for s in range(1, seeds + 1)]
    mj = engines.MuJoCoEngine(P)
    dist = [engines.rollout(mj, t, 8.0).displacement for t in thetas]
    best = [thetas[i] for i in np.argsort(dist)[-2:]]
    rng = np.random.default_rng(0)
    lo = np.array([b[0] for b in controller.THETA_BOUNDS]); hi = np.array([b[1] for b in controller.THETA_BOUNDS])
    near = [np.clip(b + rng.normal(0, 0.03, len(b)) * (hi - lo), lo, hi) for b in best for _ in range(8)]
    rand = [lo + rng.random(len(lo)) * (hi - lo) for _ in range(n_random)]
    return thetas + near + rand


def run(eng, thetas, horizon):
    out = []
    for t in thetas:
        r, tr = engines.rollout(eng, t, 8.0, trace=horizon)
        out.append((r.displacement, r.fell, np.array(tr)))
    return out


def trace_loss(a, b, n):
    """pseudo-huber of the normalised (x, z, pitch) trace difference. a fallen run is padded with its last state."""
    def pad(tr):
        tr = tr if len(tr) else np.zeros((1, 3))
        return np.vstack([tr, np.repeat(tr[-1:], max(0, n - len(tr)), axis=0)])[:n]
    e = (pad(a) - pad(b)) / SCALE
    return float(np.mean(np.sqrt(1 + e ** 2) - 1))


def metrics(real, twin, n):
    dr = np.array([r[0] for r in real]); dt = np.array([t[0] for t in twin])
    fr = np.array([r[1] for r in real]); ft = np.array([t[1] for t in twin])
    walkers = (~fr) & (dr > 1.0)
    return {
        "fall agree": float(np.mean(fr == ft)),
        "dist corr": float(np.corrcoef(dr, dt)[0, 1]),
        "mean |dd| m": float(np.mean(np.abs(dr - dt))),
        "trace loss": float(np.mean([trace_loss(r[2], t[2], n) for r, t in zip(real, twin)])),
        "walkers": int(walkers.sum()),
        "twin walks": float(np.mean(~ft[walkers])) if walkers.any() else float("nan"),
        "twin dist/real": float(np.mean(dt[walkers] / dr[walkers])) if walkers.any() else float("nan"),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--random", type=int, default=20)
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    P = morphology.NOMINAL_PARAMS
    n = int(round(args.horizon / engines.DT))
    thetas = gait_set(args.seeds, args.random, args.workers)
    real = run(engines.MuJoCoEngine(P), thetas, args.horizon)
    print(f"{len(thetas)} gaits. mujoco (real): {sum(1 for r in real if not r[1])} do not fall, "
          f"{sum(1 for r in real if (not r[1]) and r[0] > 1.0)} walk > 1 m, best {max(r[0] for r in real):.2f} m", flush=True)
    keys = ["fall agree", "dist corr", "mean |dd| m", "trace loss", "walkers", "twin walks", "twin dist/real"]
    print(f"{'rung':28s} " + " ".join(f"{k:>14s}" for k in keys))

    def row(label, fixes):
        m = metrics(real, run(engines.PyBulletEngine(P, fixes=fixes), thetas, args.horizon), n)
        print(f"{label:28s} " + " ".join(f"{m[k]:14.3f}" for k in keys), flush=True)

    print("-- cumulative, in the order a person would do it")
    row("0 raw import", ())
    for i, fx in enumerate(engines.LADDER, 1):
        row(f"{i} + {fx}", tuple(engines.LADDER[:i]))
    print("-- each fix alone on top of the raw import")
    for fx in engines.LADDER:
        row(f"raw + {fx}", (fx,))
