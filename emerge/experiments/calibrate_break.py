"""pick the connector vibration factor from a published hardware number, not from the sim.
paper (PMC8287515): 6 of 30 real robots had connections break (median 9 modules). so the target is a break rate of 0.20.
rule, fixed before running: on a fixed panel of 30 random chains (2 to 8 actuated modules, random phases) x 3 worlds,
the factor low end (connector threshold = 1.2 Nm x U(low, 1)) whose share of robots with at least one broken connection
is closest to 0.20. caveats: the paper's robots were evolved against a breaking penalty, so a random panel should break
more than they did. the panel is chains only.

    uv run python -m emerge.experiments.calibrate_break
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from emerge import morph, real_mujoco as Rm

LOWS = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6]
TARGET = 6 / 30


def panel(n=30, seed=123):
    rng = np.random.default_rng(seed)
    return [morph.chain(int(k), rng.uniform(0, 360, int(k)).tolist()) for k in rng.integers(2, 9, n)]


def one(task):
    low, gi, world = task
    Rm.BREAK_FACTOR_LOW = low
    g = panel()[gi]
    r = Rm.rollout(g, world)
    return low, gi, world, r["broken"], r["distance"]


if __name__ == "__main__":
    tasks = [(low, gi, w) for low in LOWS for gi in range(30) for w in range(3)]
    with ProcessPoolExecutor(8) as ex:
        out = list(ex.map(one, tasks, chunksize=4))
    print(f"{'low':>5s} {'share with a break':>20s} {'mean broken':>12s} {'mean distance':>14s}")
    best = None
    for low in LOWS:
        rows = [o for o in out if o[0] == low]
        share = np.mean([o[3] > 0 for o in rows])
        print(f"{low:5.2f} {share:20.2f} {np.mean([o[3] for o in rows]):12.2f} {np.mean([o[4] for o in rows]):14.3f}")
        if best is None or abs(share - TARGET) < abs(best[1] - TARGET):
            best = (low, share)
    print(f"\npaper: {TARGET:.2f}. closest: low = {best[0]} (share {best[1]:.2f})")
