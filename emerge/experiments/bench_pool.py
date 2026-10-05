import sys, time
import numpy as np
from emerge import morph, pool

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gs = [morph.chain(int(rng.integers(2, 6)), rng.uniform(0, 360, 5).tolist()) for _ in range(32)]
    for n in (1, 4, 8):
        t = time.time()
        r = pool.map("emerge.sim_fast:rollout", [dict(genome=g, duration=38.0) for g in gs[:32 if n > 1 else 8]], n_workers=n)
        dt = time.time() - t
        print(f"{n} workers: {len(r)} rollouts in {dt:.1f} s -> {len(r) / dt:.2f} rollouts/s; distances {np.round([x['distance'] for x in r[:4]], 3)}", flush=True)
