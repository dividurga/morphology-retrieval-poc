"""does a smaller physics step make the sim repeatable, and does it change the answer? 6 random chains, 4 repeats each, 38 s."""
import numpy as np
from emerge import morph, pool

if __name__ == "__main__":
    rng = np.random.default_rng(1)
    gs = [morph.chain(int(rng.integers(2, 6)), rng.uniform(0, 360, 5).tolist()) for _ in range(6)]
    for dt in (0.005, 0.0025):
        tasks = [dict(genome=g, physics_dt=dt, seed=r) for g in gs for r in range(4)]
        res = pool.map("emerge.sim_fast:rollout", tasks, n_workers=8)
        d = np.array([x["distance"] for x in res]).reshape(len(gs), 4)
        print(f"physics step {dt * 1000:.1f} ms")
        for gi, g in enumerate(gs):
            print(f"  {len(g.modules) - 1} act: mean {d[gi].mean():.3f} std {d[gi].std():.3f}  runs {np.round(d[gi], 3).tolist()}", flush=True)
        print(f"  mean within-genome std {d.std(axis=1).mean():.3f}, std of genome means {d.mean(axis=1).std():.3f}", flush=True)
