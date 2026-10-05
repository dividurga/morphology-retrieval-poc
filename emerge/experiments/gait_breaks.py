import numpy as np
from emerge import morph, pool

if __name__ == "__main__":
    g = morph.chain(3, [166.0, 128.0, 294.0])
    res = pool.map("emerge.sim_fast:rollout", [dict(genome=g, duration=20.0, t_start=4.0, seed=0) for _ in range(16)], n_workers=8)
    print("cma-best gait, 16 runs: (distance, broken)")
    for x in res:
        print(f"  {x['distance']:.3f}  broken {x['broken']}")
    d = np.array([x["distance"] for x in res]); b = np.array([x["broken"] for x in res])
    print(f"runs with a broken connector: {int((b > 0).sum())}/16, mean distance when intact {d[b == 0].mean():.3f} (std {d[b == 0].std():.3f}), when broken {d[b > 0].mean() if (b > 0).any() else float('nan'):.3f}")
