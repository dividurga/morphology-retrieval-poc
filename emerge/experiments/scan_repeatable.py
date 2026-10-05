"""what share of random gaits is repeatable in the sim? 3-actuated-module chains, 40 random phase vectors, 6 runs each (20 s)."""
import numpy as np
from emerge import morph, pool

if __name__ == "__main__":
    rng = np.random.default_rng(11)
    P = rng.uniform(0, 360, (40, 3))
    res = pool.map("emerge.sim_fast:rollout", [dict(genome=morph.chain(3, p.tolist()), duration=20.0, t_start=4.0) for p in P for _ in range(6)], n_workers=8)
    d = np.array([x["distance"] for x in res]).reshape(40, 6)
    b = np.array([x["broken"] for x in res]).reshape(40, 6)
    ok = (b == 0).all(axis=1) & (d.std(axis=1) < 0.01)
    never = (b > 0).all(axis=1)
    print(f"no break in any of 6 runs and std < 1 cm: {ok.sum()}/40 ({ok.mean():.0%})")
    print(f"a break in every run: {never.sum()}/40; mixed: {(~ok & ~never).sum()}/40")
    print(f"distance of the repeatable gaits (median over runs): {np.round(np.sort(np.median(d[ok], axis=1)), 3).tolist()}")
    print(f"within-gait std: repeatable {d[ok].std(axis=1).mean():.4f} m, others {d[~ok].std(axis=1).mean():.3f} m; spread of gait medians {np.median(d, axis=1).std():.3f} m")
    best = np.argsort(-np.median(d, axis=1) * ok)[:3]
    print("best repeatable:", [(np.round(P[i]).tolist(), round(float(np.median(d[i])), 3), round(float(d[i].std()), 4)) for i in best if ok[i]])
