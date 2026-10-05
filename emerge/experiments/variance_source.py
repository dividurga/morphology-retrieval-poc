"""how much of the sim's run-to-run spread is connector breaks? 8 random chains x 10 runs, 20 s. runs are grouped by broken count."""
import numpy as np
from emerge import morph, pool

if __name__ == "__main__":
    rng = np.random.default_rng(5)
    gs = [morph.chain(int(rng.integers(2, 6)), rng.uniform(0, 360, 5).tolist()) for _ in range(8)]
    res = pool.map("emerge.sim_fast:rollout", [dict(genome=g, duration=20.0, t_start=4.0, seed=0) for g in gs for _ in range(10)], n_workers=8)
    d = np.array([x["distance"] for x in res]).reshape(8, 10)
    b = np.array([x["broken"] for x in res]).reshape(8, 10)
    print("genome  n_act  all-runs std | share with a break | std among intact runs (n) | std among runs with the same break count")
    allv, intv, samev = [], [], []
    for i, g in enumerate(gs):
        intact = d[i][b[i] == 0]
        same = np.mean([d[i][b[i] == k].std() for k in np.unique(b[i]) if (b[i] == k).sum() > 1] or [np.nan])
        print(f"  {i}     {len(g.modules) - 1}      {d[i].std():.3f}        {np.mean(b[i] > 0):.1f}            {intact.std() if len(intact) > 1 else float('nan'):.3f} ({len(intact)})            {same:.3f}")
        allv.append(d[i].std()); samev.append(same)
        if len(intact) > 1: intv.append(intact.std())
    print(f"mean within-genome std: all runs {np.mean(allv):.3f}, intact-only {np.mean(intv):.3f}, same break count {np.nanmean(samev):.3f}; std of genome means {d.mean(axis=1).std():.3f}")
