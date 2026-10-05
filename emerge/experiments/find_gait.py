"""is a good gait repeatable? cma-es on the phases of a 3-actuated-module chain in the sim (mean of 2 runs per candidate),
then 12 repeats of the best gait and of a few random ones. 20 s episodes (4 s settle).

    uv run python -m emerge.experiments.find_gait
"""
import cma
import numpy as np

from emerge import morph, pool

N_ACT, DUR, T0 = 3, 20.0, 4.0


def evaluate(phases_list, reps=2, robust=False):
    tasks = [dict(genome=morph.chain(N_ACT, list(np.mod(p, 360.0))), duration=DUR, t_start=T0, seed=r)
             for p in phases_list for r in range(reps)]
    res = pool.map("emerge.sim_fast:rollout", tasks, n_workers=8)
    d = np.array([x["distance"] for x in res]).reshape(len(phases_list), reps)
    if robust:  # a connector break in any repeat costs most of the distance: ask for gaits that hold together every time
        b = np.array([x["broken"] for x in res]).reshape(len(phases_list), reps)
        d = np.where(b > 0, 0.2 * d, d)
    return d


if __name__ == "__main__":
    es = cma.CMAEvolutionStrategy([180.0] * N_ACT, 90.0, {"popsize": 8, "maxiter": 12, "seed": 3, "verbose": -9})
    best = (-1, None)
    while not es.stop():
        xs = es.ask()
        d = evaluate(xs, reps=3, robust=True)
        es.tell(xs, (-d.mean(axis=1)).tolist())
        i = int(np.argmax(d.mean(axis=1)))
        if d.mean(axis=1)[i] > best[0]:
            best = (float(d.mean(axis=1)[i]), np.mod(xs[i], 360.0))
        print(f"gen {es.countiter}: best so far {best[0]:.3f} m, phases {np.round(best[1]).tolist()}", flush=True)
    rng = np.random.default_rng(0)
    cands = {"robust best": best[1]}
    print("\nrepeatability: 12 runs each, same genome and seed")
    for name, p in cands.items():
        from emerge import sim_fast
        r = sim_fast.repeatability(morph.chain(N_ACT, list(p)), k=16, duration=DUR, t_start=T0, n_workers=8)
        print(f"  {name} phases {np.round(p).tolist()}: {r}", flush=True)
        d = evaluate([p], reps=12)[0]
        print(f"  {name:10s} phases {np.round(p).tolist()}  mean {d.mean():.3f} std {d.std():.3f}  runs {np.round(d, 3).tolist()}", flush=True)
