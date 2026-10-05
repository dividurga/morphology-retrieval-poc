"""baseline, experiment 1 (5 spread designs, fit), experiment 2 (5-step idw loop), oracle check.

    uv run python experiment_disparity.py --target 1.0
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import disparity as D
from surrogate.idw import IDW

N_REAL, KAPPA, WINDOW = 5, 0.5, 0.1


def measure(pool, i, cache):
    """the real rollout of pool row i. the pool file holds the (deterministic) rollout results, so this is a lookup.
    the loop only calls it for designs it has chosen. every row is read at the end for the oracle check."""
    if i not in cache:
        r = pool.iloc[i]
        cache[i] = dict(d_real=r.d_real, fell=r.real_fell, disparity=abs(r.d_real - r.distance))
    return cache[i]


def score(pool, idw, X):
    """upper bound on real error: twin error + predicted disparity."""
    pred, near = idw.predict(X)
    return np.abs(pool.distance.values - pool.attrs["T"]) + pred, near


def farthest_points(X, k, exclude, rng):
    ok = [i for i in range(len(X)) if i not in exclude]
    chosen = [int(rng.choice(ok))]
    while len(chosen) < k:
        dmin = np.min(np.linalg.norm(X[ok][:, None] - X[chosen][None], axis=2), axis=1)
        chosen.append(ok[int(np.argmax(dmin))])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=1.0)
    ap.add_argument("--prefix", default="", help="pool file prefix, e.g. pybullet_R1_")
    a = ap.parse_args()
    T = a.target
    pool = pd.read_csv(f"results/pool_{a.prefix}{T}.csv")
    pool = pool[(~pool.twin_fell.astype(bool)) & ((pool.d_twin_rerun - pool.distance).abs() < 0.05)].reset_index(drop=True)
    pool.attrs["T"] = T
    X = D.features(pool)
    twin_err = (pool.distance - T).abs().values
    print(f"target {T} m, pool {len(pool)} rows from {pool.morph_id.nunique()} bodies", flush=True)

    base = int(np.argmin(twin_err))  # ties: lowest index (argmin takes the first)
    cache = {}

    # experiment 1: 5 spread designs, then fit
    rng = np.random.default_rng(0)
    e1 = farthest_points(X, N_REAL, {base}, rng)
    for i in e1:
        measure(pool, i, cache)
    idw1 = IDW().fit(X[e1], [cache[i]["disparity"] for i in e1])
    assert base not in e1
    s1, _ = score(pool, idw1, X)
    s1[base] = np.inf
    pick1 = int(np.argmin(s1))

    # experiment 2: propose 1, roll out, update
    e2, picks2 = [], []
    excl = {base}
    for it in range(N_REAL):
        if not e2:
            c = X.mean(axis=0)
            dist = np.linalg.norm(X - c, axis=1)
            dist[list(excl)] = np.inf
            nxt = int(np.argmin(dist))
        else:
            s, near = score(pool, idw2, X)
            acq = s - KAPPA * near
            acq[list(excl)] = np.inf
            nxt = int(np.argmin(acq))
        m = measure(pool, nxt, cache)
        e2.append(nxt)
        excl.add(nxt)
        idw2 = IDW().fit(X[e2], [cache[i]["disparity"] for i in e2])
        s, _ = score(pool, idw2, X)
        s[base] = np.inf
        picks2.append(int(np.argmin(s)))
        print(f"loop {it + 1}: design {nxt} twin {pool.distance[nxt]:.3f} real {m['d_real']:.3f} "
              f"disparity {m['disparity']:.3f} fell {m['fell']:.1f}", flush=True)
    assert base not in e1 and base not in e2

    # oracle: everything in mujoco, evaluation only
    for i in range(len(pool)):
        measure(pool, i, cache)
    pool["fell"] = pool.real_fell
    pool["real_err"] = (pool.d_real - T).abs()
    pool["pred_exp1"] = idw1.predict(X)[0]
    pool["pred_exp2"] = idw2.predict(X)[0]
    pool["role"] = ""
    for i in e1:
        pool.loc[i, "role"] += "e1 "
    for i in e2:
        pool.loc[i, "role"] += "e2 "
    pool.loc[base, "role"] += "baseline"
    pool.to_csv(f"results/{a.prefix}oracle.csv", index=False)

    # summary
    def line(method, n, i):
        r = pool.iloc[i]
        return dict(method=method, n_real_designs=n, design=i, morph_id=int(r.morph_id), d_twin=r.distance,
                    d_real=r.d_real, real_err=r.real_err, disparity=r.disparity, fell=r.fell)

    rows = [line("baseline best-nominal", 0, base), line("exp1 argmin after 5", N_REAL, pick1)]
    rows += [line(f"exp2 argmin after {k + 1}", k + 1, p) for k, p in enumerate(picks2)]
    best = int(np.argmin(pool.real_err.values))
    rows.append(line("oracle best in pool", len(pool), best))
    summ = pd.DataFrame(rows)
    summ.to_csv(f"results/{a.prefix}summary.csv", index=False)
    print(summ.round(3).to_string(index=False))

    # does the fitted surrogate have bearing
    print("\nsurrogate vs truth over the pool (measured points included)")
    mask = np.ones(len(pool), bool)
    for name, idw, ev in (("exp1", idw1, e1), ("exp2", idw2, e2)):
        m = mask.copy()
        m[ev] = False
        y, p = pool.disparity.values[m], pool[f"pred_{name}"].values[m]
        rho = spearmanr(y, p).statistic
        rmse, rmse0 = np.sqrt(np.mean((y - p) ** 2)), np.sqrt(np.mean((y - y.mean()) ** 2))
        print(f"  {name}: spearman {rho:.2f}, rmse {rmse:.3f} vs mean-predictor {rmse0:.3f} (held-out rows {m.sum()})")
    rank = int((pool.real_err < pool.real_err[base]).sum()) + 1
    print(f"  baseline ranks {rank}/{len(pool)} by true real error; spread of real_err: "
          f"median {pool.real_err.median():.3f}, best {pool.real_err.min():.3f}")
    print(f"  disparity: median {pool.disparity.median():.3f}, fell share {pool.fell.mean():.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot([0] + list(range(1, N_REAL + 1)), [summ.real_err[0]] + list(summ.real_err[2:2 + N_REAL]), "o-", label="exp2 loop argmin")
    ax.plot([N_REAL], [summ.real_err[1]], "s", label="exp1 argmin")
    ax.axhline(pool.real_err.min(), ls="--", c="gray", label="oracle best")
    ax.set_xlabel("real designs spent (0 = baseline)")
    ax.set_ylabel("real error |d_mujoco - T| (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"results/{a.prefix}loop.png", dpi=150)
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    for ax, name in zip(axs, ("exp1", "exp2")):
        ax.scatter(pool[f"pred_{name}"], pool.disparity, s=8)
        ax.set_title(name)
        ax.set_xlabel("predicted disparity (m)")
    axs[0].set_ylabel("true disparity (m)")
    fig.tight_layout()
    fig.savefig(f"results/{a.prefix}pred_vs_true.png", dpi=150)


if __name__ == "__main__":
    main()
