"""surrogate with a failure signal, on the dr dataset (results/dr/run1.jsonl), diverse candidates.

twin = pb_nominal, real = mj_nominal (stored evals, [distance, fell]). pool = every (body, regime) row
for the target, no distance window. real designs are picked by farthest-point sampling over
[morphology params, theta, regime], not by closeness to the target.

    uv run python experiment_dr_fail.py --world mj_nominal
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from controller import THETA_BOUNDS
from morphology import PARAM_BOUNDS
from surrogate.idw import IDW

PN = list(PARAM_BOUNDS)
REG = ["nominal", "dr_mult", "dr_lag"]


def load(world, path="results/dr/run1.jsonl"):
    rows = []
    for l in open(path):
        r = json.loads(l)
        e = r["eval"]
        rows.append(dict(morph_id=r["morph_id"], T=r["T"], regime=r["regime"], theta=r["theta"], params=r["params"],
                         d_twin=e["pb_nominal"][0], fell_twin=e["pb_nominal"][1] > 0.5,
                         d_real=e[world][0], fell_real=e[world][1] > 0.5))
    df = pd.DataFrame(rows)
    lo = np.array([PARAM_BOUNDS[k][0] for k in PN] + [b[0] for b in THETA_BOUNDS])
    hi = np.array([PARAM_BOUNDS[k][1] for k in PN] + [b[1] for b in THETA_BOUNDS])
    X = np.array([[r.params[k] for k in PN] + list(r.theta) for r in df.itertuples()])
    X = np.clip((X - lo) / (hi - lo), 0, 1)
    oh = np.array([[r == g for g in REG] for r in df.regime], float)
    df["i"] = range(len(df))
    return df, np.hstack([X, oh])


def fps(X, k, rng):
    chosen = [int(rng.integers(len(X)))]
    dmin = np.linalg.norm(X - X[chosen[0]], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dmin = np.minimum(dmin, np.linalg.norm(X - X[nxt], axis=1))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", default="mj_nominal")
    ap.add_argument("--mu", type=float, default=0.5, help="m of predicted error charged per unit fall probability")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    df, X = load(a.world)
    df = df[~df.fell_twin].reset_index(drop=True)  # twin must not fall (relabelling rule)
    X = X[df.i.values]
    df["i"] = range(len(df))
    df["disp"] = (df.d_real - df.d_twin).abs()
    print(f"world {a.world}: {len(df)} twin-ok rows, {df.morph_id.nunique()} bodies, real fell share {df.fell_real.mean():.2f}")
    print(df.groupby("regime").fell_real.mean().round(2).to_dict(), "(real fell share by regime)")

    out = []
    for T in sorted(df["T"].unique()):
        m = (df["T"] == T).values
        P, XP = df[m].reset_index(drop=True), X[m]
        twin_err = (P.d_twin - T).abs().values
        real_err = (P.d_real - T).abs().values
        base = int(np.argmin(twin_err))
        oracle = float(real_err.min())
        # real error of a fall-free oracle (cheating, for reference)
        ok = ~P.fell_real.values
        oracle_nf = float(real_err[ok].min()) if ok.any() else np.nan
        print(f"\nT={T}: pool {len(P)}, baseline real err {real_err[base]:.3f} (fell {int(P.fell_real[base])}), "
              f"oracle {oracle:.3f}, oracle among non-fallers {oracle_nf:.3f}, pool real fell {P.fell_real.mean():.2f}")
        for N in (5, 10, 20, 40):
            res = {k: [] for k in ("rho_disp", "auc", "e_idw", "e_fail", "fell_idw", "fell_fail", "rand_e", "ok_idw", "ok_fail")}
            for rep in range(a.reps):
                rng = np.random.default_rng(rep)
                tr = fps(XP, N, rng)
                te = np.setdiff1d(np.arange(len(P)), tr)
                disp, fell = P.disp.values, P.fell_real.values.astype(float)
                d_model = IDW().fit(XP[tr], disp[tr])
                f_model = IDW().fit(XP[tr], fell[tr])
                pd_, _ = d_model.predict(XP)
                pf, _ = f_model.predict(XP)
                if len(set(fell[te])) == 2:
                    res["auc"].append(roc_auc_score(fell[te], pf[te]))
                res["rho_disp"].append(spearmanr(disp[te], pd_[te]).statistic)
                for key, score in (("idw", twin_err + pd_), ("fail", twin_err + pd_ + a.mu * pf)):
                    s = score.copy()
                    s[tr] = np.inf if False else s[tr]  # tested rows stay eligible: their score is still a prediction
                    j = int(np.argmin(s))
                    res[f"e_{key}"].append(real_err[j])
                    res[f"fell_{key}"].append(fell[j])
                    res[f"ok_{key}"].append(float((not fell[j]) and real_err[j] <= 0.1))
                res["rand_e"].append(real_err[rng.choice(len(P))])
            row = dict(T=T, N=N, **{k: float(np.nanmean(v)) for k, v in res.items()}, baseline=real_err[base], base_ok=float((not P.fell_real[base]) and real_err[base] <= 0.1), oracle=oracle, orc_ok=float(((real_err <= 0.1) & ok).any()))
            out.append(row)
    o = pd.DataFrame(out)
    pd.options.display.width = 200
    print("\nmean over reps. e_* = real error of the picked row (m), fell_* = share of picks that fall in real")
    print(o.round(3).to_string(index=False))
    o.to_csv(f"results/dr_fail_{a.world}.csv", index=False)


if __name__ == "__main__":
    main()
