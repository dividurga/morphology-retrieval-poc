"""can we predict real (mujoco) falls for dr-trained rows? sim-only features vs morphology vs few real tests.

    uv run python experiment_fail_predict.py
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment_dr_fail import PN, load
from controller import THETA_BOUNDS

df, X = load("mj_nominal")
ev = [json.loads(l)["eval"] for l in open("results/dr/run1.jsonl")]
for w in ("pb_R0", "pb_R1", "pb_R2"):
    df[f"{w}_d"] = [e[w][0] for e in ev]
    df[f"{w}_f"] = [e[w][1] for e in ev]
keep = (~df.fell_twin) & df.regime.isin(["dr_mult", "dr_lag"])
df, X = df[keep].reset_index(drop=True), X[keep.values]
y = df.fell_real.values.astype(int)
print(f"dr-trained, twin-ok rows: {len(df)}, {df.morph_id.nunique()} bodies, mujoco fell share {y.mean():.2f}")
print("fell share by T:", df.groupby("T").fell_real.mean().round(2).to_dict())
print("fell share by regime:", df.groupby("regime").fell_real.mean().round(2).to_dict())

SIM = ["d_twin", "pb_R0_d", "pb_R0_f", "pb_R1_d", "pb_R1_f", "pb_R2_d", "pb_R2_f"]
S = df[SIM].values
S = np.hstack([S, (df.regime == "dr_lag").values[:, None], df["T"].values[:, None]])
feats = {"sim behaviour only": S, "morph+theta only": X, "both": np.hstack([S, X])}

print("\nsingle sim signals vs mujoco fall (AUC of 'higher signal => fall'):")
for c in SIM:
    v = df[c].values
    print(f"  {c:8s} AUC {roc_auc_score(y, v):.2f}  (fell-share when flag>0: {y[v > 0].mean() if c.endswith('_f') and (v>0).any() else float('nan'):.2f})" if c.endswith("_f")
          else f"  {c:8s} AUC {roc_auc_score(y, v):.2f}")

print("\ngrouped 5-fold CV by body (no body in train and test), AUC for mujoco fall:")
models = {"logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)),
          "gboost": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=0)}
oof = {}
for fn, F in feats.items():
    for mn, mk in models.items():
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(F, y, df.morph_id):
            p[te] = mk().fit(F[tr], y[tr]).predict_proba(F[te])[:, 1]
        oof[(fn, mn)] = p
        print(f"  {fn:20s} {mn:7s} AUC {roc_auc_score(y, p):.2f}")

# how many real tests does it take, and which ones? calibrate logistic on sim features with N real labels
print("\nN real tests -> held-out AUC (sim-feature logreg refit on N real labels, evaluated on the rest). 40 reps.")
rng_all = np.random.default_rng(0)
Ssc = StandardScaler().fit_transform(S)
proxy = df[["pb_R0_f", "pb_R1_f", "pb_R2_f"]].values.mean(axis=1) + 0.01 * rng_all.random(len(df))  # sim-only fail prior
for N in (5, 10, 20, 40):
    out = {"random": [], "hi-proxy (likely fail)": [], "mixed 50/50": [], "FPS morph": []}
    for rep in range(40):
        rng = np.random.default_rng(rep)
        picks = {"random": rng.choice(len(y), N, replace=False)}
        order = np.argsort(-proxy)
        picks["hi-proxy (likely fail)"] = order[:N]
        picks["mixed 50/50"] = np.r_[order[: N // 2], order[::-1][: N - N // 2]]
        from experiment_dr_fail import fps
        picks["FPS morph"] = np.array(fps(X, N, rng))
        for k, tr in picks.items():
            te = np.setdiff1d(np.arange(len(y)), tr)
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            m = LogisticRegression(C=0.3, max_iter=2000).fit(Ssc[tr], y[tr])
            out[k].append(roc_auc_score(y[te], m.predict_proba(Ssc[te])[:, 1]))
    print(f"  N={N:2d} " + "  ".join(f"{k}: {np.mean(v):.2f} (n={len(v)})" for k, v in out.items()))
print(f"  reference, zero real labels, raw proxy (mean fell share over pb_R0/R1/R2): AUC {roc_auc_score(y, proxy):.2f}")
print("  label balance of the proxy-chosen top-5 (share that really fell in mujoco):",
      f"{y[np.argsort(-proxy)[:5]].mean():.2f}; top-20 {y[np.argsort(-proxy)[:20]].mean():.2f}; bottom-20 {y[np.argsort(proxy)[:20]].mean():.2f}")

# recommendation: not falling and farthest. rows are target-conditioned, so 'farthest' = largest twin distance among safe rows
print("\nrecommendation: among rows, keep P(fall) <= tau (grouped-CV predictions), pick the largest twin distance.")
p = oof[("both", "logreg")]
dist_real = df.d_real.values
ok = ~df.fell_real.values
print(f"  oracle: farthest non-falling row, real distance {dist_real[ok].max():.2f} m")
base = int(np.argmax(df.d_twin.values))
print(f"  baseline (farthest in twin, no safety): fell={y[base]}, real distance {dist_real[base]:.2f}")
for tau in (0.5, 0.3, 0.2, 0.1):
    cand = np.where(p <= tau)[0]
    if len(cand) == 0:
        print(f"  tau {tau}: no candidates"); continue
    top = cand[np.argsort(-df.d_twin.values[cand])[:5]]
    print(f"  tau {tau}: {len(cand)} candidates; top-5 by twin distance: fell {y[top].mean():.2f}, mean real dist {dist_real[top].mean():.2f}, "
          f"mean twin dist {df.d_twin.values[top].mean():.2f}, mean real dist of non-fallers {dist_real[top][~y[top].astype(bool)].mean() if (~y[top].astype(bool)).any() else float('nan'):.2f}")
