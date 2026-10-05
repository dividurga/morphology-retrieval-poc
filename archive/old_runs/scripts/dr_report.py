"""summarise results/dr/*.jsonl to a markdown table. safe to run while the experiment is still going.

    uv run python dr_report.py results/dr/run1.jsonl > results/dr/REPORT.md
"""

import json
import sys

import numpy as np
import pandas as pd

HIT = 0.1  # m. a controller "hits" when |distance - target| <= HIT
WORLDS = ["pb_nominal", "pb_R0", "pb_R1", "pb_R2", "mj_nominal", "mj_cross", "mj_R2"]


def load(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        for w, (d, f) in r["eval"].items():
            rows.append(dict(morph_id=r["morph_id"], T=r["T"], regime=r["regime"], world=w, d=d, fell=f,
                             err=abs(d - r["T"])))
    return pd.DataFrame(rows)


def table(df, title):
    g = df.groupby(["world", "regime"]).agg(n=("err", "size"), median_err=("err", "median"),
                                           hit=("err", lambda e: (e <= HIT).mean()), fell=("fell", "mean")).round(3)
    out = [f"### {title}", "", "| world | regime | n | median err (m) | hit rate | fell share |", "|---|---|---|---|---|---|"]
    for w in WORLDS:
        for r in ("nominal", "dr_mult", "dr_lag"):
            if (w, r) in g.index:
                x = g.loc[(w, r)]
                out.append(f"| {w} | {r} | {int(x.n)} | {x.median_err} | {x.hit} | {x.fell} |")
    return "\n".join(out)


def paired(df):
    p = df.pivot_table(index=["morph_id", "T", "world"], columns="regime", values="err").reset_index()
    out = ["### paired change in error, dr minus nominal, same (body, target)", "",
           "| world | regime | n pairs | median change (m) | share improved |", "|---|---|---|---|---|"]
    for w in WORLDS:
        s = p[p.world == w]
        for r in ("dr_mult", "dr_lag"):
            if r in s and "nominal" in s:
                d = (s[r] - s["nominal"]).dropna()
                out.append(f"| {w} | {r} | {len(d)} | {d.median():.3f} | {(d < 0).mean():.2f} |")
    return "\n".join(out)


if __name__ == "__main__":
    df = load(sys.argv[1])
    print(f"# dr report: {df.morph_id.nunique()} bodies, targets {sorted(df['T'].unique())}\n")
    print(f"hit = |distance - target| <= {HIT} m. worlds: pb = pybullet twin family, mj = mujoco. none were trained on.\n")
    print(table(df, "all (body, target) pairs"), "\n")
    can = df[(df.regime == "nominal") & (df.world == "pb_nominal") & (df.err <= HIT)][["morph_id", "T"]]
    sub = df.merge(can, on=["morph_id", "T"])
    print(table(sub, f"pairs where the nominal controller hits in its own twin ({len(can)} pairs)"), "\n")
    print(paired(df), "\n")
    print("### per target, hit rate in mj_nominal\n")
    print(df[df.world == "mj_nominal"].groupby(["T", "regime"]).err.agg(lambda e: round((e <= HIT).mean(), 3)).unstack().to_string())
