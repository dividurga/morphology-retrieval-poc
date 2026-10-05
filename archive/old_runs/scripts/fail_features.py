"""per dr row: twin-trace features (sim behaviour) and mujoco survival labels. -> results/dr/features.csv

    uv run python fail_features.py --workers 2
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pybullet as p

import disparity as D
import engines

EP = D.EPISODE
IDENT = {"friction": 1.0, "mass": 1.0, "motor": 1.0, "damping": 1.0, "delay": 0}


def trace_stats(eng, theta, prefix):
    res, tr = engines.rollout(eng, theta, EP, trace=EP)
    n = int(round(EP / engines.DT))
    tr = np.array(tr)  # (steps, 3): x, z, pitch. shorter than n if it fell
    steps = len(tr)
    x, z, pitch = tr[:, 0], tr[:, 1], tr[:, 2]
    standing = eng.standing
    margin_t = np.minimum(1 - np.abs(pitch) / engines.FALL_PITCH, (z / standing - 0.5) / 0.5)
    margin = max(float(margin_t.min()), 0.0) if not res.fell else 0.0
    v = np.diff(x) / engines.DT if steps > 2 else np.zeros(1)
    return {f"{prefix}_fell": float(res.fell), f"{prefix}_t": steps * engines.DT, f"{prefix}_dist": float(res.displacement),
            f"{prefix}_margin": margin, f"{prefix}_pitch_std": float(pitch.std()), f"{prefix}_pitch_absmax": float(np.abs(pitch).max()),
            f"{prefix}_pitch_mean": float(pitch.mean()), f"{prefix}_zmin": float(z.min() / standing), f"{prefix}_zstd": float(z.std() / standing),
            f"{prefix}_vmean": float(v.mean()), f"{prefix}_vstd": float(v.std()),
            f"{prefix}_margin_t": float(margin_t.min())}


def row_task(args):
    key, params, theta = args
    out = dict(key)
    pb = engines.make_engine("pybullet", params)
    engines.apply_reality(pb, IDENT)
    out.update(trace_stats(pb, theta, "pbn"))
    for rung in ("R0", "R1", "R2"):
        acc = []
        for sp in D.reality_specs(rung):
            engines.apply_reality(pb, sp)
            acc.append(trace_stats(pb, theta, "tmp"))
        for k in acc[0]:
            out[k.replace("tmp", f"pb{rung}")] = float(np.mean([a[k] for a in acc]))
    p.disconnect(pb.c)
    mj = engines.MuJoCoEngine(params)
    out.update(trace_stats(mj, theta, "mjn"))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/dr/features.csv")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open("results/dr/run1.jsonl")]
    rows = [r for r in rows if r["regime"] != "nominal"]
    if a.limit:
        rows = rows[: a.limit]
    tasks = [(dict(morph_id=r["morph_id"], T=r["T"], regime=r["regime"]), r["params"], np.array(r["theta"])) for r in rows]
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(row_task, tasks, chunksize=4))
    pd.DataFrame(out).to_csv(a.out, index=False)
    print(len(out), "rows ->", a.out)
