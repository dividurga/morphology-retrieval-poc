"""checks on a generated dataset: determinism of the relabeled rows, coverage, failure range.

    uv run python check_dataset.py data_pilot
"""

import glob
import sys

import numpy as np

import engines
import generate_dataset as G

if __name__ == "__main__":
    root = sys.argv[1]
    bad, n_rows, bins_per, best = 0, 0, [], []
    covered = {}
    files = sorted(glob.glob(f"{root}/archive/*.npz"))
    step = max(1, len(files) // int(sys.argv[2])) if len(sys.argv) > 2 else 1  # optional: check every k-th body
    for f in files[::step]:
        z = np.load(f)
        params = dict(zip([str(k) for k in z["param_names"]], z["params"].tolist()))
        reps = {b: int(i) for b, i in enumerate(z["rep_index"]) if i >= 0}
        eng = engines.PyBulletEngine(params, fixes=tuple(engines.LADDER))
        worst = 0.0
        for b, i in reps.items():
            r = engines.rollout(eng, z["theta"][i], G.EPISODE)
            worst = max(worst, abs(r.displacement - z["distance"][i]) + (1.0 if r.fell else 0.0))
            n_rows += 1
            covered[b] = covered.get(b, 0) + 1
        bad += worst > 1e-9
        bins_per.append(len(reps)); best.append(z["distance"][~z["fell"]].max() if (~z["fell"]).any() else np.nan)

    print(f"\nrows {n_rows}; morphologies with a re-evaluation mismatch: {bad}")
    print("bins filled per morphology:", bins_per)
    print("morphologies with zero rows:", sum(1 for b in bins_per if b == 0), "of", len(bins_per))
    edges = np.arange(0, 3.2, 0.2)
    print("rows per 0.2 m of achieved distance (all morphologies):")
    for lo in edges:
        c = sum(v for b, v in covered.items() if lo <= b * G.BIN < lo + 0.2)
        print(f"  {lo:.1f}-{lo + 0.2:.1f} m: {c:3d} " + "#" * c)
