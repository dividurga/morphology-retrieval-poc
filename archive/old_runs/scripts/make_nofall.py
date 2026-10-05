"""cheating filter: keep only filled rows that do not fall in the real (mujoco, the 2 fixed draws)."""
from multiprocessing import Pool

import pandas as pd

import disparity as D


def fell(r):
    return D.real_rollout(r)[1] > 0


if __name__ == "__main__":
    df = pd.read_csv("data_v1/dataset.csv")
    df = df[df.filled]
    with Pool(8) as p:
        f = p.map(fell, [r for _, r in df.iterrows()], chunksize=20)
    out = df[[not x for x in f]]
    print(f"{len(out)} of {len(df)} filled rows survive")
    out.to_csv("data_v1/dataset_nofall.csv", index=False)
