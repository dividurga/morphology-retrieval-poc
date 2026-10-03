"""inverse-distance-weighted surrogate. swappable: fit(X, y), predict(X) -> (mean, nearest_dist)."""

import numpy as np


class IDW:
    def __init__(self, power: float = 2.0):
        self.power = power

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, float), np.asarray(y, float)
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        d = np.linalg.norm(X[:, None, :] - self.X[None, :, :], axis=2)
        near = d.min(axis=1)
        w = 1.0 / np.maximum(d, 1e-12) ** self.power
        mean = (w * self.y).sum(axis=1) / w.sum(axis=1)
        return mean, near
