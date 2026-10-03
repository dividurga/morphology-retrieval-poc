"""optimizer backend: make_model(params) / simulate(model, theta, duration) on the mujoco engine."""

import engines


def make_model(params, *args, **kwargs):
    return engines.MuJoCoEngine(params)


def simulate(model, theta, duration=8.0):
    return engines.rollout(model, theta, duration)
