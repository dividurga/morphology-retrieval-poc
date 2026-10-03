"""optimizer backend: make_model(params) / simulate(model, theta, duration) on the pybullet engine."""

import engines


def make_model(params, *args, **kwargs):
    # the twin as a person sets it up from the file: every rung of engines.LADDER. chosen by procedure.
    return engines.PyBulletEngine(params, fixes=tuple(engines.LADDER))


def simulate(model, theta, duration=8.0):
    return engines.rollout(model, theta, duration)
