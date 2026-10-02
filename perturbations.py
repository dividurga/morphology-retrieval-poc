"""Mutate a compiled MuJoCo model in place to simulate physical model mismatch.
ok so in this file we create other versions of the sim
im still a bit unclear as to what other versions of sim should be bcs
right now its just different friction, mass, motor strength, and damping.
but i think there are way more things another simulator could model that this one doesnt
this is to the point of yk an optimization overfiitting to a specific sim and not generalizing to other sims
"""

import numpy as np

ACTUATED_JOINTS = ["hip_left", "knee_left", "hip_right", "knee_right"]

PERTURBATION_RANGES = {
    "friction": (0.6, 1.4),
    "mass": (0.85, 1.15),
    "motor": (0.8, 1.2),
    "damping": (0.75, 1.25),
}

NUMERICAL_PERTURBATION_RANGES = {
    # this is 100% just off of vibes, idk what to do here bnut im just
    # trying this out for now okay
    "timestep": (0.8, 1.2),
    "iterations": (0.8, 1.2),
}

def apply_perturbation(model, friction: float = 1.0, mass: float = 1.0,
                        motor: float = 1.0, damping: float = 1.0):
    floor_id = model.geom("floor").id
    model.geom_friction[floor_id] *= friction

    model.body_mass[1:] *= mass
    model.body_inertia[1:] *= mass

    for name in ACTUATED_JOINTS:
        act_id = model.actuator(name).id
        model.actuator_gainprm[act_id, 0] *= motor
        model.actuator_biasprm[act_id, 1] *= motor

        dof_id = model.jnt_dofadr[model.joint(name).id]
        model.dof_damping[dof_id] *= damping

    return model


def apply_numerical_perturbation(model, timestep: float = 1.0, iterations: float = 1.0):
    model.opt.timestep *= timestep
    model.opt.iterations = max(1, round(model.opt.iterations * iterations))
    return model


def sample_combined(rng: np.random.Generator, n: int = 20) -> list[dict]:
    """Draw n independent joint samples of all 4 multipliers, uniform within PERTURBATION_RANGES."""
    samples = []
    for _ in range(n):
        s = {k: rng.uniform(lo, hi) for k, (lo, hi) in PERTURBATION_RANGES.items()}
        samples.append(s)
    return samples


def sample_numerical(rng: np.random.Generator, n: int = 20) -> list[dict]:
    """Draw n independent joint samples of both multipliers, uniform within NUMERICAL_PERTURBATION_RANGES."""
    samples = []
    for _ in range(n):
        s = {k: rng.uniform(lo, hi) for k, (lo, hi) in NUMERICAL_PERTURBATION_RANGES.items()}
        samples.append(s)
    return samples


if __name__ == "__main__":
    import controller
    import environment
    import morphology

    params = morphology.NOMINAL_PARAMS
    theta = controller.DEFAULT_THETA

    baseline_model = environment.make_model(params)
    baseline = environment.simulate(baseline_model, theta, duration=8.0)
    print("baseline:      ", baseline)

    for dim, (lo, hi) in PERTURBATION_RANGES.items():
        for label, value in [("low", lo), ("high", hi)]:
            m = environment.make_model(params)
            apply_perturbation(m, **{dim: value})
            r = environment.simulate(m, theta, duration=8.0)
            print(f"{dim:8s} {label:4s} ({value:.2f}x): {r}")
