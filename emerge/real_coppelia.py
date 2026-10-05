"""real A: the sim's coppeliasim models, made different in two ways.

parameters (always on): small per-world draws on the physical values.
  mass x U(0.9,1.1) with inertia, bullet friction x U(0.85,1.15), servo torque cap x U(0.9,1.1),
  setpoint noise x3, setpoints refreshed at 10 Hz (the paper's real robots). assumptions, flagged here.

model form (each can be switched off): structure the sim does not have.
  engine     the physics engine is ODE instead of Bullet 2.78. contact and constraint model change, nothing else.
  servo      the position loop is not ideal: setpoints pass a one-update delay, 0.29 deg quantisation, 0.5 deg backlash
             (assumption) and a 25 rpm slew limit (paper); the torque cap falls with joint speed, linear from the
             stall torque 1.8 Nm to zero at the no-load speed 97 rpm (paper table 1).
  connector  strength depends on direction and on vibration. torque limit 1.2 Nm and pull-out limit 58 N (paper table 1,
             the sim uses 1.0 Nm and 80 N). the torque limit drops by up to 10% with vibration, measured as a smoothed
             step-to-step change of the connector torque (paper: connectors let go lower under vibration). the 10% is
             from emerge/experiments/calibrate_break.py (the real B result). the vibration scale is an assumption.

    rollout(genome, world_seed, form=FORM) -> dict(distance, broken, fitness, wall)
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from emerge import sim_coppelia as S

SPREAD = dict(mass=0.10, friction=0.15, torque=0.10)
NOISE_FACTOR, UPDATE_EVERY = 3.0, 2
RESOLUTION = np.radians(0.29)  # AX-18A
BACKLASH = np.radians(0.5)  # assumption: play of the horn
SPEED_CAP = 25 * 2 * np.pi / 60  # rad/s, paper
NO_LOAD_SPEED = 97 * 2 * np.pi / 60  # rad/s, AX-18A at 12 V
TORQUE_LIMIT, PULLOUT_LIMIT = 1.2, 58.0  # Nm, N: paper table 1
VIBRATION_DROP, VIBRATION_SCALE = 0.10, 0.2  # at most 10% lower; 0.2 Nm step-to-step change counts as full vibration
EMA = 0.3


@dataclass
class Form:
    engine: bool = True
    servo: bool = True
    connector: bool = True


FORM = Form()


def rollout(genome, world_seed: int = 0, duration: float = 38.0, t_start: float = 6.0, form: Form = FORM, hook=None):
    """hook(sim, mods, sensors, k) runs after every sim step (used by emerge/visualize.py)."""
    n_act = sum(1 for m in genome.modules if m.type == "act")
    state = dict(base_force={}, delay=None, prev=None, last_send=None, thr_draw=None, tq_prev=None, vib=None)

    def perturb(sim, mods, sensors, rng):
        for m in mods:
            for sh in m["shapes"]:
                f = rng.uniform(1 - SPREAD["mass"], 1 + SPREAD["mass"])
                sim.setShapeMass(sh, sim.getShapeMass(sh) * f)
                inertia, com = sim.getShapeInertia(sh)
                sim.setShapeInertia(sh, [x * f for x in inertia], com)
                fp = sim.ode_body_friction if form.engine else sim.bullet_body_friction
                sim.setEngineFloatParam(fp, sh, sim.getEngineFloatParam(fp, sh)
                                        * rng.uniform(1 - SPREAD["friction"], 1 + SPREAD["friction"]))
            if m["joint"] is not None:
                state["base_force"][m["joint"]] = sim.getJointTargetForce(m["joint"]) * rng.uniform(1 - SPREAD["torque"], 1 + SPREAD["torque"])
                sim.setJointTargetForce(m["joint"], state["base_force"][m["joint"]])
        state["thr_draw"] = rng.uniform(0.9, 1.0, len(sensors))  # per-connector strength draw, the same range as real B
        state["tq_prev"] = np.zeros(len(sensors))
        state["vib"] = np.zeros(len(sensors))
        state["delay"] = {i: deque([0.0] * 1) for i in range(len(mods))}
        state["prev"] = {i: 0.0 for i in range(len(mods))}
        state["last_send"] = {i: 0.0 for i in range(len(mods))}
        if form.connector:
            for fs, u in zip(sensors, state["thr_draw"]):
                sim.setFloatProperty(fs, "torqueThreshold", TORQUE_LIMIT * u)
                sim.setFloatProperty(fs, "forceThreshold", PULLOUT_LIMIT * u)
        else:
            for fs in sensors:  # parametric only: the sim's rule with a small spread
                sim.setFloatProperty(fs, "torqueThreshold", sim.getFloatProperty(fs, "torqueThreshold") * rng.uniform(0.9, 1.1))

    def target_filter(i, k, target):
        if not form.servo:
            return target
        d = state["delay"][i]
        d.append(target)
        x = d.popleft()  # one setpoint update of delay
        x = np.round(x / RESOLUTION) * RESOLUTION
        p = state["prev"][i]
        if x > p + BACKLASH:
            p = x - BACKLASH
        elif x < p - BACKLASH:
            p = x + BACKLASH
        state["prev"][i] = p
        step = SPEED_CAP * UPDATE_EVERY * S.SIM_DT
        out = state["last_send"][i] + float(np.clip(p - state["last_send"][i], -step, step))
        state["last_send"][i] = out
        return out

    def after_step(sim, mods, sensors, k):
        if form.servo:
            for m in mods:
                j = m["joint"]
                if j is not None and j in state["base_force"]:
                    v = abs(sim.getJointVelocity(j))
                    sim.setJointTargetForce(j, state["base_force"][j] * max(0.02, 1.0 - v / NO_LOAD_SPEED))
        if form.connector:
            for n, fs in enumerate(sensors):
                r = sim.readForceSensor(fs)
                if r[0] & 2:
                    continue
                tq = float(np.linalg.norm(r[2]))
                state["vib"][n] = (1 - EMA) * state["vib"][n] + EMA * abs(tq - state["tq_prev"][n])
                state["tq_prev"][n] = tq
                drop = VIBRATION_DROP * min(1.0, state["vib"][n] / VIBRATION_SCALE)
                sim.setFloatProperty(fs, "torqueThreshold", TORQUE_LIMIT * state["thr_draw"][n] * (1 - drop))
        if hook is not None:
            hook(sim, mods, sensors, k)

    return S.rollout(genome, duration, t_start, seed=world_seed, perturb=perturb,
                     noise_std=S.NOISE_STD * NOISE_FACTOR, update_every=UPDATE_EVERY,
                     engine=S.api()[1].physics_ode if form.engine else None, target_filter=target_filter, after_step=after_step)
