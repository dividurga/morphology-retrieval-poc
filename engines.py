"""the same morphology.py mjcf, in two engines. only the physics engine differs.

mujoco loads it natively, pybullet through loadMJCF. both get torque from the same
command_to_torque(), the position law the original position actuator used. everything else
(contact, solver, integrator, joint limits, armature, damping, spring, friction import) is left to
each engine on purpose: those differences are the modelling gap this poc wants to see. nothing is
tuned to make the engines agree.
"""

import os
import tempfile

from collections import deque

import mujoco
import numpy as np
import pybullet as p

import controller
import morphology
import perturbations
from environment import ResultOfARollout

DT = 0.005  # the mjcf's timestep. both engines step at this.
FALL_PITCH, FALL_HEIGHT_FRAC = 1.0, 0.5
RANGES = {"hip": morphology.HIP_RANGE, "knee": morphology.KNEE_RANGE}
NAMES = controller.ACTUATOR_ORDER  # hip_left, knee_left, hip_right, knee_right


def command_to_torque(params: dict, q: np.ndarray, targets: np.ndarray, deadband: float = 0.0) -> np.ndarray:
    """shared by both engines: tau = kp * (clipped target - q), the original position actuator.
    deadband (rad) shrinks the error toward zero, the gearbox backlash of ModelForm. 0 = the original law."""
    kp = np.array([params["actuator_kp_hip"], params["actuator_kp_knee"]] * 2)
    lo = np.array([RANGES[n.split("_")[0]][0] for n in NAMES])
    hi = np.array([RANGES[n.split("_")[0]][1] for n in NAMES])
    e = np.clip(targets, lo, hi) - q
    if deadband:
        e = np.sign(e) * np.maximum(np.abs(e) - deadband, 0.0)
    return kp * e


class MuJoCoEngine:
    name = "mujoco"

    def __init__(self, params: dict):
        self.params = params
        self.m = mujoco.MjModel.from_xml_string(morphology.build_biped_xml(params, torque_actuators=True))
        self.d = mujoco.MjData(self.m)
        self.qa = [int(self.m.joint(n).qposadr[0]) for n in NAMES]
        self.standing = morphology.root_height(params)

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)

    def joints(self):
        return self.d.qpos[self.qa].copy()

    def velocities(self):
        return self.d.qvel[[int(self.m.joint(n).dofadr[0]) for n in NAMES]].copy()

    def apply(self, tau):
        self.d.ctrl[:] = tau

    def step(self):
        mujoco.mj_step(self.m, self.d)

    def state(self):
        """x, torso z, pitch"""
        return (float(self.d.qpos[self.m.joint("root_x").qposadr[0]]),
                self.standing + float(self.d.qpos[self.m.joint("root_z").qposadr[0]]),
                float(self.d.qpos[self.m.joint("root_pitch").qposadr[0]]))


# analytic solid-shape inertia, from geometry. pybullet's own capsule inertia is a bounding box.
def capsule_inertia(rho, r, length):
    m_cyl = rho * np.pi * r ** 2 * length
    m_sph = rho * (4.0 / 3.0) * np.pi * r ** 3
    d = length / 2 + 3 * r / 8
    t = m_cyl * (3 * r ** 2 + length ** 2) / 12 + m_sph * ((83.0 / 320.0) * r ** 2 + d ** 2)
    return (t, t, m_cyl * r ** 2 / 2 + 0.4 * m_sph * r ** 2)


def box_inertia(rho, hx, hy, hz):
    m = rho * 8 * hx * hy * hz
    a, b, c = 2 * hx, 2 * hy, 2 * hz
    return (m * (b ** 2 + c ** 2) / 12, m * (a ** 2 + c ** 2) / 12, m * (a ** 2 + b ** 2) / 12)


# what a person sets up after loading the file, in order. every one comes from the file or the
# engine's documented behaviour. none looks at the real sim's output.
LADDER = ["friction", "damping", "spring", "inertia", "armature", "substeps"]


class PyBulletEngine:
    name = "pybullet"

    def __init__(self, params: dict, fixes: tuple = (), substeps: int = 1):
        self.params, self.fixes = params, set(fixes)
        self.sub = 5 if "substeps" in self.fixes else substeps
        self.c = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.c)
        p.setTimeStep(DT / self.sub, physicsClientId=self.c)
        f = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
        f.write(morphology.build_biped_xml(params, torque_actuators=True))
        f.close()
        ids = p.loadMJCF(f.name, physicsClientId=self.c)
        os.unlink(f.name)
        self.body = [b for b in ids if p.getNumJoints(b, physicsClientId=self.c) > 0][0]
        self.idx, link = {}, {}
        for i in range(p.getNumJoints(self.body, physicsClientId=self.c)):
            info = p.getJointInfo(self.body, i, physicsClientId=self.c)
            self.idx[info[1].decode()], link[info[12].decode()] = i, i
            p.setJointMotorControl2(self.body, i, p.VELOCITY_CONTROL, force=0, physicsClientId=self.c)  # brake off
        self.ji = [self.idx[n] for n in NAMES]
        self.link = link
        self.torso = link["torso"]
        self.standing = morphology.root_height(params)
        # loadMJCF applies the torso's pos once per root joint, so the torso loads too high. a
        # loader artifact, not physics. start the z slide joint at the offset that restores the
        # designed height. every other joint starts at 0.
        self.init = [0.0] * p.getNumJoints(self.body, physicsClientId=self.c)
        self._reset_to(self.init)
        z = p.getLinkState(self.body, self.torso, computeForwardKinematics=True, physicsClientId=self.c)[0][2]
        self.init[self.idx["root_z"]] = self.standing - z
        self._apply_fixes()

    def _apply_fixes(self):
        P, c, f = self.params, self.c, self.fixes
        bodies = ["torso"] + [f"{s}_{k}" for k in ("left", "right") for s in ("thigh", "shank", "foot")]
        if "friction" in f:
            # pybullet multiplies the two surfaces' friction, mujoco takes the max. the file has robot
            # 0.05 and floor 1.0, so the effective value is 1.0. floor stays 1.0, robot becomes 1.0.
            eff = max(float(morphology.ROBOT_FRICTION.split()[0]), morphology.GROUND_FRICTION_NOMINAL)
            p.changeDynamics(0, -1, lateralFriction=morphology.GROUND_FRICTION_NOMINAL, physicsClientId=c)
            for b in bodies:
                p.changeDynamics(self.body, self.link[b], lateralFriction=eff / morphology.GROUND_FRICTION_NOMINAL, physicsClientId=c)
        if "damping" in f:  # joint damping is in the file but loadMJCF drops it
            for n in ("hip_left", "knee_left", "hip_right", "knee_right"):
                p.changeDynamics(self.body, self.idx[n], jointDamping=morphology.JOINT_DAMPING, physicsClientId=c)
            for n in ("ankle_left", "ankle_right"):
                p.changeDynamics(self.body, self.idx[n], jointDamping=P["ankle_damping"], physicsClientId=c)
        inertia = {}
        if "inertia" in f or "armature" in f:
            rho, h = P["density"], P["torso_size"] / 2.0
            inertia["torso"] = list(capsule_inertia(rho, morphology.TORSO_RADIUS_RATIO * P["torso_size"], 2 * h))
            for k in ("left", "right"):
                inertia[f"thigh_{k}"] = list(capsule_inertia(rho, morphology.THIGH_RADIUS, P[f"thigh_length_{k}"]))
                inertia[f"shank_{k}"] = list(capsule_inertia(rho, morphology.SHANK_RADIUS, P[f"shank_length_{k}"]))
                inertia[f"foot_{k}"] = list(box_inertia(rho, P[f"foot_length_{k}"] / 2, morphology.FOOT_HALF_WIDTH, morphology.FOOT_HALF_THICK))
            if "inertia" not in f:  # armature without the audit: start from pybullet's own numbers
                for b in inertia:
                    inertia[b] = list(p.getDynamicsInfo(self.body, self.link[b], physicsClientId=c)[2])
            if "armature" in f:  # the file gives every joint armature 0.01. lump it on the link the joint drives.
                for k in ("left", "right"):
                    for b in (f"thigh_{k}", f"shank_{k}", f"foot_{k}"):
                        inertia[b][1] += morphology.JOINT_ARMATURE
            for b, diag in inertia.items():
                p.changeDynamics(self.body, self.link[b], localInertiaDiagonal=diag, physicsClientId=c)

    def _reset_to(self, vals):
        for i, v in enumerate(vals):
            p.resetJointState(self.body, i, v, 0.0, physicsClientId=self.c)

    def reset(self):
        self._reset_to(self.init)

    def joints(self):
        return np.array([x[0] for x in p.getJointStates(self.body, self.ji, physicsClientId=self.c)])

    def velocities(self):
        return np.array([x[1] for x in p.getJointStates(self.body, self.ji, physicsClientId=self.c)])

    def apply(self, tau):
        p.setJointMotorControlArray(self.body, self.ji, p.TORQUE_CONTROL, forces=list(tau),
                                    physicsClientId=self.c)
        if "spring" in self.fixes:  # ankle spring is in the file but loadMJCF drops it
            for n in ("ankle_left", "ankle_right"):
                q = p.getJointState(self.body, self.idx[n], physicsClientId=self.c)[0]
                p.setJointMotorControl2(self.body, self.idx[n], p.TORQUE_CONTROL,
                                        force=-self.params["ankle_stiffness"] * q, physicsClientId=self.c)

    def step(self):
        p.stepSimulation(physicsClientId=self.c)

    def state(self):
        return (p.getJointState(self.body, self.idx["root_x"], physicsClientId=self.c)[0],
                p.getLinkState(self.body, self.torso, computeForwardKinematics=True, physicsClientId=self.c)[0][2],
                p.getJointState(self.body, self.idx["root_pitch"], physicsClientId=self.c)[0])


SETTLE = 0.5  # s. hold the spawn pose (all joint targets 0) first, so contacts relax before the gait starts


class ModelForm:
    """model-form error between a controller and the actuators it was designed for. engine-agnostic: applied in rollout,
    so pybullet and mujoco get exactly the same error. an Unitree A1/Go1-class QDD joint (33.5 Nm peak, 21 rad/s no-load,
    6.33:1 planetary gearbox) fits this biped's size (about 17 kg, 0.22 m links). values from the public spec sheets, from
    memory: check the datasheet before citing. fixed in advance, never tuned against any engine's output.
    chain per control substep: sensed q (quantised, noisy) -> PD with backlash deadband -> speed-torque limit
    -> Coulomb gear friction -> first-order lag."""

    def __init__(self, tau_max=33.5, omega_free=21.0, coulomb=0.2, backlash_deg=0.5, lag_tau=0.005,
                 enc_bits=14, enc_noise=1e-3, seed=0):
        self.tau_max, self.omega_free, self.coulomb = tau_max, omega_free, coulomb
        self.deadband = np.deg2rad(backlash_deg) / 2.0  # half the gap each side of the target
        self.lag_tau, self.enc_step, self.enc_noise, self.seed = lag_tau, 2 * np.pi / 2 ** enc_bits, enc_noise, seed

    def start(self):
        """fresh per-rollout state, so a rollout is deterministic."""
        self.rng, self.tau_state = np.random.default_rng(self.seed), np.zeros(4)

    def sense(self, q):
        return np.round((q + self.rng.normal(0.0, self.enc_noise, q.shape)) / self.enc_step) * self.enc_step

    def torque(self, tau, w, dt):
        lim = self.tau_max * np.clip(1.0 - np.abs(w) / self.omega_free, 0.0, 1.0)
        tau = np.clip(tau, -lim, lim) - self.coulomb * np.tanh(w / 0.1)
        self.tau_state = self.tau_state + (dt / (self.lag_tau + dt)) * (tau - self.tau_state)
        return self.tau_state


def rollout(eng, theta: np.ndarray, duration: float = 8.0, trace: float = 0.0, settle: float = SETTLE, mf=None):
    """open-loop rollout, same loop for both engines. fell = pitch > 1 rad or torso below half height.
    settle seconds first hold the spawn pose (all targets 0). distance, fall time and trace are all measured from the
    end of settling. a fall during settling counts as fell. trace > 0 also returns (x, z, pitch) per control step for the
    first `trace` seconds of the gait. mf = a ModelForm, or None for the plain engine. settle=0, mf=None is the old loop."""
    eng.reset()
    tr = []
    sub = getattr(eng, "sub", 1)
    motor, delay = getattr(eng, "motor", 1.0), getattr(eng, "delay", 0)
    lag = deque([np.zeros(4)] * (delay * sub))  # actuator lag: torque computed `delay` control steps ago
    if mf:
        mf.start()
    n_settle = int(round(settle / DT))
    x0 = 0.0  # the torso starts at x = 0 in both engines. after settling, x0 is where the gait starts from
    for step in range(n_settle + int(round(duration / DT))):
        t = max(0, step - n_settle) * DT
        tg = np.zeros(4) if step < n_settle else controller.joint_targets(theta, t)  # settling holds the spawn pose
        for _ in range(sub):
            q = eng.joints()
            if mf:
                tau = motor * command_to_torque(eng.params, mf.sense(q), tg, mf.deadband)
                tau = mf.torque(tau, eng.velocities(), DT / sub)
            else:
                tau = motor * command_to_torque(eng.params, q, tg)
            if lag:
                lag.append(tau)
                tau = lag.popleft()
            eng.apply(tau)
            eng.step()
        x, z, pitch = eng.state()
        if step == n_settle - 1:
            x0 = x
        t_gait = (step + 1 - n_settle) * DT
        if step >= n_settle and t_gait - DT < trace:
            tr.append((x - x0, z, pitch))
        if abs(pitch) > FALL_PITCH or z < FALL_HEIGHT_FRAC * eng.standing:
            res = ResultOfARollout(displacement=x - x0, fell=True, fell_time=max(t_gait, 0.0))
            return (res, tr) if trace else res
    res = ResultOfARollout(displacement=eng.state()[0] - x0, fell=False, fell_time=None)
    return (res, tr) if trace else res


# a "reality" is the same engine with a lag and a set of multipliers (perturbations.py), applied once after the
# engine is built. same spec for both engines, so a gap here is structured and the engine is not the variable.

def _pybullet_perturb(eng, friction, mass, damping):
    """absolute, not cumulative: nominal values are cached on first use, so one engine can be re-perturbed."""
    c = eng.c
    if not hasattr(eng, "_nominal"):
        eng._nominal = {}
        for i in range(-1, p.getNumJoints(eng.body, physicsClientId=c)):
            m, _, inertia = p.getDynamicsInfo(eng.body, i, physicsClientId=c)[:3]
            eng._nominal[i] = (m, list(inertia))
    p.changeDynamics(0, -1, lateralFriction=morphology.GROUND_FRICTION_NOMINAL * friction, physicsClientId=c)
    for i, (m, inertia) in eng._nominal.items():
        if m > 0:
            p.changeDynamics(eng.body, i, mass=m * mass, localInertiaDiagonal=[x * mass for x in inertia], physicsClientId=c)
    for n in perturbations.ACTUATED_JOINTS:
        p.changeDynamics(eng.body, eng.idx[n], jointDamping=morphology.JOINT_DAMPING * damping, physicsClientId=c)


def apply_reality(eng, spec: dict):
    """spec: delay (control steps), friction, mass, motor, damping."""
    eng.motor, eng.delay = spec["motor"], spec["delay"]
    if eng.name == "mujoco":
        perturbations.apply_perturbation(eng.m, friction=spec["friction"], mass=spec["mass"], motor=1.0,
                                         damping=spec["damping"])
    else:
        _pybullet_perturb(eng, spec["friction"], spec["mass"], spec["damping"])
    return eng


def make_engine(kind: str, params: dict):
    """the twin in each engine. pybullet = the full setup ladder (what a person does). mujoco = native."""
    if kind == "mujoco":
        return MuJoCoEngine(params)
    return PyBulletEngine(params, fixes=tuple(LADDER))
