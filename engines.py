"""the same morphology.py mjcf, in two engines. only the physics engine differs.

mujoco loads it natively, pybullet through loadMJCF. both get torque from the same
command_to_torque(), the position law the original position actuator used. everything else
(contact, solver, integrator, joint limits, armature, damping, spring, friction import) is left to
each engine on purpose: those differences are the modelling gap this poc wants to see. nothing is
tuned to make the engines agree.
"""

import os
import tempfile

import mujoco
import numpy as np
import pybullet as p

import controller
import morphology
from environment import ResultOfARollout

DT = 0.005  # the mjcf's timestep. both engines step at this.
FALL_PITCH, FALL_HEIGHT_FRAC = 1.0, 0.5
RANGES = {"hip": morphology.HIP_RANGE, "knee": morphology.KNEE_RANGE}
NAMES = controller.ACTUATOR_ORDER  # hip_left, knee_left, hip_right, knee_right


def command_to_torque(params: dict, q: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """shared by both engines: tau = kp * (clipped target - q), the original position actuator."""
    kp = np.array([params["actuator_kp_hip"], params["actuator_kp_knee"]] * 2)
    lo = np.array([RANGES[n.split("_")[0]][0] for n in NAMES])
    hi = np.array([RANGES[n.split("_")[0]][1] for n in NAMES])
    return kp * (np.clip(targets, lo, hi) - q)


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


def rollout(eng, theta: np.ndarray, duration: float = 8.0, trace: float = 0.0):
    """open-loop rollout, same loop for both engines. fell = pitch > 1 rad or torso below half height.
    trace > 0 also returns (x, z, pitch) per control step for the first `trace` seconds."""
    eng.reset()
    tr = []
    for step in range(int(round(duration / DT))):
        tg = controller.joint_targets(theta, step * DT)
        sub = getattr(eng, "sub", 1)
        for _ in range(sub):
            eng.apply(command_to_torque(eng.params, eng.joints(), tg))
            eng.step()
        x, z, pitch = eng.state()
        if step * DT < trace:
            tr.append((x, z, pitch))
        if abs(pitch) > FALL_PITCH or z < FALL_HEIGHT_FRAC * eng.standing:
            res = ResultOfARollout(displacement=x, fell=True, fell_time=(step + 1) * DT)
            return (res, tr) if trace else res
    res = ResultOfARollout(displacement=eng.state()[0], fell=False, fell_time=None)
    return (res, tr) if trace else res
