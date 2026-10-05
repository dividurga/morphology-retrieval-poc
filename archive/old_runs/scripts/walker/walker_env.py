"""one walker2d mjcf, two engines. only the physics engine differs.

the file is gymnasium's walker2d_v5.xml, unmodified. mujoco loads it natively, pybullet through
loadMJCF. both engines get torque from the same command_to_torque(). everything else (contact,
solver, integrator, limits, friction import, armature, damping) is left to each engine on purpose:
those differences are the modelling gap this project wants to see. no tuning either side.
"""

import os

import mujoco
import numpy as np
import pybullet as p

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "walker2d_scaled.xml")  # walker2d_v5 x0.532, see scale_mjcf.py
# order of the actuators in the file
JOINTS = ["thigh_joint", "leg_joint", "foot_joint", "thigh_left_joint", "leg_left_joint", "foot_left_joint"]

RANGES = {"thigh_joint": (-2.618, 0.0), "leg_joint": (-2.618, 0.0), "foot_joint": (-0.785, 0.785),
          "thigh_left_joint": (-2.618, 0.0), "leg_left_joint": (-2.618, 0.0), "foot_left_joint": (-0.785, 0.785)}
TORSO_Z0_FULL = 1.25  # torso height of the unscaled walker2d_v5.xml

DT = 0.002  # the file's timestep. both engines step at this.
SUBSTEPS = 4  # control period 0.008 s, as gymnasium's frame_skip 4
CONTROL_DT = DT * SUBSTEPS


def _torso_z0() -> float:
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return float(d.xpos[m.body("torso").id][2])


TORSO_Z0 = _torso_z0()  # torso height at the designed pose, read from the file
SCALE = TORSO_Z0 / TORSO_Z0_FULL
GEAR = float(mujoco.MjModel.from_xml_path(XML).actuator_gear[0][0])  # n*m per unit control, from the file
# pd gains: 250 n*m/rad and 8 n*m*s/rad at full size. an assumption (the file has torque motors, no
# servo), scaled with the body like the gear: torque ~ s^4, torque-per-(rad/s) ~ s^4.5.
KP, KD = 250.0 * SCALE ** 4, 8.0 * SCALE ** 4.5
FALL_ANGLE, FALL_Z_FRAC = 1.0, 0.64  # healthy if |pitch| < 1 and torso z > 0.8 (of 1.25)


def command_to_torque(q: np.ndarray, qd: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """shared by both engines: target angles -> normalised control in [-1, 1]."""
    lo = np.array([RANGES[j][0] for j in JOINTS])
    hi = np.array([RANGES[j][1] for j in JOINTS])
    tau = KP * (np.clip(targets, lo, hi) - q) - KD * qd
    return np.clip(tau / GEAR, -1.0, 1.0)


class MuJoCoWalker:
    name = "mujoco"

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(XML)
        self.d = mujoco.MjData(self.m)
        assert [self.m.actuator(i).name for i in range(self.m.nu)] == [f"{j}" for j in
                [self.m.actuator(i).name for i in range(self.m.nu)]]
        self.qa = [int(self.m.joint(j).qposadr[0]) for j in JOINTS]
        self.va = [int(self.m.joint(j).dofadr[0]) for j in JOINTS]
        # actuators are listed in the file in JOINTS order
        self.torso = self.m.body("torso").id
        self.z0 = None

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self.z0 = float(self.d.xpos[self.torso][2])

    def joints(self):
        return self.d.qpos[self.qa].copy(), self.d.qvel[self.va].copy()

    def apply(self, u):
        self.d.ctrl[:] = u

    def step(self):
        mujoco.mj_step(self.m, self.d)

    def state(self):
        """x, torso z, pitch"""
        return (float(self.d.qpos[self.m.joint("rootx").qposadr[0]]), float(self.d.xpos[self.torso][2]),
                float(self.d.qpos[self.m.joint("rooty").qposadr[0]]))


class PyBulletWalker:
    name = "pybullet"

    def __init__(self):
        self.c = p.connect(p.DIRECT)
        p.setGravity(0, 0, -9.81, physicsClientId=self.c)
        p.setTimeStep(DT, physicsClientId=self.c)
        ids = p.loadMJCF(XML, physicsClientId=self.c)
        self.body = [b for b in ids if p.getNumJoints(b, physicsClientId=self.c) > 0][0]
        self.idx, self.link = {}, {}
        for i in range(p.getNumJoints(self.body, physicsClientId=self.c)):
            info = p.getJointInfo(self.body, i, physicsClientId=self.c)
            self.idx[info[1].decode()] = i
            self.link[info[12].decode()] = i
            # pybullet brakes every joint by default. torque control needs that off.
            p.setJointMotorControl2(self.body, i, p.VELOCITY_CONTROL, force=0, physicsClientId=self.c)
        self.ji = [self.idx[j] for j in JOINTS]
        self.torso = self.link["torso"]
        # importer quirk: the torso body carries three root joints (x, z, pitch) and loadMJCF
        # applies the body's pos once per joint, so the torso loads at z=2.5 instead of the file's
        # 1.25. a loader artifact, not physics. start the z slide joint at the offset that puts
        # the torso at the designed height. every other joint starts at 0.
        self.init = [0.0] * p.getNumJoints(self.body, physicsClientId=self.c)
        for i, v in enumerate(self.init):
            p.resetJointState(self.body, i, 0.0, 0.0, physicsClientId=self.c)
        z_loaded = p.getLinkState(self.body, self.torso, computeForwardKinematics=True,
                                  physicsClientId=self.c)[0][2]
        self.init[self.idx["rootz"]] = TORSO_Z0 - z_loaded
        self.u = np.zeros(len(JOINTS))
        self.z0 = None

    def reset(self):
        for i, v in enumerate(self.init):
            p.resetJointState(self.body, i, v, 0.0, physicsClientId=self.c)
        self.z0 = float(p.getLinkState(self.body, self.torso, computeForwardKinematics=True, physicsClientId=self.c)[0][2])

    def joints(self):
        s = p.getJointStates(self.body, self.ji, physicsClientId=self.c)
        return np.array([x[0] for x in s]), np.array([x[1] for x in s])

    def apply(self, u):
        self.u = u
        p.setJointMotorControlArray(self.body, self.ji, p.TORQUE_CONTROL, forces=list(GEAR * np.asarray(u)),
                                    physicsClientId=self.c)

    def step(self):
        p.stepSimulation(physicsClientId=self.c)

    def state(self):
        return (p.getJointState(self.body, self.idx["rootx"], physicsClientId=self.c)[0],
                float(p.getLinkState(self.body, self.torso, computeForwardKinematics=True, physicsClientId=self.c)[0][2]),
                p.getJointState(self.body, self.idx["rooty"], physicsClientId=self.c)[0])


ENGINES = {"mujoco": MuJoCoWalker, "pybullet": PyBulletWalker}

# open-loop gait: omega, then (bias, amplitude, phase) for thigh_left, leg_left, thigh, leg. feet hold 0.
# bias bounds include 0 (straight legs, which stands in both engines). the start is near standing.
THETA_BOUNDS = [(2.0, 12.0)] + [(-1.5, 0.0), (0.0, 0.8), (0.0, 2 * np.pi),
                               (-1.8, 0.0), (0.0, 0.9), (0.0, 2 * np.pi)] * 2
DEFAULT_THETA = np.array([3.0, -0.02, 0.05, 0.0, -0.05, 0.05, np.pi / 2, -0.02, 0.05, np.pi, -0.05, 0.05, 3 * np.pi / 2])


def targets(theta: np.ndarray, t: float) -> np.ndarray:
    w = theta[0]
    s = lambda k: theta[1 + 3 * k] + theta[2 + 3 * k] * np.sin(w * t + theta[3 + 3 * k])
    # order of JOINTS: thigh, leg, foot, thigh_left, leg_left, foot_left
    return np.array([s(2), s(3), 0.0, s(0), s(1), 0.0])


def rollout(env, theta: np.ndarray, duration: float = 8.0):
    """returns (distance, fell, fell_time). same loop for both engines."""
    env.reset()
    x0 = env.state()[0]
    for step in range(int(round(duration / CONTROL_DT))):
        tg = targets(theta, step * CONTROL_DT)
        for _ in range(SUBSTEPS):
            q, qd = env.joints()
            env.apply(command_to_torque(q, qd, tg))
            env.step()
        x, z, pitch = env.state()
        if abs(pitch) > FALL_ANGLE or z < FALL_Z_FRAC * env.z0:
            return x - x0, True, (step + 1) * CONTROL_DT
    return env.state()[0] - x0, False, None
