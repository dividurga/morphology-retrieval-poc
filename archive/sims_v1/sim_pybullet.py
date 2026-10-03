"""pybullet sim. best-effort twin of sim_mujoco.py. built from spec.py.

built with createMultiBody. loadMJCF drops hinge limits and cannot be fixed afterwards.
planar root: mujoco puts x, z, pitch on one body. pybullet allows one joint per link, so x and z
become two massless prismatic dummy links feeding the torso, which carries pitch.

every pybullet quirk below was a real bug. do not remove one without re-running audit_sims.py.
"""

from dataclasses import dataclass

import numpy as np
import pybullet as p

import controller
import spec as spec_mod
from sim_common import ResultOfARollout, has_fallen
from spec import SPEC, Spec

LEG_JOINTS = ("hip_left", "knee_left", "ankle_left", "hip_right", "knee_right", "ankle_right")
# the 9 robot dofs, in mujoco's qpos order. pybullet re-sorts links depth-first, so a joint's
# index is NOT its creation order. always go through PyBulletModel.joint_index / .dofs.
ROBOT_DOFS = ("dummy_x", "dummy_z", "torso") + LEG_JOINTS


@dataclass(frozen=True)
class PbConfig:
    """how the pybullet twin represents the spec. every value MUST come from the spec or from
    the equations of motion. MUST NOT be tuned against mujoco output: mujoco is the real
    robot, and fitting to it would leak the very data the method is meant to get only
    sparsely. audit_sims.py measures the gap. it MUST NOT feed back in here.

    armature has no per-joint equivalent in pybullet. a is spread over links so the hip and
    knee diagonals of the mass matrix gain exactly a (derived, see arm_*).
    """
    substeps: int = 10  # physics steps per control step. targets are held. at 5 a heavily damped pad
                        # blew up on impact (pad at 14 m/s, robot at 26 m/s) and the optimizer
                        # exploited it. 10 and 20 agree with each other and with mujoco.
    armature: float | None = None  # a to emulate. None = spec.armature, 0 = none
    # hip diag = thigh + shank, knee diag = shank. both must gain a, so thigh 0, shank a. the two
    # shanks then add 2a to the pitch diagonal, which a torso trim of -2a cancels. the trim is
    # limited to keep ixx <= iyy + izz, so it is partial at small scale. cross terms stay wrong.
    arm_thigh: float = 0.0  # fraction of a added to thigh iyy
    arm_shank: float = 1.0
    arm_foot: float = 0.0
    arm_torso: float = -2.0


PB_CFG = PbConfig()


@dataclass
class PyBulletModel:
    client: int
    body_id: int
    joint_index: dict  # joint name -> pybullet joint index
    dofs: list  # joint indices of ROBOT_DOFS, in that order
    spec: Spec
    cfg: PbConfig
    params: dict
    standing_height: float
    kp: dict  # "hip" / "knee" -> position gain
    ankle_stiffness: float
    pad_mass: dict  # "left" / "right" -> pad link mass, for the implicit pad integration


def make_model(params: dict, spec: Spec = SPEC, cfg: PbConfig = PB_CFG,
               client: int | None = None) -> PyBulletModel:
    if client is None:
        client = p.connect(p.DIRECT)
    p.resetSimulation(physicsClientId=client)
    p.setGravity(0, 0, spec.gravity, physicsClientId=client)
    p.setTimeStep(spec.timestep / cfg.substeps, physicsClientId=client)

    h = params["torso_size"] / 2.0
    density = params["density"]
    root_z = spec.root_z(params)

    floor = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client),
        basePosition=[0, 0, 0], physicsClientId=client)
    # mujoco uses max(robot, floor). pybullet multiplies. pick the robot value so the product
    # equals mujoco's.
    effective = max(spec.robot_friction[0], spec.floor_friction[0])
    p.changeDynamics(floor, -1, lateralFriction=spec.floor_friction[0], physicsClientId=client)
    robot_friction = effective / spec.floor_friction[0]

    masses, col, vis, pos, com, inertia = [], [], [], [], [], {}
    parent, jtype, axis, names = [], [], [], []

    def add(name, parent_idx, mass, col_shape, joint_type, ax, position, vis_shape=-1,
            com_offset=(0, 0, 0), inertia_diag=None):
        names.append(name)
        masses.append(mass)
        col.append(col_shape)
        vis.append(vis_shape)
        pos.append(position)
        com.append(list(com_offset))
        parent.append(0 if parent_idx is None else parent_idx + 1)
        jtype.append(joint_type)
        axis.append(ax)
        if inertia_diag is not None:
            inertia[name] = inertia_diag
        return len(names) - 1

    dummy_x = add("dummy_x", None, 0.0, -1, p.JOINT_PRISMATIC, [1, 0, 0], [0, 0, root_z])
    dummy_z = add("dummy_z", dummy_x, 0.0, -1, p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0])

    torso_r = spec.torso_radius(params)
    torso = add(
        "torso", dummy_z, spec_mod.capsule_mass(density, torso_r, 2 * h),
        p.createCollisionShape(p.GEOM_CAPSULE, radius=torso_r, height=2 * h, physicsClientId=client),
        p.JOINT_REVOLUTE, [0, 1, 0], [0, 0, 0],
        vis_shape=p.createVisualShape(p.GEOM_CAPSULE, radius=torso_r, length=2 * h, physicsClientId=client),
        inertia_diag=spec_mod.capsule_inertia(density, torso_r, 2 * h))

    def add_capsule_link(name, parent_idx, radius, length, position):
        offset = [0, 0, -length / 2]  # capsule runs from the joint straight down
        return add(
            name, parent_idx, spec_mod.capsule_mass(density, radius, length),
            p.createCollisionShape(p.GEOM_CAPSULE, radius=radius, height=length,
                                   collisionFramePosition=offset, physicsClientId=client),
            p.JOINT_REVOLUTE, [0, 1, 0], position,
            vis_shape=p.createVisualShape(p.GEOM_CAPSULE, radius=radius, length=length,
                                          visualFramePosition=offset, physicsClientId=client),
            com_offset=offset, inertia_diag=spec_mod.capsule_inertia(density, radius, length))

    def add_leg(prefix, y):
        thigh, shank, foot = (params[f"{k}_length_{prefix}"] for k in ("thigh", "shank", "foot"))
        hip = add_capsule_link(f"hip_{prefix}", torso, spec.thigh_radius, thigh, [0, y, -h])
        knee = add_capsule_link(f"knee_{prefix}", hip, spec.shank_radius, shank, [0, 0, -thigh])
        half = [foot / 2, spec.foot_half_width, spec.foot_half_thick]
        offset = [foot / 2 - spec.foot_heel, 0, -spec.foot_half_thick]
        f = spec.pad_mass_fraction
        mass = spec_mod.box_mass(density, *half)
        diag = spec_mod.box_inertia(density, *half)
        # the foot keeps the mass and inertia but does not collide. the pad link collides.
        ankle = add(f"ankle_{prefix}", knee, (1 - f) * mass, -1, p.JOINT_REVOLUTE, [0, 1, 0],
                    [0, 0, -shank],
                    vis_shape=p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                                  visualFramePosition=offset, physicsClientId=client),
                    com_offset=offset, inertia_diag=tuple((1 - f) * x for x in diag))
        pads.append((prefix, ankle, half, offset, f * mass, tuple(f * x for x in diag)))

    pads = []
    add_leg("left", params["hip_y_offset"])
    add_leg("right", -params["hip_y_offset"])
    for prefix, ankle, half, offset, pad_mass, pad_diag in pads:  # after the legs: indices 0-8 stay put
        add(f"pad_{prefix}", ankle, pad_mass,
            p.createCollisionShape(p.GEOM_BOX, halfExtents=half, collisionFramePosition=offset,
                                   physicsClientId=client),
            p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0], com_offset=offset, inertia_diag=pad_diag)

    n = len(names)
    body = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
        basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1],
        linkMasses=masses, linkCollisionShapeIndices=col, linkVisualShapeIndices=vis,
        linkPositions=pos, linkOrientations=[[0, 0, 0, 1]] * n,
        linkInertialFramePositions=com, linkInertialFrameOrientations=[[0, 0, 0, 1]] * n,
        linkParentIndices=parent, linkJointTypes=jtype, linkJointAxis=axis,
        physicsClientId=client)
    # pybullet re-sorts links depth-first and names joints "joint<creation order + 1>".
    index = {}
    for i in range(p.getNumJoints(body, physicsClientId=client)):
        index[names[int(p.getJointInfo(body, i, physicsClientId=client)[1].decode()[5:]) - 1]] = i

    # inertia. pybullet's own capsule inertia is a bounding box, 26-40% off. override all links.
    # armature approximation: motor joints only, added to the link they drive.
    a = spec.armature if cfg.armature is None else cfg.armature
    share = {"hip": cfg.arm_thigh, "knee": cfg.arm_shank, "ankle": cfg.arm_foot}
    for name, diag in inertia.items():
        diag = list(diag)
        kind = name.split("_")[0]
        if kind in share:
            diag[1] += share[kind] * a
        elif name == "torso":
            ixx, iyy, izz = diag  # a trim must keep ixx <= iyy + izz
            diag[1] += max(cfg.arm_torso * a, -0.95 * max(0.0, iyy - abs(ixx - izz)))
        p.changeDynamics(body, index[name], localInertiaDiagonal=diag, physicsClientId=client)

    # only the pads touch the floor, as in mujoco. group 0 collides with nothing.
    for i in [-1, *(i for n, i in index.items() if not n.startswith("pad"))]:
        p.setCollisionFilterGroupMask(body, i, 0, 0, physicsClientId=client)
    for i in [-1, *index.values()]:
        # default linear/angular damping is 0.04 on every link. mujoco has none.
        p.changeDynamics(body, i, linearDamping=0.0, angularDamping=0.0, physicsClientId=client)
    for i in index.values():
        p.changeDynamics(body, i, lateralFriction=robot_friction, physicsClientId=client)
        # default velocity motor is a brake on every joint, root dofs included.
        p.setJointMotorControl2(body, i, p.VELOCITY_CONTROL, force=0, physicsClientId=client)
        # no native joint limits: they are hard and force-capped. hip and knee are held by the
        # servo goal clip, the ankle by spec.ankle_stop_torque.
        p.changeDynamics(body, i, jointDamping=0.0, physicsClientId=client)
    for name in LEG_JOINTS:
        # native jointDamping is applied under TORQUE_CONTROL. it is the only damping source.
        damping = params["ankle_damping"] if name.startswith("ankle") else spec.joint_damping
        p.changeDynamics(body, index[name], jointDamping=damping, physicsClientId=client)

    return PyBulletModel(
        client=client, body_id=body, joint_index=index, dofs=[index[n] for n in ROBOT_DOFS], spec=spec, cfg=cfg, params=params,
        standing_height=root_z,
        kp={"hip": params["actuator_kp_hip"], "knee": params["actuator_kp_knee"]},
        ankle_stiffness=params["ankle_stiffness"],
        pad_mass={prefix: p.getDynamicsInfo(body, index[f"pad_{prefix}"], physicsClientId=client)[0]
                  for prefix in ("left", "right")})


def apply_control(model: PyBulletModel, targets, ext: dict | None = None):
    """one torque evaluation. MUST run before every stepSimulation.

    hip/knee: the servo law from spec (same code as mujoco). ankle: passive spring plus the
    rubber stop from spec. damping is native. ext adds torque per joint, audit only.
    """
    ext, spec, c = ext or {}, model.spec, model.client
    for name, target in zip(controller.ACTUATOR_ORDER, targets):
        kind = name.split("_")[0]
        q, qd = p.getJointState(model.body_id, model.joint_index[name], physicsClientId=c)[:2]
        tau = spec.servo_torque(model.kp[kind], float(target), q, qd, kind) + ext.get(name, 0.0)
        p.setJointMotorControl2(model.body_id, model.joint_index[name], p.TORQUE_CONTROL,
                                force=tau, physicsClientId=c)
    for prefix in ("left", "right"):
        name = f"ankle_{prefix}"
        q, qd = p.getJointState(model.body_id, model.joint_index[name], physicsClientId=c)[:2]
        tau = (-model.ankle_stiffness * q + spec.ankle_stop_torque(model.params, prefix, q, qd)
               + ext.get(name, 0.0))
        p.setJointMotorControl2(model.body_id, model.joint_index[name], p.TORQUE_CONTROL,
                                force=tau, physicsClientId=c)
        k, damp, _ = spec.foot_contact(model.params, prefix)  # foot pad spring-damper
        x, v = p.getJointState(model.body_id, model.joint_index[f"pad_{prefix}"], physicsClientId=c)[:2]
        # the pad is a light mass on a stiff, damped spring. an explicit damper goes unstable once
        # c*dt/m > 2. the damper is taken implicitly, c*v / (1 + dt*c/m), as mujoco does for joint
        # damping. the spring stays explicit, so a static load sinks the pad by exactly m*g/k.
        m_pad, dt = model.pad_mass[prefix], spec.timestep / model.cfg.substeps
        p.setJointMotorControl2(model.body_id, model.joint_index[f"pad_{prefix}"], p.TORQUE_CONTROL,
                                force=-k * x - damp * v / (1.0 + dt * damp / m_pad), physicsClientId=c)


# anything faster than this is a numerical blow-up, not a gait. counted as a failed rollout so an
# optimizer cannot be paid for one. real pad speeds in a sane run are ~3 m/s, the robot ~1 m/s.
MAX_PAD_SPEED = 10.0  # m/s
MAX_BODY_SPEED = 8.0  # m/s


def _unphysical(model: PyBulletModel) -> bool:
    c, bid, ji = model.client, model.body_id, model.joint_index
    pads = max(abs(p.getJointState(bid, ji[f"pad_{s}"], physicsClientId=c)[1]) for s in ("left", "right"))
    return pads > MAX_PAD_SPEED or abs(p.getJointState(bid, ji["dummy_x"], physicsClientId=c)[1]) > MAX_BODY_SPEED


def reset(model: PyBulletModel, q0=None):
    """the model is stateful. MUST reset joints and velocities before every rollout."""
    for i in model.joint_index.values():
        p.resetJointState(model.body_id, i, 0.0, 0.0, physicsClientId=model.client)
    for i, v in zip(model.dofs, q0 if q0 is not None else []):  # q0 is in ROBOT_DOFS order
        p.resetJointState(model.body_id, i, float(v), 0.0, physicsClientId=model.client)


def simulate(model: PyBulletModel, theta: np.ndarray, duration: float = 8.0) -> ResultOfARollout:
    """open-loop rollout. no command delay: pybullet is the training sim."""
    spec, c, bid, ji = model.spec, model.client, model.body_id, model.joint_index
    reset(model)
    for step in range(int(round(duration / spec.timestep))):
        t = step * spec.timestep
        targets = controller.joint_targets(theta, t)
        for _ in range(model.cfg.substeps):
            apply_control(model, targets)
            p.stepSimulation(physicsClientId=c)
        pitch = p.getJointState(bid, ji["torso"], physicsClientId=c)[0]
        dz = p.getJointState(bid, ji["dummy_z"], physicsClientId=c)[0]
        if has_fallen(pitch, dz, model.standing_height) or _unphysical(model):
            return ResultOfARollout(p.getJointState(bid, ji["dummy_x"], physicsClientId=c)[0], True, t + spec.timestep)
    return ResultOfARollout(p.getJointState(bid, ji["dummy_x"], physicsClientId=c)[0], False, None)
