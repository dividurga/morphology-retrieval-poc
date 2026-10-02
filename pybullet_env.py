"""PyBullet-backed twin of environment.py/morphology.py -- the "reality" sim.

Built from scratch via createMultiBody, not loadMJCF. loadMJCF silently drops joint
limits for every hinge (confirmed: changeDynamics can't fix this after the fact -- a
joint created unlimited stays unlimited) and has no concept of MuJoCo's <actuator>
block at all. Topology and mass/inertia DID import correctly via loadMJCF (checked
against MuJoCo's own numbers to ~6 decimal places), so this file mirrors that same
density-based mass computation by hand.

Planar constraint: MuJoCo gets x/z-translation + pitch by putting three joints on one
body (root_x, root_z, root_pitch all on "torso"). PyBullet only allows one joint per
link, so the same three DOF become two massless dummy links feeding into torso --
the same trick loadMJCF's own importer used (visible as its auto-generated
"jointfix_*" links).
"""

from dataclasses import dataclass

import numpy as np
import pybullet as p

import controller
import morphology

GRAVITY = -9.81
TIMESTEP = 0.005  # matches morphology.py's MuJoCo <option timestep>
FALL_PITCH_THRESH = 1.0  # same thresholds as environment.py, for a comparable fall check
FALL_HEIGHT_FRAC = 0.5

RANGES = {"hip": morphology.HIP_RANGE, "knee": morphology.KNEE_RANGE, "ankle": morphology.ANKLE_RANGE}
ACTUATED_JOINTS = ["hip_left", "knee_left", "hip_right", "knee_right"]
PASSIVE_JOINTS = ["ankle_left", "ankle_right"]

ACTUATOR_MAX_FORCE = 200.0  # generous headroom -- PyBullet's POSITION_CONTROL needs an
                            # explicit force cap; MuJoCo's position actuator has no such
                            # cap (kp alone determines torque), so this MUST be high
                            # enough to not itself become the limiting factor


def _capsule_mass(density: float, radius: float, length: float) -> float:
    """Solid capsule = cylinder + two hemispherical caps, matching MuJoCo's own
    density-based mass computation for a capsule geom (verified: reproduces MuJoCo's
    reported torso mass to float precision for NOMINAL_PARAMS)."""
    return density * (np.pi * radius**2 * length + (4.0 / 3.0) * np.pi * radius**3)


def _box_mass(density: float, hx: float, hy: float, hz: float) -> float:
    return density * (2 * hx) * (2 * hy) * (2 * hz)


@dataclass
class PyBulletModel:
    client: int
    body_id: int
    joint_index: dict  # joint name -> PyBullet joint index on body_id
    standing_height: float  # torso height above ground when standing, for fall checks
    kp_hip: float
    kp_knee: float
    ankle_stiffness: float  # carried here, not re-read from params, since simulate()
    ankle_damping: float    # doesn't take params -- matches environment.py's interface


@dataclass
class ResultOfARollout:
    displacement: float
    fell: bool
    fell_time: float | None


def make_model(params: dict, client: int | None = None) -> PyBulletModel:
    """Build the planar biped directly in PyBullet. One DIRECT-mode client per model
    unless `client` is given (reuse a client across many make_model calls instead of
    spawning a new physics server every time -- spawning is the expensive part)."""
    owns_client = client is None
    if owns_client:
        client = p.connect(p.DIRECT)
    p.resetSimulation(physicsClientId=client)
    p.setGravity(0, 0, GRAVITY, physicsClientId=client)
    p.setTimeStep(TIMESTEP, physicsClientId=client)

    h = params["torso_size"] / 2.0
    torso_radius = morphology.TORSO_RADIUS_RATIO * params["torso_size"]
    root_z = (
        h
        + max(params["thigh_length_left"], params["thigh_length_right"])
        + max(params["shank_length_left"], params["shank_length_right"])
        + 2 * morphology.FOOT_HALF_THICK
        + morphology.GROUND_CLEARANCE
    )
    density = params["density"]

    floor_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=client)
    floor_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_shape,
                                  basePosition=[0, 0, 0], physicsClientId=client)
    # PyBullet defaults to lateralFriction=0.5 on everything -- never matches MuJoCo's
    # configured values (robot foot 0.05, floor 1.0) unless set explicitly. This was a
    # real bug, not a nitpick: left at defaults, the first rollout comparison fell at
    # ~0.8s regardless of PD gain tuning, which pointed at something structural rather
    # than an actuator-gain problem.
    # MuJoCo combines contact friction as max(geom1, geom2) = 1.0 (floor) here; PyBullet
    # MULTIPLIES the two lateralFriction values. Reproduce MuJoCo's effective value by
    # setting the robot to 1.0 so the product equals the floor's value.
    robot_friction = 1.0
    p.changeDynamics(floor_id, -1, lateralFriction=morphology.GROUND_FRICTION_NOMINAL,
                      physicsClientId=client)

    link_masses, link_col, link_vis = [], [], []
    link_pos, link_orn = [], []
    link_inertial_pos, link_inertial_orn = [], []
    link_parent, link_joint_type, link_axis = [], [], []
    joint_names = []

    identity_orn = [0, 0, 0, 1]

    def add_link(name, parent_array_idx, mass, col_shape, joint_type, axis, position, vis_shape=-1,
                 com=(0, 0, 0)):
        """parent_array_idx: 0-based index into these arrays, or None for the base."""
        joint_names.append(name)
        link_masses.append(mass)
        link_col.append(col_shape)
        link_vis.append(vis_shape)
        link_pos.append(position)
        link_orn.append(identity_orn)
        link_inertial_pos.append(list(com))
        link_inertial_orn.append(identity_orn)
        link_parent.append(0 if parent_array_idx is None else parent_array_idx + 1)
        link_joint_type.append(joint_type)
        link_axis.append(axis)
        return len(joint_names) - 1

    # two massless dummy links compose the x/z-translation, torso itself carries pitch
    # -- together these reproduce MuJoCo's three joints (root_x, root_z, root_pitch)
    # all declared on a single body.
    dummy_x = add_link("dummy_x", None, 0.0, -1, p.JOINT_PRISMATIC, [1, 0, 0], [0, 0, root_z])
    dummy_z = add_link("dummy_z", dummy_x, 0.0, -1, p.JOINT_PRISMATIC, [0, 0, 1], [0, 0, 0])

    torso_col = p.createCollisionShape(p.GEOM_CAPSULE, radius=torso_radius, height=2 * h,
                                        physicsClientId=client)
    torso_vis = p.createVisualShape(p.GEOM_CAPSULE, radius=torso_radius, length=2 * h,
                                     physicsClientId=client)
    torso_mass = _capsule_mass(density, torso_radius, 2 * h)
    torso = add_link("torso", dummy_z, torso_mass, torso_col, p.JOINT_REVOLUTE, [0, 1, 0], [0, 0, 0],
                      vis_shape=torso_vis)

    def add_leg(prefix: str, y: float):
        thigh_length = params[f"thigh_length_{prefix}"]
        shank_length = params[f"shank_length_{prefix}"]
        foot_length = params[f"foot_length_{prefix}"]

        thigh_frame = [0, 0, -thigh_length / 2]
        thigh_col = p.createCollisionShape(
            p.GEOM_CAPSULE, radius=morphology.THIGH_RADIUS, height=thigh_length,
            collisionFramePosition=thigh_frame, physicsClientId=client)
        thigh_vis = p.createVisualShape(
            p.GEOM_CAPSULE, radius=morphology.THIGH_RADIUS, length=thigh_length,
            visualFramePosition=thigh_frame, physicsClientId=client)
        thigh_mass = _capsule_mass(density, morphology.THIGH_RADIUS, thigh_length)
        thigh = add_link(f"hip_{prefix}", torso, thigh_mass, thigh_col, p.JOINT_REVOLUTE,
                          [0, 1, 0], [0, y, -h], vis_shape=thigh_vis, com=thigh_frame)

        shank_frame = [0, 0, -shank_length / 2]
        shank_col = p.createCollisionShape(
            p.GEOM_CAPSULE, radius=morphology.SHANK_RADIUS, height=shank_length,
            collisionFramePosition=shank_frame, physicsClientId=client)
        shank_vis = p.createVisualShape(
            p.GEOM_CAPSULE, radius=morphology.SHANK_RADIUS, length=shank_length,
            visualFramePosition=shank_frame, physicsClientId=client)
        shank_mass = _capsule_mass(density, morphology.SHANK_RADIUS, shank_length)
        shank = add_link(f"knee_{prefix}", thigh, shank_mass, shank_col, p.JOINT_REVOLUTE,
                          [0, 1, 0], [0, 0, -thigh_length], vis_shape=shank_vis, com=shank_frame)

        foot_half_extents = [foot_length / 2, morphology.FOOT_HALF_WIDTH, morphology.FOOT_HALF_THICK]
        foot_frame = [foot_length / 2 - morphology.FOOT_HEEL, 0, -morphology.FOOT_HALF_THICK]
        foot_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=foot_half_extents,
            collisionFramePosition=foot_frame, physicsClientId=client)
        foot_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=foot_half_extents,
            visualFramePosition=foot_frame, physicsClientId=client)
        foot_mass = _box_mass(density, foot_length / 2, morphology.FOOT_HALF_WIDTH, morphology.FOOT_HALF_THICK)
        add_link(f"ankle_{prefix}", shank, foot_mass, foot_col, p.JOINT_REVOLUTE,
                  [0, 1, 0], [0, 0, -shank_length], vis_shape=foot_vis, com=foot_frame)

    add_leg("left", params["hip_y_offset"])
    add_leg("right", -params["hip_y_offset"])

    body_id = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
        basePosition=[0, 0, 0], baseOrientation=identity_orn,
        linkMasses=link_masses, linkCollisionShapeIndices=link_col, linkVisualShapeIndices=link_vis,
        linkPositions=link_pos, linkOrientations=link_orn,
        linkInertialFramePositions=link_inertial_pos, linkInertialFrameOrientations=link_inertial_orn,
        linkParentIndices=link_parent, linkJointTypes=link_joint_type, linkJointAxis=link_axis,
        physicsClientId=client,
    )

    joint_index = {name: i for i, name in enumerate(joint_names)}

    for idx in joint_index.values():
        p.changeDynamics(body_id, idx, lateralFriction=robot_friction, physicsClientId=client)

    # PyBullet engages a default velocity-control motor (target velocity 0) on every
    # joint at creation -- effectively a brake. MUST disable it on every joint,
    # including dummy_x/dummy_z/torso, or the "free" root DOFs silently resist motion.
    # Earlier version of this loop skipped exactly those three while also skipping the
    # limit/damping block below for them -- which meant the x-translation DOF was never
    # un-braked, and the robot could execute the gait perfectly while barely
    # translating at all (found via the first PyBullet-vs-MuJoCo rollout comparison:
    # -0.065m vs MuJoCo's 5.986m for the identical theta, despite joint angles tracking
    # the commanded targets correctly).
    for idx in joint_index.values():
        p.setJointMotorControl2(body_id, idx, p.VELOCITY_CONTROL, force=0, physicsClientId=client)

    # joint limits + damping -- set at creation-adjacent time via changeDynamics on a
    # NATIVELY created joint (not loadMJCF-imported); unlike the loadMJCF case, this
    # reliably takes effect in the actual constraint solver -- verified empirically
    # (getJointInfo doesn't reflect it, but simulated behavior does).
    ranges = {"hip": morphology.HIP_RANGE, "knee": morphology.KNEE_RANGE, "ankle": morphology.ANKLE_RANGE}
    for name, idx in joint_index.items():
        if name in ("dummy_x", "dummy_z", "torso"):
            continue
        joint_kind = name.split("_")[0]  # "hip"/"knee"/"ankle"
        lo, hi = ranges[joint_kind]
        p.changeDynamics(body_id, idx, jointLowerLimit=lo, jointUpperLimit=hi,
                          jointLimitForce=ACTUATOR_MAX_FORCE, physicsClientId=client)
        p.changeDynamics(body_id, idx, jointDamping=morphology.JOINT_DAMPING, physicsClientId=client)

    # MuJoCo adds JOINT_ARMATURE to every joint's diagonal inertia; PyBullet has no such
    # field. Without it the ankle (foot Iyy ~5e-4) makes the explicit damping torque
    # unstable at this timestep (dt*c/I > 2). Approximate by adding it to the link's Iyy.
    for name, idx in joint_index.items():
        if name in ("dummy_x", "dummy_z"):
            continue
        info = p.getDynamicsInfo(body_id, idx, physicsClientId=client)
        inertia = list(info[2])
        inertia[1] += morphology.JOINT_ARMATURE
        p.changeDynamics(body_id, idx, localInertiaDiagonal=inertia, physicsClientId=client)

    # passive ankle: no motor at all (handled by manual spring torque in simulate());
    # disable PyBullet's own default velocity-control motor (it defaults ON with a
    # small friction-like force unless explicitly zeroed, same call as above already
    # covers this for every non-torso joint, ankles included).

    standing_height = root_z
    return PyBulletModel(client=client, body_id=body_id, joint_index=joint_index,
                          standing_height=standing_height,
                          kp_hip=params["actuator_kp_hip"], kp_knee=params["actuator_kp_knee"],
                          ankle_stiffness=params["ankle_stiffness"],
                          ankle_damping=params["ankle_damping"])


def _apply_ankle_spring(model: PyBulletModel, name: str, stiffness: float, damping: float):
    idx = model.joint_index[name]
    state = p.getJointState(model.body_id, idx, physicsClientId=model.client)
    angle, velocity = state[0], state[1]
    torque = -stiffness * angle - damping * velocity
    p.setJointMotorControl2(model.body_id, idx, p.TORQUE_CONTROL, force=torque,
                             physicsClientId=model.client)


def simulate(model: PyBulletModel, theta: np.ndarray, duration: float = 8.0) -> ResultOfARollout:
    client = model.client
    # MuJoCo's simulate() allocates fresh MjData per call; the PyBullet model is stateful
    # and gets reused across rollouts (optimize() builds it once), so reset every joint
    # to the t=0 pose and zero velocity explicitly.
    for idx in model.joint_index.values():
        p.resetJointState(model.body_id, idx, 0.0, 0.0, physicsClientId=client)
    steps = int(round(duration / TIMESTEP))
    fell = False
    fell_time = None

    for step in range(steps):
        t = step * TIMESTEP
        targets = controller.joint_targets(theta, t)  # [hip_left, knee_left, hip_right, knee_right]
        for name, target in zip(controller.ACTUATOR_ORDER, targets):
            idx = model.joint_index[name]
            kp = model.kp_hip if name.startswith("hip") else model.kp_knee
            q, qd = p.getJointState(model.body_id, idx, physicsClientId=client)[:2]
            lo, hi = RANGES[name.split("_")[0]]
            tau = kp * (float(np.clip(target, lo, hi)) - q) - morphology.JOINT_DAMPING * qd
            p.setJointMotorControl2(model.body_id, idx, p.TORQUE_CONTROL, force=tau,
                                     physicsClientId=client)
        _apply_ankle_spring(model, "ankle_left", model.ankle_stiffness, model.ankle_damping)
        _apply_ankle_spring(model, "ankle_right", model.ankle_stiffness, model.ankle_damping)
        p.stepSimulation(physicsClientId=client)

        pitch = p.getJointState(model.body_id, model.joint_index["torso"], physicsClientId=client)[0]
        x_pos = p.getJointState(model.body_id, model.joint_index["dummy_x"], physicsClientId=client)[0]
        z_pos = p.getJointState(model.body_id, model.joint_index["dummy_z"], physicsClientId=client)[0]
        height = z_pos + model.standing_height
        if abs(pitch) > FALL_PITCH_THRESH or height < FALL_HEIGHT_FRAC * model.standing_height:
            fell = True
            fell_time = t
            break

    x_final = p.getJointState(model.body_id, model.joint_index["dummy_x"], physicsClientId=client)[0]
    return ResultOfARollout(displacement=x_final, fell=fell, fell_time=fell_time)
