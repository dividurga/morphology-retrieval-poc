"""mujoco sim. this is "real". built from spec.py, nothing hardcoded here.

planar biped: torso carries x, z, pitch. each leg is hip -> thigh -> knee -> shank -> ankle
-> foot. hip and knee are driven by the servo law in spec.py, the ankle is a passive spring
with a rubber bumper stop.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

import controller
import spec as spec_mod
from sim_common import ResultOfARollout, has_fallen
from spec import SPEC, Spec

QPOS_X, QPOS_Z, QPOS_PITCH = 0, 1, 2
LEGS = ("left", "right")
# the 9 robot dofs. the foot pad slide joints sit between them in the tree, so qpos indices
# are not contiguous. always go through MujocoModel.dofs.
ROBOT_DOFS = ("root_x", "root_z", "root_pitch", "hip_left", "knee_left", "ankle_left",
              "hip_right", "knee_right", "ankle_right")


@dataclass
class MujocoModel:
    mj: mujoco.MjModel
    spec: Spec
    params: dict
    standing_height: float
    qpos_adr: dict  # joint name -> qpos index
    dof_adr: dict  # joint name -> dof index
    dofs: list  # qpos / dof index of each ROBOT_DOFS entry. equal for hinge and slide joints
    kp: dict  # "hip" / "knee" -> position gain


def _leg_xml(params: dict, spec: Spec, prefix: str, y: float, h: float) -> str:
    thigh = params[f"thigh_length_{prefix}"]
    shank = params[f"shank_length_{prefix}"]
    foot = params[f"foot_length_{prefix}"]
    # armature MUST sit on the motor joints only. the ankle is passive, the root is not a motor.
    # hip and knee have no range: the servo clips its goal instead. the ankle stop is a torque
    # from spec.ankle_stop_torque, not a mujoco limit.
    motor = f'damping="{spec.joint_damping}" armature="{spec.armature}"'
    # the foot pad is a spring-damper in series with the foot: a light pad body on a slide joint
    # (spec.foot_contact). only the pad touches the floor. its mass is taken out of the foot.
    k_pad, c_pad, _ = spec.foot_contact(params, prefix)
    f = spec.pad_mass_fraction
    box_mass = spec_mod.box_mass(params["density"], foot / 2, spec.foot_half_width, spec.foot_half_thick)
    # foam densification: past this strain the pad is a hard stop (model-form, mujoco only).
    pad_range = ("" if spec.pad_stop_strain is None else
                 f'range="-1 {spec.pad_stop_strain * spec.pad_thickness0 * spec.scale}"')
    box_pos = f"{foot / 2 - spec.foot_heel} 0 {-spec.foot_half_thick}"
    box_size = f"{foot / 2} {spec.foot_half_width} {spec.foot_half_thick}"
    if spec.pad_enabled:
        foot_body = f"""<geom name="foot_{prefix}" type="box" pos="{box_pos}" size="{box_size}" mass="{(1 - f) * box_mass}" contype="0" conaffinity="0"/>
          <body name="pad_{prefix}">
            <joint name="pad_{prefix}" type="slide" axis="0 0 1" stiffness="{k_pad}" damping="{c_pad}" armature="0" {pad_range}/>
            <geom name="pad_{prefix}" type="box" pos="{box_pos}" size="{box_size}" mass="{f * box_mass}" contype="2" conaffinity="1"/>
          </body>"""
    else:  # experiment: rigid foot, no pad dof. the pad joints are still named so dof maps work
        foot_body = f"""<geom name="foot_{prefix}" type="box" pos="{box_pos}" size="{box_size}" mass="{box_mass}" contype="2" conaffinity="1"/>
          <body name="pad_{prefix}">
            <joint name="pad_{prefix}" type="slide" axis="0 0 1" range="0 0" armature="0"/>
            <geom name="padmass_{prefix}" type="sphere" size="0.001" mass="1e-6" contype="0" conaffinity="0"/>
          </body>"""
    return f"""
    <body name="thigh_{prefix}" pos="0 {y} {-h}">
      <joint name="hip_{prefix}" type="hinge" axis="0 1 0" {motor}/>
      <geom name="thigh_{prefix}" type="capsule" fromto="0 0 0 0 0 {-thigh}" size="{spec.thigh_radius}"/>
      <body name="shank_{prefix}" pos="0 0 {-thigh}">
        <joint name="knee_{prefix}" type="hinge" axis="0 1 0" {motor}/>
        <geom name="shank_{prefix}" type="capsule" fromto="0 0 0 0 0 {-shank}" size="{spec.shank_radius}"/>
        <body name="foot_{prefix}" pos="0 0 {-shank}">
          <joint name="ankle_{prefix}" type="hinge" axis="0 1 0" stiffness="{params['ankle_stiffness']}" damping="{params['ankle_damping']}" armature="{spec.armature_ankle}"/>
          {foot_body}
        </body>
      </body>
    </body>
"""


def build_xml(params: dict, spec: Spec = SPEC) -> str:
    h = params["torso_size"] / 2.0
    dt = spec.timestep / spec.mj_substeps
    rf = " ".join(map(str, spec.robot_friction))
    ff = " ".join(map(str, spec.floor_friction))
    # only the feet touch the floor (conaffinity 0 elsewhere). a standing robot's shank would
    # otherwise rest on the floor and carry load around the pad. a fallen robot is a failed
    # rollout either way, so legs and torso never need to collide.
    # non-foot geoms (torso, thigh, shank) have no pad. stiffest contact mujoco can integrate:
    # time constant 2*dt (refsafe), critical damping. the floor takes its contact from the foot
    # (solmix 0), so left and right pads can differ.
    solref = f"{2 * dt if spec.contact_timeconst is None else spec.contact_timeconst} 1"
    return f"""
<mujoco model="planar_biped">
  <compiler angle="radian"/>
  <option timestep="{dt}" gravity="0 0 {spec.gravity}"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
  </visual>

  <default>
    <geom density="{params['density']}" friction="{rf}" solref="{solref}" contype="2" conaffinity="{1 if spec.legs_collide else 0}"/>
  </default>

  <worldbody>
    <light diffuse="0.8 0.8 0.8" pos="0 -1 3" dir="0 0.3 -1"/>
    <geom name="floor" type="plane" size="10 2 0.1" friction="{ff}" solref="{solref}" solmix="0" contype="1" conaffinity="1" rgba="0.8 0.8 0.8 1"/>

    <body name="torso" pos="0 0 {spec.root_z(params)}">
      <joint name="root_x" type="slide" axis="1 0 0" damping="0" armature="0"/>
      <joint name="root_z" type="slide" axis="0 0 1" damping="0" armature="0"/>
      <joint name="root_pitch" type="hinge" axis="0 1 0" damping="0" armature="0"/>
      <geom name="torso" type="capsule" fromto="0 0 {-h} 0 0 {h}" size="{spec.torso_radius(params)}"/>
      {_leg_xml(params, spec, "left", params["hip_y_offset"], h)}
      {_leg_xml(params, spec, "right", -params["hip_y_offset"], h)}
    </body>
  </worldbody>

  <actuator>
    <motor name="hip_left" joint="hip_left" gear="1" ctrllimited="false"/>
    <motor name="knee_left" joint="knee_left" gear="1" ctrllimited="false"/>
    <motor name="hip_right" joint="hip_right" gear="1" ctrllimited="false"/>
    <motor name="knee_right" joint="knee_right" gear="1" ctrllimited="false"/>
  </actuator>
</mujoco>
"""


def make_model(params: dict, spec: Spec = SPEC) -> MujocoModel:
    mj = mujoco.MjModel.from_xml_string(build_xml(params, spec))
    assert [mj.actuator(i).name for i in range(mj.nu)] == controller.ACTUATOR_ORDER
    names = [mj.joint(i).name for i in range(mj.njnt)]
    return MujocoModel(
        mj=mj, spec=spec, params=params, standing_height=spec.root_z(params),
        qpos_adr={n: int(mj.joint(n).qposadr[0]) for n in names},
        dof_adr={n: int(mj.joint(n).dofadr[0]) for n in names},
        dofs=[int(mj.joint(n).dofadr[0]) for n in ROBOT_DOFS],
        kp={"hip": params["actuator_kp_hip"], "knee": params["actuator_kp_knee"]})


def apply_control(model: MujocoModel, data: mujoco.MjData, targets, ext: dict | None = None):
    """one torque evaluation. MUST run before every mj_step. ext adds torque per joint (audit only)."""
    spec, ext = model.spec, ext or {}
    for i, (name, target) in enumerate(zip(controller.ACTUATOR_ORDER, targets)):
        kind = name.split("_")[0]
        q, qd = data.qpos[model.qpos_adr[name]], data.qvel[model.dof_adr[name]]
        data.ctrl[i] = spec.servo_torque(model.kp[kind], float(target), float(q), float(qd), kind)
    data.qfrc_applied[:] = 0
    if spec.joint_coulomb:  # model-form: coulomb friction, smoothed over 0.05 rad/s
        for name in controller.ACTUATOR_ORDER:
            data.qfrc_applied[model.dof_adr[name]] -= spec.joint_coulomb * np.tanh(
                data.qvel[model.dof_adr[name]] / 0.05)
    for prefix in LEGS:
        name = f"ankle_{prefix}"
        q, qd = data.qpos[model.qpos_adr[name]], data.qvel[model.dof_adr[name]]
        data.qfrc_applied[model.dof_adr[name]] = spec.ankle_stop_torque(model.params, prefix, float(q), float(qd))
    for name, tau in ext.items():
        data.qfrc_applied[model.dof_adr[name]] += tau


def simulate(model: MujocoModel, theta: np.ndarray, duration: float = 8.0) -> ResultOfARollout:
    """open-loop rollout. the command is delayed by spec.delay_steps control periods.

    before the delay has elapsed the servo sees the t=0 target.
    """
    mj, spec = model.mj, model.spec
    data = mujoco.MjData(mj)
    delay = spec.delay_steps * spec.timestep
    for step in range(int(round(duration / spec.timestep))):
        targets = controller.joint_targets(theta, max(0.0, step * spec.timestep - delay))
        for _ in range(spec.mj_substeps):
            apply_control(model, data, targets)
            mujoco.mj_step(mj, data)
        if has_fallen(data.qpos[QPOS_PITCH], data.qpos[QPOS_Z], model.standing_height):
            return ResultOfARollout(float(data.qpos[QPOS_X]), True, (step + 1) * spec.timestep)
    return ResultOfARollout(float(data.qpos[QPOS_X]), False, None)
