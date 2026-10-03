"""Procedural MJCF generation for the planar biped."""



NOMINAL_PARAMS = dict(
    thigh_length_left=0.22,
    thigh_length_right=0.22,
    shank_length_left=0.22,
    shank_length_right=0.22,
    foot_length_left=0.12,
    foot_length_right=0.12,
    torso_size=0.35,
    ankle_stiffness=8.0,
    ankle_damping=0.4,
    actuator_kp_hip=40.0,
    actuator_kp_knee=40.0,
    density=1000.0,
    hip_y_offset=0.06,
)

# Per-param (low, high) multiplier on NOMINAL_PARAMS. Shape params widened from Level 1's
# +-28% to +-50%; the new params (never swept before) get their own physically-reasoned
# ranges instead of inheriting the shape multiplier -- actuator gain and density in
# particular shouldn't swing as wide as leg length without risking degenerate bodies.
PARAM_BOUND_MULTIPLIERS = {
    "thigh_length_left": (0.5, 1.5),
    "thigh_length_right": (0.5, 1.5),
    "shank_length_left": (0.5, 1.5),
    "shank_length_right": (0.5, 1.5),
    "foot_length_left": (0.5, 1.5),
    "foot_length_right": (0.5, 1.5),
    "torso_size": (0.5, 1.5),
    "ankle_stiffness": (0.25, 2.5),
    "ankle_damping": (0.25, 3.0),
    "actuator_kp_hip": (0.5, 2.0),
    "actuator_kp_knee": (0.5, 2.0),
    "density": (0.6, 1.8),
    "hip_y_offset": (0.5, 2.0),
}

PARAM_BOUNDS = {name: (lo * NOMINAL_PARAMS[name], hi * NOMINAL_PARAMS[name])
                 for name, (lo, hi) in PARAM_BOUND_MULTIPLIERS.items()}

THIGH_RADIUS = 0.035
SHANK_RADIUS = 0.030
FOOT_HALF_WIDTH = 0.04
FOOT_HALF_THICK = 0.015
FOOT_HEEL = 0.02
TORSO_RADIUS_RATIO = 0.09 / 0.35

ROBOT_FRICTION = "0.05 0.005 0.0001"
GROUND_FRICTION_NOMINAL = 1.0

JOINT_DAMPING = 0.5
JOINT_ARMATURE = 0.01
HIP_RANGE = (-1.3, 1.3)
KNEE_RANGE = (-2.4, 0.0)
ANKLE_RANGE = (-0.6, 0.6)

GROUND_CLEARANCE = 0.02


def _leg_xml(params: dict, prefix: str, y: float, h: float) -> str:
    """Return the MJCF body subtree for one leg: hip -> thigh -> knee -> shank -> ankle -> foot.

    prefix: "left" or "right", used to name joints/bodies (e.g. "hip_left") AND to look up
    this leg's own thigh/shank/foot length -- legs are no longer forced symmetric, each
    side reads its own f"{dim}_length_{prefix}" entry from params.
    y: fixed lateral offset (world/local y) for this leg's attachment point under the torso.

    Required joint names: f"hip_{prefix}", f"knee_{prefix}", f"ankle_{prefix}"
    Required body names:  f"thigh_{prefix}", f"shank_{prefix}", f"foot_{prefix}"
    """
    thigh_length = params[f"thigh_length_{prefix}"]
    shank_length = params[f"shank_length_{prefix}"]
    foot_length = params[f"foot_length_{prefix}"]
    xml = f"""
    <body name="thigh_{prefix}" pos="0 {y} {-h}">
      <joint name="hip_{prefix}" type="hinge" axis="0 1 0" range="{HIP_RANGE[0]} {HIP_RANGE[1]}" limited="true"/>
      <geom name="thigh_{prefix}" type="capsule" fromto="0 0 0 0 0 {-thigh_length}" size="{THIGH_RADIUS}"/>
      <body name="shank_{prefix}" pos="0 0 {-thigh_length}">
        <joint name="knee_{prefix}" type="hinge" axis="0 1 0" range="{KNEE_RANGE[0]} {KNEE_RANGE[1]}" limited="true"/>
        <geom name="shank_{prefix}" type="capsule" fromto="0 0 0 0 0 {-shank_length}" size="{SHANK_RADIUS}"/>
        <body name="foot_{prefix}" pos="0 0 {-shank_length}">
          <joint name="ankle_{prefix}" type="hinge" axis="0 1 0" range="{ANKLE_RANGE[0]} {ANKLE_RANGE[1]}" limited="true" stiffness="{params['ankle_stiffness']}" damping="{params['ankle_damping']}"/>
          <geom name="foot_{prefix}" type="box" pos="{foot_length/2 - FOOT_HEEL} 0 {-FOOT_HALF_THICK}" size="{foot_length/2} {FOOT_HALF_WIDTH} {FOOT_HALF_THICK}"/>
        </body>
      </body>
    </body>

"""
    return xml

def root_height(params: dict) -> float:
    h = params["torso_size"] / 2.0
    return (h + max(params["thigh_length_left"], params["thigh_length_right"])
            + max(params["shank_length_left"], params["shank_length_right"])
            + 2 * FOOT_HALF_THICK + GROUND_CLEARANCE)


def build_biped_xml(params: dict, torque_actuators: bool = False) -> str:
    h = params["torso_size"] / 2.0
    torso_radius = TORSO_RADIUS_RATIO * params["torso_size"]
    # legs may differ in length now -- spawn at a height that clears the LONGER of each
    # pair, so the shorter leg hangs with a bit of slack rather than either leg clipping
    # into the floor at t=0 (which MuJoCo's contact solver would otherwise have to resolve
    # as an interpenetration on the very first step).
    root_z = (
        h
        + max(params["thigh_length_left"], params["thigh_length_right"])
        + max(params["shank_length_left"], params["shank_length_right"])
        + 2 * FOOT_HALF_THICK
        + GROUND_CLEARANCE
    )

    kp_hip, kp_knee = params["actuator_kp_hip"], params["actuator_kp_knee"]
    if torque_actuators:
        # plain torque motors: the shared loop computes kp*(target - q) itself, so both engines get
        # the same torque. identical law to the position actuator below, which is the original.
        actuators = "\n".join(f'    <motor name="{n}" joint="{n}" gear="1"/>'
                              for n in ("hip_left", "knee_left", "hip_right", "knee_right"))
    else:
        actuators = f"""    <position name="hip_left" joint="hip_left" kp="{kp_hip}" ctrlrange="{HIP_RANGE[0]} {HIP_RANGE[1]}"/>
    <position name="knee_left" joint="knee_left" kp="{kp_knee}" ctrlrange="{KNEE_RANGE[0]} {KNEE_RANGE[1]}"/>
    <position name="hip_right" joint="hip_right" kp="{kp_hip}" ctrlrange="{HIP_RANGE[0]} {HIP_RANGE[1]}"/>
    <position name="knee_right" joint="knee_right" kp="{kp_knee}" ctrlrange="{KNEE_RANGE[0]} {KNEE_RANGE[1]}"/>"""
    left_leg = _leg_xml(params, "left", params["hip_y_offset"], h)
    right_leg = _leg_xml(params, "right", -params["hip_y_offset"], h)

    xml = f"""
<mujoco model="planar_biped">
  <compiler angle="radian"/>
  <option timestep="0.005" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="1280" offheight="960"/>
  </visual>

  <default>
    <joint damping="{JOINT_DAMPING}" armature="{JOINT_ARMATURE}"/>
    <geom density="{params['density']}" friction="{ROBOT_FRICTION}" contype="2" conaffinity="1"/>
  </default>

  <worldbody>
    <light diffuse="0.8 0.8 0.8" pos="0 -1 3" dir="0 0.3 -1"/>
    <geom name="floor" type="plane" size="10 2 0.1" friction="{GROUND_FRICTION_NOMINAL} 0.005 0.0001" contype="1" conaffinity="1" rgba="0.8 0.8 0.8 1"/>

    <body name="torso" pos="0 0 {root_z}">
      <joint name="root_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="root_z" type="slide" axis="0 0 1" damping="0"/>
      <joint name="root_pitch" type="hinge" axis="0 1 0" damping="0"/>
      <geom name="torso" type="capsule" fromto="0 0 {-h} 0 0 {h}" size="{torso_radius}"/>

      {left_leg}
      {right_leg}
    </body>
  </worldbody>

  <actuator>
{actuators}
  </actuator>
</mujoco>
"""
    return xml


if __name__ == "__main__":
    import mujoco

    xml = build_biped_xml(NOMINAL_PARAMS)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    print("nbody:", model.nbody, "njnt:", model.njnt, "nu:", model.nu)
    print("total mass:", sum(model.body_mass))
    mujoco.mj_step(model, data)
    print("torso x,z after 1 step:", data.qpos[0], data.qpos[1])
