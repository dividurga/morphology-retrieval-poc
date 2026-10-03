"""single source of truth for both sims. no sim code in here.

sim_mujoco.py ("real") and sim_pybullet.py (best-effort twin) MUST read every physical
number from a Spec. any difference between them MUST be logged in SIM_DIFFS.md.
ablate an effect with dataclasses.replace(SPEC, armature=0.0), never by editing a builder.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np

# servo: dynamixel mx-64 (6.0 n*m stall @12v, 200:1, 0.088 deg).
# https://emanual.robotis.com/docs/en/dxl/mx/mx-64/
MX64_STALL_TORQUE = 6.0
MX64_GEAR_RATIO = 200.0
MX64_RESOLUTION_DEG = 0.088
MX64_NO_LOAD_RPM = 63.0  # at 12 v. same page.
# system-identified apparent inertia, bam params/mx64/m4.json "armature" (kg*m^2).
# https://github.com/rimim/bam. ax-12a has no published rotor inertia, so it was dropped.
MX64_ARMATURE = 0.010961212217454795

def gent_modulus(shore_a: float) -> float:
    """young's modulus in pa from shore a hardness. gent's relation, good for ~20-80 shore a.
    https://en.wikipedia.org/wiki/Shore_durometer"""
    return 0.0981 * (56 + 7.62336 * shore_a) / (0.137505 * (254 - 2.54 * shore_a)) * 1e6


# stall torque of the mx-64 at three supply voltages, all sourced from the same e-manual page.
MX64_STALL_BY_VOLT = ((11.1, 5.5), (12.0, 6.0), (14.8, 7.3))


# unit-scale shape of the body. every length-like value is multiplied by Spec.scale.
_NOMINAL_PARAMS_UNIT = dict(
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

# per-param (low, high) multiplier on the nominal. the box MUST keep a region where the
# robot cannot walk. do not narrow it to hide that.
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

# how each nominal param moves with scale s. torque ~ density*L^4, time ~ L^0.5, so
# torque-per-rad ~ s^4 and torque-per-(rad/s) ~ s^4.5. these exponents are derived, not
# measured. SHOULD be checked against the audit before trusting them.
_PARAM_SCALE_EXPONENT = {
    "thigh_length_left": 1, "thigh_length_right": 1,
    "shank_length_left": 1, "shank_length_right": 1,
    "foot_length_left": 1, "foot_length_right": 1,
    "torso_size": 1, "hip_y_offset": 1,
    "ankle_stiffness": 4.0, "actuator_kp_hip": 4.0, "actuator_kp_knee": 4.0,
    "ankle_damping": 4.5,
    "density": 0.0,
}


@dataclass(frozen=True)
class Spec:
    # robot scale. measured on the real sim, cap off: peak hip/knee torque of best gaits was
    # ~45 n*m at 1.0, ~10 at 0.7, ~3.7 at 0.55, ~1.9 at 0.45 (about s^4). 0.5 puts the nominal
    # near 40% of the 6.0 stall and leaves the heavy half of the box saturated. 0.5 made the
    # armature ~20x a thigh's inertia, so pybullet could not follow. 0.7 chosen: best agreement
    # among the scales where the robot still walks. the cap binds at nominal.
    scale: float = 0.7  # chosen. see SIM_DIFFS.md scale table.

    gravity: float = -9.81
    timestep: float = 0.005  # control period. targets are held over it.
    mj_substeps: int = 5  # mujoco steps per control period. the servo law runs every substep.

    # servo effects. each one is a dial. None / 0 switches it off.
    torque_cap: float | None = MX64_STALL_TORQUE  # n*m stall, hip + knee only
    # driving torque falls linearly with joint speed, cap*(1-|w|/wnl). braking stays at the stall
    # (current limit). first-order dc motor with a voltage limit and back-emf. sourced stall and
    # no-load only.
    speed_torque: bool = True
    no_load_speed: float = MX64_NO_LOAD_RPM * 2 * np.pi / 60.0  # rad/s
    armature: float = MX64_ARMATURE  # kg*m^2, hip + knee only. NOT scaled, the servo is fixed.
    delay_steps: int = 3  # control periods. mujoco ("real") only. pybullet MUST NOT have it.

    # base geometry at scale 1. read it through the *_ properties below.
    thigh_radius0: float = 0.035
    shank_radius0: float = 0.030
    foot_half_width0: float = 0.04
    foot_half_thick0: float = 0.015
    foot_heel0: float = 0.02
    torso_radius_ratio: float = 0.09 / 0.35
    ground_clearance0: float = 0.02

    # foot pad on a hard lab floor. soft foam. modulus is an ASSUMPTION: no citable value was
    # found (sorbothane / foam datasheets were not retrievable, and gent's relation is only valid
    # for 20-80 shore a). a sourced shore 50a rubber pad is ~rigid (5 um sink), so it would give no
    # compliance at all. pad_thickness0 is a design choice. loss factor tan(delta) 0.15 is also an
    # ASSUMPTION. both SHOULD be swept, not trusted.
    pad_modulus: float = 3.0e4  # pa. a very soft foam. stiffer pads blow up the explicit pybullet pad spring
    pad_thickness0: float = 0.010  # m at scale 1
    pad_loss_factor: float = 0.15
    # share of the foot's mass carried by the pad link. NOT physical (a foam pad weighs grams). a
    # lighter pad makes the explicit pad spring unstable at 1 ms. same in both sims.
    pad_mass_fraction: float = 0.5
    # ankle rubber bumper. shore 50a rubber, e from gent's relation (sourced). area, lever and
    # thickness are design placeholders.
    bumper_shore_a: float = 50.0
    bumper_thickness0: float = 0.003
    bumper_area0: float = 1.0e-4  # m^2
    bumper_lever0: float = 0.05  # m, bumper to ankle axis

    # effects the training sim does not model. both are zero / off in the nominal spec and only a
    # RealityDeltas turns them on (mujoco only). pybullet ignores them by construction.
    # coulomb friction on hip and knee, n*m. magnitude is bam's identified mx-64 "friction_base"
    # 0.0615 (https://github.com/rimim/bam). bam's units are unverified.
    joint_coulomb: float = 0.0
    # foam densification: the pad slide joint hits a hard stop at this compressive strain.
    # None = no stop. ~0.5 is a typical foam figure, an ASSUMPTION, not sourced.
    pad_stop_strain: float | None = None

    # experiment toggles (mujoco only). defaults are the current model. ablation.py flips them to
    # find what limits the gaits.
    armature_ankle: float = 0.0  # kg*m^2. the ankle is passive, so a real servo gives it none
    legs_collide: bool = False  # thigh/shank/torso also hit the floor
    pad_enabled: bool = True  # False: the foot box touches the floor directly, no spring-damper
    contact_timeconst: float | None = None  # mujoco contact time constant. None = 2 dt (stiffest)

    # joint ranges, rad. unscaled. hip and knee are limited by the servo's goal clip, as a
    # dynamixel angle limit is. only the passive ankle has a mechanical stop (rubber bumper).
    hip_range: tuple = (-1.3, 1.3)
    knee_range: tuple = (-2.4, 0.0)
    ankle_range: tuple = (-0.6, 0.6)

    # viscous damping on the actuated joints, n*m/(rad/s) at scale 1. derived, not sourced.
    joint_damping0: float = 0.5

    # contact. mujoco takes the max of the two friction values, pybullet multiplies them.
    # the pybullet builder MUST compensate. see SIM_DIFFS.md.
    robot_friction: tuple = (0.05, 0.005, 0.0001)
    floor_friction: tuple = (1.0, 0.005, 0.0001)

    @property
    def thigh_radius(self): return self.thigh_radius0 * self.scale
    @property
    def shank_radius(self): return self.shank_radius0 * self.scale
    @property
    def foot_half_width(self): return self.foot_half_width0 * self.scale
    @property
    def foot_half_thick(self): return self.foot_half_thick0 * self.scale
    @property
    def foot_heel(self): return self.foot_heel0 * self.scale
    @property
    def ground_clearance(self): return self.ground_clearance0 * self.scale
    @property
    def joint_damping(self): return self.joint_damping0 * self.scale ** 4.5

    @property
    def omega_scale(self):
        """multiplier for controller omega bounds. leg swing frequency ~ sqrt(g/L)."""
        return self.scale ** -0.5

    def nominal_params(self) -> dict:
        return {k: v * self.scale ** _PARAM_SCALE_EXPONENT[k]
                for k, v in _NOMINAL_PARAMS_UNIT.items()}

    def param_bounds(self) -> dict:
        nominal = self.nominal_params()
        return {k: (lo * nominal[k], hi * nominal[k])
                for k, (lo, hi) in PARAM_BOUND_MULTIPLIERS.items()}

    @property
    def pad_damping_ratio(self): return self.pad_loss_factor / 2.0

    def pad_stiffness(self, params: dict, prefix: str) -> float:
        """n/m of one foot pad, e*a/t. no shape-factor correction, so a lower bound."""
        area = params[f"foot_length_{prefix}"] * 2 * self.foot_half_width  # plan view of the box foot
        return self.pad_modulus * area / (self.pad_thickness0 * self.scale)

    def total_mass(self, params: dict) -> float:
        rho, h = params["density"], params["torso_size"] / 2.0
        m = capsule_mass(rho, self.torso_radius(params), 2 * h)
        for prefix in ("left", "right"):
            m += capsule_mass(rho, self.thigh_radius, params[f"thigh_length_{prefix}"])
            m += capsule_mass(rho, self.shank_radius, params[f"shank_length_{prefix}"])
            m += box_mass(rho, params[f"foot_length_{prefix}"] / 2, self.foot_half_width, self.foot_half_thick)
        return m

    def foot_contact(self, params: dict, prefix: str) -> tuple:
        """(k n/m, c n*s/m, supported mass kg) of one foot. two feet carry the robot."""
        k, m = self.pad_stiffness(params, prefix), self.total_mass(params) / 2.0
        return k, 2.0 * self.pad_damping_ratio * float(np.sqrt(k * m)), m

    def ankle_stop(self, params: dict, prefix: str) -> tuple:
        """(stiffness n*m/rad, damping n*m*s/rad) of the rubber bumper. damping from the loss
        factor at the foot's inertia about the ankle."""
        e = gent_modulus(self.bumper_shore_a)
        k = (e * self.bumper_area0 * self.scale ** 2 * (self.bumper_lever0 * self.scale) ** 2
             / (self.bumper_thickness0 * self.scale))
        foot = params[f"foot_length_{prefix}"]
        half = (foot / 2, self.foot_half_width, self.foot_half_thick)
        m = box_mass(params["density"], *half)
        d = foot / 2 - self.foot_heel  # box centre offset from the ankle along x
        inertia = box_inertia(params["density"], *half)[1] + m * (d ** 2 + self.foot_half_thick ** 2)
        return k, 2.0 * self.pad_damping_ratio * float(np.sqrt(k * inertia))

    def servo_torque(self, kp: float, target: float, q: float, qd: float, kind: str) -> float:
        """one servo, one control law, shared by both sims. position error -> torque, goal
        clipped to the joint range, then limited by the motor's speed-torque curve."""
        lo, hi = self.joint_range(kind)
        tau = kp * (min(max(target, lo), hi) - q)
        if self.torque_cap is None:
            return tau
        # driving torque falls with speed (back-emf). braking torque is not boosted: the driver
        # limits current, so it stays at stall.
        w = qd / self.no_load_speed if self.speed_torque else 0.0
        drive = self.torque_cap * max(0.0, 1.0 - abs(w))
        hi, lo = (drive, -self.torque_cap) if w >= 0 else (self.torque_cap, -drive)
        return min(max(tau, lo), hi)

    def ankle_stop_torque(self, params: dict, prefix: str, q: float, qd: float) -> float:
        """rubber bumper past the ankle range. zero inside it."""
        lo, hi = self.ankle_range
        over = q - hi if q > hi else q - lo if q < lo else 0.0
        if over == 0.0:
            return 0.0
        k, c = self.ankle_stop(params, prefix)
        return -k * over - c * qd

    def joint_range(self, kind: str) -> tuple:
        return {"hip": self.hip_range, "knee": self.knee_range, "ankle": self.ankle_range}[kind]

    def torso_radius(self, params: dict) -> float:
        return self.torso_radius_ratio * params["torso_size"]

    def root_z(self, params: dict) -> float:
        """torso height at t=0. clears the longer leg of each pair, so no leg starts in the floor."""
        return (
            params["torso_size"] / 2.0
            + max(params["thigh_length_left"], params["thigh_length_right"])
            + max(params["shank_length_left"], params["shank_length_right"])
            + 2 * self.foot_half_thick
            + self.ground_clearance
        )


SPEC = Spec()


@dataclass(frozen=True)
class RealityDeltas:
    """how the real robot differs from the nominal spec the training sim is built from.

    two kinds. parameter deltas: the real value differs from the nominal (representable in both
    sims). model-form effects: something the training sim has no term for (coulomb friction,
    foam densification, command delay). apply() returns the (params, spec) of the "real" robot.
    the nominal spec / params are what pybullet is built from, always.
    """
    volts: float = 12.0  # supply. sets the stall torque (sourced points) and no-load speed (~ v)
    density_scale: float = 1.0  # link mass
    floor_mu_scale: float = 1.0  # contact friction
    damping_scale: float = 1.0  # hip / knee viscous friction
    ankle_scale: float = 1.0  # ankle stiffness and damping
    armature_scale: float = 1.0  # servo reflected inertia
    pad_modulus_scale: float = 1.0
    pad_loss_scale: float = 1.0
    delay_steps: int = 3
    coulomb: float = 0.0  # n*m, model-form
    pad_stop_strain: float | None = None  # model-form

    def apply(self, params: dict, spec: Spec) -> tuple:
        stall = float(np.interp(self.volts, [v for v, _ in MX64_STALL_BY_VOLT],
                                [t for _, t in MX64_STALL_BY_VOLT]))
        real_params = dict(params)
        real_params["density"] = params["density"] * self.density_scale
        real_params["ankle_stiffness"] = params["ankle_stiffness"] * self.ankle_scale
        real_params["ankle_damping"] = params["ankle_damping"] * self.ankle_scale
        mu = tuple(x * (self.floor_mu_scale if i == 0 else 1.0) for i, x in enumerate(spec.floor_friction))
        real = dataclasses.replace(
            spec,
            torque_cap=None if spec.torque_cap is None else stall,
            no_load_speed=spec.no_load_speed * (self.volts / 12.0),
            armature=spec.armature * self.armature_scale,
            joint_damping0=spec.joint_damping0 * self.damping_scale,
            pad_modulus=spec.pad_modulus * self.pad_modulus_scale,
            pad_loss_factor=spec.pad_loss_factor * self.pad_loss_scale,
            floor_friction=mu,
            delay_steps=self.delay_steps,
            joint_coulomb=self.coulomb,
            pad_stop_strain=self.pad_stop_strain,
        )
        return real_params, real

    @classmethod
    def sample(cls, rng, level: float = 1.0) -> "RealityDeltas":
        """a random real robot. level scales every spread (0 = nominal, 1 = the ranges below).
        stall torque spread is sourced (11.1-14.8 v). every other range is an ASSUMPTION."""
        u = lambda lo, hi: float(1.0 + level * (rng.uniform(lo, hi) - 1.0))
        return cls(
            volts=float(12.0 + level * (rng.uniform(11.1, 14.8) - 12.0)),
            density_scale=u(0.95, 1.05),
            floor_mu_scale=u(0.6, 1.3),
            damping_scale=u(0.5, 2.0),
            ankle_scale=u(0.8, 1.2),
            armature_scale=u(0.8, 1.2),
            pad_modulus_scale=u(0.5, 2.0),
            pad_loss_scale=u(0.7, 2.0),
            delay_steps=int(rng.integers(2, 5)) if level > 0 else 3,
            coulomb=0.0615 * level,
            pad_stop_strain=0.5 if level > 0 else None,
        )


# analytic solid shapes. the pybullet builder MUST use these for inertia: its own capsule
# inertia is a bounding-box approximation, 26-40% off. mujoco computes the same values itself.

def capsule_mass(density: float, radius: float, length: float) -> float:
    """cylinder + two hemispherical caps. length is the cylinder part."""
    return density * (np.pi * radius ** 2 * length + (4.0 / 3.0) * np.pi * radius ** 3)


def capsule_inertia(density: float, radius: float, length: float) -> tuple:
    """(ixx, iyy, izz) about the capsule centre, z along the axis."""
    r, L, rho = radius, length, density
    m_cyl = rho * np.pi * r ** 2 * L
    m_sph = rho * (4.0 / 3.0) * np.pi * r ** 3
    d = L / 2 + 3 * r / 8  # hemisphere com offset from the centre
    transverse = m_cyl * (3 * r ** 2 + L ** 2) / 12 + m_sph * ((83.0 / 320.0) * r ** 2 + d ** 2)
    axial = m_cyl * r ** 2 / 2 + 0.4 * m_sph * r ** 2
    return (transverse, transverse, axial)


def box_mass(density: float, hx: float, hy: float, hz: float) -> float:
    return density * (2 * hx) * (2 * hy) * (2 * hz)


def box_inertia(density: float, hx: float, hy: float, hz: float) -> tuple:
    m = box_mass(density, hx, hy, hz)
    a, b, c = 2 * hx, 2 * hy, 2 * hz
    return (m * (b ** 2 + c ** 2) / 12, m * (a ** 2 + c ** 2) / 12, m * (a ** 2 + b ** 2) / 12)
