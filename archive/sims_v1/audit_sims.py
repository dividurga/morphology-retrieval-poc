"""numeric audit: how close is sim_pybullet to sim_mujoco?

anything that survives this is engine-level model-form error, not a port bug.
this is a read-only measurement of the gap. it MUST NOT be used to tune pybullet.
run after any change to spec.py / sim_mujoco.py / sim_pybullet.py, and log the result in
SIM_DIFFS.md.

    uv run python audit_sims.py [--no-armature] [--no-delay] [--no-cap]
                                [--pb-armature X] [--substeps N] [--scale S]

sections: 1 static, 2 free air (joint dynamics, no contact), 3 contact, 4 open-loop rollouts.
--no-* switch an effect off, for gap attribution. --sub N sets both sims to N substeps (diagnostic).
"""

import argparse
import dataclasses

import mujoco
import numpy as np
import pybullet as p

import controller
import sim_mujoco as sm
import sim_pybullet as sb
from sim_common import FALL_HEIGHT_FRAC, FALL_PITCH_THRESH
from spec import SPEC

DOF_NAMES = ["root_x", "root_z", "root_pitch", "hip_left", "knee_left", "ankle_left",
             "hip_right", "knee_right", "ankle_right"]
BODY_FOR_LINK = {"torso": "torso", "hip_left": "thigh_left", "knee_left": "shank_left",
                 "ankle_left": "foot_left", "hip_right": "thigh_right",
                 "knee_right": "shank_right", "ankle_right": "foot_right"}
AIR_HEIGHT = 8.0  # root_z offset for contact-free tests
CFG = {"spec": SPEC, "pb": sb.PB_CFG}


def dt():
    return CFG["spec"].timestep


def torque_scale():
    """ext torques scale like the robot's torque, ~ scale^4. keeps pushes in a sane range."""
    return CFG["spec"].scale ** 4


def random_params(rng):
    return {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in CFG["spec"].param_bounds().items()}


def theta_bounds():
    b = [list(x) for x in controller.THETA_BOUNDS]
    b[0] = [b[0][0] * CFG["spec"].omega_scale, b[0][1] * CFG["spec"].omega_scale]
    return np.array([x[0] for x in b]), np.array([x[1] for x in b])


def make_pair(params, client):
    mj = sm.make_model(params, CFG["spec"])
    assert list(sm.ROBOT_DOFS) == DOF_NAMES
    pb = sb.make_model(params, CFG["spec"], CFG["pb"], client=client)
    assert [n for n in sb.ROBOT_DOFS] == ["dummy_x", "dummy_z", "torso"] + DOF_NAMES[3:]
    return mj, pb


def random_q(rng):
    s = CFG["spec"]
    q = np.zeros(9)
    q[2] = rng.uniform(-0.5, 0.5)
    for i in (3, 6):
        q[i] = rng.uniform(*s.hip_range)
    for i in (4, 7):
        q[i] = rng.uniform(*s.knee_range)
    for i in (5, 8):
        q[i] = rng.uniform(*s.ankle_range)
    return q


# ---------------------------------------------------------------- 1. static

def audit_static(params, client, rng, label):
    mj, pb = make_pair(params, client)
    d = mujoco.MjData(mj.mj)
    mujoco.mj_forward(mj.mj, d)
    mass_err = com_err = 0.0
    for link, body in BODY_FOR_LINK.items():
        idx = pb.joint_index[link]
        m_pb = p.getDynamicsInfo(pb.body_id, idx, physicsClientId=client)[0]
        m_mj = mj.mj.body_mass[mj.mj.body(body).id]
        mass_err = max(mass_err, abs(m_pb - m_mj) / m_mj)
        com_pb = np.array(p.getLinkState(pb.body_id, idx, computeForwardKinematics=True,
                                         physicsClientId=client)[0])
        com_err = max(com_err, np.abs(com_pb - d.xipos[mj.mj.body(body).id]).max())
    M_rel, g_err, diag = [], [], []
    for _ in range(4):
        q = random_q(rng)
        d.qpos[mj.dofs] = q
        d.qvel[:] = 0
        mujoco.mj_forward(mj.mj, d)
        M_full = np.zeros((mj.mj.nv, mj.mj.nv))
        mujoco.mj_fullM(mj.mj, d, M_full)
        M_mj = M_full[np.ix_(mj.dofs, mj.dofs)]
        n_pb = len(pb.joint_index)
        q_pb = np.zeros(n_pb)
        q_pb[pb.dofs] = q  # pad dofs stay at 0. with them locked the block is the composite.
        M_pb = np.array(p.calculateMassMatrix(pb.body_id, q_pb.tolist(), physicsClientId=client))[np.ix_(pb.dofs, pb.dofs)]
        M_rel.append(np.linalg.norm(M_pb - M_mj) / np.linalg.norm(M_mj))
        diag.append(np.diag(M_pb) / np.diag(M_mj))
        tau = np.array(p.calculateInverseDynamics(pb.body_id, q_pb.tolist(), [0.0] * n_pb, [0.0] * n_pb,
                                                  physicsClientId=client))[pb.dofs]
        g_err.append(np.abs(tau - d.qfrc_bias[mj.dofs]).max())
    print(f"[static:{label}]")
    print(f"  max rel mass err                           {mass_err:.3e}")
    print(f"  max COM world-pos err (m)                  {com_err:.3e}")
    print(f"  mass-matrix rel Frobenius err (worst)      {max(M_rel):.3e}")
    print(f"  gravity-torque abs err (worst, N*m)        {max(g_err):.3e}")
    print("  mean PB/MJ mass-matrix diagonal per DOF:")
    print("   ", "  ".join(f"{n}={r:.3f}" for n, r in zip(DOF_NAMES, np.mean(diag, axis=0))))


# ---------------------------------------------------------------- run helpers

def pb_state(pb):
    s = p.getJointStates(pb.body_id, pb.dofs, physicsClientId=pb.client)
    return np.array([x[0] for x in s])


def pb_run(pb, targets_fn, duration, q0=None, ext=None):
    """same structure as sim_pybullet.simulate. targets held over each control period."""
    sb.reset(pb, q0)
    traj = [pb_state(pb)]
    for s in range(int(round(duration / dt()))):
        targets = targets_fn(s * dt())
        for _ in range(pb.cfg.substeps):
            sb.apply_control(pb, targets, ext)
            p.stepSimulation(physicsClientId=pb.client)
        traj.append(pb_state(pb))
    return np.array(traj)


def mj_run(mj, targets_fn, duration, q0=None, ext=None):
    """same structure as sim_mujoco.simulate, plus q0 / ext hooks."""
    d = mujoco.MjData(mj.mj)
    if q0 is not None:
        d.qpos[mj.dofs] = q0
    delay = mj.spec.delay_steps * dt()
    traj = [d.qpos[mj.dofs].copy()]
    for s in range(int(round(duration / dt()))):
        targets = targets_fn(max(0.0, s * dt() - delay))
        for _ in range(mj.spec.mj_substeps):
            sm.apply_control(mj, d, targets, ext)
            mujoco.mj_step(mj.mj, d)
        traj.append(d.qpos[mj.dofs].copy())
    return np.array(traj)


def hold_zero(_t):
    return np.zeros(4)


# ---------------------------------------------------------------- 2. free air

def audit_air(params, client):
    mj, pb = make_pair(params, client)
    print("[free air: contact-free joint dynamics, 1.0 s, PD holding zero targets]")
    q0 = np.zeros(9)
    q0[1] = AIR_HEIGHT
    a, b = mj_run(mj, hold_zero, 1.0, q0=q0), pb_run(pb, hold_zero, 1.0, q0=q0)
    print(f"  free fall: max|dz| = {np.abs(a[:, 1] - b[:, 1]).max():.2e} m "
          f"(MJ fell {a[0, 1] - a[-1, 1]:.4f}, PB {b[0, 1] - b[-1, 1]:.4f}; analytic 4.905)")
    for label, (dof, val) in {"ankle_left 0.4": ("ankle_left", 0.4), "knee_left -1.0": ("knee_left", -1.0),
                              "hip_left 0.8": ("hip_left", 0.8), "pitch 0.3": ("root_pitch", 0.3)}.items():
        q0 = np.zeros(9)
        q0[1] = AIR_HEIGHT
        q0[DOF_NAMES.index(dof)] = val
        a, b = mj_run(mj, hold_zero, 1.0, q0=q0), pb_run(pb, hold_zero, 1.0, q0=q0)
        err = np.abs(a - b)[:, 2:]
        amp = np.abs(a[:, 2:]).max() + 1e-9
        worst = DOF_NAMES[2:][int(err.max(axis=0).argmax())]
        print(f"  {label:16s} max|dq|={err.max():.4f} rad ({100 * err.max() / amp:5.1f}% of amplitude)"
              f"  worst={worst}  rms={np.sqrt((err ** 2).mean()):.4f}")

    s = CFG["spec"]
    kp = {"hip": params["actuator_kp_hip"], "knee": params["actuator_kp_knee"]}
    print("[servo hold: constant external torque below stall, final joint angle after 0.8 s. analytic = tau/kp]")
    for dof, tau in [("hip_left", 2.0), ("hip_left", -2.0), ("knee_left", 2.0), ("knee_left", -2.0)]:
        q0 = np.zeros(9)
        q0[1] = AIR_HEIGHT
        a_ = mj_run(mj, hold_zero, 0.8, q0=q0, ext={dof: tau})
        b_ = pb_run(pb, hold_zero, 0.8, q0=q0, ext={dof: tau})
        i = DOF_NAMES.index(dof)
        # the push also accelerates the torso, so the joint rings. report the mean of the last 0.3 s.
        print(f"  {dof:11s} tau={tau:+5.1f}  analytic {tau / kp[dof.split('_')[0]]:+.3f} rad (if the body were fixed)"
              f"  MJ {a_[-60:, i].mean():+.3f}  PB {b_[-60:, i].mean():+.3f}")
    k = torque_scale()
    print(f"[ankle rubber stop: constant external torque {20.0 * k:.2f} N*m, peak q over 0.6 s]")
    for dof, tau in [("ankle_left", 20.0), ("ankle_left", -20.0)]:
        tau *= k
        q0 = np.zeros(9)
        q0[1] = AIR_HEIGHT
        a_ = mj_run(mj, hold_zero, 0.6, q0=q0, ext={dof: tau})
        b_ = pb_run(pb, hold_zero, 0.6, q0=q0, ext={dof: tau})
        i = DOF_NAMES.index(dof)
        pk = (lambda tr: tr[:, i].max() if tau > 0 else tr[:, i].min())
        print(f"  {dof:11s} tau={tau:+7.3f}  MJ peak={pk(a_):+.3f}  PB peak={pk(b_):+.3f}")


# ---------------------------------------------------------------- 3. contact

def audit_contact(params, client):
    mj, pb = make_pair(params, client)
    print("[contact: spawn on ground, PD holding zero targets, 1.5 s]")
    a, b = mj_run(mj, hold_zero, 1.5), pb_run(pb, hold_zero, 1.5)
    for name, i in [("root_z", 1), ("root_pitch", 2), ("root_x", 0)]:
        print(f"  {name:10s} final MJ={a[-1, i]:+.4f}  PB={b[-1, i]:+.4f}   max|diff|={np.abs(a[:, i] - b[:, i]).max():.4f}")
    audit_friction(client)


def audit_friction(client):
    """a block sliding on the floor stops at v^2 / (2 mu g). checks the friction combine rule
    against physics in each engine. a standing robot is no use for this: it topples."""
    s = CFG["spec"]
    mu = max(s.robot_friction[0], s.floor_friction[0])
    v0, expect = 1.0, 1.0 / (2 * max(s.robot_friction[0], s.floor_friction[0]) * 9.81)
    rf, ff = " ".join(map(str, s.robot_friction)), " ".join(map(str, s.floor_friction))
    xml = f"""<mujoco><option timestep="0.001" gravity="0 0 -9.81"/><worldbody>
    <geom type="plane" size="5 5 .1" friction="{ff}"/>
    <body pos="0 0 0.011"><freejoint/><geom type="box" size=".04 .03 .01" mass="1" friction="{rf}"/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qvel[0] = v0
    for _ in range(2000):
        mujoco.mj_step(m, d)
    dm = d.qpos[0]
    pc = p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81, physicsClientId=pc)
    p.setTimeStep(0.001, physicsClientId=pc)
    fl = p.createMultiBody(0, p.createCollisionShape(p.GEOM_PLANE, physicsClientId=pc), physicsClientId=pc)
    bx = p.createMultiBody(1.0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[.04, .03, .01], physicsClientId=pc),
                           basePosition=[0, 0, 0.0105], physicsClientId=pc)
    p.changeDynamics(fl, -1, lateralFriction=s.floor_friction[0], physicsClientId=pc)
    p.changeDynamics(bx, -1, lateralFriction=mu / s.floor_friction[0], linearDamping=0, angularDamping=0,
                     physicsClientId=pc)
    p.resetBaseVelocity(bx, [v0, 0, 0], physicsClientId=pc)
    for _ in range(2000):
        p.stepSimulation(physicsClientId=pc)
    dp = p.getBasePositionAndOrientation(bx, physicsClientId=pc)[0][0]
    p.disconnect(pc)
    print(f"[friction: block sliding at 1 m/s, stopping distance. physics {expect:.4f} m (mu {mu})]")
    print(f"  MJ={dm:.4f} m  PB={dp:.4f} m")


# ---------------------------------------------------------------- 4. rollouts

def audit_rollouts(client, rng, n_morph=3, n_theta=15, duration=3.0):
    lo, hi = theta_bounds()
    diverge, rms_early, fm, fp, dm, dp = [], [], [], [], [], []

    def fell(tr, standing):
        f = np.where((np.abs(tr[:, 2]) > FALL_PITCH_THRESH)
                     | (tr[:, 1] + standing < FALL_HEIGHT_FRAC * standing))[0]
        return f[0] * dt() if len(f) else duration

    for mi in range(n_morph):
        params = CFG["spec"].nominal_params() if mi == 0 else random_params(rng)
        mj, pb = make_pair(params, client)
        for _ in range(n_theta):
            theta = lo + rng.random(len(lo)) * (hi - lo)
            tf = lambda t, th=theta: controller.joint_targets(th, t)
            a, b = mj_run(mj, tf, duration), pb_run(pb, tf, duration)
            err = np.abs(a[:, :3] - b[:, :3])
            bad = np.where((err[:, 2] > 0.1) | (err[:, 1] > 0.02 * CFG["spec"].scale))[0]
            diverge.append(bad[0] * dt() if len(bad) else duration)
            rms_early.append(np.sqrt(((a[:51, :3] - b[:51, :3]) ** 2).mean(axis=0)))
            fm.append(fell(a, pb.standing_height))
            fp.append(fell(b, pb.standing_height))
            dm.append(a[-1, 0])
            dp.append(b[-1, 0])
    diverge, rms_early = np.array(diverge), np.array(rms_early)
    print(f"[rollouts: {len(diverge)} random (morphology, theta) pairs, open loop, {duration}s]")
    print(f"  time until |dpitch|>0.1 or |dz|>0.02*scale:  median={np.median(diverge):.2f}s  "
          f"10th pct={np.percentile(diverge, 10):.2f}s  (cap {duration}s)")
    print(f"  RMS err over first 0.25 s: x={rms_early[:, 0].mean():.4f} m  z={rms_early[:, 1].mean():.4f} m  pitch={rms_early[:, 2].mean():.4f} rad")
    print(f"  fall time corr (MJ vs PB) = {np.corrcoef(fm, fp)[0, 1]:.3f};  median |dfall|={np.median(np.abs(np.array(fm) - np.array(fp))):.3f}s")
    print(f"  displacement corr (MJ vs PB) = {np.corrcoef(dm, dp)[0, 1]:.3f}")


def self_check(params, client):
    """the audit's pybullet loop MUST equal sim_pybullet.simulate or the audit is void."""
    _, pb = make_pair(params, client)
    theta = controller.DEFAULT_THETA.copy()
    theta[0] *= CFG["spec"].omega_scale
    tf = lambda t: controller.joint_targets(theta, t)
    xa, xs = pb_run(pb, tf, 0.5)[-1, 0], sb.simulate(pb, theta, 0.5).displacement
    print(f"[self-check] audit x={xa:.9f}  sim_pybullet.simulate x={xs:.9f}  diff={abs(xa - xs):.2e}")
    assert abs(xa - xs) < 1e-6, "audit control loop drifted from sim_pybullet.simulate"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--no-armature", action="store_true", help="mujoco armature 0")
    ap.add_argument("--no-delay", action="store_true", help="mujoco command delay 0")
    ap.add_argument("--no-cap", action="store_true", help="torque cap off in both sims")
    ap.add_argument("--pb-armature", type=float, default=None)
    ap.add_argument("--substeps", type=int, default=None, help="pybullet substeps")
    ap.add_argument("--sub", type=int, default=None, help="diagnostic: N substeps in both sims")
    args = ap.parse_args()
    spec = SPEC
    if args.scale is not None:
        spec = dataclasses.replace(spec, scale=args.scale)
    if args.no_armature:
        spec = dataclasses.replace(spec, armature=0.0)
    if args.no_delay:
        spec = dataclasses.replace(spec, delay_steps=0)
    if args.no_cap:
        spec = dataclasses.replace(spec, torque_cap=None)
    cfg = sb.PB_CFG
    if args.pb_armature is not None:
        cfg = dataclasses.replace(cfg, armature=args.pb_armature)
    if args.substeps is not None:
        cfg = dataclasses.replace(cfg, substeps=args.substeps)
    if args.sub is not None:
        cfg = dataclasses.replace(cfg, substeps=args.sub)
        spec = dataclasses.replace(spec, mj_substeps=args.sub)
    CFG["spec"], CFG["pb"] = spec, cfg
    print(f"[config] scale={spec.scale} mj_armature={spec.armature:.5f} delay={spec.delay_steps} "
          f"cap={spec.torque_cap} speed_torque={spec.speed_torque} | pb_armature={cfg.armature} "
          f"pb_substeps={cfg.substeps} mj_substeps={spec.mj_substeps}")
    rng = np.random.default_rng(0)
    client = p.connect(p.DIRECT)
    nominal = spec.nominal_params()
    self_check(nominal, client)
    audit_static(nominal, client, rng, "nominal")
    audit_static(random_params(rng), client, rng, "random morphology")
    audit_air(nominal, client)
    audit_contact(nominal, client)
    audit_rollouts(client, rng)
