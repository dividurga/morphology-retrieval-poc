"""checks for the mujoco real: masses, weld consistency, servo lag and speed cap, connector break."""
import mujoco
import numpy as np
from emerge import morph, real_mujoco as Rm

g = morph.chain(2, [0, 90])
w = Rm.world(0, 3)
w1 = dict(mass=np.ones(3), friction=w["friction"], break_factor=np.ones(3))
m = mujoco.MjModel.from_xml_string(Rm.build_xml(g, w1))
d = mujoco.MjData(m)
mass_by = {i: float(m.body_mass[m.body(f"m{i}A").id] + (m.body_mass[m.body(f"m{i}B").id] if i else 0)) for i in range(3)}
print("masses at spread 1.0 (kg): base", round(mass_by[0], 4), "act", round(mass_by[1], 4), round(mass_by[2], 4), "| table 1: base 0.502, act 0.196")

# welds hold the as-built pose: servos holding zero, module origins after 1 s
def hold(m, d, n):
    qa = [m.joint(f"j{i}").qposadr[0] for i in range(1, 3) if m.nq]
    for _ in range(n):
        for k, i in enumerate((1, 2)):
            try:
                d.ctrl[k] = np.clip(Rm.KP * (0 - d.qpos[m.joint(f"j{i}").qposadr[0]]) - Rm.KD * d.qvel[m.joint(f"j{i}").dofadr[0]], -Rm.TORQUE_CAP, Rm.TORQUE_CAP)
            except Exception:
                pass
        mujoco.mj_step(m, d)

mujoco.mj_forward(m, d)
pos0 = {i: d.xpos[m.body(f"m{i}A").id].copy() for i in range(3)}
hold(m, d, 500)
print("max drift of module origins after 1 s with servos holding zero (m):", max(np.linalg.norm(d.xpos[m.body(f"m{i}A").id] - pos0[i]) for i in range(3)))

# servo step in the air, no gravity, free base
g1 = morph.chain(2)
w1b = dict(mass=np.ones(3), friction=np.ones(3) * 0.6, break_factor=np.ones(3))
m2 = mujoco.MjModel.from_xml_string(Rm.build_xml(g1, w1b, height=2.0))
m2.opt.gravity[:] = 0
d2 = mujoco.MjData(m2)
qa, va = m2.joint("j2").qposadr[0], m2.joint("j2").dofadr[0]
q_ref, tr = 0.0, []
cmd = np.radians(60)
for k in range(int(1.0 / Rm.DT)):
    qc = cmd if d2.time >= 0.1 else 0.0
    q_ref += np.clip(qc - q_ref, -Rm.SPEED_CAP * Rm.DT, Rm.SPEED_CAP * Rm.DT)
    d2.ctrl[1] = np.clip(Rm.KP * (q_ref - d2.qpos[qa]) - Rm.KD * d2.qvel[va], -Rm.TORQUE_CAP, Rm.TORQUE_CAP)
    mujoco.mj_step(m2, d2)
    tr.append(d2.qpos[qa])
tr = np.degrees(tr)
t30 = next((i * Rm.DT for i, v in enumerate(tr) if v >= 30), None)
print(f"servo (in the air): 60 deg step at 0.1 s. reaches 30 deg {None if t30 is None else round(t30 - 0.1, 3)} s after the step; the 25 rpm cap alone gives {np.radians(30) / Rm.SPEED_CAP:.3f} s. "
      f"it stops at {tr[-1]:.1f} deg")

# connector: slow torque ramp across the weld in the air, count the break with the real logic
m3 = mujoco.MjModel.from_xml_string(Rm.build_xml(g1, w1b, height=2.0))
m3.opt.gravity[:] = 0
d3 = mujoco.MjData(m3)
widx = {1: m3.equality("w1").id}
m3.eq_active0[:] = 1
over, applied_at_break = 0.0, None
for k in range(int(4.0 / Rm.DT)):
    torque = 0.8 * d3.time  # Nm, ramps to 3.2
    d3.xfrc_applied[m3.body("m1A").id, 3:6] = [0, torque, 0]
    d3.xfrc_applied[m3.body("m0A").id, 3:6] = [0, -torque, 0]
    mujoco.mj_step(m3, d3)
    tq = Rm.weld_torques(m3, d3, widx)[1]
    over = over + Rm.DT if tq > Rm.CONNECTOR_TORQUE else 0.0
    if over >= Rm.BREAK_HOLD:
        applied_at_break = torque
        break
print("connector: weld torque threshold", Rm.CONNECTOR_TORQUE, "Nm; broke at an applied torque of", applied_at_break)
