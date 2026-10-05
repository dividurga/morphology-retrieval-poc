"""connector loads and joint motion in the sim vs the real (breaking off in the real), same genome."""
import numpy as np
import mujoco
from emerge import morph, sim_coppelia as S, real_mujoco as Rm

g = morph.chain(2, [0, 90])
# sim
client, sim = S.api()
if sim.getSimulationState() != sim.simulation_stopped:
    S.stop(client, sim)
sim.loadScene(S.SCENE); sim.setFloatParam(sim.floatparam_simulation_time_step, S.SIM_DT)
mods, sensors = S.build(sim, g)
client.setStepping(True); sim.startSimulation()
tq, fo, jq = [[], []], [[], []], [[], []]
cmd = [[], []]
for k in range(int(38 / S.SIM_DT)):
    t = sim.getSimulationTime()
    for j, i in enumerate((1, 2)):
        c = float(morph.MAX_AMPLITUDE * np.sin(morph.MAX_ANGULAR_FREQUENCY * t + np.radians(g.modules[i].phase_deg)))
        sim.setJointTargetPosition(mods[i]["joint"], c)
        cmd[j].append(c)
    client.step()
    if t > 6:
        for j in range(2):
            r = sim.readForceSensor(sensors[j])
            fo[j].append(np.linalg.norm(r[1])); tq[j].append(np.linalg.norm(r[2]))
        for j, i in enumerate((1, 2)):
            jq[j].append(sim.getJointPosition(mods[i]["joint"]))
S.stop(client, sim)
print("SIM  force-sensor torque (Nm) median/p90/p99/max and joint motion (deg) tracked vs commanded")
for j in range(2):
    a = np.array(tq[j]); q = np.degrees(jq[j]); c = np.degrees(cmd[j][-len(q):])
    print(f"  connector {j + 1}: {np.median(a):.2f} / {np.percentile(a, 90):.2f} / {np.percentile(a, 99):.2f} / {a.max():.2f}   force median {np.median(fo[j]):.1f} N   joint {j + 1}: range {q.min():.0f}..{q.max():.0f}, rms tracking error {np.sqrt(np.mean((q - c) ** 2)):.1f}")
# real, breaking off, nominal masses
w = Rm.world(0, 3); w["break_factor"][:] = 1e6; w["mass"][:] = 1.0
m = mujoco.MjModel.from_xml_string(Rm.build_xml(g, w)); d = mujoco.MjData(m)
qadr = [m.joint(f"j{i}").qposadr[0] for i in (1, 2)]; vadr = [m.joint(f"j{i}").dofadr[0] for i in (1, 2)]
widx = {1: m.equality("w1").id, 2: m.equality("w2").id}
q_ref = np.zeros(2); q_cmd = np.zeros(2); nxt = 0; tr = {1: [], 2: []}; qs = []; cs = []
for k in range(int(38 / Rm.DT)):
    t = d.time
    if t >= nxt - 1e-9:
        for j, i in enumerate((1, 2)):
            q_cmd[j] = np.round(morph.MAX_AMPLITUDE * np.sin(2 * t + np.radians(g.modules[i].phase_deg)) / Rm.RESOLUTION) * Rm.RESOLUTION
        nxt += Rm.COMMAND_PERIOD
    q_ref += np.clip(q_cmd - q_ref, -Rm.SPEED_CAP * Rm.DT, Rm.SPEED_CAP * Rm.DT)
    d.ctrl[:] = np.clip(Rm.KP * (q_ref - d.qpos[qadr]) - Rm.KD * d.qvel[vadr], -1.8, 1.8)
    mujoco.mj_step(m, d)
    if t > 6:
        r = Rm.weld_torques(m, d, widx)
        for i in (1, 2): tr[i].append(r[i])
        qs.append(d.qpos[qadr].copy()); cs.append([morph.MAX_AMPLITUDE * np.sin(2 * t + np.radians(g.modules[i].phase_deg)) for i in (1, 2)])
qs, cs = np.degrees(np.array(qs)), np.degrees(np.array(cs))
print("REAL weld torque (Nm) median/p90/p99/max and joint motion")
for j, i in enumerate((1, 2)):
    a = np.array(tr[i])
    print(f"  connector {i}: {np.median(a):.2f} / {np.percentile(a, 90):.2f} / {np.percentile(a, 99):.2f} / {a.max():.2f}   joint {i}: range {qs[:, j].min():.0f}..{qs[:, j].max():.0f}, rms tracking error {np.sqrt(np.mean((qs[:, j] - cs[:, j]) ** 2)):.1f}")
