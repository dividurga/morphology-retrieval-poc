"""does the sim's broken-connection counter work: lower the torque threshold of the sensors and see the count rise."""
from emerge import morph, sim_coppelia as S

g = morph.chain(2, [0, 90])
client, sim = S.api()
for thr in (1.0, 0.05):
    sim.loadScene(S.SCENE)
    sim.setFloatParam(sim.floatparam_simulation_time_step, S.SIM_DT)
    mods, sensors = S.build(sim, g)
    for fs in sensors:
        sim.setFloatProperty(fs, "torqueThreshold", thr)
    client.setStepping(True)
    sim.startSimulation()
    for _ in range(200):
        for i, m in enumerate(mods):
            if m["joint"] is not None and g.modules[i].type == "act":
                sim.setJointTargetPosition(m["joint"], 1.57 * __import__("math").sin(2 * sim.getSimulationTime() + i))
        client.step()
    reads = [sim.readForceSensor(fs)[0] for fs in sensors]
    S.stop(client, sim)
    print(f"torque threshold {thr} Nm: readForceSensor flags {reads} (1 = intact)")
