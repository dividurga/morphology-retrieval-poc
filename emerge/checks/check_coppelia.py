"""sanity: do the assembled modules stay attached, are the joints actuated, what are the force sensor thresholds."""
import numpy as np
from emerge import morph, sim_coppelia as S

g = morph.chain(2, [0, 90])
client, sim = S.api()
sim.loadScene(S.SCENE)
sim.setFloatParam(sim.floatparam_simulation_time_step, S.SIM_DT)
print("engine", sim.getInt32Param(sim.intparam_dynamic_engine), "dt", sim.getFloatParam(sim.floatparam_simulation_time_step),
      "bullet stepsize", sim.getEngineFloatParam(sim.bullet_global_stepsize, -1) if hasattr(sim, "bullet_global_stepsize") else "?")
mods, sensors = S.build(sim, g)
for fs in sensors:
    props = {}
    for name, getter in (("forceThreshold", sim.getFloatProperty), ("torqueThreshold", sim.getFloatProperty),
                         ("consecutiveViolations", sim.getIntProperty)):
        try:
            props[name] = getter(fs, name)
        except Exception as e:
            props[name] = "n/a " + str(e)[:50]
    print("sensor", props)
for i, m in enumerate(mods):
    print("module", i, "shape pos", [np.round(sim.getObjectPosition(s, sim.handle_world), 4).tolist() for s in m["shapes"]],
          "joint", m["joint"])
client.setStepping(True)
sim.startSimulation()
for k in range(40):
    client.step()
print("t", sim.getSimulationTime())
for i, m in enumerate(mods):
    print("module", i, "shape pos after 2s", [np.round(sim.getObjectPosition(s, sim.handle_world), 4).tolist() for s in m["shapes"]])
for fs in sensors:
    print("read sensor", sim.readForceSensor(fs))
sim.stopSimulation()
