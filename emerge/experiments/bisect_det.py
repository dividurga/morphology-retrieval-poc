"""find the smallest robot/scene that is already not repeatable. passive (no actuation), base position std over 8 runs in micrometres."""
import sys
import numpy as np
from emerge import morph, sim_coppelia as S

TIMES = [0.05, 0.1, 0.5, 1.0]


def trace(genome, engine=None, tweak=None, drop=0.0, repeats=8):
    client, sim = S.api()
    runs = []
    for _ in range(repeats):
        if sim.getSimulationState() != sim.simulation_stopped:
            S.stop(client, sim)
        sim.loadScene(S.SCENE)
        sim.setFloatParam(sim.floatparam_simulation_time_step, S.SIM_DT)
        if engine is not None:
            sim.setInt32Param(sim.intparam_dynamic_engine, engine)
        if tweak is not None:
            tweak(sim)
        mods, sensors = S.build(sim, genome, height=0.055 / 2 + drop)
        base = mods[0]["shapes"][0]
        client.setStepping(True)
        sim.startSimulation()
        out = []
        for k in range(int(1.05 / S.SIM_DT)):
            client.step()
            t = sim.getSimulationTime()
            if any(abs(t - x) < 1e-6 for x in TIMES):
                out.append(sim.getObjectPosition(base, sim.handle_world))
        S.stop(client, sim)
        runs.append(out)
    r = np.array(runs)
    return (r.std(axis=0).max(axis=1) * 1e6).round(2).tolist()  # micrometres


if __name__ == "__main__":
    client, sim = S.api()
    cases = {
        "base only (flat on floor)": (morph.Genome([morph.Module("base")]), {}),
        "base only, dropped 20 mm": (morph.Genome([morph.Module("base")]), dict(drop=0.02)),
        "base + 1 module": (morph.chain(1), {}),
        "base + 2 modules": (morph.chain(2), {}),
        "base + 4 modules": (morph.chain(4), {}),
        "base + 4 modules, ode": (morph.chain(4), dict(engine=sim.physics_ode)),
    }
    print("std of base position (micrometres) at t =", TIMES)
    for name, (g, kw) in cases.items():
        print(f"  {name:30s}", trace(g, **kw), flush=True)
