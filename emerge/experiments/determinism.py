"""why is the sim not repeatable on longer chains? run one genome several times, record the base position over time,
and see when the runs part ways. each variant changes one thing.

    COPPELIA_ZMQ_PORT=23100 uv run python -m emerge.experiments.determinism
"""
import sys
import numpy as np
from emerge import morph, sim_coppelia as S

G = morph.chain(4, [309, 12, 263, 63])
TIMES = [0.5, 1, 2, 4, 6, 10]


def trace(engine=None, noise=0.0, lift=0.0, duration=10.5, seed=0, amp=1.0, tweak=None):
    client, sim = S.api()
    if sim.getSimulationState() != sim.simulation_stopped:
        S.stop(client, sim)
    sim.loadScene(S.SCENE)
    sim.setFloatParam(sim.floatparam_simulation_time_step, S.SIM_DT)
    if engine is not None:
        sim.setInt32Param(sim.intparam_dynamic_engine, engine)
    if tweak is not None:
        tweak(sim)
    mods, sensors = S.build(sim, G, height=0.055 / 2 + lift)
    base = mods[0]["shapes"][0]
    rng = np.random.default_rng(seed)
    ph = np.radians(G.phases())
    client.setStepping(True)
    sim.startSimulation()
    out = []
    for k in range(int(duration / S.SIM_DT)):
        t = sim.getSimulationTime()
        nz = (np.pi / 2) * noise * rng.standard_normal()
        for i, m in enumerate(mods):
            if m["joint"] is not None and G.modules[i].type == "act":
                sim.setJointTargetPosition(m["joint"], float(amp * morph.MAX_AMPLITUDE * np.sin(morph.MAX_ANGULAR_FREQUENCY * t + ph[i]) + nz))
        client.step()
        tt = sim.getSimulationTime()
        if any(abs(tt - x) < 1e-6 for x in TIMES):
            out.append(sim.getObjectPosition(base, sim.handle_world))
    S.stop(client, sim)
    return np.array(out)


def report(name, times=None, **kw):
    global TIMES
    TIMES = times or [0.5, 1, 2, 4, 6, 10]
    reps = np.array([trace(**kw) for _ in range(6)])  # (6, len(TIMES), 3)
    spread = reps.std(axis=0).max(axis=1) * 1000  # mm, worst axis at each time
    print(f"{name:34s} std of base position across 6 runs (mm) at t={TIMES}: {np.round(spread, 3).tolist()}", flush=True)


if __name__ == "__main__":
    client, sim = S.api()
    mid = [0.5, 1, 2, 4, 6, 10]
    report("bullet default (5 ms, solver default)", times=mid)
    report("bullet, physics step 2.5 ms", times=mid, tweak=lambda sm: sm.setEngineFloatParam(sm.bullet_global_stepsize, -1, 0.0025))
    report("bullet, solver iterations x4", times=mid, tweak=lambda sm: sm.setEngineInt32Param(sm.bullet_global_constraintsolvingiterations, -1, 4 * sm.getEngineInt32Param(sm.bullet_global_constraintsolvingiterations, -1)))
    report("ode, physics step 2.5 ms", times=mid, engine=sim.physics_ode, tweak=lambda sm: sm.setEngineFloatParam(sm.ode_global_stepsize, -1, 0.0025))
