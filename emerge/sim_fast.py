"""the sim with the control loop inside coppeliasim. python builds the robot, installs a lua script that sets the sinusoid
targets every sim step and tracks the base, then waits for a 'done' signal. no per-step round trips.
same controller, noise and fitness as sim_coppelia.rollout.

    rollout(genome, duration=38.0, t_start=6.0, seed=0) -> dict(distance, broken, fitness, wall)
"""

import time

import numpy as np

from emerge import morph, sim_coppelia as S

LUA = """
function sysCall_init()
    joints = {%(joints)s}
    phases = {%(phases)s}
    base = %(base)d
    amp, freq, noiseStd, tStart, duration = %(amp)f, %(freq)f, %(noise)f, %(tstart)f, %(duration)f
    math.randomseed(%(seed)d)
    p0 = nil
    done = false
end
local function gauss()
    local u1, u2 = math.random(), math.random()
    return math.sqrt(-2 * math.log(u1 + 1e-12)) * math.cos(2 * math.pi * u2)
end
function sysCall_actuation()
    if done then return end
    local t = sim.getSimulationTime()
    local noise = (math.pi / 2) * noiseStd * gauss()
    for i, j in ipairs(joints) do
        sim.setJointTargetPosition(j, amp * math.sin(freq * t + phases[i]) + noise)
    end
end
function sysCall_sensing()
    if done then return end
    local t = sim.getSimulationTime()
    if p0 == nil and t >= tStart then p0 = sim.getObjectPosition(base, sim.handle_world) end
    if t >= duration - 1e-6 then
        local p1 = sim.getObjectPosition(base, sim.handle_world)
        p0 = p0 or p1
        sim.setFloatSignal('p0x', p0[1]); sim.setFloatSignal('p0y', p0[2])
        sim.setFloatSignal('p1x', p1[1]); sim.setFloatSignal('p1y', p1[2])
        sim.setFloatSignal('done', 1)
        done = true
    end
end
"""


def rollout(genome: morph.Genome, duration: float = 38.0, t_start: float = 6.0, seed: int = 0, engine=None, physics_dt=None, tweak=None):
    client, sim = S.api()
    t0 = time.time()
    if sim.getSimulationState() != sim.simulation_stopped:
        S.stop(client, sim)
    sim.loadScene(S.SCENE)
    sim.setFloatParam(sim.floatparam_simulation_time_step, S.SIM_DT)
    if engine is not None:
        sim.setInt32Param(sim.intparam_dynamic_engine, engine)
    if physics_dt is not None:
        sim.setEngineFloatParam(sim.bullet_global_stepsize, -1, physics_dt)
    if tweak is not None:
        tweak(sim)
    mods, sensors = S.build(sim, genome)
    ph = np.radians(genome.phases())
    act = [i for i, m in enumerate(genome.modules) if m.type == "act"]
    code = LUA % dict(joints=",".join(str(mods[i]["joint"]) for i in act), phases=",".join(f"{ph[i]:.9f}" for i in act),
                      base=mods[0]["shapes"][0], amp=morph.MAX_AMPLITUDE, freq=morph.MAX_ANGULAR_FREQUENCY,
                      noise=S.NOISE_STD, tstart=t_start, duration=duration, seed=seed + 1)
    script = sim.createScript(sim.scripttype_simulation, code, 0, "lua")
    client.setStepping(False)
    sim.clearFloatSignal("done")
    sim.startSimulation()
    deadline = time.time() + max(30.0, 4.0 * duration)  # a script that dies would otherwise hang the worker forever
    while sim.getFloatSignal("done") is None:
        if time.time() > deadline:
            S.stop(client, sim)
            return dict(distance=float("nan"), broken=-1, fitness=float("nan"), wall=time.time() - t0, timeout=True)
        time.sleep(0.02)
    p0 = (sim.getFloatSignal("p0x"), sim.getFloatSignal("p0y"))
    p1 = (sim.getFloatSignal("p1x"), sim.getFloatSignal("p1y"))
    broken = sum(1 for fs in sensors if sim.readForceSensor(fs)[0] & 2)
    S.stop(client, sim)
    dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    return dict(distance=dist, broken=broken, fitness=dist * S.PENALTY ** broken, wall=time.time() - t0)


def repeatability(genome: morph.Genome, k: int = 6, duration: float = 38.0, t_start: float = 6.0, n_workers: int = 1, tol: float = 0.01):
    """run the same genome k times. the sim is only repeatable when no connector breaks: the break is a threshold event
    on a borderline load, and what follows it is chaotic (see emerge/NOTES.md). returns a dict with
    ok = no break in any run and the spread of the distance below tol."""
    if n_workers > 1:
        from emerge import pool
        res = pool.map("emerge.sim_fast:rollout", [dict(genome=genome, duration=duration, t_start=t_start) for _ in range(k)], n_workers=n_workers)
    else:
        res = [rollout(genome, duration, t_start) for _ in range(k)]
    d = np.array([r["distance"] for r in res])
    b = np.array([r["broken"] for r in res])
    return dict(median=float(np.median(d)), std=float(d.std()), break_share=float((b > 0).mean()), ok=bool((b == 0).all() and d.std() < tol))
