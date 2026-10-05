"""ode: module alone and chains, with and without a fixed ode random seed (quickstep permutes constraint order with a seeded rng)."""
from emerge import morph, sim_coppelia as S
from emerge.experiments import bisect_det as D1, bisect_det2 as B



def ode(sim, seed=None):
    sim.setInt32Param(sim.intparam_dynamic_engine, sim.physics_ode)
    if seed is not None:
        sim.setEngineInt32Param(sim.ode_global_randomseed, -1, seed)


if __name__ == "__main__":
    client, sim = S.api()
    print("module alone, ode, default seed        ", B.run(lambda s: (ode(s), B.module_alone(s))[1]), flush=True)
    print("module alone, ode, fixed seed 7        ", B.run(lambda s: (ode(s, 7), B.module_alone(s))[1]), flush=True)
    print("chain of 4 (passive), ode, default seed", D1.trace(morph.chain(4), tweak=lambda s: ode(s)), flush=True)
    print("chain of 4 (passive), ode, fixed seed 7", D1.trace(morph.chain(4), tweak=lambda s: ode(s, 7)), flush=True)
