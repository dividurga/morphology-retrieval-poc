"""can a solver setting make the sim repeatable? 4 runs each of the 4-act genome, 12 s."""
import numpy as np
from emerge import morph, sim_fast as F, sim_coppelia as S
g = morph.chain(4, [309, 12, 263, 63])
client, sim = S.api()
cases = {
    "bullet default": dict(),
    "bullet solver type 1": dict(tweak=lambda s: s.setEngineInt32Param(s.bullet_global_constraintsolvertype, -1, 1)),
    "bullet solver type 2": dict(tweak=lambda s: s.setEngineInt32Param(s.bullet_global_constraintsolvertype, -1, 2)),
    "bullet solver type 3": dict(tweak=lambda s: s.setEngineInt32Param(s.bullet_global_constraintsolvertype, -1, 3)),
    "ode default": dict(engine=sim.physics_ode),
    "ode fixed random seed 1": dict(engine=sim.physics_ode, tweak=lambda s: s.setEngineInt32Param(s.ode_global_randomseed, -1, 1)),
    "ode, quickstep off": dict(engine=sim.physics_ode, tweak=lambda s: s.setEngineBoolParam(s.ode_global_quickstep, -1, False)),
}
for name, kw in cases.items():
    try:
        d = [F.rollout(g, duration=12.0, t_start=3.0, **kw)["distance"] for _ in range(4)]
        print(f"{name:26s} {np.round(d, 5).tolist()}  std {np.std(d):.5f}", flush=True)
    except Exception as e:
        print(f"{name:26s} failed: {str(e)[:100]}", flush=True)
