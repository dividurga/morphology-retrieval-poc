"""real A with each model-form error on its own and all together, against the sim, on a few genomes. 3 worlds each."""
import sys
import numpy as np
from emerge import morph, real_coppelia as A, sim_coppelia as S

GENOMES = {"2 act, 0/90": morph.chain(2, [0, 90]), "2 act, 0/0": morph.chain(2, [0, 0]), "3 act, 0/90/180": morph.chain(3, [0, 90, 180])}
FORMS = {"parameters only": A.Form(False, False, False), "+ engine (ode)": A.Form(True, False, False),
         "+ servo model": A.Form(False, True, False), "+ connector rule": A.Form(False, False, True), "all three": A.Form(True, True, True)}
for gname, g in GENOMES.items():
    s = [S.rollout(g, seed=i) for i in range(3)]
    print(f"{gname}\n  {'sim (3 runs)':20s} distance {np.mean([x['distance'] for x in s]):.3f} (min {min(x['distance'] for x in s):.3f}, max {max(x['distance'] for x in s):.3f}) broken {np.mean([x['broken'] for x in s]):.1f}", flush=True)
    for fname, f in FORMS.items():
        r = [A.rollout(g, w, form=f) for w in range(3)]
        print(f"  {'A ' + fname:20s} distance {np.mean([x['distance'] for x in r]):.3f} (min {min(x['distance'] for x in r):.3f}, max {max(x['distance'] for x in r):.3f}) broken {np.mean([x['broken'] for x in r]):.1f}", flush=True)
