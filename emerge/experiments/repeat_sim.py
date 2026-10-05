"""noise floor: the same genome, the same seed, in the unperturbed sim, five times; and real A over five worlds."""
import numpy as np
from emerge import morph, sim_coppelia as S, real_coppelia as A
g = morph.chain(2, [0, 90])
b = [S.rollout(g)["distance"] for _ in range(5)]
print("sim, same genome and seed x5:", np.round(b, 3), "std", round(float(np.std(b)), 3))
a = [A.rollout(g, w) for w in range(5)]
print("real A, 5 worlds:", np.round([x["distance"] for x in a], 3), "broken", [x["broken"] for x in a])
