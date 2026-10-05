"""is the nondeterminism hidden state that persists in a coppeliasim process? compare the FIRST rollout on fresh instances."""
import sys
from emerge import morph, sim_fast as F
g = morph.chain(4, [309, 12, 263, 63])
r = [F.rollout(g, duration=12.0, t_start=3.0)["distance"] for _ in range(int(sys.argv[1]))]
print("port-run distances", [round(x, 6) for x in r], flush=True)
