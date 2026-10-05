"""step 3: one genome, one phase vector, run in the sim (coppeliasim, edhmor models) and in two reals: A, a slightly different
coppeliasim, and B, mujoco (3 hidden worlds each).
no output of one is used to change the other.

    sh emerge/start_coppelia.sh        # once, keeps coppeliasim headless in the background
    uv run python -m emerge.run_both
"""

import numpy as np

from emerge import morph, real_coppelia, real_mujoco, sim_coppelia

PHASES = [0.0, 90.0]  # degrees, base + 2 actuated modules


def main():
    g = morph.chain(2, PHASES)
    print(f"genome: base + {len(g.modules) - 1} actuated modules, phases {PHASES} deg, 38 s episode, fitness window from 6 s\n")
    s_ = s = sim_coppelia.rollout(g)
    print(f"sim  (coppeliasim, bullet 2.78, dt 50 ms / physics 5 ms): distance {s['distance']:.3f} m, broken {s['broken']}, "
          f"fitness {s['fitness']:.3f}, wall {s['wall']:.1f} s")
    for name, mod in (("real A (slightly different coppeliasim)", real_coppelia), ("real B (mujoco)", real_mujoco)):
        rows = []
        for seed in range(3):
            r = mod.rollout(g, seed)
            rows.append(r)
            print(f"{name}, world {seed}: distance {r['distance']:.3f} m, broken {r['broken']}, fitness {r['fitness']:.3f}, wall {r['wall']:.1f} s")
        print(f"  mean fitness {np.mean([r['fitness'] for r in rows]):.3f} vs sim {s_['fitness']:.3f}\n")
    print("not expected to agree. nothing adjusted.")


if __name__ == "__main__":
    main()
