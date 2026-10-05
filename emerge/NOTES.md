# emerge: both sims running

sim = EMERGE's own CoppeliaSim models (edhmor). two reals: A, a slightly different coppeliasim, and B, our mujoco model. nothing is fitted to the other side.

## run it

    sh emerge/start_coppelia.sh                 # headless coppeliasim, zmq on port 23000. needs an open stdin, the script handles it
    uv run python -m emerge.run_both            # one genome in both
    uv run python -m emerge.checks.check_real          # mujoco checks
    uv run python -m emerge.checks.check_coppelia      # sim assembly check
    uv run python -m emerge.checks.check_break         # sim broken-connection counter
    uv run python -m emerge.visualize --phases 0 90 --duration 20   # gif of sim, real A, real B side by side -> emerge/viz/

## dataset plan

`emerge/PLAN_DATASET.md`: every row carries a task goal (target distance T, post-hoc relabeling), the admission rule, the success rule and the cpu plan. nothing of it is implemented yet.

## layout

top level: `morph.py` (genome), `sim_coppelia.py`, `sim_fast.py`, `pool.py` (the sim), `real_coppelia.py` (real A), `real_mujoco.py` (real B), `run_both.py`, `visualize.py`, `module_geometry.json` (made by `probe_geom.py`), the start/stop scripts. `checks/` sanity checks of the builds. `experiments/` diagnostics, calibration and bisection scripts with their logs, each cited below. `viz/` and `vendor/` are generated, gitignored.

## visualiser (`emerge/visualize.py`)

coppeliasim runs headless, so nothing is rendered by an engine. each engine records the world pose of every module box each 50 ms (`hook` argument of the three rollouts), matplotlib draws all three the same way: same camera, follows the base, broken connectors tinted red. writes a gif, a top-down path png and the `.npz` recording (`--trace` redraws from it, `--only real_b` needs no coppeliasim). boxes are the shape bounding boxes, so the sim's mesh hulls are drawn as boxes. a 12 s episode of base + 2 modules takes about 30 s for all three.

## versions and setup

- CoppeliaSim Edu V4.10.0 rev0, macOS 13 x86_64 build under Rosetta. the arm64 build needs macOS 15, this mac is 14.6.1. the edu licence is for students and teachers. it is unpacked in `emerge/vendor/` (gitignored).
- edhmor repo (bitbucket afaina/edhmor, master archive, commit 80817a88) in `emerge/vendor/edhmor`. `git clone` failed (truncated pack), the archive zip worked.
- python: `coppeliasim-zmqremoteapi-client` 2.0.4 added to `pyproject.toml`.

## what the sim is (read from the models and the scene, not assumed)

- physics engine: Bullet 2.78 (engine 0). scene `scenes/edhmor/default.ttt`: simulation step 50 ms, bullet step 5 ms (matches the paper). gravity 9.81.
- actuated module (`emergeModuleAX18.ttm`): two shapes of 0.0825 kg each (0.165 kg total), joint max force 1.8 Nm, position control, joint range -90 to +90 deg, bullet friction 0.71.
- base (`flatBase.ttm`): one shape 0.149 x 0.149 x 0.050 m, 0.65 kg, friction 0.5.
- force sensors between modules: torque threshold 1.0 Nm, force threshold 80 N. broken = `readForceSensor` result bit 1.
- controller (edhmor `SinusoidalControllerWNoise`): target = (pi/2) * sin(2.0 * t + phase) + noise, set each 50 ms step.

## differences between sources (recorded, not resolved)

- the paper says angular frequency 1. the edhmor module set (`ModuleSet`, `modulesMaxAngularFrequency`) says 2.0. the sim here uses the code value 2.0.
- masses: paper table 1 says actuated 196 g, base 502 g. the coppelia models are 165 g and 650 g. the real uses the paper values.
- the sim re-parenting follows `CoppeliaSimCreateRobot.addAndConnectModule`: force sensor at the parent's face, the child's base shape parented to the sensor. only straight chains and one orientation/face pattern are ported (`emerge/morph.py:poses`). general trees and link modules are not.

## sim timing and noise

- one 38 s episode: 2.6 to 3.1 s wall (Rosetta). repeated rollouts of the same genome differ by about 5% (0.0565 vs 0.0593 m in a 10 s test), also with the same noise seed. not investigated.

## the real (mujoco), `emerge/real_mujoco.py`

shared with the sim because it is CAD knowledge: box geometry of the modules (`module_geometry.json`, dumped from the edhmor models), joint axis, sinusoid, fitness. real-only, from the paper or datasheet: table-1 masses with a +-10% spread, setpoints at 10 Hz and quantised to 0.29 deg, 25 rpm speed cap, 1.8 Nm torque cap, a PD servo loop, per-module friction draws (0.4 to 0.9), compliant connectors that break above 1.2 Nm x (0.9 to 1.0) for 35 ms.

assumptions that are not in the paper: PD gains (KP 8 Nm/rad, KD 0.1), the connector compliance (`solref 0.02 1`), the spreads of the draws, the 35 ms hold. flagged in the file.

checks (`check_real`): masses equal table 1 at spread 1.0. welds hold the as-built pose (drift 2.9 mm over 1 s with servos holding). servo reaches 30 deg 0.22 s after a 60 deg step, the speed cap alone gives 0.20 s. the connector breaks at an applied torque of about 1.3 Nm against the 1.2 Nm threshold.

bugs found and fixed while building it: mujoco read the joint range in degrees (fixed with `compiler angle="radian"`), the weld pose was wrong when an anchor is set (mujoco's relpose position is where the anchor sits in body1), welded neighbours collided (contact excludes added), the sim's broken flag is bit 1, not bit 0, and the sim must be stopped before loading the next scene. mujoco's weld rotational rows carry 2x the torque, so the torque is half the row norm.

## two reals

- real A (`emerge/real_coppelia.py`): the sim's coppeliasim models, different in two ways. parameters: mass x U(0.9,1.1), friction x U(0.85,1.15), servo torque cap x U(0.9,1.1), setpoint noise x3, setpoints at 10 Hz. model form, each switchable (`Form(engine, servo, connector)`, all on by default):
  - engine: ODE instead of Bullet 2.78 (contact and constraint model change, nothing else).
  - servo: one setpoint update of delay, 0.29 deg quantisation, 0.5 deg backlash (assumption), 25 rpm slew limit (paper), and a torque cap that falls linearly with joint speed from 1.8 Nm to zero at 97 rpm (paper table 1).
  - connector: torque limit 1.2 Nm and pull-out limit 58 N (paper table 1; the sim uses 1.0 Nm and 80 N), per-connector draw U(0.9,1.0), and a torque limit that drops by up to 10% with vibration (smoothed step-to-step change of the connector torque, scale 0.2 Nm, assumption). the 10% is the real B calibration.
- real B (`emerge/real_mujoco.py`): the mujoco model above.
- sim noise floor (`emerge/experiments/repeat_sim.py`): the same genome five times in the unperturbed sim, 0.119 to 0.133 m (std 0.005). once in a while the sim breaks a connector (fitness x0.8), because its connector load peaks at 1.2 to 1.4 Nm against its 1.0 Nm threshold.

### what each model-form error does (`emerge/experiments/compare_forms.py`, mean distance in m, 3 worlds)

| genome | sim | A params only | + engine (ode) | + servo | + connector | all three |
|---|---|---|---|---|---|---|
| 2 act, phases 0/90 | 0.135 | 0.136 | 0.178 | 0.137 | 0.131 | 0.183 |
| 2 act, phases 0/0 | 0.036 | 0.027 | 0.180 | 0.026 | 0.024 | 0.174 |
| 3 act, phases 0/90/180 | 0.014 (1 broken) | 0.015 (1 broken) | 0.131 (0 broken) | 0.015 | 0.021 | 0.125 |

- the engine swap is the whole effect here. with ODE the weak gaits (sim 0.01 to 0.04 m) move 0.13 to 0.18 m, so the ranking of genomes changes.
- the servo and connector errors are inside the sim's noise for these genomes. the servo filter is active (a 0.3 rad/s slew limit cut 0.14 to 0.045 m), but at the real 25 rpm limit and 0.5 deg backlash it barely binds on a sinusoid with peak speed 3 rad/s. the connector rule is not triggered because the loads stay under 1.2 Nm.
- not tested: bullet 2.83 (no engine-version setting found in the api), coppeliasim's own mujoco engine (the run hung the app and left an unkillable process on port 23000; the client now reads the port from `cs.log`, and `emerge/restart_coppelia.sh` restarts it).

## first result (base + 2 actuated modules, phases 0 and 90 deg), `uv run python -m emerge.run_both`

| | distance (m) | broken | fitness |
|---|---|---|---|
| sim | 0.138 | 1 | 0.110 (0.13 with no break) |
| real A, worlds 0 to 2 | 0.135, 0.142, 0.131 | 0, 0, 0 | 0.135, 0.142, 0.131 |
| real B, worlds 0 to 2 | 0.007, 0.000, 0.006 | 2, 2, 2 | 0.005, 0.000, 0.004 |

real A is inside the sim's own run-to-run noise for this genome. real B is far away because of one assumption.

## connector vibration factor, set from the paper (`emerge/experiments/calibrate_break.py`)

the factor was 0.7 to 1.0, my guess, and it broke almost everything. it is now 0.9 to 1.0. rule fixed before the run: the paper (PMC8287515) has 6 of 30 real robots with broken connections, so pick the low end whose share of robots with at least one broken connection is closest to 0.20, on a fixed panel of 30 random chains (2 to 8 actuated modules) x 3 worlds.

| low end | share with a break | mean broken | mean distance (m) |
|---|---|---|---|
| 1.00 | 0.00 | 0.00 | 0.435 |
| 0.95 | 0.00 | 0.00 | 0.435 |
| **0.90** | **0.21** | 0.21 | 0.371 |
| 0.85 | 0.31 | 0.37 | 0.372 |
| 0.80 | 0.49 | 0.72 | 0.359 |
| 0.75 | 0.61 | 0.97 | 0.326 |
| 0.70 | 0.78 | 1.73 | 0.275 |
| 0.60 | 0.90 | 2.28 | 0.275 |

caveats: the paper's robots were evolved with a breaking penalty (a random panel should break more than they did), had a median of 9 modules and were trees. the panel is chains. the break rate depends steeply on this one number. the sections below were written with the old 0.7 value.

## why real B was so far from the sim (`emerge/experiments/ablate_real.py`, `emerge/experiments/compare_loads.py`)

diagnosis only. nothing in the real was changed because of it.

| real B variant | mean distance (m) | broken |
|---|---|---|
| as built | 0.005 | 2 |
| connector vibration factor 1.0 and mass spread 1.0 (no spread) | 0.159 | 0 |
| connectors never break | 0.204 | 0 |
| sim masses, no spread (factor still drawn) | 0.000 | 2 |
| sim friction 0.71 on all modules (factor still drawn) | 0.120 | 1.7 |
| servo: 20 Hz, or no speed cap, or no quantisation (factor still drawn) | 0.000 to 0.007 | 2 |
| rigid connectors, never break | 0.085 | 0 |
| no break + sim masses + sim friction | 0.238 | 0 |
| no break + sim masses + friction + sim-like servo | 0.298 | 0 |

readings:
- the dominant cause is the connector break threshold: my "vibration factor" lowers it to 0.7 to 1.0 x 1.2 Nm. with breaking off the real walks 0.2 m, the same order as the sim's 0.13 m.
- the connector loads themselves match. torque at the connectors, after 6 s: sim median 0.23 and 0.22 Nm, p99 0.81 and 1.20, max 0.97 and 1.42; real B median 0.21 and 0.25, p99 0.44 and 1.10, max 0.75 and 1.52. the distal connector peaks above 1.0 Nm in both.
- the real servo tracks worse: rms tracking error 19 deg against 10 deg in the sim (10 Hz setpoints and the 25 rpm cap).
- with a threshold of 0.84 to 1.2 Nm and a distal load p90 of 1.06 Nm, almost any draw breaks the distal connector, and then the base stops. the factor 0.7 is my assumption, not a paper number. the paper reports 6 of 30 real robots disconnecting.
- mass, friction and servo differences are second order for this genome.

## speed (`emerge/sim_fast.py`, `emerge/pool.py`, `emerge/start_pool.sh`)

- no gpu: coppeliasim's physics engines run on the cpu. gpu is only used for rendering, which headless skips.
- control loop inside coppeliasim: a lua script, installed by python, sets the sinusoid targets every sim step and records the base. one episode of base + 2 modules: 1.5 s instead of 2.7 s. scene load and build take 0.14 s, so the rest is the simulation itself (about 29x real time). the distances agree with the python loop (0.124 to 0.130 vs 0.123 to 0.136 m). it implements the sim only. real A's servo and connector logic is still in python (`real_coppelia.py`).
- parallel instances: `sh emerge/start_pool.sh 8` starts headless instances on zmq ports 23100 to 23107 (named parameter `-GzmqRemoteApi.rpcPort`), `sh emerge/stop_pool.sh` stops them. `emerge.pool.map("module:function", [kwargs, ...], n_workers)` binds each worker process to one instance. throughput on 32 random chains of 2 to 5 modules, 38 s episodes: 1 worker 0.37 rollouts/s, 4 workers 1.47, 8 workers 2.33.
- an instance can hang if a client is killed mid-simulation. `emerge/restart_coppelia.sh` restarts the single instance. a process stuck in the exit state (pid 89164, from the engine test with coppeliasim's mujoco engine) still holds port 23000 until reboot.

## the sim is not repeatable on longer chains (`emerge/experiments/determinism.py`, `step_size.py`, `solver_tweaks.py`, `fresh_first.py`)

the same genome, instance and seed gives very different distances. 4 actuated modules, phases 309/12/263/63 deg: 0.026, 0.021, 0.000, 0.012, 0.153 m (lua loop); 0.012, 0.016, 0.144, 0.043 m (python loop).

what was found:
- it is chaos amplifying a tiny nondeterminism. the base position std across 6 runs of the 4-act genome: 0.45 mm at 0.5 s, 1.6 mm at 1 s, 4 mm at 2 s, 115 mm at 4 s, 190 mm at 10 s. a passive robot (no motion) already drifts 0.01 mm at 0.05 s and 0.25 mm at 2 s. so the contact solver is not bit-reproducible, and the moving chain amplifies it.
- not the cause: setpoint noise (off gives the same), the settling drop (a 5 mm or 20 mm lift gives the same), hidden state in the process (the first rollout on 4 fresh instances: 0.004, 0.346, 0.039, 0.080 m).
- a smaller physics step does not fix it. 2.5 ms looked repeatable on one genome at 10 s (1 mm), but over 6 random chains at 38 s the within-genome std is 0.047 m at 2.5 ms and 0.046 m at 5 ms.
- bullet solver types 1, 2 and 3 are not better (4-run std 0.26, 0.085, 0.26 m, default 0.22 m).
- ode is much calmer: std 0.026 m on the same test, 0.012 m with a fixed random seed (`ode_global_randomseed`). not bit-exact. ode with quickstep off gives nan.
- the noise matters. across 6 random chains (3 to 5 actuated modules, 4 runs each, 5 ms bullet) the mean within-genome std is 0.046 m and the std of the genome means is 0.042 m. a single rollout is mostly noise for these chains. averaging K runs shrinks the noise by sqrt(K); K = 8 gives about 0.016 m.
- the paper's own re-tests show a wide spread too (simulation with noise, IQR 1.50 to 1.98 m).

so: disparity from single rollouts is not usable on chains of 3 or more modules. options: average K repeats (cost K x), run the sim on ode (near-repeatable, but it is not the engine in the edhmor scene), or treat the noise as part of the problem and report it.

### where the run-to-run difference comes from (bisection, `emerge/experiments/bisect_det*.py`, `gait_breaks.py`, `variance_source.py`, `scan_repeatable.py`)

two layers.
1. a tiny nondeterminism in the contact physics, cause not found. about 2 um after 0.05 s. exactly repeatable: a flat base, a base + force sensor + plain box, one module shape alone, a module with only one of its two shapes respondable (hinge and floor on). not repeatable: a module with both shapes respondable together on the floor. none of these changed it: smaller physics step, solver type, solver iterations, single large floor, no setpoint noise, disjoint local respondable masks for the two shapes, bullet inertia setting. ode shrinks it (0.02 um for one module) but a chain of 4 still drifts 20 to 140 um.
2. amplification by connector breaks. the sim breaks a connector when the torque exceeds 1.0 Nm for 7 consecutive steps. gait loads sit right at that limit, so the tiny difference decides whether the break happens, and what follows a break is chaotic. a good 3-module gait (phases 166/128/294, 20 s): 8 runs without a break gave 0.211 +- 0.002 m. 8 of 16 runs broke a connector, with distances 0.003 to 0.161 m.
- random 3-module gaits (40 phase vectors, 6 runs each): 3 of 40 (8%) are repeatable (no break, std < 1 cm; within-gait std 0.003 m, distances 0.136 to 0.169 m); 9 of 40 break in every run; 28 of 40 are mixed. the within-gait std of the others is 0.037 m, the spread between gait medians 0.047 m.
- a cma-es search with 3 repeats per candidate found a "robust" gait that broke in 11 of 16 repeats. the selection needs more repeats per candidate.
- not fixed. `sim_fast.repeatability(genome, k)` reports median, std, share of runs with a break and an `ok` flag (no break, std < 1 cm). use it as an admission rule for dataset rows, or average K repeats of the fitness (distance x 0.8 per break).
- the lua loop had no timeout: a script that dies would hang a worker. a deadline was added (nan distance, `timeout=True`).

## process hygiene (found 2026-10-05)

- stop instances by killing their stdin feeder (`pkill -f "tail -f /dev/null"`, now what `stop_pool.sh` does). a pkill/kill on the coppeliaSim process itself, especially within seconds of its start, leaves it in state UE (uninterruptible exit, 0% cpu) and it keeps its zmq and ws ports until reboot. 8 such processes hold ports 23200 to 23207 and 23700 to 23707 now, so use `BASE=23300 sh emerge/start_pool.sh 8` with `COPPELIA_BASE_PORT=23300` for the pool.
- an idle instance costs cpu (the single instance sat at about 58%), so stop pools when not in use.
- the machine is shared: another job (`generate_robust.py`, 11 workers, nice 5) used about 6.5 of 12 cores and a load average near 100 during the first preflight, so throughput numbers from that window are not usable.

## not done (out of scope for this step)

datasets, surrogate, dr, pybullet removal, the stash restore, general tree morphologies, link modules, the sim's per-run variability.
