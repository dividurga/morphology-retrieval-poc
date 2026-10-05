# sim diffs

mujoco = "real". pybullet = training twin. both open the same mjcf from `morphology.py`.
the final product does not exist for system identification. so:

- the twin MUST NOT be fitted, tuned or calibrated against mujoco output.
- every twin setting MUST come from the file or from the engine's documented behaviour.
- gap measurements (`human_ladder.py`) MUST NOT be used to choose a setting.
- engine differences that remain MUST be left alone. they are the gap the poc measures.

## pipeline

- `morphology.py`: original biped, mjcf generator. joint limits explicit. `torque_actuators=True` swaps position actuators for motors.
- `engines.py`: `MuJoCoEngine` (native) and `PyBulletEngine` (`loadMJCF`). one shared `command_to_torque`: `tau = kp * (clip(target) - q)`, the original position law. same dt 0.005, same open-loop loop, same fall rule.
- `backend_mujoco.py`, `backend_pybullet.py`: `make_model` / `simulate` for `optimize_controller.py`.
- `optimize_controller.py`: cma-es. `workers=N` evaluates a generation in a process pool. DR arm raises until `perturbations.py` is ported to the twin.

## what loadMJCF does to this file

| attribute in the file | pybullet after import | effect |
|---|---|---|
| masses, geometry | kept | none |
| joint limits | kept, enforced under torque control (hip 1.346 vs limit 1.3). mujoco's limit is soft (1.85) | small |
| robot friction 0.05, floor 1.0 | kept, but pybullet multiplies, mujoco takes max | twin on ice |
| joint damping (0.5, ankle 0.4) | dropped | |
| ankle spring (stiffness 8) | dropped. ankle stays where released | foot flops |
| capsule inertia | bounding box. thigh iyy 0.0136 vs 0.0065 analytic | |
| armature 0.01 | dropped. no field | |
| root height | torso loads too high (pos applied once per root joint) | start pose off |
| default motors | brake on every joint | |

## setup ladder (`human_ladder.py`)

what a person does after loading the file, in order. each rung comes from the file or the docs.
46 gaits: 10 mujoco-optimised, 16 neighbours of the best two, 20 random. mujoco: 8 walk > 1 m, best 8.12 m.
trace loss = pseudo-huber of (x, z, pitch) over the first 1 s. twin walks = share of the 8 mujoco walkers the twin also walks.

| rung | fall agree | dist corr | mean abs dist err m | trace loss | twin walks | twin dist / real |
|---|---|---|---|---|---|---|
| 0 raw import | 0.74 | -0.01 | 1.61 | 3.35 | 0/8 | 0.00 |
| 1 + friction rule | 0.85 | 0.51 | 1.15 | 1.98 | 3/8 | 0.34 |
| 2 + damping | 0.83 | 0.49 | 1.24 | 1.09 | 1/8 | 0.10 |
| 3 + ankle spring | 0.83 | 0.53 | 1.08 | 0.87 | 2/8 | 0.50 |
| 4 + analytic inertia | 0.78 | 0.59 | 1.24 | 1.18 | 0/8 | 0.12 |
| 5 + armature (lumped on link) | 0.85 | 0.65 | 0.84 | 0.48 | 2/8 | 0.46 |
| 6 + 5 substeps | 0.78 | 0.53 | 0.99 | 0.48 | 1/8 | 0.43 |

each fix alone on the raw import does ~nothing, except friction (dist corr 0.51).

readings:
- friction is the dominant error. the combine rule, not the model.
- trace loss falls at every rung (3.35 -> 0.48). distance agreement does not follow.
- fixing one error can expose another. analytic inertia is right, and it lowered the walker count until armature came in.
- 8 walkers, 46 gaits. non-monotone rows may be noise. no standard errors.

## residual gap (left alone on purpose)

- mujoco soft joint limits and soft contact. pybullet is harder.
- armature: lumped on a link in the twin, per-joint in mujoco.
- twin reproduces 1 to 2 of 8 mujoco walkers at 0.4 to 0.5 of their distance.

## decisions

- training twin = all six rungs ("what a person does"). chosen by procedure, not by the gap numbers.
- walker2d tried. it stands in both engines but finds no open-loop gait at 30x the budget (0.2 to 0.8 m). the original biped finds 6 to 8 m in mujoco. archived work in `archive/sims_v1/`, walker files in `walker/`.

## open

- target-conditioned pipeline over several morphologies. then the shortlist, surrogate, retrieval steps in `README.md`.
- DR arm: port `perturbations.py` to the twin.
- standard errors for the ladder, more walkers.

## log

- original biped restored from git history. two-engine loop added. mujoco loop reproduces the original numbers exactly.
- cross-engine search: mujoco-trained gaits 0.7 to 7.4 m. fall at once in the raw twin.
- loader findings and ladder recorded. friction rule dominant.
- dataset by post-hoc relabeling: `generate_dataset.py`, `check_dataset.py`. data_v1: 200 morphologies, 3 s episode, 12 restarts x 480 evals each, 5 cm bins over 0 to 3 m, empty bins kept. every re-evaluated row (995 rows, every 3rd body) reproduces its distance exactly.
- disparity experiments: `experiment_disparity.py`, target 1.0 m, pool 229 rows (+-0.1 m). real = mujoco with 2 fixed `perturbations.py` draws. median disparity 0.41 m, 68% of pool rows fall in real. best-nominal baseline real error 0.49 m, rank 139/229. idw on 5 points: spearman about 0, no better than the mean predictor. oracle best real error 0.001 m.
- surrogate diagnostics: `build_pools.py`, `analyze_disparity.py`, targets 0.6, 1.0, 1.5 m. held-out disparity prediction is near the mean predictor for every feature set, idw and gp, up to n=100 (spearman 0.1 to 0.28, fall auc 0.5 to 0.6). random best-of-5 beats the 5-step loop at 1.0 m (0.140 vs 0.209 mean real error). twin re-run does not reproduce the stored distance for about 2% of rows (dropped).
- gap ladder: `engines.apply_reality` (lag + multipliers, both engines), rungs R0 to R2, twins pybullet and mujoco (`data_mj`). summary in `results/GAP_LADDER.md`. open-loop twin gaits fall under a 5 ms lag. disparity stays hard to predict (spearman 0.15 to 0.4).
- dr experiment started: `dr_experiment.py`, `dr_report.py`. nominal vs dr_mult vs dr_lag training in the pybullet twin, evaluated in pybullet realities and mujoco. results in `results/dr/`.
