# morphology-retrieval-poc

Does a gait trained in a cheap sim still hit its target distance in a different "real"? Gaits for a small biped family are
trained in a pybullet twin in two arms (nominal, robust/DR), then scored in three reals.

## current pipeline
- `morphology.py`, `controller.py`, `environment.py`: biped MJCF, 13-parameter open-loop sine gait, result type.
- `engines.py`: the two engines (`PyBulletEngine`, `MuJoCoEngine`), the shared PD law, `rollout` (0.5 s settle at the
  spawn pose, then the gait), and `ModelForm` (A1/Go1-class actuator error, identical in both engines).
- `perturbations.py`: friction / mass / motor / damping ranges used for the robust arm's draws.
- `generate_dataset.py`: `sample_morphologies` (3 free parameters: thigh, shank, torso, +-30%, legs tied) and constants.
- `generate_robust.py`: trains both arms, evaluates in sim + three reals, writes `tubey/rows.jsonl` and `tubey/dataset.csv`.
- `visualize_pair.py`: one gait, two engines, side by side gif.
- `emerge/`: separate project (EMERGE modular robots in CoppeliaSim vs mujoco). See `emerge/NOTES.md`. Needs
  `emerge/vendor/` set up on each machine (not in git).

## run
    uv run python generate_robust.py --bodies 10 --workers 10 --seed 0 --out tubey/rows.jsonl

`--seed` picks the body sample and the CMA-ES seeds, ids are offset by seed*1000. Another device uses another seed, then
concatenate the `rows.jsonl` files and rerun `flatten`.

## sim and reals
sim = pybullet twin (training world). reals, never trained on: `real_pb_mf` (pybullet + model-form error), `real_mj`
(native mujoco), `real_mj_mf` (mujoco + the same model-form error).

## old work
`archive/old_runs/` (INDEX.md there). Older README and sim notes: `archive/old_runs/docs/`.
