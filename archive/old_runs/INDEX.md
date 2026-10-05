# old runs, archived 2026-10-05 (moved, not deleted)

Everything here comes from before the restart. Paths inside the scripts are not updated, so they will not run from here
without copying them back to the repo root.

## data (`data/`)
- `data_v1/`: 200 morphologies x 61 distance bins (5 cm, 0 to 3 m). Gaits found by CMA-ES on the nominal pybullet twin
  (nominal-trained only, no DR), then relabelled by the distance they achieved. 2,838 filled rows. `archive/` holds every
  evaluated gait per morphology (npz). `dataset_nofall.csv` is filtered on mujoco output, so do not use it for claims.
- `data_pilot/`: early 12-morphology pilot.

## results (`results_all/`)
- `dr/run1.jsonl`: DR experiment, 113 of 200 bodies, targets 0.6/1.0/1.5, regimes nominal / dr_mult / dr_lag, each gait
  evaluated in 7 worlds. `run2.log` ended in BrokenProcessPool. `REPORT.md` and `NOTES.md` summarise it.
- `pool_*.csv`, `*_analysis.log`, `*_loop.*`: disparity pools and surrogate diagnostics (twin pybullet or mujoco, rungs
  R0 to R2). Finding: held-out disparity prediction was near the mean predictor.

## scripts (`scripts/`)
- `dr_experiment.py`, `dr_report.py`: the DR run above and its report.
- `experiment_disparity.py`, `build_pools.py`, `analyze_disparity.py`: shortlist / surrogate loop and diagnostics.
- `experiment_dr_fail.py`, `experiment_fail_predict.py`, `fail_features.py`, `PREREG_FAIL.md`: predicting mujoco falls.
- `human_ladder.py`: the pybullet setup ladder (which loader fix helps) against mujoco.
- `check_dataset.py`, `make_nofall.py`, `cross_engine.py`, `target_pipeline.py`: small checks and one-offs.
- `walker/`: walker2d attempt, abandoned (no open-loop gait found).
- `HANDOFF_PYBULLET_DEBUG.md`: an older debugging handoff. It describes a from-scratch createMultiBody build that no
  longer exists.
