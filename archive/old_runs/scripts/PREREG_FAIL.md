# pre-registration: predicting mujoco falls for dr-trained gaits

written 2026-10-04, before the richer features or the extra bodies were looked at. numbers from the first
113 bodies (`experiment_fail_predict.py`) were seen: grouped-cv AUC 0.49 to 0.63 for fall, n=219 rows, 55 bodies.

## data
- rows: `results/dr/run1.jsonl`, regimes dr_mult and dr_lag only, twin (pb_nominal) not fallen. targets 0.6, 1.0, 1.5.
- real = mujoco nominal (`mj_nominal`), 3 s episode. nothing is fitted to mujoco except the final calibration step.
- split by body (morph_id): odd ids = design half (all model choices, feature selection, hyperparameters),
  even ids = held-out half, touched once at the end.

## labels
- primary: fell in mujoco (binary).
- secondary (continuous): mujoco survival margin = min over the episode of min(1 - |pitch|/1.0, (z/standing - 0.5)/0.5),
  clipped at 0 once fallen, and mujoco fall time (3 s if no fall).

## predictors compared
0. base rate
1. morphology + theta
2. sim behaviour: twin rollout traces (nominal, R0, R1, R2): fall flags, fall time, margin, pitch std/max, torso height min,
   speed stats, distance
3. both

## claims allowed
- a predictor "has signal" only if the 95% body-bootstrap CI of its held-out AUC excludes 0.5.
- an N-real-test calibration "helps" only if its held-out AUC beats the zero-real-test version of the same model, CI
  over repeated draws of the N tests, N in 5, 10, 20, 40. sampling rules: random, farthest-point over morphology,
  likely-to-fail (top of the sim proxy), 50/50 mix of likely-fail and likely-safe.
- recommendation rule (fixed now): per target T, among held-out rows with twin distance within 0.1 of T, keep predicted
  P(fall) <= 0.2 and take the 5 with the largest twin distance. reported: share that fall in mujoco, mean real distance,
  against (a) the baseline "largest twin distance, no safety", (b) the oracle (best non-falling row).
  success = no fall and mujoco distance within 0.1 m of T.

## not allowed
- picking a threshold, feature set or model by looking at held-out-half results.
- reporting DR vs nominal fall shares after conditioning on twin-ok rows as a robustness effect (selection).
