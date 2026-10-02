# morphology-retrieval-poc

Self-contained PoC. It MUST run standalone, without `roboCoDesign` or
`transformer-transformer` on the path.

## What this is

Level 1 (`roboCoDesign`) found that morphology affects robustness to model
mismatch. This PoC asks a different question: given a target walking
distance, retrieve the morphology-controller pair that will actually hit it
in the real world, not just in simulation.

The design was worked out over a long conversation, not written from a
spec. This file is the durable record of what was decided and why.
Implementation code does not exist yet -- this migration only copies the
reusable simulation foundation.

## Locked decisions

**Target-distance conditioning.** A controller's fitness MUST be
`-|achieved_distance - target_distance|`, not maximize-distance. The
controller stays open-loop (no state feedback) -- it is retargeted, not
made goal-aware. Rationale: the controller has no way to sense arrival at a
point, so "reach point X" can only mean "cruise at the speed that lands
near X in a fixed episode," not literal navigation.

**DR training, as an empirical comparison, not an assumption.** Every
`(morphology, target)` pair trains two arms: nominal-only and DR-randomized
(`optimize_controller.py`'s existing `m_perturbations`/`fitness_dr` path,
built in the Level 1 DR-baseline pilot, unused until now). Every dataset
row is tagged `training_regime`. `train_perturbation_seed` MUST stay
disjoint from any evaluation-time seed (`PLAN_DR_BASELINE.md`'s existing
convention), or the DR arm's measured performance becomes circular. The
question this answers: does DR training (exposure only to modeled
parametric variation) reduce the real, structural PyBullet-vs-MuJoCo gap,
even though it never saw anything MuJoCo-like during training?

**Morphology scope.** `PARAM_BOUNDS` MUST widen beyond the current ±28%.
New continuous params SHOULD be added: ankle stiffness/damping, actuator
kp, density, hip-y offset -- these are already hardcoded constants in
`morphology.py`, just not yet exposed as design variables. Left-right
symmetry MUST be broken -- `controller.py`'s mirrored gait is a real
architecture change, roughly doubling `theta`. Topology changes (more
legs, a different body plan) are explicitly deferred, not in scope: they
would break shared assumptions in `morphology.py`, `controller.py`, and
`perturbations.py` all at once, and would confound the calibration
mechanism this PoC is meant to validate.

**Two simulators. MuJoCo is reality, full stop -- no sim3.**
- Sim 1 = PyBullet (`pybullet_env.py`). Trains both arms above. Cheap:
  this is the big, free dataset -- sample as many
  `(morphology, target, training_regime)` rows as wanted, no cost
  constraint. `optimize_controller.optimize(..., backend=pybullet_env)`
  selects it. The DR arm is MuJoCo-only until `perturbations.py` is ported.
- Sim 2 = MuJoCo (`environment.py`). The real world, for the purposes of
  this project (generally regarded as more accurate for contact-rich
  locomotion, hence the choice). Deliberately tested for only a small,
  per-query shortlist, not the whole dataset -- not because MuJoCo itself
  is expensive (it's still just software), but because this PoC exists to rehearse a methodology that
  must work under genuine scarcity later (physical hardware). The
  boundedness is a deliberate methodological choice, kept even though
  nothing stops testing more right now.

There is no `dr_spread`/cheap-signal feature. An earlier draft of this
design tried to predict real (MuJoCo) disparity from a free, same-engine
DR-variance signal computed by perturbing the training sim against itself. Dropped:
that signal can only ever vary along the parameters `perturbations.py`
already models (friction, mass, motor, damping) -- it is structurally
blind to the exact thing the second simulator exists to catch (structural/model-form
mismatch), so it cannot predict real disparity by construction, not just
in practice.

**The per-query pipeline (this is the "N good morphologies" block from the
original whiteboard sketch, precisely):**
1. **Shortlist.** For a new target distance, rank the full PyBullet-only
   dataset by proximity of `achieved_distance` to the target -- a naive
   criterion, the only one available before any real data exists. Return
   the top N (small, bounded).
2. **Test.** Roll out each shortlisted row in MuJoCo, multiple trials
   each (captures both bias and spread, not a single point), producing
   `exact_disparity` for just these N.
3. **Calibrate.** Fit an IDW surrogate (GP is the documented upgrade path)
   on `[morphology_params + behavior_features] -> exact_disparity` using
   those N exact points, then apply it to **every** row in the full
   dataset -- not just the N tested.
4. **Pull.** Retrieve the final answer from the now fully-corrected
   dataset, ranked by predicted retention. The final pick MAY differ from
   anything in step 1's shortlist -- that's the point of calibrating the
   whole pool, not just re-ranking the tested subset.

Reachability alone (which bodies can technically hit the target) is close
to vacuous once controllers are target-conditioned -- most bodies can hit
most targets within range just by changing gait speed. Step 1's shortlist
still uses it as a first pass (nothing better exists pre-calibration), but
step 4's final answer MUST be ranked by calibrated retention, not raw
reachability.

## Relationship to Transformer Transformer

`transformer-transformer` (T2, local clone at
`../transformer-transformer`) already does joint diffusion over embodiment
and dynamics tokens, conditioned on a target motion, with a pluggable
`RewardFn` mechanism (Dynamics Self-Guidance) for steering generation at
inference time.

This PoC is a small-scale rehearsal of the same mechanism, not a
different idea:

| PoC | T2 |
|---|---|
| `morphology_params` + `behavior_features` | embodiment tokens (`link`/`fixed_joint`/`dynamic_joint`/`motor`) + dynamics tokens (state/action) -- physicality is already canonical here, not bolted on |
| PyBullet dataset (cheap, unlimited) | T2's own sim-generated training data (cheap, unlimited) |
| `exact_disparity` (scarce, per-query shortlist tested in MuJoCo) | real/physical-hardware disparity, same scarcity |
| IDW/GP surrogate over `[morphology_params + behavior_features]` | IDW surrogate over **embodiment-token embeddings together with dynamics-token embeddings** -- MUST include embodiment embeddings explicitly, not rely on attention having propagated enough physical-design information into the dynamics tokens alone |
| re-rank the whole dataset, pull the final answer | fold the surrogate into a `RewardFn`, steer Dynamics Self-Guidance |

Neither T2 (one-shot real-world validation, no recalibration) nor classic
calibration methods (SimOpt, BayRn, Koos et al.'s transferability
approach -- fixed morphology, controller-only) close this exact loop. The
contribution is the combination, not any one piece alone -- state it that
way, not as "not yet seen in the literature."

## Files here

Copied unmodified from `roboCoDesign`, to be extended in place:

- `morphology.py` -- MJCF generation. Needs: widened bounds, new continuous
  params, asymmetric per-leg params.
- `controller.py` -- sinusoidal gait. Needs: drop the mirrored left-right
  assumption.
- `environment.py` -- rollout + fall detection. Unchanged.
- `perturbations.py` -- DR sampling (`sample_combined`/`apply_perturbation`).
  Unchanged; consumed by `optimize_controller.py`'s existing
  `m_perturbations`/`fitness_dr` path for the DR-trained arm.
- `optimize_controller.py` -- CMA-ES. Needs: target-distance fitness. The
  DR-training path (`m_perturbations`, `fitness_dr`) already exists, unused
  until now -- no new code there, just wiring it into the target-conditioned
  fitness.
- `visualize.py` -- copy this too, alongside the above (not yet copied as
  of this writing). Rendering utility, not Level-1 methodology -- needed to
  sanity-check sampled morphologies once bounds widen.

Not copied, and MUST NOT be pulled in later without rewriting: `evaluate.py`,
`sweep*.py`, `analyze.py`, `validate_seeds.py`, `confirm_pairs.py`,
`check_extremes_reversal.py`, `mean_of_k_check.py`, `optimize_pso.py`,
`compare_fixes.py`, `cluster_baseline.py`. These encode Level 1's
best-of-K/mean-of-K methodology and the single-scalar retention metric,
which this PoC's per-query shortlist/test/calibrate/pull pipeline replaces,
not extends.
