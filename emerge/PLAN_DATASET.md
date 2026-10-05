# EMERGE pilot dataset: target-conditioned rows, sim vs real A vs real B

## Context
Project objective (README, locked): given a target distance T, retrieve the morphology and controller that actually hits T in the real world, not only in the sim. The pipeline is shortlist by sim distance near T, test the shortlist in the real, calibrate a surrogate on disparity, pull by predicted real hit. So **every row carries a task goal: a target distance T.** This replaces my first draft, which only searched for good gaits.

This plan covers `emerge/` only (another agent owns the rest of the repo; no edits to `pyproject.toml`, `uv.lock` or top-level `.py`). Chains first; trees and edhmor's full ES are a follow-up.

Decisions (user agreed to the recommended options):
- **Goal label:** post-hoc relabeling. A non-breaking, repeatable sim gait with mean distance d is a certified optimum for T = d (fitness `-|d - T|` = 0). Bins of ~2 cm (the admitted sim std is ~1 cm). Empty bins are recorded as "not reached", not dropped. Targeted CMA restarts at a ladder of Ts fill the range.
- **Success** (stored per row, thresholds changeable later): no break and |real distance - T| <= max(0.02 m, 20% of T). Raw real distances and break flags are always kept.
- **Morphology axis:** n_act in {2,3,4,5} x base_face, *if* the visualiser shows sim, real A and real B build `base_face` variants consistently. Otherwise n_act only, and say so (retrieval among 4 morphologies is nearly trivial).
- Pilot: ~100 rows, 3 hidden worlds per real, with the traces and scalars already agreed (distances, broken, fitness, wall, flags, base xy at ~1 Hz).

## What a row is
`row_id, morphology (n_act, base_face), phases_deg, target T (= sim mean), bin, sim: K raw runs + mean/std/break share/ok, real A: 3 worlds x (distance, broken, fitness, wall, timeout), real B: same, disparity per world (real - T), success flags, search metadata (cma seed, restart target)`. Traces in `emerge/data/traces/<row_id>.npz`.

## Reuse (do not rewrite)
`morph.chain`/`poses`; `sim_fast.rollout` and `repeatability`; `pool.map`; `real_coppelia.rollout(hook=)`; `real_mujoco.rollout(hook=)`; the recorders in `visualize.py`; `cma` (already a dependency). `generate_dataset.py` (other agent's area, read only) is the model for the restart ladder and bins.

## Steps
0. **Preflight (measurements only, scratchpad scripts):** time one 38 s rollout per engine; throughput vs worker count for each engine alone (see CPU plan); check `base_face` 0..7 with `emerge.visualize`; look at where the sim's distances range (sets the T ladder and bin edges).
1. **`emerge/search.py`:** per morphology, CMA-ES over phases (wrapped 0..360). Restarts with a T ladder plus one "maximise distance" restart. Fitness per candidate = `-|d - T|` x break penalty 0.8^breaks (paper form), mean of K=2 sim runs during search. Log every evaluation to an archive. No change to the sim's connector rule.
2. **Admission and bin representative:** relabel archive entries with no break by their own distance, bin them, verify the best few per bin with `sim_fast.repeatability(k=8)`. Admit if no break in all runs and std < 1 cm. Representative per bin by an explicit rule (lowest std, then lowest break share, then closest to the bin centre). Report admission rate per n_act and the bin coverage per morphology. If admission is poor for 4 to 5 module chains, report it; do not loosen the rule.
3. **`emerge/dataset.py` (resumable orchestrator):** for each admitted row run real A and real B in 3 fixed world seeds (the same seeds for all rows), write `emerge/data/rows.jsonl` (append-only, skip done ids, timeouts as nan with a flag), traces, and `dataset.csv`. Nothing from a real is used to choose or filter rows. Rows where the reals break are kept: that is the disparity signal. Add `emerge/data/` to `.gitignore`.
4. **Docs:** `emerge/NOTES.md` gets the row definition, goal and success rules, search settings, timings and admission rates.

## CPU contention plan
Facts: M3 Pro, 12 cores, 36 GB, no GPU use. CoppeliaSim instances are x86 under Rosetta, one per worker, plus a Python client each. Real A's Python loop makes thousands of zmq round trips per rollout, so a worker keeps two processes busy (client and instance). Real B is pure CPU, one process per rollout. Earlier `bench_pool.log` already shows sublinear scaling (1 worker 0.37/s, 4 workers 1.47/s, 8 workers 2.33/s), and one idle instance I started sits at ~58% CPU. The other agent's jobs (pybullet/mujoco CMA-ES with process pools, `run_dr.sh`) share the same 12 cores.

1. **Run stages, not mixes.** The pipeline is a sequence of phases, never concurrent: (a) search + verification on the CoppeliaSim pool, (b) real A on the same pool, (c) real B on a separate MuJoCo-only pool. Real B never runs while instances are busy.
2. **Size each pool from measurements** (step 0): pick the worker count where throughput stops growing, subtract what the other agent needs, and set `N_COPPELIA` and `N_MUJOCO` in one config constant. Default budget: at most 6 cores for us (instances + clients counted as 2 per Coppelia worker, so about 3 Coppelia workers, or 6 MuJoCo workers), until we agree otherwise.
3. **Limit hidden threads:** set `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1` in worker initialisers (numpy/scipy/cma/mujoco otherwise oversubscribe).
4. **Be a polite neighbour:** run the job with `nice -n 10`, and a simple lock/budget file `emerge/data/.cpu_budget` (the cores we are allowed, read at the start of each phase) so the budget can be changed without code edits. A suggestion to the other agent to read the same file, not something we enforce.
5. **No idle processes:** stop the single instance on port 23000 before the pilot (`start_pool.sh 8` ports are separate), and stop the pool when the run ends so instances do not hang on a dead client. Keep the existing deadline-with-nan in `sim_fast.rollout`; add the same wall-clock guard to the real A path.
6. **Resumability** makes contention cheap: if we are told to back off, kill the run and restart with a smaller budget; finished rows are skipped.

## Verification
- Smoke: 3 rows end to end (tiny search budget), jsonl and traces written, rerun resumes without redoing rows.
- Contention check: throughput of the sim pool alone, real B pool alone, and both together (expect combined < sum) with the chosen worker counts, while the other agent's jobs run. Record in NOTES.
- Pilot (~100 rows): per n_act admission rate and bin coverage, sim repeatability, success rate per real, disparity distribution, break share per real (real B vs the paper's ~20% as a sanity check only).
- Eyeball a few rows with `uv run python -m emerge.visualize --phases ...`.
- Prerequisite at run time: `sh emerge/start_pool.sh <N>` with N from the contention plan.

## Out of scope (follow-up)
Trees and link modules, edhmor's ES operators (add node, prune, symmetry, local search), per-run sim variability beyond the K-repeat summary, surrogate and retrieval experiments.
