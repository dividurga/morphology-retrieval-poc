# sim diffs

mujoco = "real". pybullet = best-effort twin. both build from `spec.py`.
this file MUST change in the same commit as any sim change.
numbers come from `audit_sims.py` at `scale=0.7` unless stated.
status: `exact` | `shared` | `derived` | `approx` | `assumed` | `none` | `reality-only`.

## principle

pybullet MUST NOT be tuned, fitted or calibrated against mujoco output. mujoco stands in for a
real world that does not exist for the morphologies we generate. tuning to it would leak the data
the method only gets sparsely, and would shrink the measured gap.
every parameter MUST come from the spec or from the equations of motion.
each sim MUST be checked against physics (analytic values), not against the other sim.
`audit_sims.py`, `gap_budget.py`, `scale_check.py` measure the gap. their output MUST NOT feed
back into either sim.
both sims MUST derive from physical constructs in `spec.py` (motor, material compliance).

## representation

| effect | mujoco | pybullet | status |
|---|---|---|---|
| mass, com | from geoms | analytic mass, com = shape offset | exact (1e-16) |
| inertia | from geoms | analytic `localInertiaDiagonal` | exact (mass matrix 1.5e-17, armature off) |
| gravity, free fall | native | native, damping 0 | exact (4.9099 m both) |
| servo law (hip, knee) | `spec.servo_torque`, `motor` actuator, every substep | same function, every substep | shared |
| servo limits | goal clipped to range, no mujoco range | same | shared |
| torque cap | stall 6.0 n*m (mx-64, sourced) | same | shared |
| speed-torque | driving torque cap(1-\|w\|/wnl), wnl 63 rpm (sourced). braking stays at stall | same | shared |
| servo hold | 0.213 rad under 2 n*m, analytic 0.208 | 0.212 | exact to 2-6 % (body motion) |
| armature (hip, knee) | `armature` 0.01096 (bam mx-64, sourced) | a on shank iyy, -2a torso iyy (valid-limited) | derived. hip, knee diag exact. pitch diag 1.1x. cross terms wrong |
| joint damping | `damping` | native `jointDamping` | exact |
| ankle spring | `stiffness` | explicit torque | exact |
| ankle stop | `spec.ankle_stop_torque`, rubber bumper 70 n*m/rad | same function | shared. bumper geometry assumed |
| foot pad | slide joint, native spring and damper, pad body | prismatic joint, explicit spring and damper, pad link | derived. same spec, different integration |
| floor contact | stiffest soft contact (2 dt), only pads collide | rigid, only pads collide | approx |
| friction combine | max | product, robot set so product = max | exact (block test, see below) |
| control delay | command delayed 3 steps | none | reality-only |
| timestep | 0.005 control, 5 substeps (1 ms) | same | exact |

## compliance

foot pad, a spring-damper in series with the foot. k = e*a/t, damping ratio = loss factor / 2.
at scale 0.7: e = 3e4 pa, t = 7 mm, k = 2.0e4 n/m per foot, 1.34 mm static sink, c = 35 n*s/m.
e = 3e4 pa (very soft foam), pad thickness 10 mm x scale, loss factor 0.15 are ASSUMPTIONS.
no citable value was found. a sourced shore 50a rubber (gent, 2.46 mpa) gives 5 um of sink, which
is rigid and makes compliance a non-effect. the foam is soft enough to matter.
checked against physics, time-averaged pad deflection under standing: mujoco 1.32 mm, pybullet
1.28 mm, physics 1.34 mm.

engine limits found on the way:
- pybullet ignores `contactStiffness` on multibody links (a 10000x softer foot changed nothing).
  so the pad is an explicit dof.
- mujoco's soft contact stiffness is normalised by the contact's effective mass. for a foot at the
  end of free joints that is the light foot, not the supported 2.75 kg. the sink came out 20x too soft.
  so the pad is an explicit dof there too.
- pybullet's contact damping is erratic on a rebound test. not used.
- pad mass is 50 % of the foot mass in both sims. not physical. a lighter pad makes the explicit
  spring unstable at 1 ms.
- the old geometry let the shank rest on the floor (shank radius = foot underside). only pads
  collide now, in both sims.

## reality deltas (`spec.RealityDeltas`, mujoco only)

pybullet is always built from the nominal spec. the real robot differs by deltas.
parameter deltas (representable in both):

| delta | range | source |
|---|---|---|
| supply voltage -> stall torque | 11.1-14.8 v -> 5.5-7.3 n*m, no-load ~ v | stall points sourced (mx-64 e-manual). no-load scaling assumed |
| link mass | +-5 % | assumed |
| floor friction | x0.6-1.3 | assumed |
| hip/knee viscous damping | x0.5-2 | assumed |
| ankle spring and damping | x0.8-1.2 | assumed |
| armature | x0.8-1.2 | assumed |
| pad modulus | x0.5-2 | assumed |
| pad loss factor | x0.7-2 | assumed |
| delay | 2-4 steps | assumed (plan 6b) |

model-form effects (pybullet has no term for them):
- coulomb joint friction 0.0615 n*m on hip, knee. bam mx-64 `friction_base`. bam's units unverified.
- foam densification, hard stop at 50 % strain. assumed.
- command delay.

`RealityDeltas.sample(rng, level)`: level 0 is the nominal real robot exactly, 1 is the ranges above.

## gap now (scale 0.7)

physics checks, each against analytic values: free fall exact. block friction: physics 0.0510 m,
pybullet 0.0510, mujoco 0.0523. servo hold above. pad deflection above.
free-air joint error (mujoco vs pybullet): ankle 1.7 %, knee 10 %, hip 14 %.
ankle stop under a 4.8 n*m push: 0.671 vs 0.660 rad.

rollouts, nominal real vs nominal pybullet, 45 random pairs, open loop:
- audit (1 set): early pitch rms 0.119 rad, median time to divergence 0.17 s, fall corr 0.88, disp corr 0.48.
- scale_check (4 sets): loss 0.27, pitch rms 0.125, fall corr 0.83, disp corr 0.29.

the standing test (zero targets) is not a measurement: the posture is unstable and both sims topple
in different directions (-0.17 vs +0.25 rad). it is printed, not interpreted.

## gap budget (`gap_budget.py`, one delta at a time, 8 x 45 pairs, 3 s)

loss = pseudo-huber of the normalised joint-space error over the first 0.4 s. baseline 1.446.

| delta | d loss | | delta | d loss |
|---|---|---|---|---|
| joint damping x2 | +0.064 | | density -5 % | +0.009 |
| volts 14.8 (stall 7.3) | +0.047 | | pad modulus x2 | +0.009 |
| floor mu x0.6 | +0.018 | | ankle spring x0.8 | +0.007 |
| armature x0.8 | -0.014 | | joint damping x0.5 | +0.006 |
| volts 11.1 (stall 5.5) | +0.012 | | delay 4 / 2 | +0.006 / -0.006 |
| delay 0 | -0.011 | | density +5 % | -0.006 |
| armature x1.2 | +0.010 | | coulomb, pad loss, floor mu x1.3, ankle x1.2, pad modulus x0.5 | within +-0.004 |
| foam stop 50 % | +0.010 | | | |

- no single delta moves the loss by more than 4.5 % of the baseline.
- the baseline is dominated by structure: armature emulation and servo saturation amplifying
  small state differences. parameter deltas ride on top.
- joint damping and supply voltage matter most. the sim that assumes 6.0 n*m and nominal damping
  is the one that is wrong when those move.
- every delta below ~0.01 is probably noise. no standard error was computed.

## scale (`scale_check.py`, real servo with current-limited braking)

chosen: 0.7. kept after the re-check. the choice used the audit, so it is a design choice made
with the gap in view. search = mujoco max-distance gait, default budget, 3 seeds, 8 s.

| scale | mass kg | search distance m | peak servo n*m | agreement loss | pitch rms | fall corr | disp corr |
|---|---|---|---|---|---|---|---|
| 0.6 | 3.5 | 0.5 / 1.0 / 0.7 | 4.9-6.0 | 0.47 | 0.20 | 0.67 | 0.32 |
| 0.65 | 4.4 | 0.6 / 0.5 / 0.2 | 5.2-6.0 | 0.35 | 0.16 | 0.79 | 0.41 |
| 0.7 | 5.5 | 0.3 / 0.3 / 0.6 | 6.0 (cap) | 0.27 | 0.13 | 0.83 | 0.29 |
| 0.8 | 8.2 | 0.7 / 0.7 / 0.3, two fall | 6.0 (cap) | 0.16 | 0.07 | 0.88 | 0.45 |

agreement improves with scale. the cap binds at 0.7 on every seed, so nominal has no torque
margin. 0.65 is the fallback if margin matters more. gaits at the default budget are 0.2-1 m
everywhere, so the failure range is still not measurable.

## gaps

designed:
- armature. mujoco has it. pybullet cannot represent it per joint.
- delay. mujoco only.
- coulomb friction and foam densification. mujoco only.
- reality deltas.

residual:
- armature cross terms (hip-knee, joint-pitch) and the partial torso trim.
- servo saturation amplifies small differences. not a bug.
- contact formulation. mujoco is soft at the 2 dt floor, pybullet is rigid.
- pad integration. both explicit-spring-like, different integrators.

## search (`budget_scan.py`, scale 0.7, max distance in 8 s, 4 seeds)

budget does not help. best distance per seed, mujoco:

| popsize x maxiter | evals / seed | distances m | median |
|---|---|---|---|
| 24 x 100 | 2400 | 0.27 0.34 0.55 0.38 | 0.36 |
| 48 x 200 | 9600 | 0.65 0.16 0.25 0.24 | 0.25 |
| 96 x 300 | 28800 | 0.39 0.65 0.31 0.50 | 0.45 |

pybullet at the default budget: 0.39 0.10 0.24 0.45 (median 0.32). larger pybullet budgets not run.

the servo limits are not the ceiling either (mujoco, delay off, 24 x 100, 4 seeds, median):
ideal servo no armature -3.7 (falls), ideal servo with armature 0.47, cap only 0.29, cap +
speed-torque 0.09, cap doubled 0.45. the open-loop sinusoid gait on this robot gives ~0.3-0.6 m
in 8 s whatever the servo. rare seeds reach 1.2-1.7 m.

`optimize(..., workers=N)` evaluates a generation in a process pool. same result, 3x faster on pybullet.

## tried and rejected

- `loadMJCF`. drops hinge limits, cannot be fixed.
- native pybullet joint limits. hard and force capped.
- hip and knee joint-limit constraints in either sim. replaced by the servo goal clip.
- gear-constrained rotor body for armature. unstable.
- armature on all joints incl. ankle and root (old model). real servos only drive hip and knee.
- system identification of pybullet against mujoco. removed. see principle.
- empirical limit stiffness scale, and armature placement picked by rollout agreement.
- mujoco default contact softness as "real". not a material property.
- mujoco native soft contact as the pad. stiffness depends on effective mass.
- pybullet `contactStiffness` as the pad. ignored on links.
- braking torque above stall. a real driver limits current.
- a standing robot as a friction test. it topples.

## open

- pad modulus, thickness, loss factor, bumper geometry, delay, and every delta range other than the
  stall points are assumptions. sweep them before trusting a result.
- bam friction units, full voltage model, backlash, quantisation (0.088 deg), p-gain dynamics: not in.
- failure range. max-distance search tops out near 0.3-0.6 m (see search). it cannot separate
  morphologies. needs a different axis.
- re-run `gap_budget.py` with more case sets for standard errors.
- `check/` snapshots and gifs predate the rewrite and are stale.

## log

- spec.py, sim_common.py, sim_mujoco.py, sim_pybullet.py, audit_sims.py added. static rows exact.
- scale set from torque, moved to 0.7 (armature ~20x thigh inertia at 0.5).
- pybullet_fit.py added then removed. principle added: no tuning against mujoco.
- servo law moved into spec.py with a speed-torque curve. hip and knee limits are the goal clip. ankle stop is a rubber bumper.
- mujoco runs 5 substeps. mujoco joint ranges removed.
- foot pad added as an explicit spring-damper dof in both sims. only pads collide. pybullet joint indices are now read from joint names (it re-sorts links).
- braking torque capped at stall. audit friction and servo-hold checks added.
- RealityDeltas, gap_budget.py, scale_check.py added. scale re-checked, 0.7 kept.
- optimize_controller.py, visualize.py, pybullet_visualize.py retargeted to the new sims. optimizer takes `spec` and `backend`, theta bounds follow `spec.omega_scale`. DR training draws `RealityDeltas` with the model-form effects zeroed. environment.py, pybullet_env.py, morphology.py, perturbations.py deleted.
- optimize() takes `workers`. budget_scan.py added. budget and servo limits ruled out as the cause of small distances.
