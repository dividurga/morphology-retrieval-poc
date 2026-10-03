"""per-effect gap budget. read-only: it measures, it MUST NOT tune either sim.

each row switches ONE difference on in the "real" mujoco (a RealityDeltas), pybullet stays the
nominal twin. same 45 (morphology, gait) pairs for every row (common random numbers), 3 s open
loop. metrics are the audit's rollout metrics.

    uv run python gap_budget.py [--workers 12] [--seeds 8]
"""

import argparse
import dataclasses
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pybullet as p

import audit_sims as A
import controller
import sim_pybullet as sb
from spec import SPEC, RealityDeltas

HORIZON, N_MORPH, N_THETA = 3.0, 3, 15
D0 = RealityDeltas()  # the nominal real robot: spec as is, delay 3, no model-form effects

VARIANTS = {
    "nominal real": D0,
    "volts 11.1 (stall 5.5)": dataclasses.replace(D0, volts=11.1),
    "volts 14.8 (stall 7.3)": dataclasses.replace(D0, volts=14.8),
    "density -5%": dataclasses.replace(D0, density_scale=0.95),
    "density +5%": dataclasses.replace(D0, density_scale=1.05),
    "floor mu x0.6": dataclasses.replace(D0, floor_mu_scale=0.6),
    "floor mu x1.3": dataclasses.replace(D0, floor_mu_scale=1.3),
    "joint damping x0.5": dataclasses.replace(D0, damping_scale=0.5),
    "joint damping x2": dataclasses.replace(D0, damping_scale=2.0),
    "ankle spring x0.8": dataclasses.replace(D0, ankle_scale=0.8),
    "ankle spring x1.2": dataclasses.replace(D0, ankle_scale=1.2),
    "armature x0.8": dataclasses.replace(D0, armature_scale=0.8),
    "armature x1.2": dataclasses.replace(D0, armature_scale=1.2),
    "pad modulus x0.5": dataclasses.replace(D0, pad_modulus_scale=0.5),
    "pad modulus x2": dataclasses.replace(D0, pad_modulus_scale=2.0),
    "pad loss x0.7": dataclasses.replace(D0, pad_loss_scale=0.7),
    "pad loss x2": dataclasses.replace(D0, pad_loss_scale=2.0),
    "delay 2": dataclasses.replace(D0, delay_steps=2),
    "delay 4": dataclasses.replace(D0, delay_steps=4),
    "delay 0 (no delay)": dataclasses.replace(D0, delay_steps=0),
    "coulomb friction 0.0615": dataclasses.replace(D0, coulomb=0.0615),
    "foam stop at 50% strain": dataclasses.replace(D0, pad_stop_strain=0.5),
}


def cases(seed: int, spec=SPEC):
    rng = np.random.default_rng(seed)
    A.CFG["spec"] = spec
    lo, hi = A.theta_bounds()
    out = []
    for m in range(N_MORPH):
        params = spec.nominal_params() if m == 0 else A.random_params(rng)
        out.append((params, [lo + rng.random(len(lo)) * (hi - lo) for _ in range(N_THETA)]))
    return out


def measure(deltas: RealityDeltas, case_list: list, client: int, spec=SPEC) -> dict:
    diverge, rms, fm, fp, dm, dp, losses = [], [], [], [], [], [], []
    scale = np.array([0.05, 0.02, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])  # per-dof error scale
    A.CFG["spec"] = spec

    def fell(tr, standing):
        f = np.where((np.abs(tr[:, 2]) > 1.0) | (tr[:, 1] + standing < 0.5 * standing))[0]
        return f[0] * spec.timestep if len(f) else HORIZON

    for params, thetas in case_list:
        rp, rs = deltas.apply(params, spec)
        mj = A.sm.make_model(rp, rs)
        pb = sb.make_model(params, spec, sb.PB_CFG, client=client)
        for theta in thetas:
            tf = lambda t, th=theta: controller.joint_targets(th, t)
            a, b = A.mj_run(mj, tf, HORIZON), A.pb_run(pb, tf, HORIZON)
            err = np.abs(a[:, :3] - b[:, :3])
            bad = np.where((err[:, 2] > 0.1) | (err[:, 1] > 0.02 * spec.scale))[0]
            diverge.append(bad[0] * spec.timestep if len(bad) else HORIZON)
            rms.append(np.sqrt(((a[:51, :3] - b[:51, :3]) ** 2).mean(axis=0)))
            e = (a[:81] - b[:81]) / scale  # first 0.4 s, all 9 dofs, pseudo-huber
            losses.append(float(np.mean(np.sqrt(1.0 + e ** 2) - 1.0)))
            fm.append(fell(a, pb.standing_height))
            fp.append(fell(b, pb.standing_height))
            dm.append(a[-1, 0])
            dp.append(b[-1, 0])
    corr = lambda x, y: float(np.corrcoef(x, y)[0, 1])
    return {"loss": float(np.mean(losses)), "pitch_rms": float(np.mean([r[2] for r in rms])), "t_div": float(np.median(diverge)),
            "fall_corr": corr(fm, fp), "disp_corr": corr(dm, dp),
            "mean_abs_disp_err": float(np.mean(np.abs(np.array(dm) - np.array(dp))))}


_STATE = {}


def _init():
    _STATE["client"] = p.connect(p.DIRECT)


def _job(args):
    name, deltas, seed = args
    return name, seed, measure(deltas, cases(seed), _STATE["client"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=1, help="independent case sets to average over")
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(n, d, s) for n, d in VARIANTS.items() for s in range(args.seeds)]
    res = {}
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        for name, seed, m in pool.map(_job, jobs):
            res.setdefault(name, []).append(m)
    base = {k: np.mean([m[k] for m in res["nominal real"]]) for k in res["nominal real"][0]}
    print(f"gap budget, {args.seeds} case set(s) x {N_MORPH * N_THETA} pairs, {HORIZON}s, {time.time() - t0:.0f}s")
    print(f"{'effect':28s} {'loss 0.4s':>9s} {'d loss':>8s} {'pitch rms':>9s} {'t_div':>6s} {'fall corr':>9s} {'disp corr':>9s}")
    for name, ms in res.items():
        v = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        print(f"{name:28s} {v['loss']:9.4f} {v['loss'] - base['loss']:+8.4f} {v['pitch_rms']:9.3f} {v['t_div']:6.2f} "
              f"{v['fall_corr']:9.2f} {v['disp_corr']:9.2f}")


if __name__ == "__main__":
    main()
