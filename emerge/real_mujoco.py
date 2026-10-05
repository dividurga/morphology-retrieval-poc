"""the real: an EMERGE robot as a mujoco model. stands in for hardware that does not exist.
built from paper/datasheet numbers plus effects the sim builder (edhmor) did not model. nothing is tuned to the sim's output.

what is shared with the sim, because it is known from CAD: box geometry of the modules (emerge/module_geometry.json,
dumped from the edhmor models), joint axis, the sinusoid definition, the fitness definition.
what is real-only (the sim leaves these out), all from the EMERGE paper (PMC8287515) or the AX-18A datasheet:
  - masses from the paper's table 1 (actuated 196 g, base 502 g), plus a per-module spread (printing, cabling)
  - servo: setpoint update at 10 Hz (sim 20 Hz), speed cap 25 rpm, torque cap 1.8 Nm, 0.29 deg resolution, a pd loop
  - friction: per-module draw against the substrate (the paper names friction as a main gap)
  - connectors: compliant welds that break above 1.2 Nm x a vibration factor
assumptions not in the paper, flagged: servo pd gains (KP, KD), the connector compliance (CONNECTOR_SOLREF), spreads of the draws.

    rollout(genome, world_seed, duration=38.0, t_start=6.0) -> dict(distance, broken, fitness, wall)
"""

import json
import os
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from emerge import morph

GEOM = json.load(open(os.path.join(os.path.dirname(__file__), "module_geometry.json")))
DT = 0.002
MASS_ACT, MASS_BASE = 0.196, 0.502  # kg, paper table 1
SPEED_CAP = 25 * 2 * np.pi / 60  # rad/s, paper: motors capped at 25 rpm
TORQUE_CAP = 1.8  # Nm, AX-18A at 12 V
RESOLUTION = np.radians(0.29)  # AX-18A
COMMAND_PERIOD = 0.1  # s, paper: 10 Hz update on the real robot
KP, KD = 8.0, 0.1  # assumption: stall torque at about 13 deg of error
CONNECTOR_TORQUE = 1.2  # Nm static, paper table 1
CONNECTOR_SOLREF = "0.02 1"  # assumption: compliant magnetic connector
BREAK_HOLD = 0.035  # s over threshold before it lets go (the sim uses 7 consecutive checks of 5 ms)
PENALTY = 0.8
BREAK_FACTOR_LOW = 0.9  # connectors let go at CONNECTOR_TORQUE x U(BREAK_FACTOR_LOW, 1). from the paper's 6/30 break rate, see emerge/experiments/calibrate_break.py


def world(seed: int, n_modules: int) -> dict:
    """a hidden world: draws of what the sim builder could not know."""
    rng = np.random.default_rng(seed)
    return dict(
        mass=rng.uniform(0.9, 1.1, n_modules),  # per-module mass spread
        friction=rng.uniform(0.4, 0.9, n_modules),  # module vs substrate
        break_factor=rng.uniform(BREAK_FACTOR_LOW, 1.0, n_modules),  # connector, lowered by vibration
    )


def _aabb(entry):
    """half extents of a module shape in the module frame, and its centre."""
    q = entry["pose_in_root"][3:]
    rot = R.from_quat(q)
    half = np.abs(rot.as_matrix()) @ (np.array(entry["size"]) / 2)
    return half, np.array(entry["pose_in_root"][:3])


def build_xml(genome: morph.Genome, w: dict, height: float = 0.055 / 2):
    placed = morph.poses(genome, height)
    parts = []  # per module: dict(A=body name, B=body name or None)
    bodies = []
    welds = []
    act_geom = {e["alias"]: e for e in GEOM["emergeModuleAX18"] if e["type"] == 0}
    ca, cb = act_geom["Cuboid5"], act_geom["Cuboid0"]
    ha, ca_pos = _aabb(ca)
    hb, cb_pos = _aabb(cb)
    joint = [e for e in GEOM["emergeModuleAX18"] if e["type"] == 1][0]
    j_pos = np.array(joint["pose_in_root"][:3])
    base_e = GEOM["flatBase"][0]
    hbase, _ = _aabb(base_e)
    for i, (m, (pos, rot)) in enumerate(zip(genome.modules, placed)):
        q = rot.as_quat()  # xyzw
        quat = f"{q[3]} {q[0]} {q[1]} {q[2]}"
        f = w["friction"][i]
        if m.type == "base":
            mass = MASS_BASE * w["mass"][i]
            bodies.append(
                f'<body name="m{i}A" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat}"><freejoint/>'
                f'<geom type="box" size="{hbase[0]} {hbase[1]} {hbase[2]}" mass="{mass}" friction="{f} 0.005 0.0001"/></body>')
            parts.append(("m%dA" % i, "m%dA" % i))
        else:
            mass = MASS_ACT * w["mass"][i] / 2
            pa = pos + rot.apply(ca_pos)
            pb_rel = cb_pos - ca_pos  # in the module frame, part B relative to part A (same orientation)
            bodies.append(
                f'<body name="m{i}A" pos="{pa[0]} {pa[1]} {pa[2]}" quat="{quat}"><freejoint/>'
                f'<geom type="box" size="{ha[0]} {ha[1]} {ha[2]}" mass="{mass}" friction="{f} 0.005 0.0001"/>'
                f'<body name="m{i}B" pos="{pb_rel[0]} {pb_rel[1]} {pb_rel[2]}">'
                f'<joint name="j{i}" type="hinge" axis="0 -1 0" pos="{(j_pos - cb_pos)[0]} {(j_pos - cb_pos)[1]} {(j_pos - cb_pos)[2]}" '
                f'range="-3.2 3.2" damping="0.02"/>'
                f'<geom type="box" size="{hb[0]} {hb[1]} {hb[2]}" mass="{mass}" friction="{f} 0.005 0.0001"/></body></body>')
            parts.append(("m%dA" % i, "m%dB" % i))
    for i, m in enumerate(genome.modules):
        if m.parent < 0:
            continue
        p = genome.modules[m.parent]
        pbody = parts[m.parent][0 if m.parent_face in morph.BASE_PART_FACES[p.type] else 1]
        ofp, _ = (np.array(v, float) for v in morph.FACES[p.type][m.parent_face])
        ppos, prot = placed[m.parent]
        cpos, crot = placed[i]
        pivot = ppos + prot.apply(ofp)  # connection point in the world
        # relpose of child A in the parent shape body frame. both bodies carry the module orientation
        p_origin = ppos + prot.apply(ca_pos if (p.type == "act" and pbody.endswith("A")) else
                                      (cb_pos if p.type == "act" else np.zeros(3)))
        c_origin = cpos + crot.apply(ca_pos)
        rel_pos = prot.inv().apply(pivot - p_origin)  # mujoco: where body2's anchor sits in body1's frame
        rel_rot = (prot.inv() * crot).as_quat()
        anchor = crot.inv().apply(pivot - c_origin)
        welds.append(f'<weld name="w{i}" body1="{pbody}" body2="m{i}A" anchor="{anchor[0]} {anchor[1]} {anchor[2]}" '
                     f'relpose="{rel_pos[0]} {rel_pos[1]} {rel_pos[2]} {rel_rot[3]} {rel_rot[0]} {rel_rot[1]} {rel_rot[2]}" '
                     f'solref="{CONNECTOR_SOLREF}"/>')
    excl = "".join(f'<exclude body1="{parts[genome.modules[i].parent][k]}" body2="m{i}A"/>' for i, m in enumerate(genome.modules) if m.parent >= 0 for k in (0, 1) if parts[m.parent][k] != parts[m.parent][0] or k == 0)
    acts = "".join(f'<motor name="a{i}" joint="j{i}" gear="1" ctrllimited="true" ctrlrange="-{TORQUE_CAP} {TORQUE_CAP}"/>'
                   for i, m in enumerate(genome.modules) if m.type == "act")
    return f"""<mujoco><compiler angle="radian"/><option timestep="{DT}" gravity="0 0 -9.81"/>
<worldbody><geom name="floor" type="plane" size="20 20 0.1" friction="0.01 0.005 0.0001"/>
{''.join(bodies)}</worldbody>
<contact>{excl}</contact>
<equality>{''.join(welds)}</equality>
<actuator>{acts}</actuator></mujoco>"""


def weld_torques(m, d, weld_idx) -> dict:
    """magnitude of the rotational constraint force of each weld (Nm)."""
    out = {i: 0.0 for i in weld_idx}
    if d.nefc:
        eq_rows = np.where(d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY)[0]
        ids = d.efc_id[eq_rows]
        for i, e in weld_idx.items():
            rows = eq_rows[ids == e]
            if len(rows) >= 6:
                out[i] = 0.5 * float(np.linalg.norm(d.efc_force[rows[3:6]]))  # mujoco's rotational weld rows carry 2x the torque (checked with a torque ramp)
    return out


def rollout(genome: morph.Genome, world_seed: int = 0, duration: float = 38.0, t_start: float = 6.0, trace: bool = False, hook=None):
    """hook(m, d, k) runs after every physics step (used by emerge/visualize.py)."""
    t0 = time.time()
    w = world(world_seed, len(genome.modules))
    m = mujoco.MjModel.from_xml_string(build_xml(genome, w))
    d = mujoco.MjData(m)
    act_ids = [i for i, mod in enumerate(genome.modules) if mod.type == "act"]
    qadr = [m.joint(f"j{i}").qposadr[0] for i in act_ids]
    vadr = [m.joint(f"j{i}").dofadr[0] for i in act_ids]
    weld_idx = {i: m.equality(f"w{i}").id for i, mod in enumerate(genome.modules) if mod.parent >= 0}
    over = {i: 0.0 for i in weld_idx}
    broken = set()
    ph = np.radians(genome.phases())
    q_ref = np.zeros(len(act_ids))
    q_cmd = np.zeros(len(act_ids))
    base = m.body("m0A").id
    n = int(round(duration / DT))
    next_cmd, p0 = 0.0, None
    # equality rows: the efc rows of weld k, found at runtime
    for k in range(n):
        t = d.time
        if t >= next_cmd - 1e-9:  # servo setpoint update at 10 Hz, quantised
            for j, i in enumerate(act_ids):
                target = morph.MAX_AMPLITUDE * np.sin(morph.MAX_ANGULAR_FREQUENCY * t + ph[i])
                q_cmd[j] = np.round(target / RESOLUTION) * RESOLUTION
            next_cmd += COMMAND_PERIOD
        q_ref += np.clip(q_cmd - q_ref, -SPEED_CAP * DT, SPEED_CAP * DT)
        q = d.qpos[qadr]
        qd = d.qvel[vadr]
        d.ctrl[:] = np.clip(KP * (q_ref - q) - KD * qd, -TORQUE_CAP, TORQUE_CAP)
        mujoco.mj_step(m, d)
        if hook is not None:
            hook(m, d, k)
        tq_all = weld_torques(m, d, weld_idx)
        for i, e in weld_idx.items():
            if i in broken:
                continue
            if tq_all[i] > CONNECTOR_TORQUE * w["break_factor"][i]:
                over[i] += DT
                if over[i] >= BREAK_HOLD:
                    d.eq_active[e] = 0
                    broken.add(i)
            else:
                over[i] = 0.0
        if p0 is None and d.time >= t_start:
            p0 = d.xpos[base].copy()
    p1 = d.xpos[base].copy()
    p0 = p0 if p0 is not None else p1
    dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    return dict(distance=dist, broken=len(broken), fitness=dist * PENALTY ** len(broken), wall=time.time() - t0,
                start=p0.tolist(), end=p1.tolist())
