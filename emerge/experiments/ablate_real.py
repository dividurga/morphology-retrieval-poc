"""why do the sim and the real differ so much? move the real toward the sim's values one knob at a time and see which one matters.
diagnosis only. the real's defaults are not changed by this."""
import numpy as np
from emerge import morph, real_mujoco as Rm

BASE = dict(MASS_ACT=Rm.MASS_ACT, MASS_BASE=Rm.MASS_BASE, SPEED_CAP=Rm.SPEED_CAP, RESOLUTION=Rm.RESOLUTION,
            COMMAND_PERIOD=Rm.COMMAND_PERIOD, CONNECTOR_TORQUE=Rm.CONNECTOR_TORQUE, BREAK_HOLD=Rm.BREAK_HOLD,
            CONNECTOR_SOLREF=Rm.CONNECTOR_SOLREF, KP=Rm.KP, KD=Rm.KD)
world0 = Rm.world

def run(label, consts=None, world_fn=None, seeds=(0, 1, 2), genome=None):
    for k, v in BASE.items():
        setattr(Rm, k, v)
    for k, v in (consts or {}).items():
        setattr(Rm, k, v)
    Rm.world = world_fn or world0
    g = genome or morph.chain(2, [0, 90])
    r = [Rm.rollout(g, s) for s in seeds]
    print(f"{label:46s} distance {np.mean([x['distance'] for x in r]):.3f} (min {min(x['distance'] for x in r):.3f}, max {max(x['distance'] for x in r):.3f})  broken {np.mean([x['broken'] for x in r]):.1f}")

def w_with(mass=None, friction=None, brk=None):
    def f(seed, n):
        w = world0(seed, n)
        if mass is not None: w["mass"][:] = mass
        if friction is not None: w["friction"][:] = friction
        if brk is not None: w["break_factor"][:] = brk
        return w
    return f

NOBREAK = dict(CONNECTOR_TORQUE=1e6)
run("real as built")
run("no spread (mass 1, break factor 1)", world_fn=w_with(mass=1.0, brk=1.0))
run("connectors never break", NOBREAK)
run("sim masses (165 g / 650 g), no spread", dict(MASS_ACT=0.165, MASS_BASE=0.65), w_with(mass=1.0))
run("sim friction 0.71 on all modules", world_fn=w_with(friction=0.71))
run("servo: 20 Hz command", dict(COMMAND_PERIOD=0.05))
run("servo: no speed cap", dict(SPEED_CAP=1e3))
run("servo: no quantisation", dict(RESOLUTION=1e-9))
run("servo: all three servo differences removed", dict(COMMAND_PERIOD=0.05, SPEED_CAP=1e3, RESOLUTION=1e-9))
run("rigid connectors (solref 0.004 1) + no break", dict(CONNECTOR_SOLREF="0.004 1", **NOBREAK))
run("no break + sim masses + sim friction", dict(MASS_ACT=0.165, MASS_BASE=0.65, **NOBREAK), w_with(mass=1.0, friction=0.71))
run("no break + masses + friction + servo like sim", dict(MASS_ACT=0.165, MASS_BASE=0.65, COMMAND_PERIOD=0.05, SPEED_CAP=1e3, RESOLUTION=1e-9, **NOBREAK), w_with(mass=1.0, friction=0.71))
run("... + rigid connectors", dict(MASS_ACT=0.165, MASS_BASE=0.65, COMMAND_PERIOD=0.05, SPEED_CAP=1e3, RESOLUTION=1e-9, CONNECTOR_SOLREF="0.004 1", **NOBREAK), w_with(mass=1.0, friction=0.71))
