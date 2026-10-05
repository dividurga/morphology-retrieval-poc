"""EMERGE genome and module geometry, ported from edhmor (EmergeModuleSet, Emerge18FlatBaseModuleSet).
one genome feeds both sims. a genome is a tree of modules plus one phase (degrees) per module.

module types: "base" (flatBase, 8 faces, no actuator) and "act" (emergeModuleAX18, 1 dof, 4 faces).
face 0 of "act" is the connection face to its parent. it belongs to the base part of the module.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

# (origin vector in module frame, outward normal). edhmor src/modules/Emerge18FlatBaseModuleSet.java, EmergeModuleSet.java
FACES = {
    "base": [
        ((0.0747, 0.043, 0), (1, 0, 0)), ((0.043, 0.0747, 0), (0, 1, 0)),
        ((-0.043, 0.0747, 0), (0, 1, 0)), ((-0.0747, 0.043, 0), (-1, 0, 0)),
        ((-0.0747, -0.043, 0), (-1, 0, 0)), ((-0.043, -0.0747, 0), (0, -1, 0)),
        ((0.043, -0.0747, 0), (0, -1, 0)), ((0.0747, -0.043, 0), (1, 0, 0)),
    ],
    "act": [
        ((-0.0385, 0, 0), (-1, 0, 0)), ((0.0385, 0, 0), (1, 0, 0)),
        ((0.008, -0.0305, 0), (0, -1, 0)), ((0.008, 0.0305, 0), (0, 1, 0)),
    ],
}
# faces that sit on the actuated part of the module (the rest are on the base part). "act" has only face 0 on the base part.
BASE_PART_FACES = {"base": set(range(8)), "act": {0}}
ORIENTATION_ANGLE = (0.0, np.pi / 2)  # rotation of the child about the connection normal
MAX_AMPLITUDE = 0.5 * np.pi  # edhmor ModuleSet: modulesMaxAmplitude
MAX_ANGULAR_FREQUENCY = 2.0  # edhmor ModuleSet: modulesMaxAngularFrequency (the paper text says 1)


@dataclass
class Module:
    type: str  # "base" or "act"
    parent: int = -1  # index of the parent module, -1 for the root
    parent_face: int = 0
    orientation: int = 0  # 0 or 1
    phase_deg: float = 0.0


@dataclass
class Genome:
    modules: list = field(default_factory=list)

    def phases(self):
        return np.array([m.phase_deg for m in self.modules])


def chain(n_act: int, phases_deg=None, base_face: int = 0) -> Genome:
    """a base with a straight chain of actuated modules, joint axes alternating pitch/yaw."""
    g = Genome([Module("base")])
    for i in range(n_act):
        g.modules.append(Module("act", parent=i, parent_face=base_face if i == 0 else 1, orientation=i % 2,
                                phase_deg=0.0 if phases_deg is None else phases_deg[i]))
    return g


def poses(genome: Genome, base_height: float):
    """module poses in the world: list of (position, scipy Rotation). root is flat on the floor.
    the child's face 0 is mated to the parent face: its normal opposes the parent's normal,
    then the child is spun about that normal by the orientation angle."""
    out = []
    for i, m in enumerate(genome.modules):
        if m.parent < 0:
            out.append((np.array([0.0, 0.0, base_height]), R.identity()))
            continue
        pp, pr = out[m.parent]
        ptype = genome.modules[m.parent].type
        ofp, nrm = (np.array(v, float) for v in FACES[ptype][m.parent_face])
        n_world = pr.apply(nrm)
        # yaw so the child's face-0 normal (-1,0,0) maps to -n_world (all normals lie in the xy plane)
        target = -n_world
        yaw = np.arctan2(target[1], target[0]) - np.arctan2(0.0, -1.0)
        rc = R.from_euler("z", yaw)
        rc = R.from_rotvec(ORIENTATION_ANGLE[m.orientation] * n_world) * rc
        f0 = np.array(FACES["act"][0][0], float)
        pos = pp + pr.apply(ofp) - rc.apply(f0)
        out.append((pos, rc))
    return out
