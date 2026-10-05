"""fifth: stop A and B from colliding with each other using the local respondable mask (shapes sharing a parent use the local byte),
keep both colliding with the floor (global byte). mask = (global << 8) | local."""
from emerge import sim_coppelia as S
from emerge.experiments import bisect_det2 as B


def masked(local_a, local_b, glob=0xFF):
    def build(sim):
        root = sim.loadModel(f"{S.MODELS}/emergeModuleAX18.ttm")
        sim.setObjectPose(root, [0, 0, 0.0275, 0, 0, 0, 1], sim.handle_world)
        shapes = [o for o in sim.getObjectsInTree(root, sim.handle_all, 1) if sim.getObjectType(o) == sim.sceneobject_shape]
        for sh, loc in zip(shapes, (local_a, local_b)):
            sim.setObjectInt32Param(sh, sim.shapeintparam_respondable_mask, (glob << 8) | loc)
        return shapes[0]
    return build


if __name__ == "__main__":
    for name, b in (("default masks", masked(0x0F, 0x0F)), ("A=1, B=2 (disjoint local bits)", masked(0x01, 0x02)),
                    ("A=0x0F, B=0 (B has no local bits)", masked(0x0F, 0x00))):
        print(f"{name:40s}", B.run(b), flush=True)
