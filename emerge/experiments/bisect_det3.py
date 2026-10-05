"""third bisection: does a single mesh shape of the module repeat, with no joint and no second body?"""
from emerge import sim_coppelia as S
from emerge.experiments import bisect_det2 as B


def keep(index):
    def build(sim):
        root = sim.loadModel(f"{S.MODELS}/emergeModuleAX18.ttm")
        sim.setObjectPose(root, [0, 0, 0.0275, 0, 0, 0, 1], sim.handle_world)
        objs = sim.getObjectsInTree(root, sim.handle_all, 1)
        shapes = [o for o in objs if sim.getObjectType(o) == sim.sceneobject_shape]
        joint = [o for o in objs if sim.getObjectType(o) == sim.sceneobject_joint][0]
        sim.setObjectParent(shapes[index], root, True)
        other = shapes[1 - index]
        sim.setObjectParent(other, root, True)
        sim.setObjectInt32Param(other, sim.shapeintparam_respondable, 0)
        sim.setObjectInt32Param(other, sim.shapeintparam_static, 1)
        return shapes[index]
    return build


if __name__ == "__main__":
    for name, i in (("first mesh shape alone", 0), ("second mesh shape alone", 1)):
        print(name, "std (um) at", B.TIMES, B.run(keep(i)), flush=True)
