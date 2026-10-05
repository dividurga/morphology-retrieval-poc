"""fourth bisection: the module's two shapes + hinge. which part is not repeatable: hinge, A-B contact, or floor contact?"""
from emerge import sim_coppelia as S
from emerge.experiments import bisect_det2 as B


def module(respondable=(1, 1), height=0.0275, mode=None):
    def build(sim):
        root = sim.loadModel(f"{S.MODELS}/emergeModuleAX18.ttm")
        sim.setObjectPose(root, [0, 0, height, 0, 0, 0, 1], sim.handle_world)
        objs = sim.getObjectsInTree(root, sim.handle_all, 1)
        shapes = [o for o in objs if sim.getObjectType(o) == sim.sceneobject_shape]
        joint = [o for o in objs if sim.getObjectType(o) == sim.sceneobject_joint][0]
        for sh, r in zip(shapes, respondable):
            sim.setObjectInt32Param(sh, sim.shapeintparam_respondable, r)
        if mode is not None:
            sim.setObjectInt32Param(joint, sim.jointintparam_dynctrlmode, mode)
        return shapes[0]
    return build


if __name__ == "__main__":
    client, sim = S.api()
    cases = {
        "in the air (0.5 m), non-respondable, joint default": module((0, 0), 0.5),
        "in the air (0.5 m), non-respondable, joint free": module((0, 0), 0.5, sim.jointdynctrl_free),
        "on the floor, non-respondable (falls through? no: floor still collides only if respondable)": module((0, 0), 0.0275),
        "on the floor, only A respondable": module((1, 0)),
        "on the floor, only B respondable": module((0, 1)),
        "on the floor, both respondable": module((1, 1)),
    }
    for name, b in cases.items():
        print(f"{name:60s}", B.run(b), flush=True)
