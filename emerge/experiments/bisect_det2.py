"""second bisection: is it the actuated module's joint, or the force sensor connection?"""
import numpy as np
from emerge import morph, sim_coppelia as S

TIMES = [0.05, 0.1, 0.5, 1.0]


def run(builder, repeats=8):
    client, sim = S.api()
    runs = []
    for _ in range(repeats):
        if sim.getSimulationState() != sim.simulation_stopped:
            S.stop(client, sim)
        sim.loadScene(S.SCENE)
        sim.setFloatParam(sim.floatparam_simulation_time_step, S.SIM_DT)
        track = builder(sim)
        client.setStepping(True)
        sim.startSimulation()
        out = []
        for k in range(int(1.05 / S.SIM_DT)):
            client.step()
            t = sim.getSimulationTime()
            if any(abs(t - x) < 1e-6 for x in TIMES):
                out.append(sim.getObjectPosition(track, sim.handle_world))
        S.stop(client, sim)
        runs.append(out)
    return (np.array(runs).std(axis=0).max(axis=1) * 1e6).round(2).tolist()


def module_alone(sim, joint_mode=None):
    root = sim.loadModel(f"{S.MODELS}/emergeModuleAX18.ttm")
    sim.setObjectPose(root, [0, 0, 0.0275, 0, 0, 0, 1], sim.handle_world)
    shapes = [o for o in sim.getObjectsInTree(root, sim.handle_all, 1) if sim.getObjectType(o) == sim.sceneobject_shape]
    joint = [o for o in sim.getObjectsInTree(root, sim.handle_all, 1) if sim.getObjectType(o) == sim.sceneobject_joint][0]
    if joint_mode == "free":
        sim.setObjectInt32Param(joint, sim.jointintparam_dynctrlmode, sim.jointdynctrl_free)
    return shapes[0]


def base_sensor_box(sim):
    base = sim.loadModel(f"{S.MODELS}/flatBase.ttm")
    sim.setObjectPose(base, [0, 0, 0.0275, 0, 0, 0, 1], sim.handle_world)
    bshape = [o for o in sim.getObjectsInTree(base, sim.handle_all, 1) if sim.getObjectType(o) == sim.sceneobject_shape][0]
    box = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [0.05, 0.05, 0.05], 0)
    sim.setObjectPose(box, [0.12, 0.043, 0.0275, 0, 0, 0, 1], sim.handle_world)
    sim.setShapeMass(box, 0.1)
    sim.setObjectInt32Param(box, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(box, sim.shapeintparam_respondable, 1)
    fs = sim.loadModel(f"{S.MODELS}/forceSensor.ttm")
    sim.setObjectPose(fs, [0.0747, 0.043, 0.0275, 0, 0, 0, 1], sim.handle_world)
    sim.setObjectParent(fs, bshape, True)
    sim.setObjectParent(box, fs, True)
    return box


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    print("std of tracked position (micrometres) at t =", TIMES)
    for name, b in (("one actuated module alone", module_alone),
                    ("one actuated module alone, joint free", lambda s: module_alone(s, "free")),
                    ("base + force sensor + plain box", base_sensor_box)):
        print(f"  {name:40s}", run(b), flush=True)


def masks(sim, floor_mask=7):
    """returns the default mask of the first shape in the scene, then lets a builder set masks."""
    floors = [o for o in sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_shape) if "Floor" in sim.getObjectAlias(o, 1)]
    for f in floors:
        sim.setObjectInt32Param(f, sim.shapeintparam_respondable_mask, floor_mask)
    return floors


def module_alone_masked(sim):
    """local mask (low byte) decides between shapes that share a parent: give A and B disjoint local bits.
    the global mask (high byte) stays 0xFF so both still collide with the floor."""
    root = sim.loadModel(f"{S.MODELS}/emergeModuleAX18.ttm")
    sim.setObjectPose(root, [0, 0, 0.0275, 0, 0, 0, 1], sim.handle_world)
    shapes = [o for o in sim.getObjectsInTree(root, sim.handle_all, 1) if sim.getObjectType(o) == sim.sceneobject_shape]
    glob = 0xFF00
    sim.setObjectInt32Param(shapes[0], sim.shapeintparam_respondable_mask, glob | 0x01)
    sim.setObjectInt32Param(shapes[1], sim.shapeintparam_respondable_mask, glob | 0x02)
    return shapes[0]


if __name__ == "__main__" and False:
    pass


if __name__ == "__main__" and len(__import__("sys").argv) == 2 and __import__("sys").argv[1] == "masked":
    print("module alone, A and B given masks that do not overlap; floor mask 7")
    print("  std (um) at", TIMES, run(module_alone_masked), flush=True)


def module_alone_boxes(sim):
    root = sim.loadModel(f"{S.MODELS}/emergeModuleAX18.ttm")
    sim.setObjectPose(root, [0, 0, 0.0275, 0, 0, 0, 1], sim.handle_world)
    shapes = [o for o in sim.getObjectsInTree(root, sim.handle_all, 1) if sim.getObjectType(o) == sim.sceneobject_shape]
    boxes = [S.boxify(sim, sh) for sh in shapes]
    return boxes[0]


if __name__ == "__main__" and len(__import__("sys").argv) > 2:
    print("module alone, mesh shapes replaced by cuboids")
    print("  std (um) at", TIMES, run(module_alone_boxes), flush=True)


def module_alone_tweak(tweak):
    def b(sim):
        tweak(sim)
        return module_alone(sim)
    return b


if __name__ == "__main__" and len(__import__("sys").argv) == 2 and __import__("sys").argv[1] == "inertia":
    client, sim = S.api()
    print("computeinertias currently:", sim.getEngineBoolParam(sim.bullet_global_computeinertias, -1))
    for val in (False, True):
        print(f"  computeinertias={val}: std (um)", run(module_alone_tweak(lambda s, v=val: s.setEngineBoolParam(s.bullet_global_computeinertias, -1, v))), flush=True)
