"""seventh: floor hypothesis. the scene floor is 50 small shapes. replace it by one large static cuboid and see if the module / chains repeat."""
from emerge import morph, sim_coppelia as S
from emerge.experiments import bisect_det as D1, bisect_det2 as B



def single_floor(sim):
    floors = [o for o in sim.getObjectsInTree(sim.handle_scene, sim.sceneobject_shape) if "Floor" in sim.getObjectAlias(o, 1)]
    for f in floors:
        sim.setObjectInt32Param(f, sim.shapeintparam_respondable, 0)
    big = sim.createPrimitiveShape(sim.primitiveshape_cuboid, [40.0, 40.0, 0.2], 0)
    sim.setObjectPose(big, [0, 0, -0.1, 0, 0, 0, 1], sim.handle_world)
    sim.setObjectInt32Param(big, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(big, sim.shapeintparam_respondable, 1)
    sim.setEngineFloatParam(sim.bullet_body_friction, big, 0.71)  # placeholder: same order as the module friction
    return len(floors)


if __name__ == "__main__":
    client, sim = S.api()
    print("module alone, single floor      ", B.run(lambda s: (single_floor(s), B.module_alone(s))[1]), flush=True)
    print("chain of 4 passive, single floor", D1.trace(morph.chain(4), tweak=single_floor), flush=True)
