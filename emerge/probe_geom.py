import json
import os
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
V = os.path.abspath("emerge/vendor/edhmor/edhmor")
sim = RemoteAPIClient().require("sim")
sim.loadScene(f"{V}/scenes/edhmor/default.ttt")
out = {}
for name in ("emergeModuleAX18", "flatBase"):
    root = sim.loadModel(f"{V}/models/edhmor/emergeModules/{name}.ttm")
    rows = []
    for o in sim.getObjectsInTree(root, sim.handle_all, 1):
        t = sim.getObjectType(o)
        d = dict(alias=sim.getObjectAlias(o), type=int(t), pose_in_root=sim.getObjectPose(o, root))
        if t == sim.sceneobject_shape:
            size, bbpose = sim.getShapeBB(o)
            d.update(size=size, bbpose=bbpose, mass=sim.getShapeMass(o), com=sim.getShapeInertia(o)[1],
                     inertia=sim.getShapeInertia(o)[0], respondable=bool(sim.getObjectInt32Param(o, sim.shapeintparam_respondable)),
                     friction=sim.getEngineFloatParam(sim.bullet_body_friction, o))
        if t == sim.sceneobject_joint:
            d.update(range=sim.getJointInterval(o)[1], maxforce=sim.getJointTargetForce(o))
        rows.append(d)
    out[name] = rows
    sim.removeModel(root)
json.dump(out, open("emerge/module_geometry.json", "w"), indent=1)
for k, v in out.items():
    print(k)
    for d in v:
        print(" ", d["alias"], "pose", np.round(d["pose_in_root"], 4).tolist(), {x: (np.round(d[x], 4).tolist() if x in ("size", "bbpose", "com") else d[x]) for x in d if x in ("size", "bbpose", "mass", "com", "respondable", "friction", "range", "maxforce")})
