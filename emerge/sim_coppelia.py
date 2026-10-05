"""the sim: EMERGE's own CoppeliaSim models (edhmor), assembled from python the way edhmor's CoppeliaSimCreateRobot does.
needs a running coppeliaSim (see emerge/start_coppelia.sh). nothing here is tuned. values come from edhmor's files.

    rollout(genome, duration=38.0, t_start=6.0) -> dict(distance, broken, fitness, wall)
"""

import os
import time

import numpy as np

from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from emerge import morph

V = os.path.abspath(os.path.join(os.path.dirname(__file__), "vendor/edhmor/edhmor"))
MODELS = f"{V}/models/edhmor/emergeModules"
SCENE = f"{V}/scenes/edhmor/default.ttm".replace(".ttm", ".ttt")
MODEL_FILE = {"base": "flatBase", "act": "emergeModuleAX18"}
SIM_DT, PHYSICS_DT = 0.05, 0.005  # paper: control/sim step 50 ms, physics step 5 ms
PENALTY = 0.8  # fitness x 0.8 per broken connection (simulationControl.xml)
NOISE_STD = 0.001  # setpoint noise, SinusoidalControllerWNoise: (pi/2) * std * gauss

_client = None


def _port():
    """the zmq port of the running coppeliasim: env COPPELIA_ZMQ_PORT, else the last rpcPort in emerge/cs.log.
    a new instance picks the next port when an old one still holds 23000."""
    if os.environ.get("COPPELIA_ZMQ_PORT"):
        return int(os.environ["COPPELIA_ZMQ_PORT"])
    try:
        import re
        txt = open(os.path.join(os.path.dirname(__file__), "cs.log"), errors="ignore").read()
        return int(re.findall(r"rpcPort=(\d+)", txt)[-1])
    except Exception:
        return 23000


def api():
    global _client
    if _client is None:
        _client = RemoteAPIClient(port=_port())
    return _client, _client.require("sim")


def _pose(p, rot):
    q = rot.as_quat()  # x, y, z, w (coppelia order too)
    return [float(p[0]), float(p[1]), float(p[2]), float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def boxify(sim, shape):
    """replace a mesh shape by a primitive cuboid of the same bounding box, pose, mass, inertia, friction and place in the hierarchy.
    returns the new handle. the module meshes are 16 to 24 vertex convex hulls; see emerge/NOTES.md for why."""
    size, bbpose = sim.getShapeBB(shape)
    pose = sim.getObjectPose(shape, sim.handle_world)
    parent = sim.getObjectParent(shape)
    children = [c for c in sim.getObjectsInTree(shape, sim.handle_all, 1) if sim.getObjectParent(c) == shape]
    mass = sim.getShapeMass(shape)
    inertia, com = sim.getShapeInertia(shape)
    friction = sim.getEngineFloatParam(sim.bullet_body_friction, shape)
    box = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
    sim.setObjectPose(box, pose, sim.handle_world)
    sim.setObjectInt32Param(box, sim.shapeintparam_static, 0)
    sim.setObjectInt32Param(box, sim.shapeintparam_respondable, 1)
    sim.setShapeMass(box, mass)
    sim.setShapeInertia(box, inertia, com)
    for eng_friction in (sim.bullet_body_friction, sim.ode_body_friction):
        sim.setEngineFloatParam(eng_friction, box, friction)
    sim.setObjectParent(box, parent, True)
    for c in children:
        sim.setObjectParent(c, box, True)
    sim.removeObjects([shape])
    return box


def build(sim, genome: morph.Genome, height: float = 0.055 / 2, boxes: bool = False):
    """returns (module dicts with handles, force sensor handles)."""
    placed = morph.poses(genome, height)
    mods, sensors = [], []
    for i, (m, (pos, rot)) in enumerate(zip(genome.modules, placed)):
        root = sim.loadModel(f"{MODELS}/{MODEL_FILE[m.type]}.ttm")
        sim.setObjectPose(root, _pose(pos, rot), sim.handle_world)
        objs = sim.getObjectsInTree(root, sim.handle_all, 1)  # children, without the root
        shapes = [o for o in objs if sim.getObjectType(o) == sim.sceneobject_shape]
        if boxes and m.type == "act":
            shapes = [boxify(sim, sh) for sh in shapes]
        joints = [o for o in sim.getObjectsInTree(root, sim.handle_all, 1) if sim.getObjectType(o) == sim.sceneobject_joint]
        mods.append(dict(root=root, shapes=shapes, joint=joints[0] if joints else None))
    for i, m in enumerate(genome.modules):
        if m.parent < 0:
            continue
        p = genome.modules[m.parent]
        ofp, nrm = (np.array(v, float) for v in morph.FACES[p.type][m.parent_face])
        ppos, prot = placed[m.parent]
        fs = sim.loadModel(f"{MODELS}/forceSensor.ttm")
        n_world = prot.apply(nrm)
        # sensor z axis along the face normal
        from scipy.spatial.transform import Rotation as R
        z = np.array([0.0, 0.0, 1.0])
        axis = np.cross(z, n_world)
        ang = np.arccos(np.clip(z @ n_world, -1, 1))
        srot = R.from_rotvec(axis / np.linalg.norm(axis) * ang) if np.linalg.norm(axis) > 1e-9 else R.identity()
        sim.setObjectPose(fs, _pose(ppos + prot.apply(ofp), srot), sim.handle_world)
        # parent side: the shape that carries the face (first shape = base part, last = actuated part)
        pshape = mods[m.parent]["shapes"][0 if m.parent_face in morph.BASE_PART_FACES[p.type] else -1]
        sim.setObjectParent(fs, pshape, True)
        # child side: its face 0 is on the base part, the first shape. the rest of the module hangs from it.
        sim.setObjectParent(mods[i]["shapes"][0], fs, True)
        sensors.append(fs)
    return mods, sensors


def stop(client, sim):
    """stop whatever is running, stepping or free-running, and wait until the simulator reports stopped."""
    client.setStepping(False)
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.02)


def rollout(genome: morph.Genome, duration: float = 38.0, t_start: float = 6.0, seed: int = 0, perturb=None,
            noise_std: float = NOISE_STD, update_every: int = 1, engine=None, target_filter=None, after_step=None):
    """perturb(sim, mods, sensors, rng) runs once after the robot is built. used by real_coppelia.
    update_every: setpoints are refreshed every n sim steps (n=1 is edhmor's 20 Hz)."""
    client, sim = api()
    t0 = time.time()
    if sim.getSimulationState() != sim.simulation_stopped:
        stop(client, sim)
    sim.loadScene(SCENE)
    sim.setFloatParam(sim.floatparam_simulation_time_step, SIM_DT)
    if engine is not None:
        sim.setInt32Param(sim.intparam_dynamic_engine, engine)
        for const in ('bullet_global_stepsize', 'ode_global_stepsize'):
            try:
                sim.setEngineFloatParam(getattr(sim, const), -1, PHYSICS_DT)
            except Exception:
                pass
    mods, sensors = build(sim, genome)
    base = mods[0]["shapes"][0]
    rng = np.random.default_rng(seed)
    if perturb is not None:
        perturb(sim, mods, sensors, rng)
    client.setStepping(True)
    sim.startSimulation()
    ph = np.radians(genome.phases())
    p0 = None
    n = int(round(duration / SIM_DT))
    for k in range(n):
        t = sim.getSimulationTime()
        noise = (np.pi / 2) * noise_std * rng.standard_normal()
        for i, m in enumerate(mods):
            if k % update_every == 0 and m["joint"] is not None and genome.modules[i].type == "act":
                target = float(morph.MAX_AMPLITUDE * np.sin(morph.MAX_ANGULAR_FREQUENCY * t + ph[i]) + noise)
                if target_filter is not None:
                    target = target_filter(i, k, target)
                sim.setJointTargetPosition(m["joint"], target)
        client.step()
        if after_step is not None:
            after_step(sim, mods, sensors, k)
        if p0 is None and t >= t_start:
            p0 = sim.getObjectPosition(base, sim.handle_world)
    p1 = sim.getObjectPosition(base, sim.handle_world)
    # readForceSensor result: bit 0 = reading available, bit 1 = sensor broken (checked with emerge/checks/check_break.py)
    broken = 0
    for fs in sensors:
        try:
            r = sim.readForceSensor(fs)
            broken += 1 if (r[0] & 2) else 0
        except Exception:
            broken += 1
    stop(client, sim)
    p0 = p0 or p1
    dist = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    return dict(distance=dist, broken=broken, fitness=dist * PENALTY ** broken, wall=time.time() - t0,
                start=p0, end=p1)
