"""Quick visual checks for a morphology in the "real" mujoco sim: snapshot PNG, rollout GIF, or a live viewer."""

import argparse
import json

import mujoco
import numpy as np
from PIL import Image

import controller
import sim_mujoco as sm
from spec import SPEC, RealityDeltas


def render_snapshot(params: dict, out_path: str = "results/snapshot.png", width=800, height=600,
                    spec=SPEC):
    model = sm.make_model(params, spec)
    data = mujoco.MjData(model.mj)
    mujoco.mj_forward(model.mj, data)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 45
    cam.elevation = -15
    cam.distance = 1.8
    cam.lookat = np.array([0.0, 0.0, model.standing_height * 0.6])

    renderer = mujoco.Renderer(model.mj, height=height, width=width)
    renderer.update_scene(data, camera=cam)
    Image.fromarray(renderer.render()).save(out_path)
    print(f"saved {out_path}")


def record_rollout(params: dict, theta: np.ndarray, out_path: str = "results/rollout.gif",
                    duration: float = 8.0, fps: int = 20, width: int = 640, height: int = 480,
                    spec=SPEC, deltas: RealityDeltas | None = None):
    """Re-steps sim_mujoco.simulate's loop so frames can be captured. deltas makes the robot
    differ from the nominal spec the same way gap_budget.py does."""
    if deltas is not None:
        params, spec = deltas.apply(params, spec)
    model = sm.make_model(params, spec)
    data = mujoco.MjData(model.mj)
    renderer = mujoco.Renderer(model.mj, height=height, width=width)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 45
    cam.elevation = -15
    cam.distance = 2.2

    steps_per_frame = max(1, round(1.0 / (fps * spec.timestep)))
    delay = spec.delay_steps * spec.timestep
    frames = []
    for step in range(int(round(duration / spec.timestep))):
        targets = controller.joint_targets(theta, max(0.0, step * spec.timestep - delay))
        for _ in range(spec.mj_substeps):
            sm.apply_control(model, data, targets)
            mujoco.mj_step(model.mj, data)
        if step % steps_per_frame == 0:
            cam.lookat = np.array([data.qpos[sm.QPOS_X], 0.0, model.standing_height * 0.6])
            renderer.update_scene(data, camera=cam)
            frames.append(Image.fromarray(renderer.render()))

    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"saved {out_path} ({len(frames)} frames)")


def launch_interactive(params: dict, spec=SPEC):
    import mujoco.viewer

    model = sm.make_model(params, spec)
    mujoco.viewer.launch(model.mj, mujoco.MjData(model.mj))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--rollout", action="store_true")
    parser.add_argument("--theta-file", default=None, help="JSON file with a 'theta' field (e.g. results/controller_nominal.json)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.theta_file:
        with open(args.theta_file) as f:
            theta = np.array(json.load(f)["theta"])
    else:
        theta = controller.default_theta(SPEC.omega_scale)

    params = SPEC.nominal_params()
    if args.interactive:
        launch_interactive(params)
    elif args.rollout:
        record_rollout(params, theta, out_path=args.out or "results/rollout.gif")
    else:
        render_snapshot(params, out_path=args.out or "results/snapshot.png")
