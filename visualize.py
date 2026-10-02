"""Quick visual checks for a morphology: snapshot PNG, rollout GIF, or a live interactive viewer."""

import argparse
import json

import mujoco
import numpy as np
from PIL import Image

import controller
import environment
import morphology
import perturbations


def render_snapshot(params: dict, out_path: str = "results/snapshot.png", width=800, height=600):
    xml = morphology.build_biped_xml(params)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    torso_z = data.xpos[model.body("torso").id][2]

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 45
    cam.elevation = -15
    cam.distance = 1.8
    cam.lookat = np.array([0.0, 0.0, torso_z * 0.6])

    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=cam)
    pixels = renderer.render()
    Image.fromarray(pixels).save(out_path)
    print(f"saved {out_path}")


def record_rollout(params: dict, theta: np.ndarray, out_path: str = "results/rollout.gif",
                    duration: float = 8.0, fps: int = 20, width: int = 640, height: int = 480,
                    perturbation: dict | None = None):
    model = environment.make_model(params)
    if perturbation:
        perturbations.apply_perturbation(model, **perturbation)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth = 45
    cam.elevation = -15
    cam.distance = 2.2

    standing_height = model.body("torso").pos[2]
    steps_per_frame = max(1, round(1.0 / (fps * model.opt.timestep)))

    frames = []
    step = 0
    while data.time < duration:
        data.ctrl = controller.joint_targets(theta, data.time)
        mujoco.mj_step(model, data)
        if step % steps_per_frame == 0:
            cam.lookat = np.array([data.qpos[environment.QPOS_X], 0.0, standing_height * 0.6])
            renderer.update_scene(data, camera=cam)
            frames.append(Image.fromarray(renderer.render()))
        step += 1

    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"saved {out_path} ({len(frames)} frames)")


def launch_interactive(params: dict):
    import mujoco.viewer

    xml = morphology.build_biped_xml(params)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)


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
        theta = controller.DEFAULT_THETA

    if args.interactive:
        launch_interactive(morphology.NOMINAL_PARAMS)
    elif args.rollout:
        record_rollout(morphology.NOMINAL_PARAMS, theta, out_path=args.out or "results/rollout.gif")
    else:
        render_snapshot(morphology.NOMINAL_PARAMS, out_path=args.out or "results/snapshot.png")
