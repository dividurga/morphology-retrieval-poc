"""Quick visual checks for the PyBullet twin: snapshot PNG or rollout GIF.

Mirrors visualize.py's two functions for sim_pybullet. Kept as its own module rather than
added to sim_pybullet.simulate(), since frame capture isn't that function's job.
"""

import numpy as np
import pybullet as p
from PIL import Image

import controller
import sim_pybullet as sb
from sim_common import has_fallen
from spec import SPEC

WIDTH, HEIGHT = 480, 360  # lightweight -- matches the GIFs already in check/
FPS = 15

# Known cosmetic-only quirk: capsule end-caps (e.g. the torso's top) render as a
# faceted cone, not a smooth hemisphere, 


def _view_and_proj(client, lookat, distance, width, height):
    # yaw/pitch/distance/target is PyBullet's direct analogue of MuJoCo's
    # azimuth/elevation/distance/lookat camera model -- same framing convention
    # visualize.py uses (azimuth 45, elevation -15).
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=lookat, distance=distance, yaw=45, pitch=-15, roll=0,
        upAxisIndex=2, physicsClientId=client)
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=width / height, nearVal=0.05, farVal=10,
                                         physicsClientId=client)
    return view, proj


def _capture(client, lookat, distance, width, height):
    view, proj = _view_and_proj(client, lookat, distance, width, height)
    _, _, rgb, _, _ = p.getCameraImage(width, height, view, proj, renderer=p.ER_TINY_RENDERER,
                                        physicsClientId=client)
    return Image.fromarray(np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8))


def render_snapshot(params: dict, out_path: str = "check/pybullet_snapshot.png",
                     width: int = WIDTH, height: int = HEIGHT, spec=SPEC):
    model = sb.make_model(params, spec)
    lookat = [0.0, 0.0, model.standing_height * 0.6]
    img = _capture(model.client, lookat, distance=1.8, width=width, height=height)
    img.save(out_path)
    print(f"saved {out_path}")


def record_rollout(params: dict, theta: np.ndarray, out_path: str = "check/pybullet_rollout.gif",
                    duration: float = 8.0, fps: int = FPS, width: int = WIDTH, height: int = HEIGHT,
                    spec=SPEC):
    """Re-steps sim_pybullet.simulate's loop locally (shared apply_control + substeps) so frames
    can be captured."""
    model = sb.make_model(params, spec)
    client = model.client
    sb.reset(model)
    steps_per_frame = max(1, round(1.0 / (fps * spec.timestep)))
    frames = []

    for step in range(int(round(duration / spec.timestep))):
        targets = controller.joint_targets(theta, step * spec.timestep)
        for _ in range(model.cfg.substeps):
            sb.apply_control(model, targets)
            p.stepSimulation(physicsClientId=client)

        if step % steps_per_frame == 0:
            x_pos = p.getJointState(model.body_id, model.joint_index["dummy_x"], physicsClientId=client)[0]
            lookat = [x_pos, 0.0, model.standing_height * 0.6]
            frames.append(_capture(client, lookat, distance=2.2, width=width, height=height))

        pitch = p.getJointState(model.body_id, model.joint_index["torso"], physicsClientId=client)[0]
        dz = p.getJointState(model.body_id, model.joint_index["dummy_z"], physicsClientId=client)[0]
        if has_fallen(pitch, dz, model.standing_height):
            break

    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"saved {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    render_snapshot(SPEC.nominal_params())
