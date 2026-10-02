"""Quick visual checks for the PyBullet backend: snapshot PNG or rollout GIF.

Mirrors visualize.py's two functions for pybullet_env. Kept as its own module rather
than added to pybullet_env.simulate() -- same separation visualize.py already keeps
from environment.py (that file also re-steps the sim locally in record_rollout rather
than calling environment.simulate(), since frame capture isn't that function's job).
"""

import numpy as np
import pybullet as p
from PIL import Image

import controller
import morphology
import pybullet_env

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
                     width: int = WIDTH, height: int = HEIGHT):
    model = pybullet_env.make_model(params)
    lookat = [0.0, 0.0, model.standing_height * 0.6]
    img = _capture(model.client, lookat, distance=1.8, width=width, height=height)
    img.save(out_path)
    print(f"saved {out_path}")


def record_rollout(params: dict, theta: np.ndarray, out_path: str = "check/pybullet_rollout.gif",
                    duration: float = 8.0, fps: int = FPS, width: int = WIDTH, height: int = HEIGHT):
    """Re-steps pybullet_env's control loop locally (same torque law, same ankle spring)
    so frames can be captured -- intentionally duplicated from pybullet_env.simulate()
    rather than modifying it, see module docstring."""
    model = pybullet_env.make_model(params)
    client = model.client
    for idx in model.joint_index.values():
        p.resetJointState(model.body_id, idx, 0.0, 0.0, physicsClientId=client)

    steps = int(round(duration / pybullet_env.TIMESTEP))
    steps_per_frame = max(1, round(1.0 / (fps * pybullet_env.TIMESTEP)))
    frames = []

    for step in range(steps):
        t = step * pybullet_env.TIMESTEP
        targets = controller.joint_targets(theta, t)
        for name, target in zip(controller.ACTUATOR_ORDER, targets):
            idx = model.joint_index[name]
            kp = model.kp_hip if name.startswith("hip") else model.kp_knee
            q, qd = p.getJointState(model.body_id, idx, physicsClientId=client)[:2]
            lo, hi = pybullet_env.RANGES[name.split("_")[0]]
            tau = kp * (float(np.clip(target, lo, hi)) - q) - morphology.JOINT_DAMPING * qd
            p.setJointMotorControl2(model.body_id, idx, p.TORQUE_CONTROL, force=tau,
                                     physicsClientId=client)
        pybullet_env._apply_ankle_spring(model, "ankle_left", model.ankle_stiffness, model.ankle_damping)
        pybullet_env._apply_ankle_spring(model, "ankle_right", model.ankle_stiffness, model.ankle_damping)
        p.stepSimulation(physicsClientId=client)

        if step % steps_per_frame == 0:
            x_pos = p.getJointState(model.body_id, model.joint_index["dummy_x"], physicsClientId=client)[0]
            lookat = [x_pos, 0.0, model.standing_height * 0.6]
            frames.append(_capture(client, lookat, distance=2.2, width=width, height=height))

        pitch = p.getJointState(model.body_id, model.joint_index["torso"], physicsClientId=client)[0]
        z_pos = p.getJointState(model.body_id, model.joint_index["dummy_z"], physicsClientId=client)[0]
        height_now = z_pos + model.standing_height
        if (abs(pitch) > pybullet_env.FALL_PITCH_THRESH
                or height_now < pybullet_env.FALL_HEIGHT_FRAC * model.standing_height):
            break

    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    print(f"saved {out_path} ({len(frames)} frames)")


if __name__ == "__main__":
    render_snapshot(morphology.NOMINAL_PARAMS)
