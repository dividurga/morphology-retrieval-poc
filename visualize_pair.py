"""one gait, nominal settings, in the pybullet twin (sim) and mujoco (real), side by side as a gif.

    uv run python visualize_pair.py --csv tubey/dataset.csv --row 0 --out check/pair_row0.gif

same open-loop loop as engines.rollout (no lag, nominal). each panel freezes at its own fall. a gait is a
(params, theta) pair, so this works for any row of any dataset csv that has the morphology columns and theta_*.
"""

import argparse

import mujoco
import numpy as np
import pandas as pd
import pybullet as p
from PIL import Image, ImageDraw

import controller
import engines
import morphology

def row_params(r) -> dict:
    return {k: float(r[k]) for k in morphology.PARAM_BOUNDS}


def row_theta(r) -> np.ndarray:
    return np.array([float(r[f"theta_{j}"]) for j in range(len(controller.THETA_BOUNDS))])


W, H, FPS = 400, 300, 20
DUR = 3.0


def _frame_pb(eng, x):
    view = p.computeViewMatrixFromYawPitchRoll([x, 0.0, 0.4], 2.2, 0, -10, 0, 2, physicsClientId=eng.c)
    proj = p.computeProjectionMatrixFOV(50, W / H, 0.05, 20, physicsClientId=eng.c)
    rgb = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER, physicsClientId=eng.c)[2]
    return np.reshape(rgb, (H, W, 4))[:, :, :3].astype(np.uint8)


def _frame_mj(eng, renderer, x):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [x, 0.0, 0.4]
    cam.distance, cam.azimuth, cam.elevation = 2.2, 90, -10  # azimuth 90: looking along +y, same side view as the pybullet camera
    renderer.update_scene(eng.d, cam)
    return renderer.render()


def record(kind, params, theta):
    eng = engines.make_engine(kind, params)
    renderer = mujoco.Renderer(eng.m, H, W) if kind == "mujoco" else None
    eng.reset()
    sub = getattr(eng, "sub", 1)
    every = round(1.0 / (FPS * engines.DT))
    frames, fell_t, last = [], None, 0.0
    for step in range(int(round(DUR / engines.DT))):
        tg = controller.joint_targets(theta, step * engines.DT)
        for _ in range(sub):
            eng.apply(engines.command_to_torque(eng.params, eng.joints(), tg))
            eng.step()
        x, z, pitch = eng.state()
        if step % every == 0:
            frames.append((_frame_mj(eng, renderer, x) if renderer else _frame_pb(eng, x), x))
        if abs(pitch) > engines.FALL_PITCH or z < engines.FALL_HEIGHT_FRAC * eng.standing:
            fell_t = (step + 1) * engines.DT
            frames.append((frames[-1][0], x))
            break
        last = x
    x_end = eng.state()[0]
    if kind == "pybullet":
        p.disconnect(eng.c)
    return frames, x_end, fell_t


def label(img, title, x, fell_t):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 34], fill=(255, 255, 255))
    d.text((6, 4), title, fill=(0, 0, 0))
    d.text((6, 18), f"x = {x:.2f} m" + (f"   FELL at {fell_t:.2f} s" if fell_t else ""), fill=(200, 0, 0) if fell_t else (0, 110, 0))
    return im


def pair_gif(params, theta, out):
    res = {k: record(k, params, theta) for k in ("pybullet", "mujoco")}
    n = max(len(r[0]) for r in res.values()) + int(0.6 * FPS)  # hold the last frame a moment
    names = {"pybullet": "sim: pybullet (nominal)", "mujoco": "real: mujoco (nominal)"}
    frames = []
    for i in range(n):
        panels = []
        for k in ("pybullet", "mujoco"):
            fr, x_end, fell_t = res[k]
            img, x = fr[min(i, len(fr) - 1)]
            done = i >= len(fr) - 1
            panels.append(label(img, names[k], x, fell_t if (done and fell_t) else None))
        canvas = Image.new("RGB", (2 * W + 4, H), (255, 255, 255))
        canvas.paste(panels[0], (0, 0)); canvas.paste(panels[1], (W + 4, 0))
        frames.append(canvas)
    # one shared palette for every frame. per-frame palettes make the two very different panels flicker with colour streaks.
    sample = Image.new("RGB", (2 * W + 4, H * 3))
    for j, i in enumerate((0, len(frames) // 2, len(frames) - 1)):
        sample.paste(frames[i], (0, H * j))
    pal = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    frames = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0)
    print(f"saved {out}. pybullet x={res['pybullet'][1]:.2f} fell={res['pybullet'][2]}, "
          f"mujoco x={res['mujoco'][1]:.2f} fell={res['mujoco'][2]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--row", type=int, default=0, help="row index of the csv (after any --filled filter)")
    ap.add_argument("--out", default="check/pair.gif")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    if "filled" in df:
        df = df[df.filled]
    r = df.iloc[a.row]
    pair_gif(row_params(r), row_theta(r), a.out)
