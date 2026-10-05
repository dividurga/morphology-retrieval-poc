"""render one genome in the sim and both reals, side by side, from recorded module poses.
coppeliasim runs headless here, so nothing is rendered by the engines: each engine records the world pose of every module box
at every control step, and matplotlib draws all of them the same way (same camera, same scale, follows the base).

    sh emerge/start_coppelia.sh
    uv run python -m emerge.visualize --phases 0 90 --duration 20 --out emerge/viz/chain2.gif
    uv run python -m emerge.visualize --only real_b --duration 10        # one engine, no coppeliasim needed
    uv run python -m emerge.visualize --trace emerge/viz/chain2.npz      # redraw from a saved recording

engines: sim (edhmor bullet), real_a (coppeliasim ode + servo/connector effects, world seed), real_b (mujoco, world seed).
boxes are the bounding boxes of the module shapes (the mujoco model is built from the same boxes). a red tint = a broken connector.
"""

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from emerge import morph

ENGINES = {"sim": "sim (coppeliasim, bullet 2.78)", "real_a": "real A (coppeliasim, ode + servo/connector)", "real_b": "real B (mujoco)"}
OUT = os.path.join(os.path.dirname(__file__), "viz")
FRAME_DT = 0.05  # s between recorded frames (the sim's control step)
CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
FACES = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]  # corner indices per box face


class Recorder:
    """collects per-frame box poses. frames: t, centres (n,3), rotation matrices (n,3,3), broken (n_connectors,) bool."""

    def __init__(self):
        self.t, self.centre, self.rot, self.broken = [], [], [], []
        self.half = None  # (n,3)
        self.module = None  # (n,) module index of each box
        self.connector_module = None  # (c,) module index of the child side of each connector

    def add(self, t, centre, rot, broken):
        self.t.append(t)
        self.centre.append(np.asarray(centre, float))
        self.rot.append(np.asarray(rot, float))
        self.broken.append(np.asarray(broken, bool))

    def arrays(self):
        return dict(t=np.array(self.t), centre=np.array(self.centre), rot=np.array(self.rot), broken=np.array(self.broken),
                    half=self.half, module=self.module, connector_module=self.connector_module)


# ---- recording per engine ----

def _coppelia_hook(rec: Recorder, genome: morph.Genome, every: int = 1):
    state = {}

    def hook(sim, mods, sensors, k):
        if k % every:
            return
        if not state:  # first call: box sizes and offsets are constant
            shapes = [(i, sh) for i, m in enumerate(mods) for sh in m["shapes"]]
            bb = [sim.getShapeBB(sh) for _, sh in shapes]
            state["shapes"] = [sh for _, sh in shapes]
            state["bb"] = [(np.array(s, float) / 2, np.array(p[:3], float), R.from_quat(p[3:])) for s, p in bb]
            rec.half = np.array([b[0] for b in state["bb"]])
            rec.module = np.array([i for i, _ in shapes])
            rec.connector_module = np.array([i for i, m in enumerate(genome.modules) if m.parent >= 0])
        c, rm = [], []
        for sh, (_, pbb, rbb) in zip(state["shapes"], state["bb"]):
            p = sim.getObjectPose(sh, sim.handle_world)
            rw = R.from_quat(p[3:])
            c.append(np.array(p[:3]) + rw.apply(pbb))
            rm.append((rw * rbb).as_matrix())
        br = [bool(sim.readForceSensor(fs)[0] & 2) for fs in sensors]
        rec.add((k + 1) * 0.05, c, rm, br)

    return hook


def _mujoco_hook(rec: Recorder, genome: morph.Genome):
    every = int(round(FRAME_DT / 0.002))
    state = {}

    def hook(m, d, k):
        if k % every:
            return
        if not state:
            import mujoco
            gids = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]
            state["gids"] = gids
            state["welds"] = [m.equality(f"w{i}").id for i, mod in enumerate(genome.modules) if mod.parent >= 0]
            rec.half = np.array([m.geom_size[g] for g in gids])
            rec.module = np.array([int(m.body(m.geom_bodyid[g]).name[1:-1]) for g in gids])
            rec.connector_module = np.array([i for i, mod in enumerate(genome.modules) if mod.parent >= 0])
        rec.add(d.time, d.geom_xpos[state["gids"]], d.geom_xmat[state["gids"]].reshape(-1, 3, 3),
                [not d.eq_active[e] for e in state["welds"]])

    return hook


def record(engine: str, genome: morph.Genome, duration: float, world: int = 0, t_start: float = 6.0):
    """run one engine and return (arrays, rollout result). coppeliasim engines need a running instance."""
    rec = Recorder()
    if engine == "sim":
        from emerge import sim_coppelia
        res = sim_coppelia.rollout(genome, duration, t_start, seed=world, after_step=_coppelia_hook(rec, genome))
    elif engine == "real_a":
        from emerge import real_coppelia
        res = real_coppelia.rollout(genome, world, duration, t_start, hook=_coppelia_hook(rec, genome))
    elif engine == "real_b":
        from emerge import real_mujoco
        res = real_mujoco.rollout(genome, world, duration, t_start, hook=_mujoco_hook(rec, genome))
    else:
        raise ValueError(engine)
    return rec.arrays(), res


# ---- drawing ----

def _boxes(a, f):
    """verts per box at frame f: (n,6,4,3)."""
    corners = a["centre"][f][:, None, :] + np.einsum("nij,nkj->nki", a["rot"][f], CORNERS[None] * a["half"][:, None, :])
    return corners[:, FACES, :]


def _colors(a, f):
    cols = []
    broken = set(a["connector_module"][a["broken"][f]].tolist())
    cmap = ["#8a8f98", "#4c78a8", "#59a14f", "#f28e2b", "#b07aa1", "#76b7b2", "#edc948", "#9c755f", "#ff9da7"]
    for mod in a["module"]:
        c = np.array(matplotlib_rgb(cmap[mod % len(cmap)]))
        if mod in broken:
            c = 0.4 * c + 0.6 * np.array([0.85, 0.1, 0.1])
        cols.append((*c, 0.9))
    return cols


def matplotlib_rgb(h):
    import matplotlib.colors as mc
    return mc.to_rgb(h)


def _base_xy(a):
    base = np.where(a["module"] == 0)[0][0]
    return a["centre"][:, base, :2]


def render(traces: dict, results: dict, out: str, fps: int = 20, window: float = 0.35, t_start: float = 6.0, title: str = ""):
    """traces: engine -> arrays. a gif (or mp4 by extension) with one 3d panel per engine, camera following the base."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    names = list(traces)
    n_frames = min(len(a["t"]) for a in traces.values())
    fig = plt.figure(figsize=(4.6 * len(names), 4.8))
    axes, colls, fl = [], [], {}
    for j, name in enumerate(names):
        ax = fig.add_subplot(1, len(names), j + 1, projection="3d")
        ax.set_box_aspect((1, 1, 0.5))
        ax.view_init(elev=24, azim=-60)
        pc = Poly3DCollection(_boxes(traces[name], 0).reshape(-1, 4, 3), edgecolor="k", linewidth=0.4)
        ax.add_collection3d(pc)
        fl[name] = Poly3DCollection([[(0, 0, 0)] * 4], facecolor="#cfcfcf", alpha=0.35, edgecolor="none")
        ax.add_collection3d(fl[name])
        axes.append(ax)
        colls.append(pc)
    fig.suptitle(title, fontsize=10)

    def update(f):
        for name, ax, pc in zip(names, axes, colls):
            a = traces[name]
            pc.set_verts(_boxes(a, f).reshape(-1, 4, 3))
            fc = np.repeat(np.array(_colors(a, f)), 6, axis=0)
            pc.set_facecolor(fc)
            c = _base_xy(a)[f]
            ax.set_xlim(c[0] - window, c[0] + window)
            ax.set_ylim(c[1] - window, c[1] + window)
            ax.set_zlim(0, 2 * window * 0.5)
            fl[name].set_verts([[(c[0] - window, c[1] - window, 0), (c[0] + window, c[1] - window, 0),
                                 (c[0] + window, c[1] + window, 0), (c[0] - window, c[1] + window, 0)]])
            i0 = int(np.searchsorted(a["t"], t_start))
            d = float(np.linalg.norm(_base_xy(a)[f] - _base_xy(a)[min(i0, f)])) if f >= i0 else 0.0
            nb = int(a["broken"][f].sum())
            r = results[name]
            ax.set_title(f"{ENGINES[name]}\nt={a['t'][f]:5.1f}s  moved {d:.3f} m  broken {nb}   (final {r['distance']:.3f} m)", fontsize=8,
                         color="#b01010" if nb else "black")
        return colls

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps)
    writer = animation.FFMpegWriter(fps=fps) if out.endswith(".mp4") else animation.PillowWriter(fps=fps)
    anim.save(out, writer=writer)
    plt.close(fig)


def render_paths(traces: dict, results: dict, out: str, t_start: float = 6.0):
    """top-down base trajectories of all engines, from the start of the fitness window, one png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, a in traces.items():
        xy = _base_xy(a)
        i0 = int(np.searchsorted(a["t"], t_start))
        p = xy[i0:] - xy[min(i0, len(xy) - 1)]
        ax.plot(p[:, 0], p[:, 1], label=f"{name}: {results[name]['distance']:.3f} m, broken {results[name]['broken']}")
        ax.plot(*p[-1], "o")
    ax.plot(0, 0, "k+")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("base path from the fitness window start")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def save_traces(traces, results, path):
    flat = {}
    for name, a in traces.items():
        for k, v in a.items():
            flat[f"{name}/{k}"] = v
        flat[f"{name}/result"] = np.array([results[name]["distance"], results[name]["broken"]])
    np.savez_compressed(path, **flat)


def load_traces(path):
    z = np.load(path)
    traces, results = {}, {}
    for key in z.files:
        name, k = key.split("/")
        if k == "result":
            results[name] = dict(distance=float(z[key][0]), broken=int(z[key][1]))
        else:
            traces.setdefault(name, {})[k] = z[key]
    return traces, results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phases", type=float, nargs="+", default=[0.0, 90.0], help="phase (deg) of each actuated module of a straight chain")
    ap.add_argument("--duration", type=float, default=20.0, help="episode length in s (the fitness window starts at --t-start)")
    ap.add_argument("--t-start", type=float, default=6.0)
    ap.add_argument("--world", type=int, default=0, help="hidden world / noise seed for the reals and the sim")
    ap.add_argument("--only", nargs="+", choices=list(ENGINES), help="engines to run (default all three)")
    ap.add_argument("--out", default=os.path.join(OUT, "chain.gif"), help=".gif or .mp4; a .npz recording and _paths.png go next to it")
    ap.add_argument("--trace", help="redraw from a saved .npz instead of running")
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    stem = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.trace:
        traces, results = load_traces(args.trace)
    else:
        genome = morph.chain(len(args.phases), args.phases)
        traces, results = {}, {}
        for name in args.only or list(ENGINES):
            traces[name], results[name] = record(name, genome, args.duration, args.world, args.t_start)
            r = results[name]
            print(f"{name}: distance {r['distance']:.3f} m, broken {r['broken']}, {len(traces[name]['t'])} frames")
        save_traces(traces, results, stem + ".npz")
    title = f"phases {args.phases} deg" if not args.trace else os.path.basename(args.trace)
    render(traces, results, args.out, args.fps, t_start=args.t_start, title=title)
    render_paths(traces, results, stem + "_paths.png", args.t_start)
    print("wrote", args.out, stem + "_paths.png", stem + ".npz")


if __name__ == "__main__":
    main()
