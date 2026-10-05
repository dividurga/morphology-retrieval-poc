"""run coppeliasim rollouts in parallel over the instances from emerge/start_pool.sh (ports 23200 ..).
each worker process binds to one instance.

    from emerge import pool
    results = pool.map("emerge.sim_fast:rollout", [dict(genome=g, seed=s), ...], n_workers=8)
"""

import importlib
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

BASE_PORT = int(os.environ.get("COPPELIA_BASE_PORT", 23200))


def _init(counter, n):
    with counter.get_lock():
        k = counter.value
        counter.value += 1
    os.environ["COPPELIA_ZMQ_PORT"] = str(BASE_PORT + k % n)


def _run(args):
    target, kwargs = args
    mod, fn = target.split(":")
    return getattr(importlib.import_module(mod), fn)(**kwargs)


def map(target: str, tasks: list, n_workers: int = 8):
    """target 'module:function'. tasks: list of kwargs dicts. returns results in order."""
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    with ProcessPoolExecutor(n_workers, mp_context=ctx, initializer=_init, initargs=(counter, n_workers)) as ex:
        return list(ex.map(_run, [(target, t) for t in tasks]))
