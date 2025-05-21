"""
Vectorised & JIT timing utilities for :class:`BraxPassiveWalker`.

Run as a script:

    $ python -m passive_walker.brax.benchmarks           # default sizes
    $ python -m passive_walker.brax.benchmarks 32 256    # custom batch sizes

Or import the helpers inside a notebook:

    from passive_walker.brax.benchmarks import *
    env32   = make_vec_env(32)
    ms      = bench(32)
    ps_list = take_batch(env32.reset(jax.random.PRNGKey(0)).pipeline_state)
"""

from __future__ import annotations

import sys, time
from pathlib import Path
from typing import Dict, Sequence

import jax, jax.numpy as jnp
from brax.envs.wrappers.training import VmapWrapper
from brax.io import html as brax_html
from IPython.display import HTML  # safe import outside notebooks

from passive_walker.envs.brax_env import BraxPassiveWalker
from passive_walker.brax import SYSTEM_PICKLE


# -------------------------------------------------------------------------
# helpers – vectorise, JIT, timing
# -------------------------------------------------------------------------
def make_vec_env(num_envs: int):
    """Returns a BraxPassiveWalker wrapped to act on `num_envs` in parallel."""
    base_env = BraxPassiveWalker(walker_sys)
    return VmapWrapper(base_env, batch_size=num_envs)


def jit_env(env):
    """Return (`jit_reset`, `jit_step`) for the provided environment."""
    return jax.jit(env.reset), jax.jit(env.step)


def bench(num_envs: int, *, warmups: int = 1, runs: int = 500) -> float:
    """
    Measure wall-clock *execution* time (JIT already compiled).

    Returns
    -------
    ms_per_step : float
        Milliseconds for **one batched env step**.
    """
    env = make_vec_env(num_envs)
    reset_jit, step_jit = jit_env(env)

    # 1) compile
    _ = reset_jit(jax.random.PRNGKey(0))

    # 2) warm-up runs (optional)
    state = reset_jit(jax.random.PRNGKey(1))
    action = jnp.zeros((num_envs, env.action_size))
    for _ in range(warmups):
        state = step_jit(state, action)

    # 3) timed loop
    t0 = time.time()
    for _ in range(runs):
        state = step_jit(state, action)
    ms = (time.time() - t0) * 1e3 / runs
    return ms


# -------------------------------------------------------------------------
# nice utilities for HTML replay inside notebooks
# -------------------------------------------------------------------------
def take_batch(ps, idx: int = 0):
    """Extract element `idx` from every leaf in the pipeline_state pytree."""
    return jax.tree_util.tree_map(lambda arr: arr[idx], ps)


def html_replay(pipeline_state_batch, height: int = 400) -> HTML:
    """Render a batch of QP’s (usually after a vmap rollout) to HTML."""
    from . import SYSTEM_PICKLE  # lazy import to avoid cost unless needed
    import pickle, gzip, brax
    import pathlib

    # quick loader
    pkl = Path(SYSTEM_PICKLE)
    opener = gzip.open if pkl.suffix == ".gz" else open
    with opener(pkl, "rb") as f:
        sys = pickle.load(f)

    # turn the batch into a list of single QP’s
    N = pipeline_state_batch.x.pos.shape[0]
    ps_list = [take_batch(pipeline_state_batch, i) for i in range(N)]
    return HTML(brax_html.render(sys, ps_list, height=height))


# -------------------------------------------------------------------------
# CLI entry-point
# -------------------------------------------------------------------------
def _as_main(argv: Sequence[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    sizes = list(map(int, argv)) or [1, 32, 128, 512, 1024]

    print("JAX backend :", jax.lib.xla_bridge.get_backend().platform)
    print("Devices     :", jax.devices(), "\n")

    for n in sizes:
        ms = bench(n)
        print(f"{n:5d} envs : {ms:6.2f} ms/step   |  {ms/n*1e3:6.1f} µs/env/step")


if __name__ == "__main__":  # pragma: no cover
    _as_main()
