# NEW / CHANGED – fixed walker_sys reference + tiny import clean-ups.

from __future__ import annotations
import sys, time
from pathlib import Path
from typing import Sequence

import jax, jax.numpy as jnp
from brax.envs.wrappers.training import VmapWrapper
from brax.io import html as brax_html
from IPython.display import HTML

from passive_walker.envs.brax_env import BraxPassiveWalker
from passive_walker.brax import WALKER_SYS


# --------------------------------------------------------------------- utils
def make_vec_env(num_envs: int):
    base_env = BraxPassiveWalker(WALKER_SYS)
    return VmapWrapper(base_env, batch_size=num_envs)


def jit_env(env):
    return jax.jit(env.reset), jax.jit(env.step)


def bench(num_envs: int, *, runs: int = 500) -> float:
    env = make_vec_env(num_envs)
    reset_jit, step_jit = jit_env(env)

    _ = reset_jit(jax.random.PRNGKey(0))          # compile
    state  = reset_jit(jax.random.PRNGKey(1))
    action = jnp.zeros((num_envs, env.action_size))

    t0 = time.time()
    for _ in range(runs):
        state = step_jit(state, action)
    return (time.time() - t0) * 1e3 / runs        # ms / batched-step


# ---------------- HTML replay helper --------------------------------------
def _take(ps, idx: int):
    return jax.tree_map(lambda a: a[idx], ps)


def html_replay(pipeline_state_batch, height: int = 400) -> HTML:
    ps_list = [_take(pipeline_state_batch, i)
               for i in range(pipeline_state_batch.x.pos.shape[0])]
    return HTML(brax_html.render(WALKER_SYS, ps_list, height=height))


# ---------------- CLI -----------------------------------------------------
def _as_main(argv: Sequence[str] | None = None):
    sizes = list(map(int, argv or sys.argv[1:])) or [1, 32, 128, 512, 1024]
    print("JAX backend :", jax.lib.xla_bridge.get_backend().platform)
    print("Devices     :", jax.devices(), "\n")
    for n in sizes:
        ms = bench(n)
        print(f"{n:5d} envs : {ms:6.2f} ms/step   |  {ms/n*1e3:6.1f} µs/env/step")


if __name__ == "__main__":   # pragma: no-cover
    _as_main()
