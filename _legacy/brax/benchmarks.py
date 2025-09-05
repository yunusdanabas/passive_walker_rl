# NEW / CHANGED – fixed walker_sys reference + tiny import clean-ups.

from __future__ import annotations
import sys, time
from pathlib import Path
from typing import Sequence
import mujoco
from brax.io import mjcf
import jax, jax.numpy as jnp
from brax.envs.wrappers.training import VmapWrapper
from brax.io import html as brax_html
from IPython.display import HTML

from passive_walker.envs.brax_env import BraxPassiveWalker

from passive_walker.constants import XML_PATH
from passive_walker.brax.utils import uint64_patch


# --------------------------------------------------------------------- utils
def make_vec_env(num_envs: int, walker_sys: mjcf.MjcfModel | None = None) -> BraxPassiveWalker:
    """Returns a BraxPassiveWalker wrapped to act on `num_envs` in parallel."""
    base_env = BraxPassiveWalker(walker_sys)
    return VmapWrapper(base_env, batch_size=num_envs)


def jit_env(env):
    return jax.jit(env.reset), jax.jit(env.step)


def bench(num_envs: int, *, runs: int = 500, walker_sys) -> float:
    env = make_vec_env(num_envs, walker_sys=walker_sys)
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


def html_replay(pipeline_state_batch, height: int = 400,  walker_sys: mjcf.MjcfModel | None = None) -> HTML:
    ps_list = [_take(pipeline_state_batch, i)
               for i in range(pipeline_state_batch.x.pos.shape[0])]
    return HTML(brax_html.render(walker_sys, ps_list, height=height))


# ---------------- CLI -----------------------------------------------------
def _as_main(argv: Sequence[str] | None = None):

    import os
    os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"

    # Load into a raw MuJoCo model
    mj_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    # Convert MuJoCo model → Brax System (JAX‑native datastructure)
    walker_sys = mjcf.load_model(mj_model)
    print('MuJoCo XML successfully loaded and converted to Brax System')

    sizes = list(map(int, argv or sys.argv[1:])) or [1, 32, 128, 512, 1024]
    print("JAX backend :", jax.lib.xla_bridge.get_backend().platform)
    print("Devices     :", jax.devices(), "\n")
    for n in sizes:
        ms = bench(n, walker_sys=walker_sys)
        print(f"{n:5d} envs : {ms:6.2f} ms/step   |  {ms/n*1e3:6.1f} µs/env/step")


if __name__ == "__main__":   # pragma: no-cover
    _as_main()
