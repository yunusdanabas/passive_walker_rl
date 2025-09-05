"""
Utility helpers for the Brax sub-package
========================================

* ``summarize_brax_system`` – pretty HTML / DataFrame dump of a
  :class:`brax.System`.
* ``check_and_save``           – quick numeric sanity check **+** gzip-pickle
  to disk (with an optional MuJoCo joint/actuator dump for debugging).
* ``uint64_patch``            – helper for Brax <-> JAX printing / logging.
* ``visualize_in_mujoco``     – visualize learned Brax policy in MuJoCo GUI.

Both functions are *pure helpers* – they never mutate the Brax ``System`` you
pass in.
"""

from __future__ import annotations
from pathlib import Path
import gzip, pickle, sys as _sys
import time
import os
import numpy as np
import pandas as pd
import jax, jax.numpy as jnp
from IPython.display import Markdown, display

from passive_walker.brax import XML_PATH

import brax.training.types as _bt


# ---------------------------------------------------------------------------
# Pretty HTML / DataFrame dump
# ---------------------------------------------------------------------------
def summarize_brax_system(sys, *, style: str = "html"):
    """
    Return three :class:`pandas.DataFrame` objects summarising **joints**,
    **actuators** and **bodies** of a Brax ``System``.

    Parameters
    ----------
    sys
        The Brax :class:`brax.System` to visualise.
    style  : {'html', 'df'}
        ``'html'``  – display nice, coloured tables inside a Jupyter notebook  
        ``'df'``    – *do not* display, simply **return** the three dataframes.

    Returns
    -------
    joints_df, actuators_df, bodies_df : pandas.DataFrame
    """
    joint_type_map = {0: "free", 1: "ball", 2: "slide", 3: "hinge", 4: "weld"}
    names_blob = sys.names
    _name = lambda off: names_blob[off:].split(b"\x00", 1)[0].decode()
    body_names = ["world"] + list(sys.link_names)

    tilt = float(jnp.degrees(jnp.arctan2(-sys.gravity[0], sys.gravity[2])))

    # ---------- joints -----------------------------------------------------
    joints_df = pd.DataFrame(
        {
            "idx": np.arange(sys.njnt),
            "type": [joint_type_map.get(int(t), "?") for t in sys.jnt_type],
            "name": [_name(int(o)) for o in sys.name_jntadr],
            "body": [body_names[int(b)] for b in sys.jnt_bodyid],
            "axis": [tuple(a.tolist()) for a in sys.jnt_axis],
            "limited": sys.jnt_limited.astype(bool),
            "lo": sys.jnt_range[:, 0],
            "hi": sys.jnt_range[:, 1],
        }
    )

    # ---------- actuators --------------------------------------------------
    q2joint = {int(qp): idx for idx, qp in enumerate(sys.jnt_qposadr)}
    rows = []
    for i, (lo, hi) in enumerate(sys.actuator.ctrl_range):
        qid, jidx = int(sys.actuator.q_id[i]), q2joint[int(sys.actuator.q_id[i])]
        rows.append(
            dict(
                idx=i,
                name=_name(int(sys.name_actuatoradr[i]))
                if hasattr(sys, "name_actuatoradr")
                else f"act_{i}",
                joint=joints_df.loc[jidx, "name"],
                body=joints_df.loc[jidx, "body"],
                ctrl_lo=lo,
                ctrl_hi=hi,
                unit="rad" if i == 0 else "m",
            )
        )
    actuators_df = pd.DataFrame(rows)

    # ---------- bodies -----------------------------------------------------
    bodies_df = pd.DataFrame(
        {
            "idx": np.arange(len(sys.link_names)),
            "name": sys.link_names,
            "parent": [
                "world" if p == -1 else sys.link_names[p] for p in sys.link_parents
            ],
            "mass": sys.link.inertia.mass,
            "inertia": [tuple(np.diag(I)) for I in sys.link.inertia.i],
        }
    )

    # ---------- optional pretty-print --------------------------------------
    if style == "html":
        display(Markdown("## ⟪ Brax System Summary ⟫"))
        print("‾" * 45)
        print(f" bodies           : {sys.link_names}")
        print(f" nq | nv | nu     : {sys.nq} | {sys.nv} | {sys.nu}")
        print(f" gravity tilt     : {tilt:+.1f} deg")
        display(Markdown("### Bodies"));    display(bodies_df.style.hide(axis="index"))
        display(Markdown("### Joints"));    display(joints_df.style.hide(axis="index"))
        display(Markdown("### Actuators")); display(actuators_df.style.hide(axis="index"))

    return joints_df, actuators_df, bodies_df


# ---------------------------------------------------------------------------
# Sanity-check & pickle helper
# ---------------------------------------------------------------------------
def check_and_save(sys, out: Path, *, verbose: bool = True) -> None:
    """
    * Checks*: finite inertia + positive masses  
    * Side-effect*: gzip-pickle the Brax ``System`` to *out*.

    Parameters
    ----------
    sys   : brax.System
    out   : pathlib.Path
        Destination (``.pkl`` or ``.pkl.gz`` – compression auto-detected).
    verbose : bool
        If *True*, prints per-joint type/name dump **and** a success banner.
    """
    # 1 – verbose dump of MuJoCo joint codes → names
    if verbose:
        print("Raw MuJoCo joint codes ➜ names")
        for idx, (code, name_off) in enumerate(
            zip(sys.jnt_type, sys.name_jntadr, strict=False)
        ):
            name = sys.names[name_off:].split(b"\0", 1)[0]
            print(f"  {idx:2d}  code={int(code)}  name={name.decode()}")

    # 2 – numeric sanity checks
    assert np.all(np.isfinite(sys.link.inertia.i)), "non-finite inertia detected"
    assert np.all(
        sys.link.inertia.mass > 0
    ), "found link(s) with zero or negative mass"

    # 3 – serialise
    out.parent.mkdir(parents=True, exist_ok=True)
    pickler = gzip.open if out.suffix == ".gz" else open
    with pickler(out, "wb") as f:
        pickle.dump(sys, f)

    if verbose:
        print("Basic numeric sanity checks passed ✓")
        print(f"System pickled to: {out.resolve()}")
        print("-" * 40)


# ---------------------------------------------------------------------------
# Brax <-> JAX printing / logging helpers
# ---------------------------------------------------------------------------
def uint64_patch():
    """Patch Brax's UInt64 type for better JAX compatibility."""
    def _uint64_to_numpy(self):
        return (int(self.hi) << 32) | int(self.lo)

    _bt.UInt64.to_numpy = _uint64_to_numpy          # type: ignore[attr-defined]
    _bt.UInt64.__int__  = lambda self: _uint64_to_numpy(self)     # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────────────────
# Sweep-results plotting helpers
# ──────────────────────────────────────────────────────────────────────────
import pandas as _pd, matplotlib.pyplot as _plt
from passive_walker.brax import RESULTS_BRAX, DATA_BRAX

def load_sweep_df(csv: str | Path | None = None) -> _pd.DataFrame:
    """
    Load the aggregated CSV produced by ``sweep_ppo.py``.
    """
    csv = Path(csv) if csv else RESULTS_BRAX / "sweep_agg.csv"
    if not csv.exists():
        raise FileNotFoundError(f"sweep CSV not found: {csv}")
    return _pd.read_csv(csv)

def barplot_mean_reward(df: _pd.DataFrame | None = None,
                        *, save: bool | str = True):
    """
    Bar-plot **mean ± std** reward for every (reward_scale, lr) combo.

    Parameters
    ----------
    df   : DataFrame returned by :func:`load_sweep_df`
    save : ``True`` → write to *results/brax/sweep_barplot.png* or give
           an explicit filepath.  ``False`` → no file output.
    """
    if df is None:
        df = load_sweep_df()
    pl = (df.groupby(["reward_scale","lr"], as_index=False)
            .agg(mean_reward=("reward","mean"),
                 std_reward =("reward","std")))

    ax = pl.pivot(index="reward_scale", columns="lr",
                  values="mean_reward").plot.bar(
            yerr = pl.pivot(index="reward_scale", columns="lr",
                            values="std_reward"),
            capsize=3, rot=0, figsize=(6,4))
    ax.set_xlabel("Reward scaling"); ax.set_ylabel("Final episode reward")
    ax.set_title("Passive Walker — PPO sweep")
    _plt.tight_layout()
    if save:
        fp = (RESULTS_BRAX / "sweep_barplot.png") if save is True else Path(save)
        _plt.savefig(fp, dpi=150); print("saved →", fp)
    _plt.show()

def heatmap_lr_arch(df: _pd.DataFrame | None = None,
                    *, save: bool | str = False, cmap="viridis"):
    """
    Convenience heat-map: **learning-rate × network-arch** → mean reward.
    """
    if df is None:
        df = load_sweep_df()
    hm = (df.groupby(["arch","lr"], as_index=False)
            .agg(mean_reward=("reward","mean")))
    pivot = hm.pivot(index="arch", columns="lr", values="mean_reward")
    _plt.figure(figsize=(8,3))
    _plt.imshow(pivot.values, aspect="auto", cmap=cmap,
                origin="lower", interpolation="nearest")
    _plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    _plt.yticks(range(len(pivot.index)),   pivot.index)
    _plt.colorbar(label="mean reward")
    _plt.title("Reward heat-map")
    _plt.tight_layout()
    if save:
        fp = (RESULTS_BRAX / "sweep_heatmap.png") if save is True else Path(save)
        _plt.savefig(fp, dpi=150); print("saved →", fp)
    _plt.show()


# ---------------------------------------------------------------------------
# MuJoCo visualization helper
# ---------------------------------------------------------------------------
def visualize_in_mujoco(
    policy_fn,
    xml_path: str,
    duration_s: float = 10.0,
    rng_seed: int = 0,
    use_nn_for_hip: bool = True,
    use_nn_for_knees: bool = True,
):
    """
    Launch the given policy in MuJoCo for a GUI rollout.

    Args:
      policy_fn:           Callable(obs: jnp.ndarray, key) → (action, _)
      xml_path:            Path to passiveWalker_model.xml
      duration_s:          Simulation time (seconds)
      rng_seed:            RNG seed for env reset
      use_nn_for_hip:      Whether to use NN control on the hip
      use_nn_for_knees:    Whether to use NN control on the knees
    """
    import time
    import numpy as np
    import jax, jax.numpy as jnp
    from mujoco.glfw import glfw
    from passive_walker.envs.mujoco_env import PassiveWalkerEnv

    # 1) build the env & force window/context creation
    mj_env = PassiveWalkerEnv(
        xml_path=str(xml_path),
        simend=duration_s,
        use_nn_for_hip=use_nn_for_hip,
        use_nn_for_knees=use_nn_for_knees,
        use_gui=True,
        rng_seed=rng_seed,
    )
    # do one render to ensure the GLFW window and context are initialized
    obs = mj_env.reset()
    mj_env.render()

    # 2) rollout loop
    key = jax.random.PRNGKey(rng_seed)
    done = False
    total_reward = 0.0
    t0 = time.time()

    while (not done
           and mj_env.window is not None
           and not glfw.window_should_close(mj_env.window)):
        key, sub = jax.random.split(key)

        # policy_fn may return (action, extras)
        out = policy_fn(jnp.array(obs), sub)
        act_jax = out[0] if isinstance(out, (tuple, list)) else out
        
        # Ensure action is 1D array
        if isinstance(act_jax, (tuple, list)):
            act_jax = jnp.array(act_jax)
        act_jax = jnp.squeeze(act_jax)
        
        # Convert to numpy and ensure correct shape
        act = np.asarray(act_jax, dtype=np.float32)
        if act.ndim > 1:
            act = act.flatten()

        obs, reward, done, _ = mj_env.step(act)
        total_reward += float(reward)

        mj_env.render(mode="human")

    elapsed = time.time() - t0
    print(f"\nEpisode finished in {elapsed:.1f}s  |  total reward = {total_reward:.2f}")

    # 3) clean up
    mj_env.close()
