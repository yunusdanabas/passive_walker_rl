"""
Large-scale PPO hyper-parameter sweep for Brax Passive Walker
============================================================

Run **the whole 120-job grid** (≈ few minutes on a good GPU):

    python -m passive_walker.brax.sweep_ppo

Run **only one combination** (e.g. deepXL arch, seed 2):

    python -m passive_walker.brax.sweep_ppo  --arch deepXL --seed 2

After the sweep finishes you will have:

data/brax/
└─ sweep_results/
   ├─ run_<hash>.msgpack     ← 120 serialized runs (metrics + params)
   └─ best_policy.msgpack    ← network bytes of the best model

results/brax/
└─ sweep_barplot.png         ← aggregation figure
"""

from __future__ import annotations
import argparse, hashlib, json, time, gzip, pickle
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any

import jax, jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import flax, msgpack

import mujoco
from brax.io import mjcf

import brax
from brax.training import types
from brax.training.agents.ppo import train as ppo_train
from brax.training.agents.ppo import networks as ppo_nets

from passive_walker.brax import RESULTS_BRAX, DATA_BRAX, XML_PATH
from passive_walker.envs.brax_env import BraxPassiveWalker

from passive_walker.brax.utils import visualize_in_mujoco

# -------------------------------------------------------------------------
# tiny UInt64 → numpy patch (Brax 0.10+)
import brax.training.types as _bt
_bt.UInt64.to_numpy = lambda self: (int(self.hi) << 32) | int(self.lo)     # type: ignore
_bt.UInt64.__int__  = _bt.UInt64.to_numpy                                  # type: ignore

# -------------------------------------------------------------------------
# sweep definition ---------------------------------------------------------
@dataclass(frozen=True)
class SweepConfig:
    seed: int
    reward_scale: float
    lr: float
    arch: str

    def hash(self) -> str:
        return hashlib.sha1(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()[:8]
        
    @property
    def tag(self) -> str:
        return f"arch={self.arch} seed={self.seed} rs={self.reward_scale} lr={self.lr}"

# grid
SEEDS          = [0, 1, 2]
REWARD_SCALES  = [0.5, 1.0]
LRS            = [1e-3, 5e-4, 1e-4, 1e-5]
ARCHS          = ["tiny", "small", "medium", "deep", "deepXL"]

GRID = [SweepConfig(s, rs, lr, a)
        for s, rs, lr, a in product(SEEDS, REWARD_SCALES, LRS, ARCHS)]

# -------------------------------------------------------------------------
# network factory ----------------------------------------------------------
def make_networks(observation_size: int,
                  action_size: int,
                  cfg: SweepConfig,
                  preprocess_fn=types.identity_observation_preprocessor):

    match cfg.arch:
        case "tiny":   ph, vh = (64, 64),               (128, 128, 128)
        case "small":  ph, vh = (128,)*4,               (256,)*6
        case "medium": ph, vh = (256,)*6,               (512,)*8
        case "deep":   ph, vh = (512,)*6,               (1024,)*8
        case "deepXL": ph, vh = (512,)*12,              (1024,)*14
        case other:
            raise ValueError(f"Unknown arch {other}")

    return ppo_nets.make_ppo_networks(
        observation_size           = observation_size,
        action_size                = action_size,
        policy_hidden_layer_sizes  = ph,
        value_hidden_layer_sizes   = vh,
        preprocess_observations_fn = preprocess_fn,
        activation                 = jax.nn.tanh,
    )

# -------------------------------------------------------------------------
# single run ---------------------------------------------------------------
NUM_TIMESTEPS   =  1024        # 24 M
NUM_ENVS        = 128
EPISODE_LENGTH  = 1024
BATCH_SIZE      = 4096

SWEEP_DIR = DATA_BRAX / "sweep_results"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)

def run_one(cfg: SweepConfig, walker_sys: brax.System, overwrite: bool = False) -> Path:
    out_file = SWEEP_DIR / f"run_{cfg.hash()}.msgpack"
    if out_file.exists() and not overwrite:
        return out_file

    env = BraxPassiveWalker(walker_sys)
    net_factory = lambda obs, act, **_: make_networks(obs, act, cfg)

    policy_fn, params, metrics = ppo_train.train(
        environment     = env,
        network_factory = net_factory,
        reward_scaling  = cfg.reward_scale,
        learning_rate   = cfg.lr,
        num_timesteps   = NUM_TIMESTEPS,
        num_envs        = NUM_ENVS,
        episode_length  = EPISODE_LENGTH,
        batch_size      = BATCH_SIZE,
        entropy_cost    = 1e-3,
        wrap_env        = True,
        num_evals       = 1,
        seed            = cfg.seed,
        progress_fn     = lambda *_: None,   # silent
    )

    # flatten metrics to python scalars
    metrics_py = jax.tree_util.tree_map(lambda x: float(x) if isinstance(x, jax.Array) else x,
                                        metrics)

    packed = {
        "cfg"         : asdict(cfg),
        "metrics"     : metrics_py,
        "params_bytes": flax.serialization.to_bytes(params),
    }
    with open(out_file, "wb") as f:
        msgpack.pack(packed, f, use_bin_type=True)

    return out_file

# -------------------------------------------------------------------------
# aggregation & plotting ---------------------------------------------------
def aggregate():
    files = list(SWEEP_DIR.glob("run_*.msgpack"))
    rows  = []
    for f in files:
        d = msgpack.unpack(open(f, "rb"), raw=False)
        reward = (
            d["metrics"].get("evaluation/episode_reward")
            or d["metrics"].get("eval/episode_reward")
            or next(iter(d["metrics"].values()))
        )
        rows.append({**d["cfg"], "reward": reward, "file": str(f)})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_BRAX / "sweep_agg.csv", index=False)

    # barplot (reward_scale × lr, average over seeds)
    plot_df = (df.groupby(["reward_scale","lr"], as_index=False)
                 .agg(mean_reward=("reward","mean"), std_reward=("reward","std")))

    ax = plot_df.pivot(index="reward_scale",
                       columns="lr",
                       values="mean_reward").plot.bar(
            yerr = plot_df.pivot(index="reward_scale",
                                 columns="lr",
                                 values="std_reward"),
            rot=0, capsize=3, figsize=(6,4))
    ax.set_ylabel("Final reward")
    ax.set_title("Passive Walker – PPO sweep")
    plt.tight_layout()
    plt.savefig(RESULTS_BRAX / "sweep_barplot.png", dpi=150)
    plt.close()

    # best policy bytes
    best_file = df.loc[df.reward.idxmax(), "file"]
    best_data = msgpack.unpack(open(best_file, "rb"), raw=False)
    with open(DATA_BRAX / "best_policy.msgpack", "wb") as f:
        msgpack.pack(best_data, f, use_bin_type=True)
    print("✓ aggregation done – best run:", best_file)


# ---------------------------------------------------------------------------
# μ-GUI *** BEST POLICY REPLAY **********************************************
# ---------------------------------------------------------------------------
def replay_best_in_mujoco(duration_s: float = 10.0):
    """
    Loads `best_policy.msgpack` (written by sweep_ppo.aggregate) and visualises
    one episode in MuJoCo's OpenGL viewer.

    Parameters
    ----------
    duration_s : float
        How long to run the episode in simulated seconds.
    """
    import msgpack, flax, jax, numpy as np, jax.numpy as jnp, time, warnings
    from mujoco.glfw import glfw
    from passive_walker.envs.mujoco_env import PassiveWalkerEnv
    from passive_walker.brax.utils        import uint64_patch                # <- already defined
    from passive_walker.brax.utils        import visualize_in_mujoco        # <- if you prefer

    uint64_patch()          # just in case

    BEST_FILE = DATA_BRAX / "best_policy.msgpack"
    if not BEST_FILE.exists():
        print("No best_policy.msgpack found – run the sweep first.")
        return

    # --- load -------------------------------------------------------------
    payload = msgpack.unpack(open(BEST_FILE, "rb"), raw=False)
    cfg       = SweepConfig(**payload["cfg"])
    net_bytes = payload["params_bytes"]

    print("Loaded best run:", cfg.tag)

    # --- rebuild Brax policy ---------------------------------------------
    # 1) build the BraxSystem again
    mj_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    walker_sys = mjcf.load_model(mj_model)

    env = BraxPassiveWalker(walker_sys)
    # Get observation and action sizes from environment
    obs_size = env.observation_size
    act_size = env.action_size

    nets = make_networks(obs_size, act_size, cfg)
    policy_factory = ppo_nets.make_inference_fn(nets)

    # dummy params tree – same structure as saved bytes
    dummy_params = (
        None,
        nets.policy_network.init(jax.random.PRNGKey(0)),
        nets.value_network.init(jax.random.PRNGKey(1)),
    )
    params = flax.serialization.from_bytes(dummy_params, net_bytes)
    policy = policy_factory(params, deterministic=True)

    # --- MuJoCo replay ----------------------------------------------------
    print("Launching MuJoCo viewer …")
    visualize_in_mujoco(
        policy_fn=lambda obs, key: (jnp.squeeze(policy(obs, key_sample=key)[0]), None),  # extract action from tuple
        xml_path=str(XML_PATH),
        duration_s=duration_s,
        rng_seed=cfg.seed,
        use_nn_for_hip=True,
        use_nn_for_knees=True
    )


# -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",          type=int,   help="Run only this seed")
    p.add_argument("--reward-scale",  type=float, help="Run only this reward scaling")
    p.add_argument("--lr",            type=float, help="Run only this learning rate")
    p.add_argument("--arch",          type=str,   help="Run only this network arch")
    p.add_argument("--overwrite",     action="store_true",
                   help="Overwrite existing *.msgpack files")
    p.add_argument("--play",          action="store_true",
                   help="Replay the best policy in MuJoCo when sweep finishes")
    args = p.parse_args()

    # Load the MuJoCo XML file and convert it to a Brax System
    assert XML_PATH.exists(), f'XML file not found: {XML_PATH.resolve()}'
    mj_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    walker_sys = mjcf.load_model(mj_model)
    print('MuJoCo XML successfully loaded and converted to Brax System')

    subset = [cfg for cfg in GRID
              if (args.seed          is None or cfg.seed == args.seed)
              and (args.reward_scale is None or cfg.reward_scale == args.reward_scale)
              and (args.lr           is None or cfg.lr == args.lr)
              and (args.arch         is None or cfg.arch == args.arch)]

    if not subset:
        print("→ running full grid:", len(GRID), "jobs")
        subset = GRID
    else:
        print("→ running subset:", len(subset), "job(s)")

    for cfg in tqdm(subset, desc="sweep", unit="run"):
        t0 = time.time()
        run_one(cfg, walker_sys, overwrite=args.overwrite)
        print(f"   {cfg.tag}  ({time.time()-t0:.1f}s)")

    aggregate()
    
    # Generate visualizations
    from passive_walker.brax.utils import load_sweep_df, barplot_mean_reward, heatmap_lr_arch
    df = load_sweep_df()
    print("\nGenerating visualizations...")
    barplot_mean_reward(df)                 # shows & saves bar-plot
    heatmap_lr_arch(df, save=True)          # shows & saves heat-map
    
    print("✓ sweep complete.  Results →", RESULTS_BRAX)

    if args.play:
        replay_best_in_mujoco(duration_s=10.0)


if __name__ == "__main__":
    main()
