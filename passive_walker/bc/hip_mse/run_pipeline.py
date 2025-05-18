# passive_walker/bc/hip_mse/run_pipeline.py
"""
Run the complete hip-only behavior cloning pipeline.

This module implements a complete pipeline for training and evaluating a neural network
controller for the passive walker's hip joint. The pipeline consists of:
1. Collecting demonstration data from FSM controller
2. Training neural network via MSE behavior cloning
3. Saving the trained controller
4. Testing in GUI (hip controlled by NN, knees by FSM)

Usage:
    python -m passive_walker.bc.hip_mse.run_pipeline [--steps N] [--epochs E]
                                                    [--batch B] [--hidden-size H]
                                                    [--lr LR] [--sim-duration S]
                                                    [--seed SEED] [--gpu] [--plot]
"""

import os
import pickle
import numpy as np
from mujoco.glfw import glfw

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from passive_walker.envs.mujoco_fsm_env import PassiveWalkerEnv
from passive_walker.controllers.nn.hip_nn import HipController
from passive_walker.bc.hip_mse.train import train_nn_controller
from passive_walker.bc.hip_mse.collect import collect_demo_data
from . import DATA_DIR, XML_PATH, set_device

def main():
    """Run the complete behavior cloning pipeline."""
    import argparse
    p = argparse.ArgumentParser(description="Full hip-only BC pipeline")
    p.add_argument("--steps",         type=int,   default=20_000, help="Demo steps")
    p.add_argument("--epochs",        type=int,   default=50,    help="Training epochs")
    p.add_argument("--batch",         type=int,   default=32,     help="Batch size")
    p.add_argument("--hidden-size",   type=int,   default=128,    help="Hidden layer size")
    p.add_argument("--lr",            type=float, default=1e-4,   help="Learning rate")
    p.add_argument("--sim-duration",  type=float, default=30.0,   help="Test sim duration (s)")
    p.add_argument("--seed",          type=int,   default=42,     help="PRNG seed")
    p.add_argument("--gpu",           action="store_true",        help="Use GPU if available")
    p.add_argument("--plot",          action="store_true",        help="Plot training loss curve")
    args = p.parse_args()

    # 0. Configure device
    set_device(args.gpu)

    # 1. Collect demonstration data
    print("1) Collecting FSM demos…")
    env_demo = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=30,
        use_nn_for_hip=False,
        use_gui=False,
    )
    demo_obs, demo_labels = collect_demo_data(env_demo, num_steps=args.steps)
    env_demo.close()
    print(f"   Collected obs={demo_obs.shape}, labels={demo_labels.shape}")

    # 2. Instantiate & train NN controller
    print("2) Training NN controller via BC…")
    input_size = demo_obs.shape[1]
    key = jax.random.PRNGKey(args.seed)
    nn_controller = HipController(
        input_size=input_size,
        hidden_size=args.hidden_size,
        output_size=1,
        key=key
    )
    optimizer = optax.adam(args.lr)

    nn_controller, loss_history = train_nn_controller(
        nn_controller,
        optimizer,
        demo_obs,
        demo_labels,
        num_epochs=args.epochs,
        batch_size=args.batch,
        plot_loss=args.plot,
    )

    # 3. Save the trained model and loss history
    out_file = DATA_DIR / "hip_mse_controller.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(nn_controller, f)
    print(f"3) Saved trained controller → {out_file}")

    if args.plot:
        loss_file = DATA_DIR / "training_loss_history.pkl"
        with open(loss_file, "wb") as f:
            pickle.dump(loss_history, f)
        print(f"   Saved loss history → {loss_file}")

    # 4. Test the controller in GUI
    print("4) Evaluating in GUI (hip by NN, knees by FSM)…")
    env_test = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=args.sim_duration,
        use_nn_for_hip=True,
        use_gui=True,
    )
    obs = env_test.reset()
    done = False
    total_reward = 0.0

    while not done and not glfw.window_should_close(env_test.window):
        act = np.array(nn_controller(jnp.array(obs)))
        obs, rew, done, _ = env_test.step(act)
        total_reward += rew
        env_test.render()

    env_test.close()
    print(f"Episode finished, total reward = {total_reward:.3f}")

if __name__ == "__main__":
    main()
