"""
Run the complete hip+knee behavior cloning pipeline.

This module implements a complete pipeline for training and evaluating a neural network
controller for the passive walker's hip and knee joints. The pipeline consists of:
1. Collecting demonstration data from FSM controller
2. Training neural network via MSE behavior cloning
3. Saving the trained controller
4. Testing in GUI (hip and knees controlled by NN)

Usage:
    python -m passive_walker.bc.hip_knee_mse.run_pipeline [--steps N] [--epochs E]
                                                    [--batch B] [--hidden-size H]
                                                    [--lr LR] [--sim-duration S]
                                                    [--seed SEED] [--hz H] [--gpu] [--plot]
"""

import argparse
import pickle
import numpy as np
from mujoco.glfw import glfw
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from passive_walker.envs.mujoco_fsm_env import PassiveWalkerEnv
from passive_walker.controllers.nn.hip_knee_nn import HipKneeController
from passive_walker.bc.hip_knee_mse.collect import collect_demo_data
from passive_walker.bc.hip_knee_mse.train import train_nn_controller
from passive_walker.bc.hip_knee_mse import DATA_BC_HIP_KNEE_MSE, XML_PATH, set_device
from passive_walker.utils.io import save_pickle


def main():
    """Run the complete behavior cloning pipeline."""
    p = argparse.ArgumentParser(description="Full hip+knee BC pipeline")
    p.add_argument("--steps",         type=int,   default=100_000, help="Demo steps")
    p.add_argument("--epochs",        type=int,   default=100,    help="Training epochs")
    p.add_argument("--batch",         type=int,   default=32,     help="Batch size")
    p.add_argument("--hidden-size",   type=int,   default=256,    help="Hidden layer size")
    p.add_argument("--lr",            type=float, default=5e-5,   help="Learning rate")
    p.add_argument("--sim-duration",  type=float, default=15.0,   help="Test sim duration (s)")
    p.add_argument("--seed",          type=int,   default=42,     help="PRNG seed")
    p.add_argument("--hz",            type=int,   default=1000,    help="Simulation frequency (Hz)")
    p.add_argument("--gpu",           action="store_true",        help="Use GPU if available")
    p.add_argument("--plot",          action="store_true",        help="Plot training loss curve")
    args = p.parse_args()

    # 0. Configure device
    set_device(args.gpu)

    # 1. Check for existing demo data or collect new data
    demo_file = DATA_BC_HIP_KNEE_MSE / f"hip_knee_mse_demos_{args.steps}steps.pkl"
    if demo_file.exists():
        print(f"1) Loading existing demo data from {demo_file}…")
        with open(demo_file, "rb") as f:
            demos = pickle.load(f)
        obs = jnp.array(demos["obs"])
        labels = jnp.array(demos["labels"])
    else:
        print("1) Collecting FSM demos…")
        env_demo = PassiveWalkerEnv(
            xml_path=str(XML_PATH),
            simend=args.steps / float(args.hz),
            use_nn_for_hip=False,
            use_nn_for_knees=False,
            use_gui=False
        )
        obs, labels = collect_demo_data(env_demo, args.steps)
        env_demo.close()
        
        # Save the collected demos
        save_pickle(
            {"obs": np.array(obs, dtype=np.float32),
             "labels": np.array(labels, dtype=np.float32)},
            demo_file
        )
    
    print(f"   Using obs={obs.shape}, labels={labels.shape}")

    # 2. Train
    print("2) Training NN controller via BC…")
    model = HipKneeController(
        input_size=obs.shape[1],
        hidden_size=args.hidden_size,
        output_size=3,  # For hip, left knee, right knee
        key=jax.random.PRNGKey(args.seed)
    )
    optimizer = optax.adam(args.lr)
    model, loss_hist = train_nn_controller(
        model, optimizer, obs, labels,
        epochs=args.epochs, batch=args.batch,
        plot_loss=args.plot
    )

    # 3. Save the trained model and loss history
    out_file = DATA_BC_HIP_KNEE_MSE / f"hip_knee_mse_controller_{args.steps}steps.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(model, f)
    print(f"3) Saved trained controller → {out_file}")

    if args.plot:
        loss_file = DATA_BC_HIP_KNEE_MSE / f"training_loss_history_{args.steps}steps.pkl"
        with open(loss_file, "wb") as f:
            pickle.dump(loss_hist, f)
        print(f"   Saved loss history → {loss_file}")

    # 4. Test the controller in GUI
    print("4) Evaluating in GUI (hip and knees by NN)…")
    env_test = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=args.sim_duration,
        use_nn_for_hip=True,
        use_nn_for_knees=True,
        use_gui=True
    )
    obs = env_test.reset()
    done = False
    total_reward = 0.0

    while not done and not glfw.window_should_close(env_test.window):
        act = np.array(model(jnp.array(obs)))
        obs, rew, done, _ = env_test.step(act)
        total_reward += rew
        env_test.render()

    env_test.close()
    print(f"Episode finished, total reward = {total_reward:.3f}")


if __name__ == "__main__":
    main()
