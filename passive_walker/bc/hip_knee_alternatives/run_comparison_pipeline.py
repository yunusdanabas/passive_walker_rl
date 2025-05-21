# passive_walker/bc/hip_knee_alternatives/run_comparison_pipeline.py
"""
Run and compare all hip+knee BC variants (MSE, Huber, L1, Combined), plot losses,
evaluate each in Mujoco GUI, and play out the best-performing controller.

Usage:
    python -m passive_walker.bc.hip_knee_alternatives.run_comparison_pipeline \
        [--steps N] [--hz H] [--gpu]
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from mujoco.glfw import glfw
from pathlib import Path
import seaborn as sns
import pickle

from passive_walker.envs.mujoco_fsm_env import PassiveWalkerEnv
from passive_walker.bc.hip_knee_alternatives import (
    DATA_BC_HIP_KNEE_ALTERNATIVES,
    XML_PATH,
    set_device,
    load_demo_data,
    train_model,
    save_model,
    load_model,
)


def plot_comparison(results, rewards, out_dir: Path):
    """
    Create comparison plots for different methods.
    
    Args:
        results: Dictionary containing training results for each method
        rewards: Dictionary containing evaluation rewards for each method
        out_dir: Directory to save the plots
    """
    # Set style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Training Loss Comparison
    ax1 = plt.subplot(2, 2, 1)
    for name, res in results.items():
        if "loss_history" in res:
            ax1.plot(res["loss_history"], label=name.upper(), alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Comparison")
    ax1.legend()
    ax1.grid(True)
    
    # 2. Reward Comparison
    ax2 = plt.subplot(2, 2, 2)
    methods = list(rewards.keys())
    reward_values = [rewards[m] for m in methods]
    bars = ax2.bar(methods, reward_values)
    ax2.set_xlabel("Method")
    ax2.set_ylabel("Total Reward")
    ax2.set_title("Evaluation Reward Comparison")
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    # 3. Action Distribution Comparison
    ax3 = plt.subplot(2, 2, 3)
    for name, res in results.items():
        if "action_dist" in res:
            # Flatten the action distribution for plotting
            flat_actions = res["action_dist"].flatten()
            sns.kdeplot(data=flat_actions, label=name.upper(), ax=ax3)
    ax3.set_xlabel("Action Value")
    ax3.set_ylabel("Density")
    ax3.set_title("Action Distribution Comparison")
    ax3.legend()
    
    # 4. Training Time Comparison
    ax4 = plt.subplot(2, 2, 4)
    training_times = [res.get("training_time", 0) for res in results.values()]
    bars = ax4.bar(methods, training_times)
    ax4.set_xlabel("Method")
    ax4.set_ylabel("Training Time (s)")
    ax4.set_title("Training Time Comparison")
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom')
    
    # Adjust layout and save
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "method_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def rollout(model, sim_duration, use_gpu):
    """Run one Mujoco GUI rollout and return total reward."""
    env = PassiveWalkerEnv(
        xml_path=str(XML_PATH),
        simend=sim_duration,
        use_nn_for_hip=True,
        use_nn_for_knees=True,
        use_gui=True
    )
    obs = env.reset()
    done = False
    total = 0.0
    actions = []  # Store actions for distribution analysis
    
    while not done and not glfw.window_should_close(env.window):
        act = np.array(model(jnp.array(obs)))
        actions.append(act)
        obs, r, done, _ = env.step(act)
        total += r
        env.render()
    env.close()
    return total, np.array(actions)


def load_or_train_model(name: str, obs, labels, steps: int, loss_type: str):
    """
    Load existing model if available, otherwise train a new one.
    
    Args:
        name: Name of the model variant
        obs: Observation data
        labels: Label data
        steps: Number of steps used for training
        loss_type: Type of loss function to use
    
    Returns:
        Tuple of (model, loss_history, training_time)
    """
    model_file = DATA_BC_HIP_KNEE_ALTERNATIVES / f"hip_knee_alternatives_{loss_type}_{steps}steps.eqx"
    history_file = DATA_BC_HIP_KNEE_ALTERNATIVES / f"hip_knee_alternatives_{loss_type}_{steps}steps_history.pkl"
    
    if model_file.exists() and history_file.exists():
        print(f"[pipeline] Loading existing {name.upper()} model from {model_file}")
        model = load_model(model_file)
        with open(history_file, 'rb') as f:
            loss_history = pickle.load(f)
        training_time = 0.0  # No training time for loaded models
    else:
        print(f"[pipeline] Training new {name.upper()} model...")
        import time
        start_time = time.time()
        model, loss_history = train_model(obs, labels, loss_type=loss_type)
        training_time = time.time() - start_time
        
        # Save the model and history
        save_model(model, model_file)
        with open(history_file, 'wb') as f:
            pickle.dump(loss_history, f)
        print(f"[pipeline] Saved {name.upper()} model to {model_file}")
    
    return model, loss_history, training_time


def main():
    """Main function to run the comparison pipeline."""
    p = argparse.ArgumentParser(
        description="Run comparison pipeline for all BC variants"
    )
    p.add_argument(
        "--steps", type=int, default=20_000,
        help="Number of simulation steps to use for training"
    )
    p.add_argument(
        "--hz", type=int, default=200,
        help="Simulation frequency in Hz"
    )
    p.add_argument(
        "--gpu", action="store_true",
        help="Use GPU if available (sets JAX_PLATFORM_NAME=gpu)"
    )
    p.add_argument(
        "--sim-duration", type=float, default=15.0,
        help="GUI simulation duration in seconds"
    )
    p.add_argument(
        "--force-retrain", action="store_true",
        help="Force retraining even if models exist"
    )
    args = p.parse_args()

    # configure JAX backend
    set_device(args.gpu)

    # 1) load demonstration data
    print("\n=== Step 1: Loading demos ===")
    demo_file = DATA_BC_HIP_KNEE_ALTERNATIVES / f"hip_knee_alternatives_demos_{args.steps}steps.pkl"
    if not demo_file.exists():
        print(f"[pipeline] No demo file found with {args.steps} steps → {demo_file}")
        return

    print(f"[pipeline] loading demos from {demo_file}")
    obs, labels = load_demo_data(demo_file)
    print(f"[pipeline] loaded {len(obs)} samples → obs={obs.shape}, labels={labels.shape}")

    # 2) train or load each variant
    print("\n=== Step 2: Loading/Training variants ===")
    variants = ["mse", "huber", "l1", "combined"]
    results = {}
    for name in variants:
        print(f"\nProcessing {name.upper()} variant...")
        if args.force_retrain:
            print(f"[pipeline] Force retraining {name.upper()} model...")
            model, loss_history = train_model(obs, labels, loss_type=name)
            training_time = 0.0  # Training time not tracked in force retrain mode
        else:
            model, loss_history, training_time = load_or_train_model(
                name, obs, labels, args.steps, name
            )
        
        # Store results
        results[name] = {
            "model": model,
            "training_time": training_time,
            "loss_history": loss_history
        }

    # 3) evaluate each variant in GUI
    print("\n=== Step 3: GUI rollouts ===")
    rewards = {}
    for name, res in results.items():
        print(f"\n{name.upper()} rollout:")
        total, actions = rollout(res["model"], args.sim_duration, args.gpu)
        print(f"  total reward = {total:.3f}")
        rewards[name] = total
        # Store action distribution for plotting
        results[name]["action_dist"] = actions

    # 4) plot comparisons
    print("\n=== Step 4: Generating comparison plots ===")
    plot_comparison(results, rewards, DATA_BC_HIP_KNEE_ALTERNATIVES / "plots")
    print(f"Saved comparison plots to {DATA_BC_HIP_KNEE_ALTERNATIVES / 'plots'}")

    # 5) select best and play longer
    best = max(rewards, key=rewards.get)
    print(f"\nBest variant: {best.upper()} (reward {rewards[best]:.3f})")
    print("Playing best variant for full sim…")
    final_total, _ = rollout(results[best]["model"], args.sim_duration, args.gpu)
    print(f"\nFinal long rollout reward = {final_total:.3f}")


if __name__ == "__main__":
    main()
