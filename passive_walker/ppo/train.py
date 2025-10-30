"""
PPO Training CLI

Command-line interface for training PPO agents with enhanced environment integration.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from passive_walker.ppo.models import create_actor_critic
from passive_walker.common.device import pick_torch_device
from passive_walker.ppo.config import PPOConfig, create_default_configs
from passive_walker.ppo.trainer import PPOTrainer
from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.config.paths import PPO_MODELS_DIR, PPO_RUNS_DIR, METRICS_DIR, ensure_dir_exists
from passive_walker.config.paths_redirect import redirect_legacy_dir


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="PPO training")
    
    # Basic arguments
    parser.add_argument("--experiment_name", type=str, default="ppo_experiment",
                       help="Name of the experiment")
    parser.add_argument("--model_type", type=str, default="mlp", 
                       choices=["mlp", "lstm", "gru"],
                       help="Type of actor-critic model")
    parser.add_argument("--timesteps", type=int, default=100000,
                       help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                       choices=["cpu", "cuda"],
                       help="Device to train on (deprecated; prefer --gpu)")
    parser.add_argument("--gpu", action="store_true",
                       help="Use CUDA if available (optional)")
    # Deprecated --out retained for compatibility; redirected if used
    parser.add_argument("--out", type=str, default=str(PPO_RUNS_DIR),
                       help="Output directory (deprecated; use centralized runs dir)")
    
    # Model arguments
    parser.add_argument("--hidden_size", type=int, default=64,
                       help="Hidden size for temporal models")
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64, 64],
                       help="Hidden sizes for MLP models")
    parser.add_argument("--num_layers", type=int, default=1,
                       help="Number of layers for temporal models")
    
    # Training arguments
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=2048,
                       help="Steps per rollout")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=10,
                       help="PPO epochs per rollout")
    
    # Enhanced environment arguments
    parser.add_argument("--use_curriculum", action="store_true",
                       help="Use reward curriculum")
    parser.add_argument("--use_domain_randomization", action="store_true",
                       help="Use domain randomization")
    parser.add_argument("--randomization_profile", type=str, default="moderate",
                       choices=["light", "moderate", "aggressive"],
                       help="Domain randomization profile")
    
    # Evaluation arguments
    parser.add_argument("--eval_freq", type=int, default=0,
                       help="Evaluation frequency (0 = auto: every 5%% of total timesteps)")
    parser.add_argument("--n_eval_episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    
    # Config presets
    parser.add_argument("--config", type=str, default=None,
                       help="Use predefined config (ppo_mlp_basic, ppo_lstm_temporal, etc.)")
    
    args = parser.parse_args()
    
    # Create config
    if args.config:
        # Use predefined config
        defaults = create_default_configs()
        if args.config in defaults:
            config = defaults[args.config]
            config.experiment_name = args.experiment_name
            config.total_timesteps = args.timesteps
            config.use_curriculum = args.use_curriculum
            config.use_domain_randomization = args.use_domain_randomization
            config.randomization_profile = args.randomization_profile
            # Set eval_freq: auto if 0, else use specified value
            if args.eval_freq == 0:
                config.eval_freq = max(args.timesteps // 20, 2048)  # Every 5% of total steps
            else:
                config.eval_freq = args.eval_freq
            config.n_eval_episodes = args.n_eval_episodes
        else:
            print(f"Unknown config: {args.config}")
            print(f"Available configs: {list(defaults.keys())}")
            return
    else:
        # Create custom config
        # Set eval_freq: auto if 0, else use specified value
        eval_freq = args.eval_freq
        if eval_freq == 0:
            eval_freq = max(args.timesteps // 20, 2048)  # Every 5% of total steps
        
        config = PPOConfig(
            experiment_name=args.experiment_name,
            model_type=args.model_type,
            hidden_sizes=args.hidden_sizes,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            total_timesteps=args.timesteps,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            use_curriculum=args.use_curriculum,
            use_domain_randomization=args.use_domain_randomization,
            randomization_profile=args.randomization_profile,
            eval_freq=eval_freq,
            n_eval_episodes=args.n_eval_episodes
        )
    
    print(f"Starting PPO training: {config.experiment_name}")
    print(f"Model: {config.model_type}")
    print(f"Timesteps: {config.total_timesteps}")
    print(f"Eval frequency: {config.eval_freq}")
    # Resolve device with optional CUDA preference
    if getattr(args, "gpu", False):
        device = pick_torch_device(True)
        if device == "cpu":
            print("[WARN] --gpu requested but CUDA not available; using CPU")
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Curriculum: {config.use_curriculum}")
    print(f"Domain randomization: {config.use_domain_randomization}")
    
    # Set random seed
    import numpy as np
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create model
    if config.model_type == "mlp":
        model = create_actor_critic(
            config.model_type,
            obs_dim=17,
            action_dim=3,
            hidden_sizes=config.hidden_sizes
        )
    else:
        model = create_actor_critic(
            config.model_type,
            obs_dim=17,
            action_dim=3,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers
        )
    
    # Redirect legacy out path and ensure dir exists
    out_dir = str(redirect_legacy_dir(args.out))
    ensure_dir_exists(out_dir)

    # Create trainer
    trainer = PPOTrainer(model, config, device=device, output_dir=out_dir)
    
    # Train
    try:
        results = trainer.train()
        
        print(f"\nTraining completed!")
        print(f"Final timestep: {results['final_timestep']}")
        print(f"Best eval return: {results['best_eval_return']:.2f}")
        print(f"Training time: {results['training_time']:.1f}s")
        
        # Save final model
        trainer.save_model("final_model.pth")
        print(f"Model saved to: {trainer.run_dir}/final_model.pth")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_model("interrupted_model.pth")
        print(f"Model saved to: {trainer.run_dir}/interrupted_model.pth")
    
    finally:
        pass


if __name__ == "__main__":
    main()