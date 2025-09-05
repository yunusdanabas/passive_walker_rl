#!/usr/bin/env python3
"""
Example usage of the legacy passive walker environment.
This demonstrates how to use the old codebase independently.
"""

import sys
import os
import numpy as np

# Add the legacy directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def example_fsm_environment():
    """Example using the FSM environment."""
    print("FSM Environment Example")
    print("-" * 30)
    
    from envs.mujoco_fsm_env import PassiveWalkerEnv as MuJoCoFSMEnv
    
    # Create FSM environment
    env = MuJoCoFSMEnv(use_gui=False)
    
    # Reset environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs[:5]}...")
    
    # Run for a few steps
    total_reward = 0
    for step in range(10):
        # FSM mode uses zeros (FSM overrides actions)
        action = np.zeros(3, dtype=np.float32)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step+1}: reward={reward:.3f}, done={done}")
        
        if done:
            print("Episode ended early")
            break
    
    print(f"Total reward: {total_reward:.3f}")
    env.close()

def example_research_environment():
    """Example using the research environment."""
    print("\nResearch Environment Example")
    print("-" * 30)
    
    from envs.mujoco_env import PassiveWalkerEnv as MuJoCoEnv, WalkerCfg
    from constants import XML_PATH
    
    # Create research environment with custom config
    cfg = WalkerCfg()
    env = MuJoCoEnv(str(XML_PATH), cfg=cfg, use_gui=False)
    
    # Reset environment
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Run for a few steps with random actions
    total_reward = 0
    for step in range(10):
        # Random actions in research mode
        action = np.random.uniform(-1, 1, 3).astype(np.float32)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step+1}: action={action[:2]}, reward={reward:.3f}")
        
        if done:
            print("Episode ended early")
            break
    
    print(f"Total reward: {total_reward:.3f}")
    env.close()

def example_control_utilities():
    """Example using control utilities."""
    print("\nControl Utilities Example")
    print("-" * 30)
    
    from utils.control import denormalize_action, compute_pd_control, get_joint_ranges, get_pd_gains, get_ctrl_limits
    
    # Test action denormalization
    action_norm = np.array([0.5, -0.3, 0.8])
    joint_ranges = get_joint_ranges()
    action_denorm = denormalize_action(action_norm, joint_ranges)
    print(f"Normalized action: {action_norm}")
    print(f"Denormalized action: {action_denorm}")
    
    # Test PD control
    q = np.array([0.1, 0.2, 0.3])
    qd = np.array([0.01, 0.02, 0.03])
    q_des = np.array([0.0, 0.0, 0.0])
    joint_ranges = get_joint_ranges()
    joint_names = ["hip", "left_knee", "right_knee"]
    kp_kv_dict = get_pd_gains()
    ctrl_limits = get_ctrl_limits()
    u = compute_pd_control(q, qd, q_des, joint_names, kp_kv_dict, ctrl_limits, joint_ranges)
    print(f"PD control output: {u}")
    
    # Test reward function
    from utils.smooth_rewards import get_default_reward_config
    cfg = get_default_reward_config()
    print(f"Reward config: c_fp={cfg.c_fp}, c_up={cfg.c_up}")

def main():
    """Run all examples."""
    print("Legacy Passive Walker Environment Examples")
    print("=" * 50)
    
    try:
        example_fsm_environment()
        example_research_environment()
        example_control_utilities()
        
        print("\n" + "=" * 50)
        print("✓ All examples completed successfully!")
        print("\nNote: This is the LEGACY codebase.")
        print("For new development, use the main passive_walker package.")
        
    except Exception as e:
        print(f"\n✗ Example failed: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
