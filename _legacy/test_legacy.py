#!/usr/bin/env python3
"""
Test script for legacy passive walker environment.
Verifies that the legacy code can be imported and run independently.
"""

import sys
import os
import numpy as np

# Add the legacy directory to the path
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test that all legacy modules can be imported."""
    print("Testing legacy imports...")
    
    try:
        from envs.mujoco_env import PassiveWalkerEnv as MuJoCoEnv
        print("✓ MuJoCoEnv (PassiveWalkerEnv) imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MuJoCoEnv: {e}")
        return False
    
    try:
        from envs.mujoco_fsm_env import PassiveWalkerEnv as MuJoCoFSMEnv
        print("✓ MuJoCoFSMEnv (PassiveWalkerEnv) imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import MuJoCoFSMEnv: {e}")
        return False
    
    try:
        from utils.control import compute_pd_control
        print("✓ compute_pd_control imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import compute_pd_control: {e}")
        return False
    
    try:
        from utils.smooth_rewards import compute_smooth_reward
        print("✓ compute_smooth_reward imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import compute_smooth_reward: {e}")
        return False
    
    try:
        from utils.rollout_buffer import RolloutBuffer
        print("✓ RolloutBuffer imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import RolloutBuffer: {e}")
        return False
    
    return True

def test_environment_creation():
    """Test that environments can be created."""
    print("\nTesting environment creation...")
    
    try:
        from envs.mujoco_fsm_env import PassiveWalkerEnv as MuJoCoFSMEnv
        
        # Create environment
        env = MuJoCoFSMEnv(use_gui=False)
        print("✓ MuJoCoFSMEnv created successfully")
        
        # Test reset
        obs = env.reset()
        print(f"✓ Reset successful, obs shape: {obs.shape}")
        
        # Test step
        action = np.zeros(3, dtype=np.float32)
        obs, reward, done, info = env.step(action)
        print(f"✓ Step successful, reward: {reward:.3f}")
        
        env.close()
        print("✓ Environment closed successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_control_utilities():
    """Test control utilities."""
    print("\nTesting control utilities...")
    
    try:
        from utils.control import denormalize_action, compute_pd_control, get_joint_ranges
        
        # Test denormalization
        action_norm = np.array([0.5, -0.3, 0.8])
        joint_ranges = get_joint_ranges()
        action_denorm = denormalize_action(action_norm, joint_ranges)
        print(f"✓ Action denormalization: {action_norm} -> {action_denorm}")
        
        # Test PD control
        from utils.control import get_pd_gains, get_ctrl_limits
        q = np.array([0.1, 0.2, 0.3])
        qd = np.array([0.01, 0.02, 0.03])
        q_des = np.array([0.0, 0.0, 0.0])
        joint_names = ["hip", "left_knee", "right_knee"]
        kp_kv_dict = get_pd_gains()
        ctrl_limits = get_ctrl_limits()
        u = compute_pd_control(q, qd, q_des, joint_names, kp_kv_dict, ctrl_limits, joint_ranges)
        print(f"✓ PD control: u = {u}")
        
        return True
        
    except Exception as e:
        print(f"✗ Control utilities test failed: {e}")
        return False

def test_reward_functions():
    """Test reward functions."""
    print("\nTesting reward functions...")
    
    try:
        from utils.smooth_rewards import get_default_reward_config
        
        # Test reward config creation
        cfg = get_default_reward_config()
        print(f"✓ Reward config created: c_fp={cfg.c_fp}, c_up={cfg.c_up}")
        
        return True
        
    except Exception as e:
        print(f"✗ Reward functions test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Legacy Passive Walker Environment Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_environment_creation,
        test_control_utilities,
        test_reward_functions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Legacy environment is working.")
        return 0
    else:
        print("✗ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
