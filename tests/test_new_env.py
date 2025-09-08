#!/usr/bin/env python3
"""
Test script for the unified passive walker environment.

Tests both FSM and research modes to verify basic functionality.
"""

import warnings
import os
from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.core.io import load_walker_config

# Suppress gym deprecation warnings
os.environ["GYM_DISABLE_WARNINGS"] = "1"
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")


def test_fsm_mode():
    """Test FSM data collection mode."""
    print("Testing FSM mode...")

    # Load FSM configuration
    cfg = load_walker_config("passive_walker/configs/fsm_collect.yaml")
    env = PassiveWalkerEnv(cfg, use_gui=False)

    # Run a few steps to test basic functionality
    obs, info = env.reset()
    print(f"FSM - Initial obs shape: {obs.shape}")

    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()  # Random action (FSM overrides)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"FSM - Total reward: {total_reward:.3f} (10 steps)")
    env.close()


def test_research_mode():
    """Test RL research mode."""
    print("\nTesting research mode...")

    # Load research configuration
    cfg = load_walker_config("passive_walker/configs/ppo_train.yaml")
    env = PassiveWalkerEnv(cfg, use_gui=False)

    # Run a few steps to test basic functionality
    obs, info = env.reset()
    print(f"Research - Initial obs shape: {obs.shape}")

    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break

    print(f"Research - Total reward: {total_reward:.3f} (10 steps)")
    env.close()


def main():
    """Run all environment tests."""
    print("New Unified Environment Test")
    print("=" * 40)

    try:
        test_fsm_mode()
        test_research_mode()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
