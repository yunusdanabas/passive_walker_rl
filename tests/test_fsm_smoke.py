"""
Test FSM stability and basic functionality.
"""
import numpy as np
import pytest
from passive_walker.core.env import PassiveWalkerEnv


def test_fsm_smoke():
    """Test FSM mode runs without falling for 5 seconds."""
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    
    try:
        obs, _ = env.reset(seed=123)
        zero_action = np.zeros(3, dtype=np.float32)
        
        step_count = 0
        max_steps = 250  # 5 seconds at 50Hz
        
        while step_count < max_steps:
            obs, reward, done, info = env.step(zero_action)
            step_count += 1
            
            # Should not fall in FSM mode
            assert not info.get("fell", False), f"Walker fell at step {step_count}"
            
            if done:
                break
                
        # Should have run for reasonable time
        assert step_count > 100, f"Episode ended too early: {step_count} steps"
        
    finally:
        env.close()


def test_fsm_forward_progress():
    """Test that FSM mode produces forward progress."""
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    
    try:
        obs, _ = env.reset(seed=123)
        zero_action = np.zeros(3, dtype=np.float32)
        
        initial_x = obs[0]  # x position
        step_count = 0
        max_steps = 100
        
        while step_count < max_steps:
            obs, reward, done, info = env.step(zero_action)
            step_count += 1
            
            if done:
                break
                
        final_x = obs[0]
        forward_progress = final_x - initial_x
        
        # Should have moved forward
        assert forward_progress > 0.1, f"Insufficient forward progress: {forward_progress}"
        
    finally:
        env.close()
