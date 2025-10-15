"""
Test FSM walking for 20+ seconds (critical success requirement).
"""
import numpy as np
import pytest
from passive_walker.core.env import PassiveWalkerEnv


@pytest.mark.slow
def test_fsm_20_second_walking():
    """Test that FSM can walk for at least 20 seconds without falling."""
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    
    try:
        obs, _ = env.reset(seed=123)
        zero_action = np.zeros(3, dtype=np.float32)
        
        # Target: 20 seconds at 100Hz = 2000 steps
        target_steps = 2000
        step_count = 0
        gait_cycles = 0
        last_hip_state = None
        hip_state_changes = 0
        total_distance = 0.0
        initial_x = obs[0]
        
        print(f"Testing 20-second FSM walking (target: {target_steps} steps)...")
        
        while step_count < target_steps:
            obs, reward, done, info = env.step(zero_action)
            step_count += 1
            
            # Track gait cycles
            current_hip_state = env.fsm.fsm_hip
            if last_hip_state is not None and current_hip_state != last_hip_state:
                hip_state_changes += 1
                gait_cycles = hip_state_changes // 2
            last_hip_state = current_hip_state
            
            # Track distance
            total_distance = obs[0] - initial_x
            
            # Progress indicator every 5 seconds
            if step_count % 500 == 0:
                elapsed_seconds = step_count / 100.0
                print(f"  {elapsed_seconds:.1f}s: {gait_cycles} cycles, {total_distance:.2f}m forward")
            
            # CRITICAL: Should not fall before 20 seconds
            if done:
                elapsed_seconds = step_count / 100.0
                pytest.fail(f"Walker fell at {elapsed_seconds:.1f} seconds (step {step_count}) - need 20+ seconds!")
        
        # Success! Check quality metrics
        elapsed_seconds = step_count / 100.0
        print(f"✅ SUCCESS: Walked for {elapsed_seconds:.1f} seconds!")
        print(f"  Gait cycles: {gait_cycles}")
        print(f"  Forward distance: {total_distance:.2f}m")
        print(f"  Average speed: {total_distance/elapsed_seconds:.3f} m/s")
        
        # Quality assertions
        assert gait_cycles >= 8, f"Too few gait cycles: {gait_cycles} (expected ≥8)"
        assert total_distance > 0.5, f"Too little forward progress: {total_distance:.2f}m (expected >0.5m)"
        assert elapsed_seconds >= 20.0, f"Not enough walking time: {elapsed_seconds:.1f}s (expected ≥20s)"
        
    finally:
        env.close()


@pytest.mark.slow
def test_fsm_20_second_multiple_seeds():
    """Test 20-second walking across multiple random seeds for robustness."""
    seeds = [123, 456, 789, 999, 42]
    successful_seeds = 0
    
    for seed in seeds:
        env = PassiveWalkerEnv(mode="fsm", use_gui=False)
        
        try:
            obs, _ = env.reset(seed=seed)
            zero_action = np.zeros(3, dtype=np.float32)
            
            target_steps = 2000  # 20 seconds
            step_count = 0
            
            while step_count < target_steps:
                obs, reward, done, info = env.step(zero_action)
                step_count += 1
                
                if done:
                    elapsed_seconds = step_count / 100.0
                    print(f"Seed {seed}: Failed at {elapsed_seconds:.1f}s")
                    break
            else:
                # Completed 20 seconds
                successful_seeds += 1
                elapsed_seconds = step_count / 100.0
                distance = obs[0] - obs[0]  # Reset distance calculation
                print(f"Seed {seed}: SUCCESS - {elapsed_seconds:.1f}s, {distance:.2f}m")
                
        finally:
            env.close()
    
    # Require at least 80% success rate (4/5 seeds)
    success_rate = successful_seeds / len(seeds)
    print(f"Success rate: {successful_seeds}/{len(seeds)} = {success_rate:.1%}")
    
    assert success_rate >= 0.8, f"FSM not robust enough: only {success_rate:.1%} success rate (need ≥80%)"


def test_fsm_quick_smoke():
    """Quick smoke test for FSM - should walk for at least 5 seconds."""
    env = PassiveWalkerEnv(mode="fsm", use_gui=False)
    
    try:
        obs, _ = env.reset(seed=123)
        zero_action = np.zeros(3, dtype=np.float32)
        
        # Quick test: 5 seconds
        target_steps = 500
        step_count = 0
        
        while step_count < target_steps:
            obs, reward, done, info = env.step(zero_action)
            step_count += 1
            
            if done:
                elapsed_seconds = step_count / 100.0
                pytest.fail(f"Quick smoke test failed: fell at {elapsed_seconds:.1f}s")
        
        elapsed_seconds = step_count / 100.0
        print(f"✅ Quick smoke test passed: {elapsed_seconds:.1f}s")
        
    finally:
        env.close()
