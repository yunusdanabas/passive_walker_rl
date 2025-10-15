"""
Test FSM walking for 20+ seconds (critical success requirement).
"""
import numpy as np
import pytest
import argparse
import sys
from passive_walker.core.env import PassiveWalkerEnv


@pytest.mark.slow
def test_fsm_20_second_walking(use_gui=False):
    """Test that FSM can walk for at least 20 seconds without falling."""
    env = PassiveWalkerEnv(mode="fsm", use_gui=use_gui)
    
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
        
        mode_str = "GUI" if use_gui else "headless"
        print(f"Testing 20-second FSM walking in {mode_str} mode (target: {target_steps} steps)...")
        
        if use_gui:
            print("  GUI Mode: Watch the simulation window - it will run for 20 seconds...")
        
        while step_count < target_steps:
            obs, reward, done, info = env.step(zero_action)
            step_count += 1
            
            # Render GUI if enabled
            if use_gui:
                env.render()
            
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
def test_fsm_20_second_walking_gui():
    """Test 20-second FSM walking with GUI for visual verification."""
    print("\n" + "="*60)
    print("GUI MODE: 20-second FSM walking test")
    print("="*60)
    print("You should see a MuJoCo window with the walking simulation.")
    print("The test will run for 20 seconds - watch for stable walking gait.")
    print("Press Ctrl+C if you need to interrupt.")
    print("="*60)
    
    test_fsm_20_second_walking(use_gui=True)


@pytest.mark.slow
@pytest.mark.parametrize("use_gui", [False, True], ids=["headless", "gui"])
def test_fsm_20_second_walking_parametrized(use_gui):
    """Parametrized test for both headless and GUI modes."""
    if use_gui:
        print(f"\n🎬 Running GUI test - watch the simulation window!")
    test_fsm_20_second_walking(use_gui=use_gui)


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


def main():
    """CLI entry point for running 20-second walking test with options."""
    parser = argparse.ArgumentParser("20-Second FSM Walking Test")
    parser.add_argument("--gui", action="store_true", default=False,
                       help="Run with GUI for visual verification")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick 5-second smoke test instead")
    parser.add_argument("--seeds", action="store_true",
                       help="Run multiple seeds robustness test")
    args = parser.parse_args()
    
    if args.quick:
        print("Running quick smoke test...")
        test_fsm_quick_smoke()
    elif args.seeds:
        print("Running multiple seeds robustness test...")
        test_fsm_20_second_multiple_seeds()
    else:
        print(f"Running 20-second walking test (GUI: {args.gui})...")
        if args.gui:
            print("\n" + "="*60)
            print("GUI MODE: 20-second FSM walking test")
            print("="*60)
            print("You should see a MuJoCo window with the walking simulation.")
            print("The test will run for 20 seconds - watch for stable walking gait.")
            print("Press Ctrl+C if you need to interrupt.")
            print("="*60)
        
        try:
            test_fsm_20_second_walking(use_gui=args.gui)
            print("\n✅ SUCCESS: 20-second walking test passed!")
        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ FAILED: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
