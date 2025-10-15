#!/usr/bin/env python3
"""
Test FSM walking on all structured physics conditions.

This script tests each physics condition to see if the FSM can successfully walk
and reports the results for each condition. Supports both headless and GUI modes.
"""

import os
import sys
import time
import argparse
import numpy as np
import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from passive_walker.core.env import PassiveWalkerEnv

# Physics condition presets from collect.py
PHYSICS_PRESETS = {
    "nominal": {"ramp_deg": 10.0, "friction": 0.9, "randomize": False},
    "gentle": {"ramp_deg": 8.0, "friction": 0.9, "randomize": False},
    "low_friction": {"ramp_deg": 10.0, "friction": 0.6, "randomize": False},
    "high_friction": {"ramp_deg": 10.0, "friction": 1.0, "randomize": False},
    "mass_jitter": {"ramp_deg": 10.0, "friction": 0.9, "randomize": True},
    "sweep_gentle_low": {"ramp_deg": 8.0, "friction": 0.6, "randomize": False},
    "very_gentle": {"ramp_deg": 7.0, "friction": 0.9, "randomize": False},
    "moderate": {"ramp_deg": 9.0, "friction": 0.9, "randomize": False},
    "medium_friction": {"ramp_deg": 10.0, "friction": 0.7, "randomize": False},
    "gentle_high": {"ramp_deg": 8.0, "friction": 1.0, "randomize": False},
}

@pytest.mark.parametrize("condition_name,physics_params", list(PHYSICS_PRESETS.items()))
def test_condition(condition_name, physics_params, test_duration=15.0, use_gui=False):
    """Test FSM walking on a specific physics condition."""
    print(f"\n--- Testing {condition_name} ---")
    print(f"Parameters: ramp={physics_params['ramp_deg']}°, friction={physics_params['friction']}, randomize={physics_params['randomize']}")
    print(f"Mode: {'GUI' if use_gui else 'Headless'}")
    print(f"Duration: {test_duration}s")
    
    try:
        # Create environment with specific physics
        env = PassiveWalkerEnv(
            mode="fsm",
            use_gui=use_gui,
            ramp_deg=physics_params["ramp_deg"],
            friction=physics_params["friction"],
            randomize_physics=physics_params["randomize"]
        )
        
        if use_gui:
            print("  GUI opened - you should see the walking simulation")
            print("  Press Ctrl+C to interrupt if needed")
        
        # Reset environment
        obs, _ = env.reset(seed=123)
        
        # Test parameters
        steps = int(test_duration * 100)  # 100 Hz
        gait_cycles = 0
        last_hip_state = None
        hip_state_changes = 0
        total_reward = 0.0
        max_distance = 0.0
        min_torso_z = float('inf')
        max_pitch = 0.0
        
        print(f"Running {test_duration}s test ({steps} steps)...")
        
        # Add progress indicator for GUI mode
        if use_gui:
            print("  Watch the simulation - it will run for 15 seconds...")
        
        for t in range(steps):
            # Get FSM action
            hip_des = env.fsm.desired_hip()
            lk_des, rk_des = env.fsm.desired_knees()
            action = np.array([hip_des, lk_des, rk_des], dtype=np.float32)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # Render GUI if enabled
            if use_gui:
                env.render()
            
            # Track gait cycles
            current_hip_state = env.fsm.fsm_hip
            if last_hip_state is not None and current_hip_state != last_hip_state:
                hip_state_changes += 1
                gait_cycles = hip_state_changes // 2
            last_hip_state = current_hip_state
            
            # Track metrics
            x_pos = obs[0]  # x position
            torso_z = obs[1]  # z position  
            pitch = abs(obs[2])  # absolute pitch
            
            max_distance = max(max_distance, x_pos)
            min_torso_z = min(min_torso_z, torso_z)
            max_pitch = max(max_pitch, pitch)
            
            # Check for failure
            if done:
                print(f"  ❌ FAILED at step {t} (fell)")
                print(f"  Reason: pitch={pitch:.3f} rad, torso_z={torso_z:.3f} m")
                env.close()
                pytest.fail(f"Condition {condition_name} failed at step {t}: pitch={pitch:.3f} rad, torso_z={torso_z:.3f} m")
        
        # Success!
        avg_reward = total_reward / steps
        print(f"  ✅ SUCCESS: {gait_cycles} gait cycles, {max_distance:.2f}m distance")
        print(f"  Metrics: avg_reward={avg_reward:.3f}, min_torso_z={min_torso_z:.3f}m, max_pitch={max_pitch:.3f}rad")
        
        env.close()
        
        # Assert success for pytest
        assert True, f"Condition {condition_name} completed successfully"
        
        # Don't return values from pytest test functions
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        pytest.fail(f"Condition {condition_name} failed with error: {e}")

def run_test_suite(use_gui=False, test_duration=15.0):
    """Run the complete test suite."""
    mode_name = "GUI" if use_gui else "Headless"
    print(f"FSM Physics Condition Test - {mode_name} Mode")
    print("=" * 60)
    
    if use_gui:
        print("⚠️  GUI Mode: Each test will run for 15 seconds")
        print("⚠️  You can interrupt with Ctrl+C if needed")
        print("⚠️  Make sure you can see the MuJoCo window")
        input("Press Enter to start GUI testing...")
    
    results = {}
    successful_conditions = []
    failed_conditions = []
    
    for condition_name, physics_params in PHYSICS_PRESETS.items():
        success, metrics = test_condition(condition_name, physics_params, test_duration, use_gui)
        results[condition_name] = metrics
        
        if success:
            successful_conditions.append(condition_name)
        else:
            failed_conditions.append(condition_name)
    
    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY - {mode_name} Mode")
    print("=" * 60)
    print(f"Successful conditions: {len(successful_conditions)}/{len(PHYSICS_PRESETS)}")
    print(f"Failed conditions: {len(failed_conditions)}/{len(PHYSICS_PRESETS)}")
    
    if successful_conditions:
        print(f"\n✅ Working conditions:")
        for condition in successful_conditions:
            metrics = results[condition]
            print(f"  {condition:20} - {metrics['gait_cycles']:2d} cycles, {metrics['max_distance']:6.2f}m")
    
    if failed_conditions:
        print(f"\n❌ Failed conditions:")
        for condition in failed_conditions:
            metrics = results[condition]
            if 'error' in metrics:
                print(f"  {condition:20} - ERROR: {metrics['error']}")
            else:
                print(f"  {condition:20} - Fell at step {metrics['steps']}")
    
    # Recommendations
    print(f"\n📋 RECOMMENDATIONS - {mode_name} Mode:")
    if len(successful_conditions) >= 8:
        print("  ✅ Most conditions work - structured physics diversity is excellent!")
        print("  ✅ Can proceed with physics sweep collection")
    elif len(successful_conditions) >= 6:
        print("  ✅ Most conditions work - structured physics diversity is viable!")
        print("  ✅ Can proceed with physics sweep collection")
    elif len(successful_conditions) >= 4:
        print("  ⚠️  Some conditions work - consider removing failed ones")
        print("  ⚠️  May need to adjust physics parameter ranges")
    else:
        print("  ❌ Many conditions fail - physics parameters may be too extreme")
        print("  ❌ Consider more conservative parameter ranges")
    
    return results, successful_conditions, failed_conditions

def main():
    """Main entry point with CLI arguments."""
    parser = argparse.ArgumentParser("FSM Physics Condition Test")
    parser.add_argument("--gui", action="store_true", help="Enable GUI mode for visual testing")
    parser.add_argument("--duration", type=float, default=15.0, help="Test duration in seconds")
    parser.add_argument("--headless-first", action="store_true", 
                       help="Run headless test first, then GUI if all pass")
    args = parser.parse_args()
    
    if args.headless_first:
        # Run headless test first
        print("Running headless test first...")
        headless_results, headless_success, headless_failed = run_test_suite(use_gui=False, test_duration=args.duration)
        
        if len(headless_failed) == 0:
            print(f"\n🎉 All {len(headless_success)} conditions passed headless test!")
            print("Proceeding to GUI test...")
            
            # Run GUI test
            gui_results, gui_success, gui_failed = run_test_suite(use_gui=True, test_duration=args.duration)
            
            # Compare results
            print(f"\n" + "=" * 60)
            print("HEADLESS vs GUI COMPARISON")
            print("=" * 60)
            print(f"Headless: {len(headless_success)}/{len(PHYSICS_PRESETS)} passed")
            print(f"GUI:      {len(gui_success)}/{len(PHYSICS_PRESETS)} passed")
            
            if len(gui_failed) == 0:
                print("🎉 All conditions work in both headless and GUI modes!")
            else:
                print(f"⚠️  {len(gui_failed)} conditions failed in GUI mode")
                for condition in gui_failed:
                    print(f"    - {condition}")
        else:
            print(f"\n❌ {len(headless_failed)} conditions failed headless test - skipping GUI test")
            print("Fix headless issues before testing with GUI")
    else:
        # Run single test mode
        run_test_suite(use_gui=args.gui, test_duration=args.duration)

if __name__ == "__main__":
    main()
