#!/usr/bin/env python3
"""
Contact Information GUI Test

Visual test for the enhanced environment with contact information.
Shows real-time contact status overlay and prints contact data to screen.
"""

import numpy as np
import time
from passive_walker.core.env import PassiveWalkerEnv


def test_contact_gui():
    """Test contact information with GUI visualization."""
    print("=" * 60)
    print("CONTACT INFORMATION GUI TEST")
    print("=" * 60)
    
    # Create environment with GUI enabled
    env = PassiveWalkerEnv(mode="fsm", use_gui=True)
    
    print(f"✅ Environment created with GUI")
    print(f"✅ Observation space: {env.observation_space.shape[0]}D")
    print(f"✅ Action space: {env.action_space.shape[0]}D")
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    print(f"✅ Environment reset")
    
    # Print initial observation info
    print(f"\nInitial Observation:")
    print(f"  Position: x={obs[0]:.3f}, z={obs[1]:.3f}")
    print(f"  Pitch: {obs[2]:.3f} rad ({np.degrees(obs[2]):.1f}°)")
    print(f"  Velocities: x={obs[3]:.3f}, z={obs[4]:.3f}")
    print(f"  Joint angles: hip={obs[5]:.3f}, lk={obs[6]:.3f}, rk={obs[7]:.3f}")
    
    # Print contact information
    print(f"\nContact Information:")
    print(f"  Left contact: {obs[11]} ({'ON' if obs[11] > 0.5 else 'OFF'})")
    print(f"  Right contact: {obs[12]} ({'ON' if obs[12] > 0.5 else 'OFF'})")
    print(f"  Left force: {obs[13]:.3f} N")
    print(f"  Right force: {obs[14]:.3f} N")
    print(f"  Left duration: {obs[15]:.3f} s")
    print(f"  Right duration: {obs[16]:.3f} s")
    
    print(f"\nStarting simulation...")
    print(f"Press Ctrl+C to stop")
    print("=" * 60)
    
    # Simulation loop
    step_count = 0
    last_contact_print = 0
    
    try:
        while True:
            # FSM mode - send zero action (FSM controls internally)
            action = np.array([0.0, 0.0, 0.0])
            obs, reward, terminated, info = env.step(action)
            
            step_count += 1
            
            # Print contact status every 50 steps or when contact changes
            left_contact = obs[11] > 0.5
            right_contact = obs[12] > 0.5
            
            if (step_count % 50 == 0 or 
                left_contact != (obs[11] > 0.5) or 
                right_contact != (obs[12] > 0.5)):
                
                print(f"\nStep {step_count:4d} | Time: {env.data.time:.2f}s")
                print(f"  Position: x={obs[0]:.3f}, z={obs[1]:.3f}")
                print(f"  Pitch: {np.degrees(obs[2]):.1f}°")
                print(f"  Velocities: x={obs[3]:.3f}, z={obs[4]:.3f}")
                print(f"  Joint angles: hip={np.degrees(obs[5]):.1f}°, lk={obs[6]:.3f}, rk={obs[7]:.3f}")
                
                # Contact status with visual indicators
                left_status = "🟢 ON " if left_contact else "🔴 OFF"
                right_status = "🟢 ON " if right_contact else "🔴 OFF"
                
                print(f"  Contact: Left {left_status} Right {right_status}")
                print(f"  Forces: Left={obs[13]:.2f}N, Right={obs[14]:.2f}N")
                print(f"  Duration: Left={obs[15]:.2f}s, Right={obs[16]:.2f}s")
                
                # Gait phase detection
                if left_contact and not right_contact:
                    gait_phase = "Left Support"
                elif right_contact and not left_contact:
                    gait_phase = "Right Support"
                elif left_contact and right_contact:
                    gait_phase = "Double Support"
                else:
                    gait_phase = "Airborne"
                
                print(f"  Gait Phase: {gait_phase}")
                
                # Reward and termination info
                print(f"  Reward: {reward:.3f}")
                if terminated:
                    print(f"  ⚠️  Episode terminated!")
                    break
            
            # Check for episode termination
            if terminated:
                print(f"\nEpisode terminated at step {step_count}")
                print(f"Final position: x={obs[0]:.3f}, z={obs[1]:.3f}")
                print(f"Final pitch: {np.degrees(obs[2]):.1f}°")
                break
            
            # Small delay for visualization
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print(f"\n\nSimulation stopped by user")
        print(f"Total steps: {step_count}")
        print(f"Final time: {env.data.time:.2f}s")
    
    print(f"\n✅ Contact GUI test completed")
    print(f"✅ Total steps: {step_count}")
    print(f"✅ Final observation shape: {obs.shape}")


def test_contact_visualization():
    """Test contact visualization with simple overlay."""
    print("\n" + "=" * 60)
    print("CONTACT VISUALIZATION TEST")
    print("=" * 60)
    
    # Create environment
    env = PassiveWalkerEnv(mode="fsm", use_gui=True)
    obs, _ = env.reset(seed=123)
    
    print("Contact visualization test - watch the console for contact changes")
    print("The GUI window should show the walker simulation")
    print("Contact status will be printed to console with visual indicators")
    print("=" * 60)
    
    step_count = 0
    contact_history = []
    
    try:
        for _ in range(1000):  # Run for 1000 steps max
            action = np.array([0.0, 0.0, 0.0])
            obs, reward, terminated, info = env.step(action)
            step_count += 1
            
            # Track contact changes
            left_contact = obs[11] > 0.5
            right_contact = obs[12] > 0.5
            
            contact_state = (left_contact, right_contact)
            if not contact_history or contact_state != contact_history[-1]:
                contact_history.append(contact_state)
                
                # Print contact change
                left_indicator = "🟢" if left_contact else "🔴"
                right_indicator = "🟢" if right_contact else "🔴"
                
                print(f"Step {step_count:4d}: Contact change - Left {left_indicator} Right {right_indicator}")
                print(f"  Forces: L={obs[13]:.2f}N, R={obs[14]:.2f}N")
                print(f"  Duration: L={obs[15]:.2f}s, R={obs[16]:.2f}s")
            
            if terminated:
                print(f"\nEpisode terminated at step {step_count}")
                break
                
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print(f"\nTest stopped by user")
    
    print(f"\nContact change summary:")
    print(f"Total contact changes: {len(contact_history)}")
    print(f"Total steps: {step_count}")
    
    # Analyze contact patterns
    if len(contact_history) > 1:
        print(f"\nContact pattern analysis:")
        for i, (left, right) in enumerate(contact_history[:10]):  # Show first 10 changes
            left_str = "ON " if left else "OFF"
            right_str = "ON " if right else "OFF"
            print(f"  Change {i+1}: Left {left_str} Right {right_str}")


if __name__ == "__main__":
    print("Starting Contact Information GUI Tests...")
    print("This will open a MuJoCo GUI window showing the walker simulation")
    print("Contact information will be displayed in the console")
    print()
    
    try:
        # Run the main contact GUI test
        test_contact_gui()
        
        # Run additional visualization test
        test_contact_visualization()
        
    except Exception as e:
        print(f"Error during test: {e}")
        print("Make sure MuJoCo GUI dependencies are installed")
    
    print("\n✅ All contact GUI tests completed!")
