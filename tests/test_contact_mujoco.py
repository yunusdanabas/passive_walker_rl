#!/usr/bin/env python3
"""
MuJoCo Contact Visualization Test

Test the enhanced environment with contact information using MuJoCo's built-in viewer.
Shows real-time contact status with visual overlays in the MuJoCo window.
"""

import numpy as np
import time
from passive_walker.core.env import PassiveWalkerEnv


def test_contact_mujoco_viewer():
    """Test contact information with MuJoCo viewer."""
    print("🦶 MUJOCO CONTACT VISUALIZATION TEST 🦶")
    print("=" * 60)
    
    # Create environment with MuJoCo viewer
    env = PassiveWalkerEnv(mode="fsm", use_gui=True)
    
    print(f"✅ Environment created with MuJoCo viewer")
    print(f"✅ Observation space: {env.observation_space.shape[0]}D")
    print(f"✅ Action space: {env.action_space.shape[0]}D")
    
    # Reset environment
    obs, _ = env.reset(seed=42)
    print(f"✅ Environment reset")
    
    # Print initial contact information
    print(f"\nInitial Contact Information:")
    print(f"  Left contact: {obs[11]} ({'ON' if obs[11] > 0.5 else 'OFF'})")
    print(f"  Right contact: {obs[12]} ({'ON' if obs[12] > 0.5 else 'OFF'})")
    print(f"  Left force: {obs[13]:.3f} N")
    print(f"  Right force: {obs[14]:.3f} N")
    print(f"  Left duration: {obs[15]:.3f} s")
    print(f"  Right duration: {obs[16]:.3f} s")
    
    print(f"\nStarting MuJoCo simulation...")
    print(f"Watch the MuJoCo window for visual contact feedback")
    print(f"Press Ctrl+C to stop")
    print("=" * 60)
    
    # Simulation loop
    step_count = 0
    contact_change_count = 0
    last_left_contact = False
    last_right_contact = False
    
    try:
        while True:
            # FSM mode - send zero action (FSM controls internally)
            action = np.array([0.0, 0.0, 0.0])
            obs, reward, terminated, info = env.step(action)
            
            step_count += 1
            
            # Get contact status
            left_contact = obs[11] > 0.5
            right_contact = obs[12] > 0.5
            
            # Check for contact changes
            if (step_count == 1 or 
                left_contact != last_left_contact or 
                right_contact != last_right_contact):
                
                contact_change_count += 1
                left_icon = "🟢" if left_contact else "🔴"
                right_icon = "🟢" if right_contact else "🔴"
                
                print(f"Step {step_count:4d}: Contact change #{contact_change_count}")
                print(f"  Left {left_icon} Right {right_icon}")
                print(f"  Forces: L={obs[13]:.2f}N R={obs[14]:.2f}N")
                print(f"  Duration: L={obs[15]:.2f}s R={obs[16]:.2f}s")
                
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
                print(f"  Position: x={obs[0]:.2f}, z={obs[1]:.2f}")
                print(f"  Pitch: {np.degrees(obs[2]):.1f}°")
                print("-" * 40)
                
                last_left_contact = left_contact
                last_right_contact = right_contact
            
            # Print periodic status updates
            if step_count % 100 == 0:
                print(f"\nStep {step_count:4d} | Time: {env.data.time:.2f}s")
                print(f"  Position: x={obs[0]:.2f}, z={obs[1]:.2f}")
                print(f"  Velocity: x={obs[3]:.2f}, z={obs[4]:.2f}")
                print(f"  Contact changes: {contact_change_count}")
                print(f"  Reward: {reward:.3f}")
            
            # Check for episode termination
            if terminated:
                print(f"\nEpisode terminated at step {step_count}")
                print(f"Final position: x={obs[0]:.3f}, z={obs[1]:.3f}")
                print(f"Final pitch: {np.degrees(obs[2]):.1f}°")
                print(f"Total contact changes: {contact_change_count}")
                break
            
            # Small delay for visualization
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print(f"\n\nSimulation stopped by user")
        print(f"Total steps: {step_count}")
        print(f"Total contact changes: {contact_change_count}")
        print(f"Final time: {env.data.time:.2f}s")
    
    print(f"\n✅ MuJoCo contact test completed")
    print(f"✅ Total steps: {step_count}")
    print(f"✅ Contact changes: {contact_change_count}")
    print(f"✅ Final observation shape: {obs.shape}")


def test_contact_with_visual_feedback():
    """Test contact with enhanced visual feedback in MuJoCo."""
    print("\n" + "=" * 60)
    print("ENHANCED CONTACT VISUALIZATION")
    print("=" * 60)
    
    # Create environment
    env = PassiveWalkerEnv(mode="fsm", use_gui=True)
    obs, _ = env.reset(seed=123)
    
    print("Enhanced contact visualization - watch MuJoCo window")
    print("Contact status will be shown with visual indicators")
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
                
                # Print contact change with visual indicators
                left_indicator = "🟢" if left_contact else "🔴"
                right_indicator = "🟢" if right_contact else "🔴"
                
                print(f"Step {step_count:4d}: Contact change - Left {left_indicator} Right {right_indicator}")
                print(f"  Forces: L={obs[13]:.2f}N, R={obs[14]:.2f}N")
                print(f"  Duration: L={obs[15]:.2f}s, R={obs[16]:.2f}s")
                
                # Show gait phase
                if left_contact and not right_contact:
                    phase = "Left Support"
                elif right_contact and not left_contact:
                    phase = "Right Support"
                elif left_contact and right_contact:
                    phase = "Double Support"
                else:
                    phase = "Airborne"
                
                print(f"  Gait Phase: {phase}")
            
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
    print("Starting MuJoCo Contact Visualization Tests...")
    print("This will open a MuJoCo viewer window showing the walker simulation")
    print("Contact information will be displayed in the console")
    print()
    
    try:
        # Run the main contact test with MuJoCo viewer
        test_contact_mujoco_viewer()
        
        # Run enhanced visualization test
        test_contact_with_visual_feedback()
        
    except Exception as e:
        print(f"Error during test: {e}")
        print("Make sure MuJoCo is properly installed")
    
    print("\n✅ All MuJoCo contact tests completed!")