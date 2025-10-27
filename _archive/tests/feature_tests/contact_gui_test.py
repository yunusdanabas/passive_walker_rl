#!/usr/bin/env python3
"""
Contact Visualization with GUI

Shows the walker simulation with contact information overlay.
"""

import numpy as np
import time
from passive_walker.core.env import PassiveWalkerEnv

def contact_gui_test():
    """Test contact visualization with GUI."""
    print("🦶 CONTACT VISUALIZATION WITH GUI")
    print("=" * 60)
    
    # Create environment with GUI
    env = PassiveWalkerEnv(mode="fsm", use_gui=True)
    obs, _ = env.reset(seed=42)
    
    print(f"✅ Environment created with GUI")
    print(f"✅ Observation space: {env.observation_space.shape[0]}D")
    print(f"✅ Window: 1200x900 'Passive Walker - v2.1'")
    
    print(f"\n🖥️  GUI WINDOW SHOULD BE VISIBLE NOW!")
    print(f"Look for: 'Passive Walker - v2.1' window")
    print(f"If not visible: Alt+Tab, check taskbar, or try different desktop")
    print(f"Press Ctrl+C to stop")
    print("=" * 60)
    
    step_count = 0
    last_contact_print = 0
    
    try:
        while True:
            # FSM mode - zero action
            action = np.array([0.0, 0.0, 0.0])
            obs, reward, terminated, info = env.step(action)
            
            # Render the current state
            env.render()
            
            step_count += 1
            
            # Get contact status
            left_contact = obs[11] > 0.5
            right_contact = obs[12] > 0.5
            
            # Print contact changes every 50 steps
            if step_count % 50 == 0:
                left_icon = "🟢" if left_contact else "🔴"
                right_icon = "🟢" if right_contact else "🔴"
                
                print(f"Step {step_count:4d}: Contact - Left {left_icon} Right {right_icon}")
                print(f"  Forces: L={obs[13]:.2f}N R={obs[14]:.2f}N")
                print(f"  Duration: L={obs[15]:.2f}s R={obs[16]:.2f}s")
                print(f"  Position: x={obs[0]:.2f}, z={obs[1]:.2f}")
                print(f"  Reward: {reward:.3f}")
                print("-" * 40)
            
            if terminated:
                print(f"\nEpisode terminated at step {step_count}")
                break
                
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print(f"\nStopped by user at step {step_count}")
    
    print(f"\n✅ Contact GUI test completed")
    print(f"✅ Total steps: {step_count}")
    print(f"✅ Final observation shape: {obs.shape}")
    
    env.close()

if __name__ == "__main__":
    print("Starting Contact Visualization with GUI...")
    print("This will show the walker simulation with contact information")
    print()
    
    contact_gui_test()
    
    print("\n✅ Contact visualization test completed!")
    print("If you didn't see the GUI window:")
    print("1. Try Alt+Tab to switch windows")
    print("2. Check your taskbar/dock for the window")
    print("3. Check all virtual desktops")
    print("4. The window might be off-screen")
