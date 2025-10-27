#!/usr/bin/env python3
"""
Simple Contact Visualization Test

Minimal test showing contact information with visual indicators.
"""

import numpy as np
import time
from passive_walker.core.env import PassiveWalkerEnv


def simple_contact_test():
    """Simple contact visualization test."""
    print("🦶 CONTACT VISUALIZATION TEST 🦶")
    print("=" * 50)
    
    # Create environment
    env = PassiveWalkerEnv(mode="fsm", use_gui=True)
    obs, _ = env.reset(seed=42)
    
    print(f"Environment: {env.observation_space.shape[0]}D observations")
    print(f"Contact threshold: 0.1N")
    print("Starting simulation...")
    print("=" * 50)
    
    step = 0
    last_left = False
    last_right = False
    
    try:
        while step < 500:  # Run for 500 steps
            # FSM mode - zero action
            action = np.array([0.0, 0.0, 0.0])
            obs, reward, terminated, info = env.step(action)
            step += 1
            
            # Get contact status
            left_contact = obs[11] > 0.5
            right_contact = obs[12] > 0.5
            
            # Print when contact changes
            if left_contact != last_left or right_contact != last_right:
                left_icon = "🟢" if left_contact else "🔴"
                right_icon = "🟢" if right_contact else "🔴"
                
                print(f"Step {step:3d}: Left {left_icon} Right {right_icon} | "
                      f"Forces: L={obs[13]:.2f}N R={obs[14]:.2f}N")
                
                last_left = left_contact
                last_right = right_contact
            
            if terminated:
                print(f"\nEpisode ended at step {step}")
                break
                
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print(f"\nStopped at step {step}")
    
    print(f"\n✅ Test completed - {step} steps")


if __name__ == "__main__":
    simple_contact_test()
