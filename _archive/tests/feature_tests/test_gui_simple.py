#!/usr/bin/env python3
"""
Simple GUI Test for Passive Walker Environment

Test the GUI functionality of the environment.
"""

import numpy as np
import time
from passive_walker.core.env import PassiveWalkerEnv

def test_gui():
    """Test GUI functionality."""
    print("🖥️  TESTING GUI FUNCTIONALITY")
    print("=" * 50)
    
    try:
        # Create environment with GUI
        print("Creating environment with GUI...")
        env = PassiveWalkerEnv(mode="fsm", use_gui=True)
        print("✅ Environment created")
        
        # Reset environment
        obs, _ = env.reset(seed=42)
        print(f"✅ Environment reset, obs shape: {obs.shape}")
        
        # Check if window was created
        if env.window is not None:
            print("✅ GUI window created successfully")
        else:
            print("⚠️  GUI window not created yet (will be created on first render)")
        
        # Run simulation with explicit rendering
        print("Running simulation with rendering...")
        print("Look for a window titled 'Passive Walker - v2.1'")
        print("If you don't see it, try Alt+Tab to switch windows")
        print("Press Ctrl+C to stop")
        
        step_count = 0
        try:
            while step_count < 100:  # Run for 100 steps
                action = np.array([0.0, 0.0, 0.0])  # FSM mode
                obs, reward, done, info = env.step(action)
                
                # Render the current state
                env.render()
                
                step_count += 1
                
                if step_count % 20 == 0:
                    print(f"Step {step_count}: time={info.get('time', 0):.2f}s, "
                          f"position={obs[0]:.2f}, reward={reward:.3f}")
                
                if done:
                    print(f"Episode ended at step {step_count}")
                    break
                
                time.sleep(0.01)  # Small delay for visualization
                
        except KeyboardInterrupt:
            print(f"\nStopped by user at step {step_count}")
        
        print(f"✅ Simulation completed - {step_count} steps")
        
        # Check window status
        if env.window is not None:
            print("✅ GUI window was active")
        else:
            print("⚠️  GUI window was not created")
        
        # Clean up
        env.close()
        print("✅ Environment closed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_display_info():
    """Check display information."""
    print("\n" + "=" * 50)
    print("DISPLAY INFORMATION")
    print("=" * 50)
    
    import os
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'Not set')}")
    
    # Check if we can import GLFW
    try:
        from mujoco.glfw import glfw
        print("✅ GLFW available")
        
        # Try to initialize GLFW
        if glfw.init():
            print("✅ GLFW initialized successfully")
            glfw.terminate()
        else:
            print("❌ GLFW initialization failed")
            
    except Exception as e:
        print(f"❌ GLFW error: {e}")
    
    # Check MuJoCo version
    import mujoco
    print(f"MuJoCo version: {mujoco.__version__}")

if __name__ == "__main__":
    print("Starting GUI test for Passive Walker Environment...")
    print()
    
    # Check display info first
    test_display_info()
    
    # Test GUI functionality
    success = test_gui()
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"GUI test: {'✅ PASS' if success else '❌ FAIL'}")
    
    if not success:
        print("\nPossible issues:")
        print("1. No display server (X11/Wayland) running")
        print("2. SSH without X11 forwarding")
        print("3. Running in headless environment")
        print("4. Graphics drivers not available")
        print("\nSolutions:")
        print("- Use SSH with X11 forwarding: ssh -X user@server")
        print("- Use VNC or remote desktop")
        print("- Run in headless mode: use_gui=False")
