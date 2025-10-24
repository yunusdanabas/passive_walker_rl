#!/usr/bin/env python3
"""
Minimal MuJoCo Viewer Test

Simple test to check if MuJoCo viewer works.
"""

import mujoco
import mujoco_viewer
import numpy as np
import time
import os

def test_mujoco_viewer():
    """Test basic MuJoCo viewer functionality."""
    print("Testing MuJoCo viewer...")
    
    # Get the model file path
    model_path = "passive_walker/assets/passiveWalker_model.xml"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return False
    
    try:
        # Load model
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        print(f"✅ Model loaded successfully")
        print(f"Model has {model.nq} DOFs, {model.nv} velocities")
        
        # Create viewer
        viewer = mujoco_viewer.MujocoViewer(model, data)
        print(f"✅ Viewer created successfully")
        
        # Run simulation for a few steps
        print("Running simulation for 100 steps...")
        for i in range(100):
            mujoco.mj_step(model, data)
            viewer.render()
            time.sleep(0.01)
            
            if i % 20 == 0:
                print(f"Step {i}: qpos[0]={data.qpos[0]:.3f}")
        
        print("✅ Simulation completed successfully")
        viewer.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_environment_viewer():
    """Test environment with MuJoCo viewer."""
    print("\nTesting environment with MuJoCo viewer...")
    
    try:
        from passive_walker.core.env import PassiveWalkerEnv
        
        # Create environment
        env = PassiveWalkerEnv(mode="fsm", use_gui=True)
        print(f"✅ Environment created")
        
        # Reset
        obs, _ = env.reset(seed=42)
        print(f"✅ Environment reset, obs shape: {obs.shape}")
        
        # Run a few steps
        print("Running 50 steps...")
        for i in range(50):
            action = np.array([0.0, 0.0, 0.0])  # FSM mode
            obs, reward, done, info = env.step(action)
            
            if i % 10 == 0:
                print(f"Step {i}: reward={reward:.3f}, done={done}")
            
            if done:
                print(f"Episode ended at step {i}")
                break
        
        print("✅ Environment test completed")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("MUJOCO VIEWER DIAGNOSTIC TEST")
    print("=" * 50)
    
    # Test 1: Basic MuJoCo viewer
    success1 = test_mujoco_viewer()
    
    # Test 2: Environment with viewer
    success2 = test_environment_viewer()
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Basic MuJoCo viewer: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Environment viewer: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if not success1 and not success2:
        print("\n❌ No GUI is working. Possible issues:")
        print("1. No display available (SSH, headless server)")
        print("2. MuJoCo viewer not properly installed")
        print("3. Graphics drivers not available")
        print("4. X11 forwarding not enabled")
        
        print("\nTo check display:")
        print("echo $DISPLAY")
        print("xrandr")
        
        print("\nTo install MuJoCo viewer:")
        print("pip install mujoco-viewer")
        
        print("\nFor headless servers, you can:")
        print("1. Use SSH with X11 forwarding: ssh -X user@server")
        print("2. Use VNC or similar remote desktop")
        print("3. Run tests without GUI (headless mode)")
