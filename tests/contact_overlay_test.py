#!/usr/bin/env python3
"""
Contact Visualization with MuJoCo Overlay

Shows the walker simulation with contact information overlaid on the MuJoCo viewer.
Based on the mujoco_lqr_controller_interactive.py example.
"""

import numpy as np
import time
import mujoco
from mujoco.glfw import glfw
from passive_walker.core.env import PassiveWalkerEnv

# Global overlay dictionary
_overlay = {}

def add_overlay(gridpos, text1, text2=""):
    """Add text to overlay at specified grid position."""
    if gridpos not in _overlay:
        _overlay[gridpos] = ["", ""]
    _overlay[gridpos][0] += text1 + "\n"
    _overlay[gridpos][1] += text2 + "\n"

def create_contact_overlay(env, obs, step_count, contact_change_count, reward):
    """Create contact information overlay."""
    # Get contact status
    left_contact = obs[11] > 0.5
    right_contact = obs[12] > 0.5
    
    # Contact icons
    left_icon = "🟢" if left_contact else "🔴"
    right_icon = "🟢" if right_contact else "🔴"
    
    # Gait phase detection
    if left_contact and not right_contact:
        gait_phase = "Left Support"
    elif right_contact and not left_contact:
        gait_phase = "Right Support"
    elif left_contact and right_contact:
        gait_phase = "Double Support"
    else:
        gait_phase = "Airborne"
    
    # Top-left: Contact status and gait phase
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        "Contact Status:",
        f"Left {left_icon} Right {right_icon}"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        "Gait Phase:",
        gait_phase
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        "Contact Forces:",
        f"L: {obs[13]:.2f}N  R: {obs[14]:.2f}N"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        "Contact Duration:",
        f"L: {obs[15]:.2f}s  R: {obs[16]:.2f}s"
    )
    
    # Top-right: Position and movement info
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        "Position:",
        f"x: {obs[0]:.2f}m  z: {obs[1]:.2f}m"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        "Velocity:",
        f"x: {obs[3]:.2f}m/s  z: {obs[4]:.2f}m/s"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        "Pitch:",
        f"{np.degrees(obs[2]):.1f}°"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        "Joint Angles:",
        f"Hip: {np.degrees(obs[5]):.1f}°"
    )
    
    # Bottom-left: Simulation info
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
        "Step:",
        f"{step_count}"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
        "Time:",
        f"{env.data.time:.2f}s"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
        "Contact Changes:",
        f"{contact_change_count}"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
        "Reward:",
        f"{reward:.3f}"
    )
    
    # Bottom-right: Controls
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
        "Controls:",
        "R - Reset"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
        "",
        "ESC - Exit"
    )

def keyboard_callback(window, key, scancode, act, mods):
    """Handle keyboard input."""
    if act == glfw.PRESS:
        if key == glfw.KEY_R:
            # Reset environment
            env.reset(seed=42)
        elif key == glfw.KEY_ESCAPE:
            # Close window
            glfw.set_window_should_close(window, True)

def contact_gui_overlay_test():
    """Test contact visualization with MuJoCo overlay."""
    print("🦶 CONTACT VISUALIZATION WITH MUJOCO OVERLAY")
    print("=" * 60)
    
    # Create environment with GUI
    env = PassiveWalkerEnv(mode="fsm", use_gui=True)
    obs, _ = env.reset(seed=42)
    
    print(f"✅ Environment created with GUI")
    print(f"✅ Observation space: {env.observation_space.shape[0]}D")
    
    # Ensure window is created
    env._ensure_window()
    
    if env.window is None:
        print("❌ GUI window creation failed")
        return
    
    print(f"✅ Window: 1200x900 'Passive Walker - v2.1'")
    
    print(f"\n🖥️  GUI WINDOW WITH OVERLAY SHOULD BE VISIBLE NOW!")
    print(f"Look for: 'Passive Walker - v2.1' window")
    print(f"Contact information will be overlaid on the viewer")
    print(f"Controls: R - Reset, ESC - Exit")
    print("=" * 60)
    
    # Get window and rendering components
    window = env.window
    cam = env.cam
    opt = env.opt
    scene = env.scene
    context = env.context
    
    # Set up keyboard callback
    glfw.set_key_callback(window, keyboard_callback)
    
    step_count = 0
    contact_change_count = 0
    last_left_contact = False
    last_right_contact = False
    
    try:
        while not glfw.window_should_close(window):
            # FSM mode - zero action
            action = np.array([0.0, 0.0, 0.0])
            obs, reward, terminated, info = env.step(action)
            
            # Check for contact changes
            left_contact = obs[11] > 0.5
            right_contact = obs[12] > 0.5
            
            if (step_count == 0 or 
                left_contact != last_left_contact or 
                right_contact != last_right_contact):
                contact_change_count += 1
                last_left_contact = left_contact
                last_right_contact = right_contact
            
            step_count += 1
            
            # Get viewport
            w, h = glfw.get_framebuffer_size(window)
            viewport = mujoco.MjrRect(0, 0, w, h)
            
            # Update camera to follow walker
            cam.lookat[0] = obs[0]  # Follow x position
            
            # Update scene
            mujoco.mjv_updateScene(env.model, env.data, opt, None, cam,
                                   mujoco.mjtCatBit.mjCAT_ALL.value, scene)
            
            # Render scene
            mujoco.mjr_render(viewport, scene, context)
            
            # Create and render overlay
            create_contact_overlay(env, obs, step_count, contact_change_count, reward)
            
            # Render overlay
            for gridpos, [t1, t2] in _overlay.items():
                mujoco.mjr_overlay(
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    gridpos,
                    viewport,
                    t1,
                    t2,
                    context
                )
            
            # Clear overlay for next frame
            _overlay.clear()
            
            # Swap buffers
            glfw.swap_buffers(window)
            glfw.poll_events()
            
            if terminated:
                print(f"\nEpisode terminated at step {step_count}")
                break
                
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print(f"\nStopped by user at step {step_count}")
    
    print(f"\n✅ Contact GUI overlay test completed")
    print(f"✅ Total steps: {step_count}")
    print(f"✅ Contact changes: {contact_change_count}")
    
    env.close()

if __name__ == "__main__":
    print("Starting Contact Visualization with MuJoCo Overlay...")
    print("This will show the walker simulation with contact information overlaid")
    print()
    
    contact_gui_overlay_test()
    
    print("\n✅ Contact visualization with overlay completed!")
    print("The overlay showed:")
    print("- Real-time contact status (🟢/🔴)")
    print("- Gait phase (Left Support, Right Support, etc.)")
    print("- Contact forces and durations")
    print("- Position, velocity, and joint angles")
    print("- Simulation progress and controls")