#!/usr/bin/env python3
"""
Comprehensive Observation Visualization with MuJoCo Overlay

Shows the walker simulation with comprehensive observation information overlaid on the MuJoCo viewer.
Displays all background calculations, contact information, physics state, and more.
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

def add_overlay2(gridpos, text1, text2=""):
    """Add text to overlay at specified grid position, with value first then label."""
    if gridpos not in _overlay:
        _overlay[gridpos] = ["", ""]
    _overlay[gridpos][0] += text2 + "\n"
    _overlay[gridpos][1] += text1 + "\n"

def create_comprehensive_overlay(env, obs, step_count, contact_change_count, reward, info):
    """Create comprehensive observation information overlay."""
    # Get contact status
    left_contact = obs[11] > 0.5
    right_contact = obs[12] > 0.5
    
    # Contact icons
    left_icon = "ON" if left_contact else "OFF"
    right_icon = "ON" if right_contact else "OFF"
    
    # Gait phase detection
    if left_contact and not right_contact:
        gait_phase = "Left Support"
    elif right_contact and not left_contact:
        gait_phase = "Right Support"
    elif left_contact and right_contact:
        gait_phase = "Double Support"
    else:
        gait_phase = "Airborne"
    
    # Calculate additional metrics
    velocity_magnitude = np.sqrt(obs[3]**2 + obs[4]**2)
    pitch_deg = np.degrees(obs[2])
    hip_deg = np.degrees(obs[5])
    
    # Energy calculations (kinetic + potential)
    kinetic_energy = 0.5 * velocity_magnitude**2  # Simplified
    potential_energy = 9.81 * abs(obs[1])  # Simplified PE
    total_energy = kinetic_energy + potential_energy
    
    # Stability metrics
    stability_score = 1.0 / (1.0 + abs(pitch_deg))  # Higher is more stable
    forward_progress = obs[0]  # Total distance traveled
    
    # =====================
    # TOP-LEFT: Contact & Gait Information
    # =====================
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        "CONTACT STATUS:",
        f"Left {left_icon} Right {right_icon}"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        "Gait Phase:",
        gait_phase
    )
    
    # =====================
    # LEFT CENTER: Additional Contact Info
    # =====================
    add_overlay2(
        mujoco.mjtGridPos.mjGRID_LEFT,
        "Contact Forces:",
        f"L: {obs[13]:.2f}N  R: {obs[14]:.2f}N",
    )
    
    add_overlay2(
        mujoco.mjtGridPos.mjGRID_LEFT,
        "Contact Duration:",
        f"L: {obs[15]:.2f}s  R: {obs[16]:.2f}s",
    )
    
    add_overlay2(
        mujoco.mjtGridPos.mjGRID_LEFT,
        "Observation Space:",
        f"17D (11D + 6D contact)",
    )
    
    add_overlay2(
        mujoco.mjtGridPos.mjGRID_LEFT,
        "Contact Threshold:",
        "0.1N",
    )
    
    add_overlay2(
        mujoco.mjtGridPos.mjGRID_LEFT,
        "SIMULATION:",
        f"Step: {step_count}  Time: {env.data.time:.2f}s",
    )
    
    add_overlay2(
        mujoco.mjtGridPos.mjGRID_LEFT,
        "Contact Changes:",
        f"{contact_change_count}",
    )

    add_overlay2(
        mujoco.mjtGridPos.mjGRID_LEFT,
        "Reward:",
        f"{reward:.3f}",
    )
    
    add_overlay2(
        mujoco.mjtGridPos.mjGRID_LEFT,
        "Episode Progress:",
        f"{env.data.time/env.simend*100:.1f}%",
    )
    
    # =====================
    # TOP-RIGHT: Position & Movement
    # =====================
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        "POSITION:",
        f"x: {obs[0]:.2f}m  z: {obs[1]:.2f}m"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        "Velocity:",
        f"x: {obs[3]:.2f}m/s  z: {obs[4]:.2f}m/s"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        "Speed:",
        f"{velocity_magnitude:.2f}m/s"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
        "Progress:",
        f"{forward_progress:.2f}m"
    )
    
    # =====================
    # BOTTOM-LEFT: Joint States
    # =====================
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
        "JOINT ANGLES:",
        f"Hip: {hip_deg:.1f}deg"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
        "Knee Positions:",
        f"L: {obs[6]:.3f}m  R: {obs[7]:.3f}m"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
        "Joint Velocities:",
        f"Hip: {np.degrees(obs[8]):.1f}deg/s"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
        "Knee Velocities:",
        f"L: {obs[9]:.3f}m/s  R: {obs[10]:.3f}m/s"
    )
    
    # =====================
    # BOTTOM-RIGHT: Physics & Stability
    # =====================
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
        "PHYSICS STATE:",
        f"Pitch: {pitch_deg:.1f}deg"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
        "Stability:",
        f"{stability_score:.3f}"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
        "Energy:",
        f"KE: {kinetic_energy:.2f}J"
    )
    
    add_overlay(
        mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
        "Total Energy:",
        f"{total_energy:.2f}J"
    )

    
    # =====================
    # RIGHT CENTER: Additional Info from Step
    # =====================
    if info:
        add_overlay(
            mujoco.mjtGridPos.mjGRID_RIGHT,
            "STEP INFO:",
            f"dx: {info.get('dx', 0):.4f}m"
        )
        
        add_overlay(
            mujoco.mjtGridPos.mjGRID_RIGHT,
            "Pitch Abs:",
            f"{info.get('pitch_abs', 0):.3f}rad"
        )
        
        add_overlay(
            mujoco.mjtGridPos.mjGRID_RIGHT,
            "Torso Z:",
            f"{info.get('torso_z', 0):.3f}m"
        )
        
        add_overlay(
            mujoco.mjtGridPos.mjGRID_RIGHT,
            "Control Sum:",
            f"{info.get('u_abs_sum', 0):.3f}"
        )

def keyboard_callback(window, key, scancode, act, mods):
    """Handle keyboard input."""
    global env
    if act == glfw.PRESS:
        if key == glfw.KEY_R:
            # Reset environment
            env.reset(seed=42)
            print("Environment reset!")
        elif key == glfw.KEY_ESCAPE:
            # Close window
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            # Pause/unpause
            print("Pause/Resume (implement pause logic if needed)")
        elif key == glfw.KEY_S:
            # Save screenshot
            print("Screenshot saved (implement if needed)")

def comprehensive_observation_test():
    """Test comprehensive observation visualization with MuJoCo overlay."""
    print("COMPREHENSIVE OBSERVATION VISUALIZATION")
    print("=" * 70)
    
    # Create environment with GUI
    env = PassiveWalkerEnv(mode="fsm", use_gui=True)
    obs, _ = env.reset(seed=42)
    
    print(f"Environment created with GUI")
    print(f"Observation space: {env.observation_space.shape[0]}D")
    
    # Ensure window is created
    env._ensure_window()
    
    if env.window is None:
        print("GUI window creation failed")
        return
    
    print(f"Window: 1200x900 'Passive Walker - v2.1'")
    
    print(f"\nCOMPREHENSIVE OVERLAY VISUALIZATION ACTIVE!")
    print(f"Look for: 'Passive Walker - v2.1' window")
    print(f"All observation data will be overlaid on the viewer")
    print(f"Controls: R - Reset, ESC - Exit, SPACE - Pause, S - Screenshot")
    print("=" * 70)
    
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
    
    # Statistics tracking
    max_velocity = 0.0
    max_pitch = 0.0
    total_distance = 0.0
    last_x = obs[0]
    
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
            
            # Update statistics
            velocity_magnitude = np.sqrt(obs[3]**2 + obs[4]**2)
            max_velocity = max(max_velocity, velocity_magnitude)
            max_pitch = max(max_pitch, abs(np.degrees(obs[2])))
            total_distance += abs(obs[0] - last_x)
            last_x = obs[0]
            
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
            
            # Create and render comprehensive overlay
            create_comprehensive_overlay(env, obs, step_count, contact_change_count, reward, info)
            
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
    
    # Print final statistics
    print(f"\nFINAL STATISTICS:")
    print(f"Total steps: {step_count}")
    print(f"Contact changes: {contact_change_count}")
    print(f"Max velocity: {max_velocity:.2f}m/s")
    print(f"Max pitch: {max_pitch:.1f}°")
    print(f"Total distance: {total_distance:.2f}m")
    print(f"Final observation shape: {obs.shape}")
    
    env.close()

def print_observation_breakdown():
    """Print detailed breakdown of observation space."""
    print("\nOBSERVATION SPACE BREAKDOWN:")
    print("=" * 50)
    print("Original 11D observations:")
    print("  [0]  x position (m)")
    print("  [1]  z position (m)")
    print("  [2]  pitch angle (rad)")
    print("  [3]  x velocity (m/s)")
    print("  [4]  z velocity (m/s)")
    print("  [5]  hip angle (rad)")
    print("  [6]  left knee slider (m)")
    print("  [7]  right knee slider (m)")
    print("  [8]  hip angular velocity (rad/s)")
    print("  [9]  left knee velocity (m/s)")
    print("  [10] right knee velocity (m/s)")
    print("\nEnhanced 6D contact observations:")
    print("  [11] left contact flag (0/1)")
    print("  [12] right contact flag (0/1)")
    print("  [13] left contact force (N)")
    print("  [14] right contact force (N)")
    print("  [15] left contact duration (s)")
    print("  [16] right contact duration (s)")
    print("=" * 50)

if __name__ == "__main__":
    print("Starting Comprehensive Observation Visualization...")
    print("This will show ALL observation data overlaid on the MuJoCo viewer")
    print()
    
    print_observation_breakdown()
    
    comprehensive_observation_test()
    
    print("\nComprehensive observation visualization completed!")
    print("The overlay showed:")
    print("- Contact status and gait phases")
    print("- Position, velocity, and movement metrics")
    print("- Joint angles and velocities")
    print("- Physics state and stability metrics")
    print("- Simulation progress and statistics")
    print("- Step-by-step information")
    print("\nThis comprehensive view helps understand:")
    print("- How contact information changes during walking")
    print("- Relationship between joint states and gait phases")
    print("- Physics calculations and stability metrics")
    print("- Real-time observation space composition")
