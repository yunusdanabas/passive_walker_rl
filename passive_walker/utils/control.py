"""
VLL Walker Control Utilities

PD control with neutral general actuators for MuJoCo/Brax compatibility.
Provides centralized control logic for both FSM and research environments.
"""

import numpy as np

# ============================================================================
# Physical Parameters
# ============================================================================

# Physical joint ranges
HIP_RANGE = (-0.5, 0.5)      # radians
KNEE_RANGE = (-0.5, 0.5)     # meters (slider displacement)

# PD gains (kp, kv) for each joint
# Conservative gains for stability
PD_GAINS = {
    'hip': (5.0, 1.0),           # hip hinge
    'left_knee': (1000.0, 100.0),  # left knee slider
    'right_knee': (1000.0, 100.0), # right knee slider
}

# Control limits (must match XML ctrlrange)
CTRL_LIMITS = {
    'hip': (-50, 50),           # hip torque
    'left_knee': (-800, 800),   # left knee force
    'right_knee': (-800, 800),  # right knee force
}


# ============================================================================
# Control Functions
# ============================================================================

def denormalize_action(action, joint_ranges):
    """
    Convert [-1, 1] normalized actions to physical joint ranges.
    
    This function maps normalized actions from [-1, 1] to the physical ranges
    of each joint. For example:
    - action = 0.0 → middle of range
    - action = 1.0 → maximum of range  
    - action = -1.0 → minimum of range
    
    Args:
        action: Normalized action array in [-1, 1]
        joint_ranges: List of (min, max) tuples for each joint
        
    Returns:
        list: Denormalized joint positions in physical units
    """
    return [
        lo + 0.5 * (val + 1.0) * (hi - lo) 
        for val, (lo, hi) in zip(action, joint_ranges)
    ]


def clamp_desired_positions(q_desired, joint_ranges):
    """
    Clamp desired joint positions to physical ranges.
    
    Ensures that desired positions stay within physically achievable limits
    to prevent the PD controller from trying to reach impossible positions.
    
    Args:
        q_desired: Desired joint positions
        joint_ranges: List of (min, max) tuples for each joint
        
    Returns:
        list: Clamped desired positions within physical ranges
    """
    return [
        np.clip(q_desired[i], joint_ranges[i][0], joint_ranges[i][1])
        for i in range(len(q_desired))
    ]


def compute_pd_control(q_current, qd_current, q_desired, joint_names, kp_kv_dict, ctrl_limits, joint_ranges=None):
    """
    Compute PD control torques/forces for all joints.
    
    Args:
        q_current: Current joint positions
        qd_current: Current joint velocities  
        q_desired: Desired joint positions
        joint_names: List of joint names ['hip', 'left_knee', 'right_knee']
        kp_kv_dict: Dict mapping joint names to (kp, kv) tuples
        ctrl_limits: Dict mapping joint names to (min, max) control limits
        joint_ranges: Optional list of (min, max) tuples for clamping desired positions
        
    Returns:
        list: Control torques/forces for each joint
    """

    # Clamp desired positions to physical ranges if provided
    if joint_ranges is not None:
        q_desired = clamp_desired_positions(q_desired, joint_ranges)
            
    controls = []
    
    for i, joint_name in enumerate(joint_names):
        kp, kv = kp_kv_dict[joint_name]
        min_ctrl, max_ctrl = ctrl_limits[joint_name]
        
        # PD control: u = kp * error - kv * velocity
        error = q_desired[i] - q_current[i]
        u = kp * error - kv * qd_current[i]
        
        # Clamp to control limits
        u = np.clip(u, min_ctrl, max_ctrl)
        controls.append(u)
    
    return controls


# ============================================================================
# Getter Functions
# ============================================================================

def get_joint_ranges():
    """Get joint ranges for denormalization."""
    return [HIP_RANGE, KNEE_RANGE, KNEE_RANGE]


def get_pd_gains():
    """Get PD gains for all joints."""
    return PD_GAINS


def get_ctrl_limits():
    """Get control limits for all joints."""
    return CTRL_LIMITS
