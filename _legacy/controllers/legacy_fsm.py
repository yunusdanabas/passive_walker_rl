"""
Legacy FSM Controller Methods

This file contains the original FSM controller methods that were used in the early
versions of the passive walker environment. These methods directly set control values
and have been replaced by the new architecture that separates FSM state management
from PD control.

The new architecture uses:
- _update_fsm_hip_state() and _update_fsm_knee_states() for state management
- _apply_pd_control() for PD control and actuator commands

These legacy methods are preserved for reference and potential future use.
"""

import numpy as np
import mujoco
from passive_walker.envs.mujoco_fsm_env import quat2euler

# FSM target positions (same as in main environments)
HIP_SWING_POS = 0.5    # Hip forward swing (radians)
HIP_SWING_NEG = -0.5   # Hip backward swing (radians)
KNEE_STANCE = 0.0      # Knee stance phase (meters)
KNEE_RETRACT = -0.25   # Knee retraction phase (meters)


class LegacyFSMController:
    """
    Legacy FSM Controller that directly sets control values.
    
    This is the original FSM implementation that directly manipulated data.ctrl
    values. It has been replaced by the new architecture that separates state
    management from control application.
    """
    
    def __init__(self, model, data, left_leg_body_id, right_leg_body_id, 
                 left_foot_body_id, right_foot_body_id, hip_pos_actuator_id,
                 left_knee_pos_actuator_id, right_knee_pos_actuator_id):
        """
        Initialize the legacy FSM controller.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            left_leg_body_id: Left leg body ID
            right_leg_body_id: Right leg body ID
            left_foot_body_id: Left foot body ID
            right_foot_body_id: Right foot body ID
            hip_pos_actuator_id: Hip actuator ID
            left_knee_pos_actuator_id: Left knee actuator ID
            right_knee_pos_actuator_id: Right knee actuator ID
        """
        self.model = model
        self.data = data
        self.left_leg_body_id = left_leg_body_id
        self.right_leg_body_id = right_leg_body_id
        self.left_foot_body_id = left_foot_body_id
        self.right_foot_body_id = right_foot_body_id
        self.hip_pos_actuator_id = hip_pos_actuator_id
        self.left_knee_pos_actuator_id = left_knee_pos_actuator_id
        self.right_knee_pos_actuator_id = right_knee_pos_actuator_id
        
        # FSM state definitions for hip control
        self.FSM_HIP_LEG1_SWING = 0  # Left leg swinging
        self.FSM_HIP_LEG2_SWING = 1  # Right leg swinging
        self.fsm_hip = self.FSM_HIP_LEG2_SWING  # Initial state
        
        # FSM state definitions for knee control
        self.FSM_KNEE1_STANCE = 0    # Left knee in stance phase
        self.FSM_KNEE1_RETRACT = 1   # Left knee in retraction phase
        self.FSM_KNEE2_STANCE = 0    # Right knee in stance phase
        self.FSM_KNEE2_RETRACT = 1   # Right knee in retraction phase
        self.fsm_knee1 = self.FSM_KNEE1_STANCE
        self.fsm_knee2 = self.FSM_KNEE2_STANCE

    def controller_fsm_hip(self):
        """
        Legacy FSM logic for the hip joint control.
        
        This method directly sets control values based on FSM state transitions.
        It has been replaced by the new architecture that separates state management
        from control application.
        
        WARNING: This method directly modifies data.ctrl values and should not be
        used with the new PD control system as it will cause conflicts.
        """
        # Get left leg state using MuJoCo's optimized quat2euler
        quat_left = self.data.xquat[self.left_leg_body_id, :]
        euler_left = quat2euler(quat_left)
        abs_left = -euler_left[1]  # Negative because of coordinate system
        pos_leftFoot = self.data.xpos[self.left_foot_body_id, :]

        # Get right leg state using MuJoCo's optimized quat2euler
        quat_right = self.data.xquat[self.right_leg_body_id, :]
        euler_right = quat2euler(quat_right)
        abs_right = -euler_right[1]  # Negative because of coordinate system
        pos_rightFoot = self.data.xpos[self.right_foot_body_id, :]

        # State transitions based on foot contact and leg angles
        if self.fsm_hip == self.FSM_HIP_LEG2_SWING and pos_rightFoot[2] < 0.05 and abs_left < 0.0:
            self.fsm_hip = self.FSM_HIP_LEG1_SWING
        elif self.fsm_hip == self.FSM_HIP_LEG1_SWING and pos_leftFoot[2] < 0.05 and abs_right < 0.0:
            self.fsm_hip = self.FSM_HIP_LEG2_SWING

        # LEGACY: Direct control value setting (replaced by PD control)
        if self.fsm_hip == self.FSM_HIP_LEG1_SWING:
            self.data.ctrl[self.hip_pos_actuator_id] = HIP_SWING_NEG
        else:
            self.data.ctrl[self.hip_pos_actuator_id] = HIP_SWING_POS

    def controller_fsm_knees(self):
        """
        Legacy FSM logic for the knee joints control.
        
        This method directly sets control values based on FSM state transitions.
        It has been replaced by the new architecture that separates state management
        from control application.
        
        WARNING: This method directly modifies data.ctrl values and should not be
        used with the new PD control system as it will cause conflicts.
        """
        # Get left leg state
        quat_leftLeg = self.data.xquat[self.left_leg_body_id, :]
        euler_leftLeg = quat2euler(quat_leftLeg)
        abs_leftLeg = -euler_leftLeg[1]  # Negative because of coordinate system
        pos_leftFoot = self.data.xpos[self.left_foot_body_id, :]

        # Get right leg state
        quat_rightLeg = self.data.xquat[self.right_leg_body_id, :]
        euler_rightLeg = quat2euler(quat_rightLeg)
        abs_rightLeg = -euler_rightLeg[1]  # Negative because of coordinate system
        pos_rightFoot = self.data.xpos[self.right_foot_body_id, :]

        # Left knee state transitions
        if self.fsm_knee1 == self.FSM_KNEE1_STANCE and pos_rightFoot[2] < 0.05 and abs_leftLeg < 0.0:
            self.fsm_knee1 = self.FSM_KNEE1_RETRACT
        elif self.fsm_knee1 == self.FSM_KNEE1_RETRACT and abs_leftLeg > 0.1:
            self.fsm_knee1 = self.FSM_KNEE1_STANCE

        # Right knee state transitions
        if self.fsm_knee2 == self.FSM_KNEE2_STANCE and pos_leftFoot[2] < 0.05 and abs_rightLeg < 0.0:
            self.fsm_knee2 = self.FSM_KNEE2_RETRACT
        elif self.fsm_knee2 == self.FSM_KNEE2_RETRACT and abs_rightLeg > 0.1:
            self.fsm_knee2 = self.FSM_KNEE2_STANCE

        # LEGACY: Direct control value setting (replaced by PD control)
        if self.fsm_knee1 == self.FSM_KNEE1_STANCE:
            self.data.ctrl[self.left_knee_pos_actuator_id] = KNEE_STANCE
        else:
            self.data.ctrl[self.left_knee_pos_actuator_id] = KNEE_RETRACT

        if self.fsm_knee2 == self.FSM_KNEE2_STANCE:
            self.data.ctrl[self.right_knee_pos_actuator_id] = KNEE_STANCE
        else:
            self.data.ctrl[self.right_knee_pos_actuator_id] = KNEE_RETRACT

    def reset_fsm_states(self):
        """Reset FSM states to initial values."""
        self.fsm_hip = self.FSM_HIP_LEG2_SWING
        self.fsm_knee1 = self.FSM_KNEE1_STANCE
        self.fsm_knee2 = self.FSM_KNEE2_STANCE

    def get_fsm_states(self):
        """Get current FSM states."""
        return {
            'hip': self.fsm_hip,
            'knee1': self.fsm_knee1,
            'knee2': self.fsm_knee2
        }


# Standalone functions for backward compatibility
def legacy_controller_fsm_hip(data, left_leg_body_id, right_leg_body_id, 
                             left_foot_body_id, right_foot_body_id, 
                             hip_pos_actuator_id, fsm_hip):
    """
    Legacy standalone FSM hip controller function.
    
    This is the original implementation that directly set control values.
    It has been replaced by the new PD control architecture.
    
    Args:
        data: MuJoCo data
        left_leg_body_id: Left leg body ID
        right_leg_body_id: Right leg body ID
        left_foot_body_id: Left foot body ID
        right_foot_body_id: Right foot body ID
        hip_pos_actuator_id: Hip actuator ID
        fsm_hip: Current FSM hip state (modified in place)
    
    Returns:
        Updated FSM hip state
    """
    # FSM state constants
    FSM_HIP_LEG1_SWING = 0
    FSM_HIP_LEG2_SWING = 1
    
    # Get left leg state
    quat_left = data.xquat[left_leg_body_id, :]
    euler_left = quat2euler(quat_left)
    abs_left = -euler_left[1]
    pos_leftFoot = data.xpos[left_foot_body_id, :]

    # Get right leg state
    quat_right = data.xquat[right_leg_body_id, :]
    euler_right = quat2euler(quat_right)
    abs_right = -euler_right[1]
    pos_rightFoot = data.xpos[right_foot_body_id, :]

    # State transitions
    if fsm_hip == FSM_HIP_LEG2_SWING and pos_rightFoot[2] < 0.05 and abs_left < 0.0:
        fsm_hip = FSM_HIP_LEG1_SWING
    elif fsm_hip == FSM_HIP_LEG1_SWING and pos_leftFoot[2] < 0.05 and abs_right < 0.0:
        fsm_hip = FSM_HIP_LEG2_SWING

    # LEGACY: Direct control setting
    if fsm_hip == FSM_HIP_LEG1_SWING:
        data.ctrl[hip_pos_actuator_id] = HIP_SWING_NEG
    else:
        data.ctrl[hip_pos_actuator_id] = HIP_SWING_POS

    return fsm_hip


def legacy_controller_fsm_knees(data, left_leg_body_id, right_leg_body_id,
                               left_foot_body_id, right_foot_body_id,
                               left_knee_pos_actuator_id, right_knee_pos_actuator_id,
                               fsm_knee1, fsm_knee2):
    """
    Legacy standalone FSM knees controller function.
    
    This is the original implementation that directly set control values.
    It has been replaced by the new PD control architecture.
    
    Args:
        data: MuJoCo data
        left_leg_body_id: Left leg body ID
        right_leg_body_id: Right leg body ID
        left_foot_body_id: Left foot body ID
        right_foot_body_id: Right foot body ID
        left_knee_pos_actuator_id: Left knee actuator ID
        right_knee_pos_actuator_id: Right knee actuator ID
        fsm_knee1: Current FSM left knee state (modified in place)
        fsm_knee2: Current FSM right knee state (modified in place)
    
    Returns:
        Tuple of updated FSM knee states (fsm_knee1, fsm_knee2)
    """
    # FSM state constants
    FSM_KNEE1_STANCE = 0
    FSM_KNEE1_RETRACT = 1
    FSM_KNEE2_STANCE = 0
    FSM_KNEE2_RETRACT = 1
    
    # Get left leg state
    quat_leftLeg = data.xquat[left_leg_body_id, :]
    euler_leftLeg = quat2euler(quat_leftLeg)
    abs_leftLeg = -euler_leftLeg[1]
    pos_leftFoot = data.xpos[left_foot_body_id, :]

    # Get right leg state
    quat_rightLeg = data.xquat[right_leg_body_id, :]
    euler_rightLeg = quat2euler(quat_rightLeg)
    abs_rightLeg = -euler_rightLeg[1]
    pos_rightFoot = data.xpos[right_foot_body_id, :]

    # Left knee state transitions
    if fsm_knee1 == FSM_KNEE1_STANCE and pos_rightFoot[2] < 0.05 and abs_leftLeg < 0.0:
        fsm_knee1 = FSM_KNEE1_RETRACT
    elif fsm_knee1 == FSM_KNEE1_RETRACT and abs_leftLeg > 0.1:
        fsm_knee1 = FSM_KNEE1_STANCE

    # Right knee state transitions
    if fsm_knee2 == FSM_KNEE2_STANCE and pos_leftFoot[2] < 0.05 and abs_rightLeg < 0.0:
        fsm_knee2 = FSM_KNEE2_RETRACT
    elif fsm_knee2 == FSM_KNEE2_RETRACT and abs_rightLeg > 0.1:
        fsm_knee2 = FSM_KNEE2_STANCE

    # LEGACY: Direct control setting
    if fsm_knee1 == FSM_KNEE1_STANCE:
        data.ctrl[left_knee_pos_actuator_id] = KNEE_STANCE
    else:
        data.ctrl[left_knee_pos_actuator_id] = KNEE_RETRACT

    if fsm_knee2 == FSM_KNEE2_STANCE:
        data.ctrl[right_knee_pos_actuator_id] = KNEE_STANCE
    else:
        data.ctrl[right_knee_pos_actuator_id] = KNEE_RETRACT

    return fsm_knee1, fsm_knee2
