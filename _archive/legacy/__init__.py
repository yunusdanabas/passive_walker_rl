"""
Legacy Passive Walker RL Environment

This package contains the original codebase that has been migrated to the new unified core architecture.
This code is FROZEN and kept for reference only.

For new development, use the main passive_walker package instead.
"""

__version__ = "0.1.0-legacy"

# Import main components for backward compatibility
from .envs.mujoco_env import MuJoCoEnv
from .envs.mujoco_fsm_env import MuJoCoFSMEnv
from .utils.control import PDController as LegacyPDController
from .utils.smooth_rewards import get_reward_fn as legacy_get_reward_fn
from .utils.rollout_buffer import RolloutBuffer as LegacyRolloutBuffer
from .utils.data_collection import collect_fsm_data as legacy_collect_fsm_data

__all__ = [
    "MuJoCoEnv",
    "MuJoCoFSMEnv", 
    "LegacyPDController",
    "legacy_get_reward_fn",
    "LegacyRolloutBuffer",
    "legacy_collect_fsm_data",
]
