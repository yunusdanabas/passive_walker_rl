"""
Passive Walker RL Environment

A unified, high-performance reinforcement learning environment for a Variable Length Leg (VLL) bipedal walker.
"""

__version__ = "0.1.0"

# Core modules
from .core.env import PassiveWalkerEnv
from .core.io import load_walker_config
from .core.reward import get_reward_fn
from .core.rollout_buffer import RolloutBuffer, MultiEnvRolloutBuffer

__all__ = [
    "PassiveWalkerEnv",
    "load_walker_config",
    "get_reward_fn",
    "RolloutBuffer",
    "MultiEnvRolloutBuffer",
]
