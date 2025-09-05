# Legacy Code

This directory contains the original codebase that has been migrated to the new unified core architecture.

## Status: FROZEN

This code is **frozen** and should not be modified. It is kept for reference only.

## Migration

See `MIGRATION.md` in the root directory for the complete mapping of old components to new ones.

## What's Here

- `envs/` - Original environment implementations (mujoco_env.py, mujoco_fsm_env.py)
- `controllers/` - FSM controller methods
- `utils/` - Utility functions (control, data collection, rewards, etc.)
- `ppo/` - PPO training scripts
- `bc/` - Behavioral cloning scripts
- `brax/` - Brax integration code
- `mujoco_*_oldCode.py` - Old code files

## New Architecture

All functionality has been migrated to the new unified core:

- **Single Environment**: `passive_walker/core/env.py`
- **Reward System**: `passive_walker/core/reward.py`
- **JAX Utils**: `passive_walker/core/jax_utils.py`
- **Rollout Buffer**: `passive_walker/core/rollout_buffer.py`
- **Training Scripts**: `passive_walker/scripts/`

## Usage

**Do not import from this directory.** Use the new core modules instead:

```python
# OLD (don't use)
from passive_walker.envs.mujoco_env import MuJoCoEnv

# NEW (use this)
from passive_walker.core.env import PassiveWalkerEnv
```

## Cleanup

This directory can be safely removed once you're confident the new architecture meets all your needs. Consider creating a git tag before removal:

```bash
git tag legacy-frozen
git rm -r _legacy/
```
