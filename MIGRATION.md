# Migration Map

This document summarizes where each old component will land in the new unified architecture.

## Core Components

* `envs/mujoco_env.py` + `envs/mujoco_fsm_env.py` → `core/env.py` (modes via cfg)
* `utils/smooth_rewards.py` → `core/reward.py` (presets from YAML)
* `utils/jax_control.py` → `core/jax_utils.py` (PD/quat/batched rewards)
* `utils/rollout_buffer.py` → `core/rollout_buffer.py`
* `utils/data_collection.py` → `scripts/collect_fsm_data.py`
* `passiveWalker_model.xml` → `assets/passiveWalker_model.xml`

## Configuration

* Hardcoded parameters → `configs/walker.yaml` + `configs/reward_presets.yaml`
* Dataclasses for type safety → `core/config.py`
* YAML loading utilities → `core/io.py`

## Legacy Code

* Everything else → `_legacy/` (kept for reference)
  - `_legacy/envs/` - Original environment implementations
  - `_legacy/controllers/` - FSM controller methods
  - `_legacy/utils/` - Utility functions
  - `_legacy/ppo/` - PPO training modules
  - `_legacy/bc/` - Behavior cloning modules
  - `_legacy/brax/` - Brax integration

## Migration Steps

1. **Step 1**: Implement `core/env.py` with unified environment
2. **Step 2**: Implement `core/reward.py` with preset system
3. **Step 3**: Implement `core/jax_utils.py` with optimized functions
4. **Step 4**: Implement `core/rollout_buffer.py` with memory pooling
5. **Step 5**: Implement `scripts/collect_fsm_data.py` for data collection
