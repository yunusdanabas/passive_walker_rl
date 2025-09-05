# Migration Map

This document summarizes where each old component will land in the new unified architecture.

## ✅ Migration Status

- [x] **Step 0**: Stabilize + Set Target Skeleton
- [x] **Step 1**: Implement unified environment (`core/env.py`)
- [x] **Step 2**: Reward module (presets + tiny API)
- [x] **Step 3**: JAX utils (pd + quat + batched reward)
- [x] **Step 4**: Rollout buffer (single + optional multi-env)
- [x] **Step 5**: Wire-up + Docs + Cleanup

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

## New Training Scripts

* `scripts/train_bc.py` - Behavioral cloning training
* `scripts/train_ppo.py` - PPO training
* `configs/fsm_collect.yaml` - FSM data collection config
* `configs/bc_eval.yaml` - BC evaluation config
* `configs/ppo_train.yaml` - PPO training config

## Legacy Code

* Everything else → `_legacy/` (kept for reference)
  - `_legacy/envs/` - Original environment implementations
  - `_legacy/controllers/` - FSM controller methods
  - `_legacy/utils/` - Utility functions
  - `_legacy/ppo/` - PPO training modules
  - `_legacy/bc/` - Behavior cloning modules
  - `_legacy/brax/` - Brax integration

## Step 5 Tasks Completed

- [x] Create `train_bc.py` script for behavioral cloning
- [x] Create `train_ppo.py` script for PPO training
- [x] Create workflow configs (fsm_collect, bc_eval, ppo_train)
- [x] Update README.md with comprehensive quickstart guide
- [x] Clean up legacy code and ensure no imports from `_legacy/`
- [x] Run end-to-end tests and final acceptance
