# Pipeline Overview

**Goal:** show `core → fsm → bc → ppo` at a glance.

## Pipeline Stages

1. **`core/`** — Environment, controller, rewards, configuration
2. **`fsm/`** — Collect episodes with FSM control → `data/bc/raw/*.npz`
3. **`bc/`** — Train BC policy on collected episodes → `results/bc/.../policy.pt`
4. **`ppo/`** — RL fine-tuning, optionally init from BC

## Fast Paths

### Collect FSM Data
```bash
walker-collect-fsm --num_episodes 100 --rollout_len 200
```

### Train BC Policy
```bash
walker-train-bc --data_dir data/bc/raw
```

### Train PPO Policy
```bash
walker-train-ppo --bc_init results/bc/.../policy.pt
```

## Configuration

Each stage has its own YAML config:
- `passive_walker/fsm/fsm_collect.yaml`
- `passive_walker/bc/bc_train.yaml`
- `passive_walker/ppo/ppo_train.yaml`

Use `--constraints` to override with additional configs:
```bash
walker-collect-fsm --constraints passive_walker/configs/constraints/dr_light.yaml
```

## Output Structure

```
data/
  bc/raw/                    # FSM episodes
    episode_000000.npz
    episode_000001.npz
    collection_summary.json

results/
  bc/20250908-123456-bc/     # BC training run
    meta.json
    train.csv
    policy.pt
    policy_best.pt
    tb/                       # TensorBoard logs (if enabled)

  ppo/20250908-123456-ppo/   # PPO training run
    meta.json
    train.csv
    ppo_final.pt
    tb/                       # TensorBoard logs (if enabled)
```

## Logging

- **CSV logs**: Training metrics in `train.csv`
- **Meta files**: Reproducibility info in `meta.json`
- **TensorBoard**: Optional with `--enable_tb` in config
- **Weights & Biases**: Optional with `--enable_wandb` in config

