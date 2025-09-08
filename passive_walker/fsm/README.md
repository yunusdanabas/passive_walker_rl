# FSM Data Collection

This module handles collection of expert demonstration data using the Finite State Machine (FSM) controller.

## Overview

The FSM controller provides expert demonstrations by using a state machine to control the walker's hip and knee joints. This data is then used to train behavioral cloning policies.

## Usage

### Collect FSM Data

```bash
# Basic collection
walker.collect_fsm --data-dir data/bc/raw --num-episodes 100

# With custom config
walker.collect_fsm --config passive_walker/fsm/fsm_collect.yaml --num-episodes 200

# With GUI for monitoring
walker.collect_fsm --gui --num-episodes 50

# Headless mode (default)
walker.collect_fsm --no-gui --num-episodes 1000
```

### Configuration

The FSM collection uses `passive_walker/fsm/fsm_collect.yaml` by default. Key parameters:

- `env.simend`: Episode length in seconds (default: 20.0)
- `physics.ramp_deg_min/max`: Ramp angle range (default: 10.0, 10.0)
- `physics.friction`: Friction coefficient range (default: [0.8, 0.8])
- `fsm.*`: FSM controller parameters (contact height, thresholds, etc.)

### Output

Episodes are saved to `data/bc/raw/YYYYmmdd-HHMMSS-fsm_collection_*/` with:

- `episode_XXXXXX.npz`: Individual episode data (observations, actions, rewards, dones, infos)
- `meta.json`: Collection metadata (git SHA, config, seeds, etc.)
- `collection_summary.json`: Summary statistics

## FSM Controller

The FSM controller uses contact detection to determine when to swing or stance each leg:

1. **Contact Detection**: Uses foot height to detect ground contact
2. **Hip Control**: Swings hip forward/backward based on contact state
3. **Knee Control**: Retracts knee during swing, extends during stance

## Data Format

Each episode NPZ file contains:

- `observations`: [T, 11] - State observations
- `actions`: [T, 3] - Actions (all zeros for FSM mode)
- `rewards`: [T] - Step rewards
- `dones`: [T] - Episode termination flags
- `infos`: [T] - Additional info dicts

## Troubleshooting

- **Short episodes**: Check termination conditions in config
- **No contact detection**: Verify foot body IDs in FSM setup
- **GUI issues**: Use `--no-gui` for headless collection

