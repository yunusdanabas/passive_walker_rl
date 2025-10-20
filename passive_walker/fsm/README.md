# FSM (Finite State Machine) Module

Rule-based controller for bipedal walking using finite state machine. Generates stable walking gaits for expert demonstration data collection.

## Purpose

The FSM controller produces expert demonstrations for Behavior Cloning (BC) training. It maintains stable walking without falling, making it ideal for training data generation.

## Data Collection

### Basic Usage

```bash
# Collect FSM data for BC training
python -m passive_walker.fsm.collect --episodes 200 --duration 25.0 --out data/fsm_training
```

### Key Parameters

- `--episodes`: Number of episodes to collect (default: 80)
- `--duration`: Episode duration in seconds (default: 25.0)
- `--physics`: Physics preset ("nominal", "gentle", "low_friction") (default: "nominal")
- `--out`: Output directory (default: "data/fsm_runs")
- `--seed`: Random seed for reproducibility (default: 123)

### Quality Guidelines

- Use 25.0s episodes for sufficient gait cycles (target: 8-10+ per episode)
- Collect 200+ episodes minimum for diversity
- Fall rate should be <10% (ideally 0%)

## Output Data Schema

Each episode generates:
- `episode_XXXXXX.npz` - Episode data with observations, actions, rewards, and FSM states
- `meta.json` - Collection metadata
- `README.json` - Quality analysis report

Episode data contains:
- `obs`: Observations (T, 11)
- `act`: Actions (T, 3) - zeros for FSM mode
- `rew`: Rewards (T,)
- `done`: Termination flags (T,)
- `info_fsm_hip/k1/k2`: FSM state sequences
- `info_qdes`: Desired joint positions (T, 3)
