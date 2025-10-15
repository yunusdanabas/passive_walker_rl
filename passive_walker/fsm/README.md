# FSM (Finite State Machine) Module

This module implements a rule-based controller for bipedal walking using a finite state machine. The FSM generates stable walking gaits by transitioning between discrete states based on foot contact detection and leg pitch angles.

## Overview

The FSM controller is designed to produce expert demonstrations for Behavior Cloning (BC) training. It uses simple rules to maintain stable walking without falling, making it an ideal source of training data.

### Key Features

- ✅ **Stable walking**: Achieves 20+ second episodes without falling
- ✅ **Discrete state space**: 8 possible states (0-7) for predictable patterns
- ✅ **High-quality data**: Generates consistent gait cycles and forward progress
- ✅ **Robust performance**: 0% fall rate under nominal conditions
- ✅ **Visual confirmation**: Stable walking gait verified through GUI testing

## FSM State Machine Logic

### State Components

The FSM consists of three independent state machines:

1. **Hip State Machine** (2 states):
   - `0`: Left leg forward swing
   - `1`: Right leg forward swing

2. **Left Knee State Machine** (2 states):
   - `0`: Stance phase (knee extended)
   - `1`: Retract phase (knee retracted)

3. **Right Knee State Machine** (2 states):
   - `0`: Stance phase (knee extended)
   - `1`: Retract phase (knee retracted)

### Combined State Space

The combined state is calculated as: `hip_state * 4 + left_knee_state * 2 + right_knee_state`

This gives 8 possible states (0-7):

| State | Hip | Left Knee | Right Knee | Description |
|-------|-----|-----------|------------|-------------|
| 0 | Left (0) | Stance (0) | Stance (0) | Left forward, both stance |
| 1 | Left (0) | Stance (0) | Retract (1) | Left forward, LK stance, RK retract |
| 2 | Left (0) | Retract (1) | Stance (0) | Left forward, LK retract, RK stance |
| 3 | Left (0) | Retract (1) | Retract (1) | Left forward, both retract |
| 4 | Right (1) | Stance (0) | Stance (0) | Right forward, both stance |
| 5 | Right (1) | Stance (0) | Retract (1) | Right forward, LK stance, RK retract |
| 6 | Right (1) | Retract (1) | Stance (0) | Right forward, LK retract, RK stance |
| 7 | Right (1) | Retract (1) | Retract (1) | Right forward, both retract |

### State Transitions

**Hip Transitions:**
- Switch from right to left when right foot contacts ground AND left leg is forward
- Switch from left to right when left foot contacts ground AND right leg is forward

**Knee Transitions:**
- Retract knee when opposite foot lands AND leg is forward
- Extend knee when leg moves past release threshold

### Target Positions

When in each state, the FSM sets target joint positions:

- **Hip targets**: +0.5 rad (right forward) or -0.5 rad (left forward)
- **Knee targets**: 0.0 m (extended) or -0.25 m (retracted)

## Data Collection

### Basic Usage

```bash
# Collect FSM data for BC training
python -m passive_walker.fsm.collect --episodes 200 --duration 25.0 --out data/fsm_training
```

### Parameters

- `--episodes`: Number of episodes to collect (default: 10)
- `--duration`: Episode duration in seconds (default: 5.0)
- `--physics`: Physics preset ("nominal", "light", "heavy") (default: "nominal")
- `--out`: Output directory (default: "data/fsm")
- `--seed`: Random seed for reproducibility (default: 123)

### Quality Guidelines

For high-quality BC training data:

- **Duration**: Use 25.0s episodes (target: 10+ gait cycles)
- **Episodes**: Collect 200+ episodes minimum
- **Physics**: Start with "nominal" for stability
- **Fall rate**: Should be <10% (ideally 0%)
- **Gait cycles**: Target 8-10+ cycles per 10 seconds

## Output Data Schema

### File Structure

```
output_directory/
├── episode_000000.npz    # Episode data
├── episode_000001.npz
├── ...
├── meta.json            # Collection metadata
└── README.json          # Quality analysis report
```

### Episode Data (episode_XXXXXX.npz)

Each episode file contains:

```python
{
    'obs': np.array,           # Shape: (T, 11) - observations
    'act': np.array,           # Shape: (T, 3) - actions (zeros for FSM)
    'rew': np.array,           # Shape: (T,) - rewards
    'done': np.array,          # Shape: (T,) - episode termination flags
    
    # FSM-specific info
    'info_fsm_hip': np.array,  # Shape: (T,) - hip FSM states (0 or 1)
    'info_fsm_k1': np.array,   # Shape: (T,) - left knee FSM states (0 or 1)
    'info_fsm_k2': np.array,   # Shape: (T,) - right knee FSM states (0 or 1)
    'info_qdes': np.array,     # Shape: (T, 3) - desired joint positions
    
    # Episode metadata
    'seed': int,               # Random seed used
    'ramp_deg': float,         # Ramp angle in degrees
    'friction': float,         # Ground friction coefficient
}
```

### Quality Analysis (README.json)

```json
{
    "episode_length_stats": {"mean": 2500, "median": 2500},
    "gait_cycles_stats": {"mean": 12.5, "min": 10},
    "distance_stats": {"mean": 35.2, "max": 40.1},
    "fall_rate": 0.05,
    "max_pitch": 0.85,
    "quality_assessment": "high"
}
```

## Performance Characteristics

### Typical Performance

Based on testing with v2.1:

- **Episode duration**: 10-25 seconds (stable walking)
- **Gait cycles**: 6-13 cycles per 10 seconds
- **Forward distance**: ~14m in 10 seconds, ~35m in 25 seconds
- **Fall rate**: 0% under nominal conditions
- **Average speed**: ~1.4 m/s

### State Distribution

Typical state usage in 10-second episodes:

- **State 0**: 27.2% - Left forward, both stance
- **State 2**: 21.5% - Left forward, LK retract, RK stance
- **State 4**: 30.1% - Right forward, both stance
- **State 5**: 21.2% - Right forward, LK stance, RK retract

### State Transitions

- **Hip changes**: ~13 transitions per 10 seconds
- **Knee changes**: ~14 (left), ~13 (right) per 10 seconds
- **Total transitions**: ~40 state changes per 10 seconds

## Known Limitations

### Discontinuous Outputs

**Issue**: FSM produces instant state transitions with discontinuous joint trajectories.

**Impact**: 
- Hip jumps: Instant ±0.5 rad transitions
- Knee jumps: Instant ±0.25 m transitions
- Creates challenging training data for BC models

**Solution**: Train BC models on FSM states rather than joint trajectories.

### Slew Rate Limiting

**Issue**: Any smoothing (slew rate limiting) breaks FSM balance.

**Root Cause**: FSM requires instant transitions for stability.

**Documentation**: Slew rate limiting is disabled by design in `controller.py`:
```python
# Slew rate limiting (disabled - FSM requires instant transitions for balance)
HIP_SLEW = None            # rad/s - disabled: any smoothing breaks FSM balance
KNEE_SLEW = None           # m/s - disabled: any smoothing breaks FSM balance
```

## BC Training Recommendations

### State-Based Training

Instead of training on discontinuous joint trajectories:

1. **Predict FSM states** from observations
2. **Use state transitions** as training targets
3. **Apply smoothing** at the state level
4. **Convert states to actions** using lookup table

### Training Data Quality

- ✅ **Use 25-second episodes** for sufficient gait cycles
- ✅ **Collect 200+ episodes** for diversity
- ✅ **Monitor fall rate** (should be <10%)
- ✅ **Verify gait cycle count** (8-10+ per 10 seconds)

### Model Architecture

- Use **temporal context** (frame stacking) for better state predictions
- Consider **state transition probabilities** for smoothing
- Train **separate models** for hip and knee states if needed

## Testing and Validation

### Visual Testing

```bash
# Test FSM with GUI
python -m passive_walker.core.env --mode fsm --seconds 10 --gui

# Collect with visual output
python -m passive_walker.fsm.collect --episodes 3 --duration 10.0 --out test_results/fsm_visual
```

### Automated Tests

```bash
# Run FSM test suite
pytest tests/test_fsm_collect.py tests/test_fsm_smoke.py -v

# Test 20-second walking criterion
python tests/test_20_second_walking.py --gui
```

### Success Criteria

- ✅ **No falls** during target duration
- ✅ **Stable gait** - no tumbling or wobbling
- ✅ **Forward progress** - consistent movement
- ✅ **Gait cycles** - 8-10+ cycles per 10 seconds
- ✅ **20-second test** - critical success criterion

## Troubleshooting

### Common Issues

**FSM not transitioning states:**
- Check leg angle detection in `controller.py`
- Verify contact threshold (`CONTACT_Z = 0.06`)
- Ensure leg body indices are correct

**Walker falling quickly:**
- Use nominal physics preset initially
- Check contact detection is working
- Verify FSM state logic is correct

**Poor data quality:**
- Increase episode duration to 25.0s
- Collect more episodes (200+)
- Monitor fall rate and gait cycle count

**GUI not showing:**
- Check Wayland/X11 display server
- Try Alt+Tab to find window
- Use headless mode for automated collection

### Performance Optimization

- Use **JAX backend** for faster collection (`--jax`)
- **Parallel collection** for large datasets
- **Quality filtering** to remove poor episodes

## Examples

### Basic Collection

```bash
# Collect 10 episodes of 5 seconds each
python -m passive_walker.fsm.collect --episodes 10 --duration 5.0 --out data/basic

# Collect high-quality training data
python -m passive_walker.fsm.collect --episodes 200 --duration 25.0 --physics nominal --out data/training
```

### Visual Inspection

```bash
# Run GUI demo
python -m passive_walker.core.env --mode fsm --seconds 10 --gui

# Test 20-second criterion
python tests/test_20_second_walking.py --gui
```

### Data Analysis

```python
import numpy as np

# Load episode data
data = np.load("data/training/episode_000000.npz")

# Analyze state transitions
hip_changes = np.sum(np.diff(data['info_fsm_hip']) != 0)
print(f"Hip state changes: {hip_changes}")

# Check gait quality
total_steps = len(data['obs'])
duration = total_steps / 100.0  # 100 Hz
print(f"Episode duration: {duration:.1f}s")
```

## Related Files

- `passive_walker/core/controller.py` - FSM implementation
- `passive_walker/fsm/collect.py` - Data collection script
- `tests/test_fsm_*.py` - Test suite
- `tests/test_20_second_walking.py` - Critical success test
