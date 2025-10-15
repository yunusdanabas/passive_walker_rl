# Passive Walker Core Environment

Core environment implementation for bipedal walking with FSM and neural network control modes.

## Environment Modes

### FSM Mode (`mode="fsm"`)
- **Purpose**: Uses built-in Finite State Machine for stable walking
- **Actions**: Ignored (FSM controls all joints automatically)
- **Use case**: Data collection, baseline walking, testing
- **Success rate**: ~90%+ for 20+ seconds walking

### Research Mode (`mode="research"`)
- **Purpose**: Neural network controls all joints
- **Actions**: 3D vector `[hip, left_knee, right_knee]` in range [-1, 1]
- **Use case**: BC training, policy learning, research
- **Success rate**: Depends on trained model quality

### Hybrid Hip Mode (`mode="hybrid_hip"`)
- **Purpose**: Neural network controls hip, FSM controls knees
- **Actions**: 1D scalar `[hip]` in range [-1, 1]
- **Use case**: Simplified learning, hip-only BC training
- **Success rate**: Higher than full research mode

## Key Parameters

```python
# Simulation settings
CTRL_HZ = 100.0          # Control frequency (Hz)
SIM_SECONDS = 25.0       # Default episode length (s)

# Physics parameters
RAMP_DEG = 10.0          # Incline angle (degrees, positive = downhill)
FRICTION = 0.9           # Contact friction coefficient
RANDOMIZE_PHYSICS = False # Enable domain randomization
MASS_JITTER = 0.05       # ±5% torso mass variation

# Rendering parameters
DEFAULT_GUI = True       # Enable GUI when run as module
CAM_DISTANCE = 8.0       # Camera distance from walker
```

## Observation Space

**Shape**: `(11,)` - `[x, z, pitch, ẋ, ż, hip, lk, rk, hiṗ, lk̇, rk̇]`

- `x`: Horizontal position (m)
- `z`: Vertical position (m)  
- `pitch`: Torso pitch angle (rad)
- `ẋ`: Horizontal velocity (m/s)
- `ż`: Vertical velocity (m/s)
- `hip`: Hip joint position (rad)
- `lk`: Left knee position (m)
- `rk`: Right knee position (m)
- `hiṗ`: Hip joint velocity (rad/s)
- `lk̇`: Left knee velocity (m/s)
- `rk̇`: Right knee velocity (m/s)

## Action Space

**Shape**: `(3,)` - `[hip, left_knee, right_knee]`

- All actions normalized to range [-1, 1]
- Automatically denormalized to physical joint ranges
- Hip: [-0.5, +0.5] rad
- Knees: [-0.5, +0.5] m (slider joints)

## Usage Examples

### Basic Environment Creation

```python
from passive_walker.core.env import PassiveWalkerEnv

# FSM mode (recommended for testing)
env = PassiveWalkerEnv(mode="fsm", use_gui=False)

# Research mode (for BC training)
env = PassiveWalkerEnv(mode="research", use_gui=False)

# With GUI for visual inspection
env = PassiveWalkerEnv(mode="fsm", use_gui=True)
```

### Running Episodes

```python
import numpy as np

# Reset environment
obs, info = env.reset(seed=123)
print(f"Initial observation: {obs.shape}")  # (11,)

# FSM mode - actions ignored
action = np.zeros(3, dtype=np.float32)
obs, reward, done, info = env.step(action)

# Research mode - provide meaningful actions
action = np.array([0.1, -0.2, 0.3], dtype=np.float32)  # [hip, lk, rk]
obs, reward, done, info = env.step(action)

# Clean up
env.close()
```

### CLI Usage

```bash
# FSM mode with GUI (visual inspection)
walker-demo --mode fsm --seconds 20 --gui

# Research mode headless (fast)
walker-demo --mode research --seconds 10 --no-gui

# Quick test
walker-demo --seconds 5
```

## GUI Rendering

### Display Requirements
- **X11**: Works out of the box
- **Wayland**: Window created but may not be visible (known limitation)
- **Headless**: Use `--no-gui` or `use_gui=False`

### GUI Tips
- If window not visible: Try Alt+Tab or check behind other windows
- On Wayland: Window might be created but not visible due to display server limitations
- Fallback: GUI automatically falls back to headless if window creation fails

### Performance
- **Headless**: ~1000 steps/second
- **GUI**: ~50-60 FPS (limited by rendering)

## Success Criteria

### FSM Mode
- ✅ **20+ seconds walking** without falling
- ✅ **8+ gait cycles** per 20-second episode
- ✅ **Forward progress** >0.5m
- ✅ **Stable pitch** <1.0 rad

### Research Mode
- ✅ **20+ seconds walking** (user requirement)
- ✅ **50%+ success rate** across multiple seeds
- ✅ **Forward progress** >0.3m
- ✅ **Stable control** without oscillations

## Troubleshooting

### Common Issues

1. **Import Error**: `ModuleNotFoundError: No module named 'passive_walker'`
   ```bash
   pip install -e .
   ```

2. **GUI Not Visible**: Window created but not shown
   - Try Alt+Tab to find window
   - Use `--no-gui` for headless operation
   - Check display server (X11 vs Wayland)

3. **FSM Not Walking**: Walker falls immediately
   - Check physics parameters (ramp_deg, friction)
   - Verify FSM state machine initialization
   - Test with different seeds

4. **Slow Performance**: Low FPS or slow execution
   - Use headless mode: `use_gui=False`
   - Check PD backend: `--numpy-pd` (faster than JAX for single env)
   - Reduce episode length for testing

### Performance Tips

- **Single environment**: Use NumPy PD backend (default)
- **Multiple environments**: Use JAX PD backend with `--jax-pd`
- **Testing**: Use headless mode for speed
- **Visual inspection**: Use GUI mode for debugging

## API Reference

### PassiveWalkerEnv

```python
class PassiveWalkerEnv(gym.Env):
    def __init__(self, 
                 mode: str = "fsm",           # Control mode
                 use_gui: bool = False,       # Enable GUI
                 use_jax_pd: bool = False,    # Use JAX PD backend
                 ramp_deg: float = None,      # Override ramp angle
                 friction: float = None,      # Override friction
                 randomize_physics: bool = None)  # Override randomization
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]
    def render(self, mode: str = "human") -> None
    def close(self) -> None
```

### Info Dictionary

```python
info = {
    "time": float,           # Current simulation time (s)
    "dx": float,            # Horizontal velocity (m/s)
    "pitch_abs": float,     # Absolute pitch angle (rad)
    "torso_z": float,       # Torso height (m)
    "qdes": np.ndarray,     # Desired joint positions (3,)
    "fsm_hip": int,         # FSM hip state
    "fsm_k1": int,          # FSM left knee state
    "fsm_k2": int,          # FSM right knee state
    "u_abs_sum": float,     # Total control effort
    "fell": bool,           # Whether walker fell
    "seed": int             # Random seed used (if set)
}
```
