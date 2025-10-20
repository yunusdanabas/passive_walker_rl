# Passive Walker Core Environment

Core environment implementation for bipedal walking with FSM and neural network control modes.

## Environment Modes

### FSM Mode (`mode="fsm"`)
- Uses built-in Finite State Machine for stable walking
- Actions ignored (FSM controls all joints automatically)
- Use case: Data collection, baseline walking, testing

### Research Mode (`mode="research"`)
- Neural network controls all joints
- Actions: 3D vector `[hip, left_knee, right_knee]` in range [-1, 1]
- Use case: BC training, policy learning, research

### Hybrid Hip Mode (`mode="hybrid_hip"`)
- Neural network controls hip, FSM controls knees
- Actions: 1D scalar `[hip]` in range [-1, 1]
- Use case: Simplified learning, hip-only BC training

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

All actions normalized to range [-1, 1], automatically denormalized to physical joint ranges:
- Hip: [-0.5, +0.5] rad
- Knees: [-0.5, +0.5] m (slider joints)

## Usage Examples

```python
from passive_walker.core.env import PassiveWalkerEnv

# FSM mode (recommended for testing)
env = PassiveWalkerEnv(mode="fsm", use_gui=False)

# Research mode (for BC training)
env = PassiveWalkerEnv(mode="research", use_gui=False)

# Reset and run episode
obs, info = env.reset(seed=123)
action = np.zeros(3, dtype=np.float32)  # For FSM mode
obs, reward, done, info = env.step(action)

env.close()
```
