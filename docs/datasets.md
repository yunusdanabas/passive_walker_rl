# Dataset Format

## FSM Episode Format

FSM episodes are saved as `.npz` files with the following structure:

```python
data = np.load("episode_000000.npz")
print(data.files)
# ['obs', 'act', 'rew', 'done', 'extras']

# Shapes
print(data['obs'].shape)  # (T, 11) - observations
print(data['act'].shape)  # (T, 3)  - actions (zeros for FSM)
print(data['rew'].shape)  # (T,)    - rewards
print(data['done'].shape) # (T,)    - done flags
print(data['extras'].shape) # (T,)  - info dictionaries
```

## Observation Space (11D)

```
obs[0]  = hip_angle
obs[1]  = left_knee_angle  
obs[2]  = right_knee_angle
obs[3]  = hip_angular_velocity
obs[4]  = left_knee_angular_velocity
obs[5]  = right_knee_angular_velocity
obs[6]  = torso_pitch
obs[7]  = torso_pitch_velocity
obs[8]  = left_foot_contact
obs[9]  = right_foot_contact
obs[10] = forward_velocity
```

## Action Space (3D)

For FSM episodes, actions are always zeros since the FSM controller handles everything internally.

For BC/PPO training, actions are:
```
act[0] = hip_torque
act[1] = left_knee_torque
act[2] = right_knee_torque
```

## Info Dictionary

Each step includes an info dictionary with:
- `fell`: Boolean indicating if walker fell
- `stalled`: Boolean indicating if walker stalled
- `unstable`: Boolean indicating if walker is unstable
- `quality_score`: Float quality score (if enabled)
- `fsm_state`: FSM state information (if enabled)

## Collection Summary

After collection, a `collection_summary.json` is created:

```json
{
  "episodes_collected": 95,
  "collection_rate": 0.95,
  "episode_metrics": [
    {
      "episode": 0,
      "steps": 200,
      "total_reward": 15.2,
      "accepted": true,
      "fell": false,
      "stalled": false,
      "quality": 0.8
    }
  ]
}
```

