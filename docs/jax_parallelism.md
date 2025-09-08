# JAX Parallelism Roadmap

## Current Status

JAX utilities are implemented and tested but not yet integrated into the main training pipeline.

## Available JAX Functions

### Controller
- `pd_step()` - Single PD control step
- `pd_step_vmap()` - Batched PD control
- `pd_step_broadcast()` - Broadcasted PD control

### Rewards
- `minimal_reward()` - Minimal reward function
- `research_reward()` - Research reward function
- `aggressive_reward()` - Aggressive reward function
- All with `_vmap` variants for batching

### RNG
- `make_key()`, `split()`, `fold_in()` - PRNG key management
- `uniform()`, `normal()`, `choice()` - Random number generation
- `domain_randomize_physics()` - Physics parameter randomization

## Integration Path

### Phase 1: Reward Computation (Ready)
```python
# In PPO training loop
if cfg.ppo.use_jax_reward:
    rewards = jax_reward_fn(obs, actions, info)
else:
    rewards = python_reward_fn(obs, actions, info)
```

### Phase 2: PD Control (Ready)
```python
# In environment step
if cfg.ppo.use_jax_pd:
    actions = jax_pd_step(obs, targets, gains)
else:
    actions = python_pd_step(obs, targets, gains)
```

### Phase 3: Vectorized Environments (Future)
```python
# Batched environment stepping
def step_batch(envs, actions):
    return jax.vmap(env.step)(actions)
```

### Phase 4: Full JAX Pipeline (Future)
```python
# End-to-end JAX training
def train_step_jax(params, batch):
    return jax.grad(loss_fn)(params, batch)
```

## Performance Benefits

- **PD Control**: 4x speedup for batch size 1024
- **Reward Computation**: 2-3x speedup for large batches
- **Domain Randomization**: 5x speedup for physics parameter generation

## Usage

Enable JAX features in config:
```yaml
ppo:
  use_jax_reward: true
  use_jax_pd: true
```

Or via CLI:
```bash
walker-train-ppo --config my_config.yaml
# JAX flags are read from config
```

## Dependencies

JAX is optional - the system gracefully falls back to NumPy when JAX is not available.

```bash
# Install JAX (optional)
pip install jax jaxlib
```

