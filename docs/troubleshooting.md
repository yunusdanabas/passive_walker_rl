# Troubleshooting Guide

## Common Issues

### Environment Issues

**Problem**: `MuJoCo GL context creation failed`
```bash
# Solution: Use headless mode
walker-demo --no-gui --seconds 5
```

**Problem**: `Gym has been unmaintained` warning
```bash
# Solution: This is just a warning, can be ignored
# Or upgrade to gymnasium (future migration)
```

### Configuration Issues

**Problem**: `TypeError: EnvCfg.__init__() got an unexpected keyword argument`
```bash
# Solution: Check YAML structure matches dataclass
# Common fix: move parameters to correct sections
```

**Problem**: `No NPZ files under data/bc/raw`
```bash
# Solution: Run FSM collection first
walker-collect-fsm --num_episodes 10
```

### Training Issues

**Problem**: BC training loss not decreasing
```bash
# Solution: Check data quality
walker-collect-fsm --min_quality 0.5
walker-train-bc --data_dir data/bc/raw
```

**Problem**: PPO training crashes
```bash
# Solution: Use overfit_tiny for debugging
walker-train-bc --overfit_tiny
walker-train-ppo --config passive_walker/ppo/ppo_train.yaml
```

### JAX Issues

**Problem**: `ImportError: cannot import name 'jax'`
```bash
# Solution: JAX is optional, install if needed
pip install jax jaxlib
# Or disable JAX features in config
```

**Problem**: JAX functions return wrong shapes
```bash
# Solution: Check input shapes match expected
# Use test_jax_pd_reward.py to verify parity
```

## Debugging Tips

### Enable Debug Logging
```yaml
debug:
  log_quality: true
  log_fsm: true
```

### Check Configuration
```bash
# Print resolved config
walker-demo --config my_config.yaml --print-interval 0.1
```

### Verify Data Quality
```bash
# Check episode statistics
python -c "
import numpy as np
data = np.load('data/bc/raw/episode_000000.npz')
print('Obs shape:', data['obs'].shape)
print('Reward range:', data['rew'].min(), data['rew'].max())
"
```

### Test Individual Components
```bash
# Test environment
python -c "from passive_walker.core.env import PassiveWalkerEnv; env = PassiveWalkerEnv(); print('OK')"

# Test JAX functions
python -m pytest tests/test_jax_pd_reward.py -v
```

## Performance Issues

### Slow Training
- Use `--overfit_tiny` for quick tests
- Reduce `rollout_len` in FSM collection
- Use smaller batch sizes

### Memory Issues
- Reduce `num_envs` in PPO config
- Use smaller `rollout_len`
- Enable gradient checkpointing

### GPU Issues
- Check CUDA availability: `torch.cuda.is_available()`
- Use CPU fallback: `--device cpu`

## Getting Help

1. Check this troubleshooting guide
2. Run tests: `python -m pytest tests/ -v`
3. Check logs in `results/*/meta.json`
4. Enable debug logging for more info

