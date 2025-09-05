#!/bin/bash
set -e

echo "Running smoke matrix tests..."
echo "================================"

# Create results directory
mkdir -p results/smoke

# Run smoke matrix
python - << 'PY'
from passive_walker.core.io import load_walker_config
from passive_walker.core.env import PassiveWalkerEnv
import numpy as np

def run(mode, preset, jax_enable):
    cfg = load_walker_config('passive_walker/configs/walker.yaml')
    cfg.mode = mode
    cfg.jax.enable = jax_enable
    if mode == 'research':
        cfg.reward.preset = preset
    env = PassiveWalkerEnv(cfg, use_gui=False)
    obs, _ = env.reset()
    tot=0.0
    for t in range(60):
        obs, r, done, info = env.step(np.zeros(3, dtype=np.float32))
        tot += r
        if done: break
    env.close()
    print(f'{mode=:8s} {preset=:11s} jax={jax_enable} -> steps={t+1} sum={tot:.3f} fell={info.get("fell")}')

print("Testing all mode/preset/JAX combinations...")
for m in ['fsm','research']:
    for p in ['minimal','default','aggressive']:
        run(m, p, False)
        run(m, p, True)
print("Smoke matrix tests completed successfully!")
PY

echo "================================"
echo "Smoke matrix tests completed!"
