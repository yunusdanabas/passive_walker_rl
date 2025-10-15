# Passive Walker RL Environment

A unified, high-performance reinforcement learning environment for a Variable Length Leg (VLL) bipedal walker using MuJoCo physics simulation.

## Quick Start

```bash
# Collect 10 episodes → Train hip (Torch) → Play
python -m passive_walker.fsm.collect --episodes 10 --steps 512 --out data/fsm_runs
python -m passive_walker.bc.train --backend torch --section hip --data-dir data/fsm_runs
python -m passive_walker.bc.play --model results/bc/torch_hip_seed123_ep0_steps512.pt
```