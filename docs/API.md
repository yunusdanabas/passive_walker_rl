# API Reference

## Core Modules

### Environment

```python
from passive_walker.core.env import PassiveWalkerEnv

env = PassiveWalkerEnv(
    mode='fsm',              # 'fsm', 'bc', or 'ppo'
    backend='mujoco',        # Physics backend
    control_freq=100,        # Control frequency (Hz)
    obs_noise=0.0,           # Observation noise
    gui=False                # Show GUI
)

# Gymnasium interface
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

**Modes:**
- `fsm`: Finite State Machine controller (deterministic)
- `bc`: Behavior Cloning model (neural network policy)
- `ppo`: PPO trained policy (neural network)

### FSM Controller

```python
from passive_walker.fsm.controller import FSMPassiveWalker

controller = FSMPassiveWalker(
    control_freq=100,
    hip_pd_gain=10.0,
    knee_pd_gain=5.0
)

# Collect demonstrations
from passive_walker.fsm.collect import collect_demonstrations

collect_demonstrations(
    episodes=10,
    duration=20.0,
    out_dir='experiments/data/fsm_demos',
    noise_level=0.5
)
```

## BC Module

### Training

```python
from passive_walker.bc.training.train import train_bc

model = train_bc(
    data_dir='experiments/data/fsm_demos',
    epochs=100,
    model_type='mlp',      # 'mlp', 'lstm', 'gru'
    hidden_size=64,
    learning_rate=1e-3
)

# Save model
model.save('experiments/models/bc/my_model.pt')
```

### Evaluation

```python
from passive_walker.bc.evaluation.evaluate import evaluate_bc_model

metrics = evaluate_bc_model(
    model_path='experiments/models/bc/my_model.pt',
    episodes=10
)

# Play model
from passive_walker.bc.evaluation.play import play_bc_model

play_bc_model(
    model_path='experiments/models/bc/my_model.pt',
    episodes=3,
    gui=True
)
```

### Models

```python
from passive_walker.bc.models.models_torch import MLPModel, LSTMModel, GRUModel

# Simple MLP
model = MLPModel(input_size=17, hidden_sizes=[64, 64], output_size=2)

# Temporal models
model = LSTMModel(input_size=17, hidden_size=128, output_size=2)
model = GRUModel(input_size=17, hidden_size=128, output_size=2)
```

## PPO Module

### Training

```python
from passive_walker.ppo.train import main as train_ppo

# Basic usage via CLI
train_ppo(
    experiment_name='my_ppo',
    timesteps=100000,
    seed=42
)
```

Or via trainer:

```python
from passive_walker.ppo.trainer import PPOTrainer
from passive_walker.ppo.config import PPOConfig

config = PPOConfig()
trainer = PPOTrainer(config)
trainer.train()
```

### Evaluation

```python
from passive_walker.ppo.evaluate import evaluate_ppo

metrics = evaluate_ppo(
    model_path='experiments/models/ppo/final_model.pth',
    episodes=10
)

# Play model
from passive_walker.ppo.play_ppo_model import play_ppo

play_ppo(
    model_path='experiments/models/ppo/final_model.pth',
    episodes=3,
    gui=True
)
```

### Models

```python
from passive_walker.ppo.models import create_actor_critic

# Create actor-critic
actor, critic = create_actor_critic(
    obs_dim=17,
    action_dim=2,
    model_type='mlp',       # 'mlp', 'lstm', 'gru'
    hidden_size=64,
    num_layers=1
)
```

## Configuration

### PPO Config

```python
from passive_walker.ppo.config import PPOConfig

config = PPOConfig(
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_epsilon=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5
)
```

### BC Config

```python
from passive_walker.bc.config import BCConfig

config = BCConfig(
    learning_rate=1e-3,
    batch_size=64,
    epochs=100,
    model_type='mlp',
    hidden_size=64
)
```

## Data Loading

```python
from passive_walker.bc.data.dataset import load_xy, create_data_loader

# Load demonstrations
X, y = load_xy('experiments/data/fsm_demos')

# Create data loader
loader = create_data_loader(
    data_dir='experiments/data/fsm_demos',
    batch_size=64,
    sequence_length=10
)
```

## Utilities

```python
from passive_walker.bc.utils import set_seed, ensure_dir, save_checkpoint

set_seed(42)
ensure_dir('experiments/models/bc')
save_checkpoint(model, optimizer, epoch, path)
```

