# Configuration Constraints

## Overview

Constraints allow you to override configuration values without modifying the base YAML files. This is useful for:

- Domain randomization experiments
- Hyperparameter sweeps
- Environment variations
- Quick configuration changes

## Usage

```bash
# Single constraint
walker-collect-fsm --constraints passive_walker/configs/constraints/dr_light.yaml

# Multiple constraints (later ones override earlier ones)
walker-train-bc --constraints constraint1.yaml constraint2.yaml
```

## Precedence Order

1. **Base config** (highest precedence)
2. **Constraint files** (in order specified)
3. **CLI arguments** (if any)

## Example Constraint File

```yaml
# passive_walker/configs/constraints/dr_light.yaml
physics:
  randomize_physics: true
  ramp_deg_min: 8.0
  ramp_deg_max: 16.0
  friction: [0.7, 1.1]
  mass_jitter: 0.08
```

## Available Constraint Packs

- `dr_light.yaml` - Light domain randomization
- `dr_heavy.yaml` - Heavy domain randomization (if created)
- `fast_training.yaml` - Fast training settings (if created)

## Creating Custom Constraints

1. Create a new YAML file in `passive_walker/configs/constraints/`
2. Specify only the values you want to override
3. Use the same structure as the base configs

Example:
```yaml
# my_constraint.yaml
physics:
  ramp_deg_min: 5.0
  ramp_deg_max: 20.0
  randomize_physics: true

bc:
  epochs: 50
  lr: 1e-4
```

## Best Practices

- Keep constraint files small and focused
- Use descriptive names
- Document what each constraint changes
- Test constraints before using in production

