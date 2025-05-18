# passive_walker/controllers/nn/hip_knee_nn.py
"""HipKneeController – one small MLP that outputs **[hip, kneeL, kneeR]**.

* Two hidden layers (configurable)
* tanh output → actions in [-1, 1]
"""

from typing import Optional
import jax
import jax.numpy as jnp
import equinox as eqx


class HipKneeController(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        output_size: int = 3,
        key: Optional["jax.random.PRNGKey"] = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        self.fc1 = eqx.nn.Linear(input_size, hidden_size, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_size, hidden_size, key=k2)
        self.fc3 = eqx.nn.Linear(hidden_size, output_size, key=k3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax.nn.relu(self.fc1(x))
        x = jax.nn.relu(self.fc2(x))
        x = self.fc3(x)
        return jnp.tanh(x)          # bound each joint command to [-1, 1]
