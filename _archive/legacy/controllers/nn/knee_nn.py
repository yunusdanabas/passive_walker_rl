# passive_walker/controllers/nn/knee_nn.py
"""Neural network controller for knee joints."""

import jax
import jax.numpy as jnp
import equinox as eqx


class KneeController(eqx.Module):
    """MLP controller for both knee joints."""
    
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64, 
        output_size: int = 2,  # [kneeL, kneeR]
        key: "jax.random.PRNGKey" = None
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(key, 3)
        self.fc1 = eqx.nn.Linear(input_size, hidden_size, key=k1)
        self.fc2 = eqx.nn.Linear(hidden_size, hidden_size, key=k2)
        self.fc3 = eqx.nn.Linear(hidden_size, output_size, key=k3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass with tanh output in [-1, 1]."""
        x = jax.nn.relu(self.fc1(x))
        x = jax.nn.relu(self.fc2(x))
        x = self.fc3(x)
        return jnp.tanh(x)

def main():
    # Assume the observation dimension remains 12 (as in the original environment).
    input_size = 12
    key = jax.random.PRNGKey(42)
    
    # Initialize the knee controller with output_size=2.
    knee_controller = KneeController(input_size=input_size, hidden_size=64, output_size=2, key=key)
    
    # Create a dummy observation vector.
    dummy_obs = jnp.ones((input_size,))
    
    # Get the output from the network.
    output = knee_controller(dummy_obs)
    
    print("Dummy observation:", dummy_obs)
    print("Knee Controller output:", output)

if __name__ == "__main__":
    main()
