# passive_walker/controllers/nn/hip_nn.py
"""Neural network controller for hip joint."""

import jax
import jax.numpy as jnp
import equinox as eqx

class HipController(eqx.Module):
    """MLP controller for hip joint."""
    
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear

    def __init__(
        self, 
        input_size: int, 
        hidden_size: int = 64, 
        output_size: int = 1, 
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
    # Assume the observation dimension is 12 (consistent with our environment).
    input_size = 12
    key = jax.random.PRNGKey(42)
    
    # Initialize the neural network controller.
    nn_controller = HipController(input_size=input_size, hidden_size=64, output_size=1, key=key)
    
    # Create a dummy observation (for testing).
    dummy_obs = jnp.ones((input_size,))
    
    # Get the network's output.
    output_action = nn_controller(dummy_obs)
    
    print("Dummy observation:", dummy_obs)
    print("NN Controller output:", output_action)

if __name__ == "__main__":
    main()
