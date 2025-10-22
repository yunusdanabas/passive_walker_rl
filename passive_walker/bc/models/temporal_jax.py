"""
JAX Temporal Neural Network Models for BC

Defines LSTM and GRU architectures using Equinox for JAX-based behavior cloning.
Supports functional programming paradigm with immutable models and efficient temporal processing.
"""

from __future__ import annotations
from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import equinox as eqx


class LSTM(eqx.Module):
    """
    Equinox-based LSTM for BC with temporal modeling.
    
    Architecture: Input -> LSTM -> Hidden -> Output
    - Uses jax.lax.scan for efficient sequence processing
    - Supports both sequence and single-step inference
    - Configurable hidden size, dropout, layer normalization
    - Functional programming paradigm (immutable)
    """
    
    lstm_cell: eqx.nn.LSTMCell
    output_layer: eqx.nn.Linear
    hidden_size: int
    dropout: Optional[eqx.nn.Dropout]
    layer_norm: Optional[eqx.nn.LayerNorm]
    
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int = 128, 
                 dropout_rate: float = 0.1, use_layer_norm: bool = False, *, key):
        """
        Initialize LSTM model.
        
        Args:
            in_dim: Input dimension (observation space)
            out_dim: Output dimension (action space)
            hidden_size: LSTM hidden dimension
            dropout_rate: Dropout probability
            use_layer_norm: Whether to use layer normalization
            key: JAX random key
        """
        key1, key2, key3 = jax.random.split(key, 3)
        
        self.hidden_size = hidden_size
        
        # LSTM cell
        self.lstm_cell = eqx.nn.LSTMCell(in_dim, hidden_size, key=key1)
        
        # Output layer
        self.output_layer = eqx.nn.Linear(hidden_size, out_dim, key=key2)
        
        # Optional dropout and layer norm
        self.dropout = eqx.nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.layer_norm = eqx.nn.LayerNorm(hidden_size) if use_layer_norm else None
    
    def __call__(self, x: jnp.ndarray, hidden: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None, 
                 key: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
                - For sequences: (seq_len, in_dim) or (batch, seq_len, in_dim)
                - For single step: (in_dim,) or (batch, in_dim)
            hidden: Optional hidden state tuple (h, c)
            key: Optional JAX random key for dropout
            
        Returns:
            Tuple of (output, new_hidden)
            Output has actions in [-1, 1] range
        """
        # Handle batch dimension
        if x.ndim == 3:  # (batch, seq_len, in_dim)
            return self._forward_batch(x, hidden, key)
        elif x.ndim == 2:  # (seq_len, in_dim)
            return self._forward_sequence(x, hidden, key)
        elif x.ndim == 1:  # (in_dim,) - single step
            return self._forward_single(x, hidden, key)
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")
    
    def _forward_single(self, x: jnp.ndarray, hidden: Optional[Tuple[jnp.ndarray, jnp.ndarray]], 
                       key: Optional[jax.random.PRNGKey]) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Forward pass for single timestep."""
        if hidden is None:
            hidden = self._get_initial_hidden()
        
        # LSTM forward
        h, c = self.lstm_cell(x, hidden)
        
        # Apply layer norm if enabled
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        
        # Apply dropout if enabled
        if self.dropout is not None and key is not None:
            h = self.dropout(h, key=key)
        
        # Output layer
        output = self.output_layer(h)
        output = jnp.tanh(output)
        
        return output, (h, c)
    
    def _forward_sequence(self, x: jnp.ndarray, hidden: Optional[Tuple[jnp.ndarray, jnp.ndarray]], 
                         key: Optional[jax.random.PRNGKey]) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Forward pass for sequence."""
        if hidden is None:
            hidden = self._get_initial_hidden()
        
        def scan_fn(carry, x_t):
            h, c = carry
            
            # LSTM forward
            h, c = self.lstm_cell(x_t, (h, c))
            
            # Apply layer norm if enabled
            if self.layer_norm is not None:
                h = self.layer_norm(h)
            
            # Apply dropout if enabled (use same key for all timesteps)
            if self.dropout is not None and key is not None:
                h = self.dropout(h, key=key)
            
            # Output layer
            output = self.output_layer(h)
            output = jnp.tanh(output)
            
            return (h, c), output
        
        # Scan over sequence dimension
        (h_final, c_final), outputs = jax.lax.scan(scan_fn, hidden, x)
        
        return outputs, (h_final, c_final)
    
    def _forward_batch(self, x: jnp.ndarray, hidden: Optional[Tuple[jnp.ndarray, jnp.ndarray]], 
                      key: Optional[jax.random.PRNGKey]) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Forward pass for batch of sequences using vmap over batch dimension."""
        batch_size = x.shape[0]
        
        # Process each sequence in the batch using vmap
        def process_single_sequence(seq):
            """Process a single sequence."""
            h_0 = jnp.zeros(self.hidden_size)
            c_0 = jnp.zeros(self.hidden_size)
            init_hidden = (h_0, c_0)
            
            def scan_fn(carry, x_t):
                h, c = carry
                
                # LSTM forward
                h, c = self.lstm_cell(x_t, (h, c))
                
                # Apply layer norm if enabled
                if self.layer_norm is not None:
                    h = self.layer_norm(h)
                
                # Apply dropout if enabled
                if self.dropout is not None and key is not None:
                    h = self.dropout(h, key=key)
                
                # Output layer
                output = self.output_layer(h)
                output = jnp.tanh(output)
                
                return (h, c), output
            
            # Transpose to (seq_len, input_dim) for scan
            seq_transposed = jnp.transpose(seq, (0, 1))  # Already in correct shape
            
            # Scan over sequence
            (h_final, c_final), outputs = jax.lax.scan(scan_fn, init_hidden, seq_transposed)
            
            return outputs, (h_final, c_final)
        
        # Vectorize over batch dimension
        vmap_process = jax.vmap(process_single_sequence, in_axes=0, out_axes=(0, 0))
        outputs, (h_final, c_final) = vmap_process(x)
        
        return outputs, (h_final, c_final)
    
    def _get_initial_hidden(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get initial hidden state."""
        h_0 = jnp.zeros(self.hidden_size)
        c_0 = jnp.zeros(self.hidden_size)
        return (h_0, c_0)


class GRU(eqx.Module):
    """
    Equinox-based GRU for BC with temporal modeling.
    
    Architecture: Input -> GRU -> Hidden -> Output
    - Uses jax.lax.scan for efficient sequence processing
    - Supports both sequence and single-step inference
    - Configurable hidden size, dropout, layer normalization
    - Functional programming paradigm (immutable)
    """
    
    gru_cell: eqx.nn.GRUCell
    output_layer: eqx.nn.Linear
    hidden_size: int
    dropout: Optional[eqx.nn.Dropout]
    layer_norm: Optional[eqx.nn.LayerNorm]
    
    def __init__(self, in_dim: int, out_dim: int, hidden_size: int = 128, 
                 dropout_rate: float = 0.1, use_layer_norm: bool = False, *, key):
        """
        Initialize GRU model.
        
        Args:
            in_dim: Input dimension (observation space)
            out_dim: Output dimension (action space)
            hidden_size: GRU hidden dimension
            dropout_rate: Dropout probability
            use_layer_norm: Whether to use layer normalization
            key: JAX random key
        """
        key1, key2, key3 = jax.random.split(key, 3)
        
        self.hidden_size = hidden_size
        
        # GRU cell
        self.gru_cell = eqx.nn.GRUCell(in_dim, hidden_size, key=key1)
        
        # Output layer
        self.output_layer = eqx.nn.Linear(hidden_size, out_dim, key=key2)
        
        # Optional dropout and layer norm
        self.dropout = eqx.nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.layer_norm = eqx.nn.LayerNorm(hidden_size) if use_layer_norm else None
    
    def __call__(self, x: jnp.ndarray, hidden: Optional[jnp.ndarray] = None, 
                 key: Optional[jax.random.PRNGKey] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
                - For sequences: (seq_len, in_dim) or (batch, seq_len, in_dim)
                - For single step: (in_dim,) or (batch, in_dim)
            hidden: Optional hidden state
            key: Optional JAX random key for dropout
            
        Returns:
            Tuple of (output, new_hidden)
            Output has actions in [-1, 1] range
        """
        # Handle batch dimension
        if x.ndim == 3:  # (batch, seq_len, in_dim)
            return self._forward_batch(x, hidden, key)
        elif x.ndim == 2:  # (seq_len, in_dim)
            return self._forward_sequence(x, hidden, key)
        elif x.ndim == 1:  # (in_dim,) - single step
            return self._forward_single(x, hidden, key)
        else:
            raise ValueError(f"Invalid input shape: {x.shape}")
    
    def _forward_single(self, x: jnp.ndarray, hidden: Optional[jnp.ndarray], 
                       key: Optional[jax.random.PRNGKey]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass for single timestep."""
        if hidden is None:
            hidden = self._get_initial_hidden()
        
        # GRU forward
        h = self.gru_cell(x, hidden)
        
        # Apply layer norm if enabled
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        
        # Apply dropout if enabled
        if self.dropout is not None and key is not None:
            h = self.dropout(h, key=key)
        
        # Output layer
        output = self.output_layer(h)
        output = jnp.tanh(output)
        
        return output, h
    
    def _forward_sequence(self, x: jnp.ndarray, hidden: Optional[jnp.ndarray], 
                         key: Optional[jax.random.PRNGKey]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass for sequence."""
        if hidden is None:
            hidden = self._get_initial_hidden()
        
        def scan_fn(carry, x_t):
            h = carry
            
            # GRU forward
            h = self.gru_cell(x_t, h)
            
            # Apply layer norm if enabled
            if self.layer_norm is not None:
                h = self.layer_norm(h)
            
            # Apply dropout if enabled (use same key for all timesteps)
            if self.dropout is not None and key is not None:
                h = self.dropout(h, key=key)
            
            # Output layer
            output = self.output_layer(h)
            output = jnp.tanh(output)
            
            return h, output
        
        # Scan over sequence dimension
        h_final, outputs = jax.lax.scan(scan_fn, hidden, x)
        
        return outputs, h_final
    
    def _forward_batch(self, x: jnp.ndarray, hidden: Optional[jnp.ndarray], 
                      key: Optional[jax.random.PRNGKey]) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass for batch of sequences using vmap over batch dimension."""
        batch_size = x.shape[0]
        
        # Process each sequence in the batch using vmap
        def process_single_sequence(seq):
            """Process a single sequence."""
            h_0 = jnp.zeros(self.hidden_size)
            
            def scan_fn(carry, x_t):
                h = carry
                
                # GRU forward
                h = self.gru_cell(x_t, h)
                
                # Apply layer norm if enabled
                if self.layer_norm is not None:
                    h = self.layer_norm(h)
                
                # Apply dropout if enabled
                if self.dropout is not None and key is not None:
                    h = self.dropout(h, key=key)
                
                # Output layer
                output = self.output_layer(h)
                output = jnp.tanh(output)
                
                return h, output
            
            # Scan over sequence
            h_final, outputs = jax.lax.scan(scan_fn, h_0, seq)
            
            return outputs, h_final
        
        # Vectorize over batch dimension
        vmap_process = jax.vmap(process_single_sequence, in_axes=0, out_axes=(0, 0))
        outputs, h_final = vmap_process(x)
        
        return outputs, h_final
    
    def _get_initial_hidden(self) -> jnp.ndarray:
        """Get initial hidden state."""
        return jnp.zeros(self.hidden_size)


def make_temporal_model(model_type: str, in_dim: int, out_dim: int, hidden_size: int = 128, 
                       dropout_rate: float = 0.1, use_layer_norm: bool = False, *, key) -> eqx.Module:
    """
    Factory function to create temporal models.
    
    Args:
        model_type: Type of model ("lstm", "gru")
        in_dim: Input dimension
        out_dim: Output dimension
        hidden_size: Hidden dimension
        dropout_rate: Dropout probability
        use_layer_norm: Whether to use layer normalization
        key: JAX random key
        
    Returns:
        Instantiated model
    """
    if model_type.lower() == "lstm":
        return LSTM(in_dim, out_dim, hidden_size, dropout_rate, use_layer_norm, key=key)
    elif model_type.lower() == "gru":
        return GRU(in_dim, out_dim, hidden_size, dropout_rate, use_layer_norm, key=key)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Must be 'lstm' or 'gru'")


def temporal_loss_fn(model, x, y, mask=None, key=None):
    """
    Compute loss for temporal models with optional masking.
    
    Args:
        model: Temporal model
        x: Input sequences (batch, seq_len, in_dim)
        y: Target sequences (batch, seq_len, out_dim)
        mask: Optional mask for valid timesteps (batch, seq_len)
        key: Optional JAX random key
        
    Returns:
        Loss value
    """
    # Forward pass
    predictions, _ = model(x, key=key)
    
    # Compute L1 loss
    loss = jnp.mean(jnp.abs(predictions - y))
    
    # Apply mask if provided
    if mask is not None:
        loss = loss * mask
        loss = jnp.sum(loss) / jnp.sum(mask)
    
    return loss


def create_sequence_mask(sequence_lengths, max_length):
    """
    Create mask for variable-length sequences.
    
    Args:
        sequence_lengths: Length of each sequence in batch (batch,)
        max_length: Maximum sequence length
        
    Returns:
        Mask tensor (batch, max_length) where True indicates valid timesteps
    """
    batch_size = len(sequence_lengths)
    mask = jnp.arange(max_length)[None, :] < sequence_lengths[:, None]
    return mask


# Save/Load helpers (Equinox-native)
def save_temporal_model(path: str, model: eqx.Module) -> None:
    """
    Save temporal model to file.
    
    Args:
        path: File path to save to
        model: Model to save
    """
    eqx.tree_serialise_leaves(path, model)


def load_temporal_model(path: str, template: eqx.Module) -> eqx.Module:
    """
    Load temporal model from file.
    
    Args:
        path: File path to load from
        template: Template model with same architecture
        
    Returns:
        Loaded model
    """
    return eqx.tree_deserialise_leaves(path, template)


def load_temporal_model_with_template(path: str, model_type: str, in_dim: int, out_dim: int, 
                                    hidden_size: int = 128, dropout_rate: float = 0.1, 
                                    use_layer_norm: bool = False) -> eqx.Module:
    """
    Load temporal model with automatic template creation.
    
    Args:
        path: File path to load from
        model_type: Type of model ("lstm", "gru")
        in_dim: Input dimension
        out_dim: Output dimension
        hidden_size: Hidden dimension
        dropout_rate: Dropout probability
        use_layer_norm: Whether to use layer normalization
        
    Returns:
        Loaded model
    """
    # Create template model with dummy key (will be overwritten)
    key = jax.random.PRNGKey(0)
    template = make_temporal_model(model_type, in_dim, out_dim, hidden_size, 
                                 dropout_rate, use_layer_norm, key=key)
    return eqx.tree_deserialise_leaves(path, template)


