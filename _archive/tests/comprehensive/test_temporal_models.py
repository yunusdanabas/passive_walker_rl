"""
Unit tests for temporal models (PyTorch and JAX)

Tests LSTM, GRU, and BiLSTM implementations with various configurations.
"""

import pytest
import numpy as np
import torch
import jax
import jax.numpy as jnp


class TestTemporalModelsTorch:
    """Test PyTorch temporal models."""
    
    def test_lstm_forward_pass(self):
        """Test LSTM forward pass with dummy data."""
        from passive_walker.bc.models.temporal_torch import TorchLSTM
        
        # Create model
        model = TorchLSTM(in_dim=11, out_dim=3, hidden_size=64, num_layers=1)
        
        # Test sequence input
        batch_size, seq_len, obs_dim = 2, 10, 11
        x_seq = torch.randn(batch_size, seq_len, obs_dim)
        
        output_seq, hidden = model(x_seq)
        
        # Check output shape
        assert output_seq.shape == (batch_size, seq_len, 3)
        assert output_seq.min() >= -1.0 and output_seq.max() <= 1.0  # Tanh output
        
        # Test single-step inference
        x_single = torch.randn(batch_size, obs_dim)
        output_single, hidden = model(x_single)
        
        # Check output shape
        assert output_single.shape == (batch_size, 3)
        assert output_single.min() >= -1.0 and output_single.max() <= 1.0
    
    def test_gru_forward_pass(self):
        """Test GRU forward pass with dummy data."""
        from passive_walker.bc.models.temporal_torch import TorchGRU
        
        # Create model
        model = TorchGRU(in_dim=11, out_dim=3, hidden_size=64, num_layers=1)
        
        # Test sequence input
        batch_size, seq_len, obs_dim = 2, 10, 11
        x_seq = torch.randn(batch_size, seq_len, obs_dim)
        
        output_seq, hidden = model(x_seq)
        
        # Check output shape
        assert output_seq.shape == (batch_size, seq_len, 3)
        assert output_seq.min() >= -1.0 and output_seq.max() <= 1.0
        
        # Test single-step inference
        x_single = torch.randn(batch_size, obs_dim)
        output_single, hidden = model(x_single)
        
        # Check output shape
        assert output_single.shape == (batch_size, 3)
        assert output_single.min() >= -1.0 and output_single.max() <= 1.0
    
    def test_bilstm_forward_pass(self):
        """Test bidirectional LSTM forward pass."""
        from passive_walker.bc.models.temporal_torch import TorchBiLSTM
        
        # Create model
        model = TorchBiLSTM(in_dim=11, out_dim=3, hidden_size=64, num_layers=1)
        
        # Test sequence input
        batch_size, seq_len, obs_dim = 2, 10, 11
        x_seq = torch.randn(batch_size, seq_len, obs_dim)
        
        output_seq, hidden = model(x_seq)
        
        # Check output shape
        assert output_seq.shape == (batch_size, seq_len, 3)
        assert output_seq.min() >= -1.0 and output_seq.max() <= 1.0
    
    def test_hidden_state_management(self):
        """Test hidden state initialization and management."""
        from passive_walker.bc.models.temporal_torch import TorchLSTM
        
        model = TorchLSTM(in_dim=11, out_dim=3, hidden_size=64, num_layers=2)
        
        # Test initial hidden state
        batch_size = 3
        hidden = model.get_initial_hidden(batch_size)
        
        assert isinstance(hidden, tuple)
        assert len(hidden) == 2  # h_0, c_0
        assert hidden[0].shape == (2, batch_size, 64)  # (num_layers, batch, hidden)
        assert hidden[1].shape == (2, batch_size, 64)
        
        # Test forward pass with hidden state
        x = torch.randn(batch_size, 5, 11)
        output, new_hidden = model(x, hidden)
        
        assert output.shape == (batch_size, 5, 3)
        assert new_hidden[0].shape == hidden[0].shape
        assert new_hidden[1].shape == hidden[1].shape
    
    def test_model_factory(self):
        """Test model factory function."""
        from passive_walker.bc.models.temporal_torch import create_temporal_model
        
        # Test LSTM creation
        lstm = create_temporal_model("lstm", 11, 3, hidden_size=64)
        assert isinstance(lstm, TorchLSTM)
        
        # Test GRU creation
        gru = create_temporal_model("gru", 11, 3, hidden_size=64)
        assert isinstance(gru, TorchGRU)
        
        # Test BiLSTM creation
        bilstm = create_temporal_model("bilstm", 11, 3, hidden_size=64)
        assert isinstance(bilstm, TorchBiLSTM)
        
        # Test invalid model type
        with pytest.raises(ValueError):
            create_temporal_model("invalid", 11, 3)
    
    def test_temporal_loss_computation(self):
        """Test temporal loss computation with masking."""
        from passive_walker.bc.models.temporal_torch import compute_temporal_loss, create_padding_mask
        
        batch_size, seq_len, out_dim = 2, 10, 3
        
        # Create dummy predictions and targets
        predictions = torch.randn(batch_size, seq_len, out_dim)
        targets = torch.randn(batch_size, seq_len, out_dim)
        
        # Test without mask
        loss = compute_temporal_loss(predictions, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        
        # Test with mask
        sequence_lengths = torch.tensor([8, 6])  # Different lengths
        mask = create_padding_mask(sequence_lengths, seq_len)
        
        loss_masked = compute_temporal_loss(predictions, targets, mask)
        assert isinstance(loss_masked, torch.Tensor)
        assert loss_masked.item() >= 0
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through temporal models."""
        from passive_walker.bc.models.temporal_torch import TorchLSTM
        
        model = TorchLSTM(in_dim=11, out_dim=3, hidden_size=64)
        
        # Create dummy data
        x = torch.randn(2, 10, 11, requires_grad=True)
        target = torch.randn(2, 10, 3)
        
        # Forward pass
        output, _ = model(x)
        loss = torch.nn.functional.l1_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        for param in model.parameters():
            assert param.grad is not None
    
    def test_save_load_functionality(self):
        """Test model save/load functionality."""
        from passive_walker.bc.models.temporal_torch import TorchLSTM
        import tempfile
        import os
        
        # Create model
        model = TorchLSTM(in_dim=11, out_dim=3, hidden_size=64)
        
        # Test forward pass
        x = torch.randn(1, 5, 11)
        output1, _ = model(x)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            model_path = f.name
        
        try:
            # Create new model and load weights
            model2 = TorchLSTM(in_dim=11, out_dim=3, hidden_size=64)
            model2.load_state_dict(torch.load(model_path))
            
            # Test forward pass
            output2, _ = model2(x)
            
            # Check outputs are the same
            assert torch.allclose(output1, output2, atol=1e-6)
        
        finally:
            os.unlink(model_path)


class TestTemporalModelsJAX:
    """Test JAX temporal models."""
    
    def test_lstm_forward_pass(self):
        """Test LSTM forward pass with dummy data."""
        from passive_walker.bc.models.temporal_jax import LSTM
        
        # Create model
        key = jax.random.PRNGKey(42)
        model = LSTM(in_dim=11, out_dim=3, hidden_size=64, key=key)
        
        # Test single step input
        x_single = jnp.array(np.random.randn(11))
        output_single, hidden = model(x_single)
        
        # Check output shape and range
        assert output_single.shape == (3,)
        assert jnp.all(output_single >= -1.0) and jnp.all(output_single <= 1.0)
        
        # Test sequence input
        seq_len, obs_dim = 10, 11
        x_seq = jnp.array(np.random.randn(seq_len, obs_dim))
        output_seq, hidden = model(x_seq)
        
        # Check output shape
        assert output_seq.shape == (seq_len, 3)
        assert jnp.all(output_seq >= -1.0) and jnp.all(output_seq <= 1.0)
    
    def test_gru_forward_pass(self):
        """Test GRU forward pass with dummy data."""
        from passive_walker.bc.models.temporal_jax import GRU
        
        # Create model
        key = jax.random.PRNGKey(42)
        model = GRU(in_dim=11, out_dim=3, hidden_size=64, key=key)
        
        # Test single step input
        x_single = jnp.array(np.random.randn(11))
        output_single, hidden = model(x_single)
        
        # Check output shape and range
        assert output_single.shape == (3,)
        assert jnp.all(output_single >= -1.0) and jnp.all(output_single <= 1.0)
        
        # Test sequence input
        seq_len, obs_dim = 10, 11
        x_seq = jnp.array(np.random.randn(seq_len, obs_dim))
        output_seq, hidden = model(x_seq)
        
        # Check output shape
        assert output_seq.shape == (seq_len, 3)
        assert jnp.all(output_seq >= -1.0) and jnp.all(output_seq <= 1.0)
    
    def test_model_factory(self):
        """Test model factory function."""
        from passive_walker.bc.models.temporal_jax import make_temporal_model
        
        key = jax.random.PRNGKey(42)
        
        # Test LSTM creation
        lstm = make_temporal_model("lstm", 11, 3, hidden_size=64, key=key)
        assert isinstance(lstm, LSTM)
        
        # Test GRU creation
        gru = make_temporal_model("gru", 11, 3, hidden_size=64, key=key)
        assert isinstance(gru, GRU)
        
        # Test invalid model type
        with pytest.raises(ValueError):
            make_temporal_model("invalid", 11, 3, key=key)
    
    def test_temporal_loss_computation(self):
        """Test temporal loss computation."""
        from passive_walker.bc.models.temporal_jax import temporal_loss_fn, make_temporal_model
        
        key = jax.random.PRNGKey(42)
        model = make_temporal_model("lstm", 11, 3, hidden_size=64, key=key)
        
        # Create dummy data
        batch_size, seq_len, obs_dim, out_dim = 2, 10, 11, 3
        x = jnp.array(np.random.randn(batch_size, seq_len, obs_dim))
        y = jnp.array(np.random.randn(batch_size, seq_len, out_dim))
        
        # Test loss computation
        loss = temporal_loss_fn(model, x, y)
        assert isinstance(loss, jnp.ndarray)
        assert loss >= 0
    
    def test_jit_compilation(self):
        """Test JIT compilation for inference."""
        from passive_walker.bc.models.temporal_jax import make_temporal_model
        
        key = jax.random.PRNGKey(42)
        model = make_temporal_model("lstm", 11, 3, hidden_size=64, key=key)
        
        # Create JIT-compiled function
        @jax.jit
        def jit_forward(x):
            return model(x)
        
        # Test JIT compilation
        x = jnp.array(np.random.randn(11))
        output = jit_forward(x)
        
        assert output[0].shape == (3,)
        assert jnp.all(output[0] >= -1.0) and jnp.all(output[0] <= 1.0)
    
    def test_save_load_functionality(self):
        """Test model save/load functionality."""
        from passive_walker.bc.models.temporal_jax import make_temporal_model, save_temporal_model, load_temporal_model
        import tempfile
        import os
        
        # Create model
        key = jax.random.PRNGKey(42)
        model = make_temporal_model("lstm", 11, 3, hidden_size=64, key=key)
        
        # Test forward pass
        x = jnp.array(np.random.randn(11))
        output1, _ = model(x)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.eqx', delete=False) as f:
            save_temporal_model(f.name, model)
            model_path = f.name
        
        try:
            # Load model
            model2 = load_temporal_model(model_path, model)
            
            # Test forward pass
            output2, _ = model2(x)
            
            # Check outputs are the same
            assert jnp.allclose(output1, output2, atol=1e-6)
        
        finally:
            os.unlink(model_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


