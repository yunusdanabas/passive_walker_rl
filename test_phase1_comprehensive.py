#!/usr/bin/env python3
"""
Comprehensive Test Suite for Phase 1: Temporal Modeling Enhancement

Tests all components:
- PyTorch temporal models (LSTM, GRU, BiLSTM)
- JAX temporal models (LSTM, GRU)
- Sequence dataset loading and augmentation
- Temporal training pipeline
- Configuration system

Run with: python test_phase1_comprehensive.py
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 70)
print("PHASE 1 COMPREHENSIVE TEST SUITE")
print("=" * 70)

# Test counters
tests_passed = 0
tests_failed = 0
test_errors = []


def test_section(name):
    """Decorator to mark test sections."""
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print('='*70)


def run_test(test_name, test_fn):
    """Run a single test and track results."""
    global tests_passed, tests_failed, test_errors
    
    try:
        print(f"\n  ► {test_name}...", end=" ")
        test_fn()
        print("✅ PASS")
        tests_passed += 1
        return True
    except Exception as e:
        print(f"❌ FAIL")
        print(f"    Error: {str(e)}")
        tests_failed += 1
        test_errors.append((test_name, str(e)))
        return False


# =============================================================================
# TEST 1: PyTorch Temporal Models
# =============================================================================

test_section("PyTorch Temporal Models")

def test_pytorch_lstm_forward():
    """Test PyTorch LSTM forward pass."""
    import torch
    from passive_walker.bc.models.temporal_torch import TorchLSTM
    
    model = TorchLSTM(in_dim=11, out_dim=3, hidden_size=32, num_layers=1)
    
    # Test sequence input
    x_seq = torch.randn(4, 10, 11)  # (batch, seq_len, in_dim)
    output, hidden = model(x_seq)
    assert output.shape == (4, 10, 3), f"Expected (4, 10, 3), got {output.shape}"
    assert torch.all(output >= -1) and torch.all(output <= 1), "Output not in [-1, 1]"
    
    # Test single-step input
    x_single = torch.randn(4, 11)  # (batch, in_dim)
    output_single, _ = model(x_single)
    assert output_single.shape == (4, 3), f"Expected (4, 3), got {output_single.shape}"

run_test("PyTorch LSTM forward pass", test_pytorch_lstm_forward)


def test_pytorch_gru_forward():
    """Test PyTorch GRU forward pass."""
    import torch
    from passive_walker.bc.models.temporal_torch import TorchGRU
    
    model = TorchGRU(in_dim=11, out_dim=3, hidden_size=32, num_layers=1)
    x = torch.randn(4, 10, 11)
    output, hidden = model(x)
    assert output.shape == (4, 10, 3)
    assert torch.all(output >= -1) and torch.all(output <= 1)

run_test("PyTorch GRU forward pass", test_pytorch_gru_forward)


def test_pytorch_bilstm_forward():
    """Test PyTorch BiLSTM forward pass."""
    import torch
    from passive_walker.bc.models.temporal_torch import TorchBiLSTM
    
    model = TorchBiLSTM(in_dim=11, out_dim=3, hidden_size=32, num_layers=1)
    x = torch.randn(4, 10, 11)
    output, hidden = model(x)
    assert output.shape == (4, 10, 3)
    assert torch.all(output >= -1) and torch.all(output <= 1)

run_test("PyTorch BiLSTM forward pass", test_pytorch_bilstm_forward)


def test_pytorch_factory():
    """Test PyTorch factory function."""
    from passive_walker.bc.models.temporal_torch import create_temporal_model
    
    lstm = create_temporal_model("lstm", 11, 3, hidden_size=32)
    gru = create_temporal_model("gru", 11, 3, hidden_size=32)
    bilstm = create_temporal_model("bilstm", 11, 3, hidden_size=32)
    
    assert lstm is not None
    assert gru is not None
    assert bilstm is not None

run_test("PyTorch factory function", test_pytorch_factory)


# =============================================================================
# TEST 2: JAX Temporal Models
# =============================================================================

test_section("JAX Temporal Models")

def test_jax_lstm_forward():
    """Test JAX LSTM forward pass."""
    import jax
    import jax.numpy as jnp
    from passive_walker.bc.models.temporal_jax import LSTM
    
    key = jax.random.PRNGKey(42)
    model = LSTM(in_dim=11, out_dim=3, hidden_size=32, key=key)
    
    # Test batch of sequences
    x = jnp.ones((4, 10, 11))  # (batch, seq_len, in_dim)
    output, hidden = model(x)
    assert output.shape == (4, 10, 3), f"Expected (4, 10, 3), got {output.shape}"
    assert jnp.all(output >= -1) and jnp.all(output <= 1), "Output not in [-1, 1]"

run_test("JAX LSTM forward pass", test_jax_lstm_forward)


def test_jax_gru_forward():
    """Test JAX GRU forward pass."""
    import jax
    import jax.numpy as jnp
    from passive_walker.bc.models.temporal_jax import GRU
    
    key = jax.random.PRNGKey(42)
    model = GRU(in_dim=11, out_dim=3, hidden_size=32, key=key)
    
    x = jnp.ones((4, 10, 11))
    output, hidden = model(x)
    assert output.shape == (4, 10, 3)
    assert jnp.all(output >= -1) and jnp.all(output <= 1)

run_test("JAX GRU forward pass", test_jax_gru_forward)


def test_jax_factory():
    """Test JAX factory function."""
    import jax
    from passive_walker.bc.models.temporal_jax import make_temporal_model
    
    key = jax.random.PRNGKey(42)
    lstm = make_temporal_model("lstm", 11, 3, hidden_size=32, key=key)
    
    key = jax.random.PRNGKey(43)
    gru = make_temporal_model("gru", 11, 3, hidden_size=32, key=key)
    
    assert lstm is not None
    assert gru is not None

run_test("JAX factory function", test_jax_factory)


# =============================================================================
# TEST 3: Sequence Dataset
# =============================================================================

test_section("Sequence Dataset and Data Loading")

def create_dummy_episodes(data_dir, num_episodes=5):
    """Create dummy episode data for testing."""
    os.makedirs(data_dir, exist_ok=True)
    
    for i in range(num_episodes):
        episode_length = np.random.randint(50, 100)
        obs = np.random.randn(episode_length, 11).astype(np.float32)
        actions = np.random.uniform(-1, 1, (episode_length, 3)).astype(np.float32)
        fsm_states = np.random.randint(0, 4, episode_length).astype(np.int32)
        
        np.savez(os.path.join(data_dir, f"episode_{i:03d}.npz"),
                 obs=obs, act=actions, fsm_state=fsm_states,
                 rew=np.random.uniform(0, 1, episode_length).astype(np.float32),
                 done=np.zeros(episode_length, dtype=bool))


def test_load_sequences():
    """Test sequence loading function."""
    from passive_walker.bc.dataset import load_sequences
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_episodes(temp_dir, num_episodes=3)
        files = [os.path.join(temp_dir, f"episode_{i:03d}.npz") for i in range(3)]
        
        sequences = load_sequences(files, section="both", label_type="act")
        
        assert len(sequences) > 0, "No sequences loaded"
        obs_seq, act_seq = sequences[0]
        assert obs_seq.shape[1] == 11, "Observation dim should be 11"
        assert act_seq.shape[1] == 3, "Action dim should be 3"

run_test("Sequence loading", test_load_sequences)


def test_sequence_dataset():
    """Test SequenceDataset class."""
    from passive_walker.bc.dataset import load_sequences, SequenceDataset
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_episodes(temp_dir, num_episodes=3)
        files = [os.path.join(temp_dir, f"episode_{i:03d}.npz") for i in range(3)]
        
        sequences = load_sequences(files, section="both", label_type="act")
        dataset = SequenceDataset(sequences, padding_strategy="zero")
        
        assert len(dataset) > 0
        obs, act, mask = dataset[0]
        assert obs.dim() == 2  # (seq_len, obs_dim)
        assert act.dim() == 2  # (seq_len, act_dim)
        assert mask.dim() == 1  # (seq_len,)

run_test("SequenceDataset class", test_sequence_dataset)


def test_sequence_loader():
    """Test sequence DataLoader creation."""
    from passive_walker.bc.dataset import create_sequence_loader_from_files
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_episodes(temp_dir, num_episodes=3)
        files = [os.path.join(temp_dir, f"episode_{i:03d}.npz") for i in range(3)]
        
        loader = create_sequence_loader_from_files(
            files, section="both", batch_size=2, shuffle=False
        )
        
        assert loader is not None
        for obs_batch, act_batch, mask_batch in loader:
            assert obs_batch.dim() == 3  # (batch, seq_len, obs_dim)
            assert act_batch.dim() == 3  # (batch, seq_len, act_dim)
            assert mask_batch.dim() == 2  # (batch, seq_len)
            break  # Just test first batch

run_test("Sequence DataLoader", test_sequence_loader)


# =============================================================================
# TEST 4: Temporal Augmentation
# =============================================================================

test_section("Temporal Data Augmentation")

def test_time_warping():
    """Test time warping augmentation."""
    import numpy as np
    from passive_walker.bc.augmentation import TimeWarping
    
    aug = TimeWarping(warp_range=(0.8, 1.2))
    obs = np.random.randn(10, 11).astype(np.float32)
    act = np.random.randn(10, 3).astype(np.float32)
    
    obs_aug, act_aug = aug(obs, act)
    assert obs_aug.shape[1] == 11  # Dimension preserved
    assert act_aug.shape[1] == 3

run_test("Time warping augmentation", test_time_warping)


def test_temporal_jittering():
    """Test temporal jittering augmentation."""
    import numpy as np
    from passive_walker.bc.augmentation import TemporalJittering
    
    aug = TemporalJittering(max_shift=3)
    obs = np.random.randn(10, 11).astype(np.float32)
    act = np.random.randn(10, 3).astype(np.float32)
    
    obs_aug, act_aug = aug(obs, act)
    assert obs_aug.shape[1] == 11

run_test("Temporal jittering", test_temporal_jittering)


def test_subsequence_extraction():
    """Test subsequence extraction."""
    import numpy as np
    from passive_walker.bc.augmentation import SubsequenceExtraction
    
    aug = SubsequenceExtraction(min_length_ratio=0.5)
    obs = np.random.randn(20, 11).astype(np.float32)
    act = np.random.randn(20, 3).astype(np.float32)
    
    obs_aug, act_aug = aug(obs, act)
    assert obs_aug.shape[0] >= 10  # At least 50% of 20
    assert obs_aug.shape[0] <= 20  # At most original length

run_test("Subsequence extraction", test_subsequence_extraction)


# =============================================================================
# TEST 5: Configuration System
# =============================================================================

test_section("Configuration System")

def test_temporal_config_validation():
    """Test TemporalTrainingConfig validation."""
    from passive_walker.bc.config import TemporalTrainingConfig
    
    with tempfile.TemporaryDirectory() as temp_dir:
        create_dummy_episodes(temp_dir, num_episodes=3)
        
        # Valid config
        config = TemporalTrainingConfig(
            backend="torch",
            section="both",
            data_dir=temp_dir,
            model_type="lstm",
            hidden_size=128
        )
        assert config.model_type == "lstm"
        assert config.hidden_size == 128
        
        # Test serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)

run_test("TemporalTrainingConfig validation", test_temporal_config_validation)


def test_config_invalid_params():
    """Test config validation catches invalid parameters."""
    from passive_walker.bc.config import TemporalTrainingConfig
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            create_dummy_episodes(temp_dir, num_episodes=1)
            
            # Invalid model type
            config = TemporalTrainingConfig(
                data_dir=temp_dir,
                model_type="invalid_type",
                hidden_size=128
            )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected

run_test("Config catches invalid parameters", test_config_invalid_params)


# =============================================================================
# TEST 6: Training Pipeline (Integration Test)
# =============================================================================

test_section("Training Pipeline Integration")

def test_pytorch_temporal_training():
    """Test PyTorch temporal training end-to-end."""
    from passive_walker.bc.config import TemporalTrainingConfig
    from passive_walker.bc.train import train_temporal_torch
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, "data")
        create_dummy_episodes(data_dir, num_episodes=5)
        
        config = TemporalTrainingConfig(
            backend="torch",
            section="both",
            data_dir=data_dir,
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            seed=42,
            model_type="lstm",
            hidden_size=32,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints")
        )
        
        train_temporal_torch(config)
        
        # Check checkpoint was created
        assert os.path.exists(config.checkpoint_dir)

run_test("PyTorch temporal training (2 epochs)", test_pytorch_temporal_training)


def test_jax_temporal_training():
    """Test JAX temporal training end-to-end."""
    from passive_walker.bc.config import TemporalTrainingConfig
    from passive_walker.bc.train import train_temporal_jax
    
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, "data")
        create_dummy_episodes(data_dir, num_episodes=5)
        
        config = TemporalTrainingConfig(
            backend="jax",
            section="both",
            data_dir=data_dir,
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            seed=42,
            model_type="lstm",
            hidden_size=32,
            checkpoint_dir=os.path.join(temp_dir, "checkpoints")
        )
        
        train_temporal_jax(config)
        
        # Check checkpoint was created
        assert os.path.exists(config.checkpoint_dir)

run_test("JAX temporal training (2 epochs)", test_jax_temporal_training)


# =============================================================================
# TEST 7: Temporal Loss Function
# =============================================================================

test_section("Temporal Loss Function")

def test_temporal_loss_computation():
    """Test temporal loss with masking."""
    import torch
    from passive_walker.bc.train import compute_temporal_loss
    
    pred = torch.randn(4, 10, 3)  # (batch, seq_len, action_dim)
    target = torch.randn(4, 10, 3)
    mask = torch.ones(4, 10, dtype=torch.bool)
    
    loss, components = compute_temporal_loss(pred, target, mask)
    
    assert isinstance(loss, torch.Tensor)
    assert "base_loss" in components
    assert "smoothness_loss" in components
    assert loss.item() >= 0

run_test("Temporal loss computation", test_temporal_loss_computation)


def test_temporal_loss_with_masking():
    """Test temporal loss properly handles masking."""
    import torch
    from passive_walker.bc.train import compute_temporal_loss
    
    pred = torch.randn(2, 10, 3)
    target = torch.randn(2, 10, 3)
    
    # Mask out last 5 timesteps
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[:, 5:] = False
    
    loss_masked, _ = compute_temporal_loss(pred, target, mask)
    
    # Full mask
    mask_full = torch.ones(2, 10, dtype=torch.bool)
    loss_full, _ = compute_temporal_loss(pred, target, mask_full)
    
    # Losses should be different
    assert loss_masked.item() != loss_full.item()

run_test("Temporal loss with masking", test_temporal_loss_with_masking)


# =============================================================================
# FINAL REPORT
# =============================================================================

print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)
print(f"Total Tests: {tests_passed + tests_failed}")
print(f"✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")

if tests_failed > 0:
    print("\n" + "="*70)
    print("FAILED TESTS:")
    print("="*70)
    for test_name, error in test_errors:
        print(f"\n  ❌ {test_name}")
        print(f"     {error}")

print("\n" + "="*70)
if tests_failed == 0:
    print("🎉 ALL TESTS PASSED! Phase 1 is production-ready!")
else:
    print(f"⚠️  {tests_failed} test(s) failed. Please review and fix.")
print("="*70)

# Exit with appropriate code
sys.exit(0 if tests_failed == 0 else 1)
