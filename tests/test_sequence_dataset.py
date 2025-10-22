"""
Unit tests for sequence dataset functionality

Tests sequence loading, SequenceDataset class, and temporal data augmentation.
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
import tempfile
import os


class TestSequenceLoading:
    """Test sequence loading functions."""
    
    def test_load_sequences(self):
        """Test loading sequences from files."""
        from passive_walker.bc.dataset import load_sequences
        
        # Create dummy NPZ files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            for i in range(3):
                obs = np.random.randn(100, 11).astype(np.float32)
                actions = np.random.randn(99, 3).astype(np.float32)
                rewards = np.random.randn(99).astype(np.float32)
                done = np.zeros(99, dtype=bool)
                
                npz_path = os.path.join(temp_dir, f"episode_{i:06d}.npz")
                np.savez(npz_path, obs=obs, act=actions, rew=rewards, done=done)
            
            # Test loading sequences
            files = [os.path.join(temp_dir, f"episode_{i:06d}.npz") for i in range(3)]
            sequences = load_sequences(files, "both", "act")
            
            assert len(sequences) == 3
            for obs_seq, action_seq in sequences:
                assert obs_seq.shape[0] == 99  # T timesteps
                assert obs_seq.shape[1] == 11  # obs_dim
                assert action_seq.shape[0] == 99  # T timesteps
                assert action_seq.shape[1] == 3  # action_dim (both section)
    
    def test_load_sequences_with_max_length(self):
        """Test loading sequences with max length constraint."""
        from passive_walker.bc.dataset import load_sequences
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            obs = np.random.randn(100, 11).astype(np.float32)
            actions = np.random.randn(99, 3).astype(np.float32)
            rewards = np.random.randn(99).astype(np.float32)
            done = np.zeros(99, dtype=bool)
            
            npz_path = os.path.join(temp_dir, "episode_000000.npz")
            np.savez(npz_path, obs=obs, act=actions, rew=rewards, done=done)
            
            # Test with max_length
            sequences = load_sequences([npz_path], "both", "act", max_length=50)
            
            assert len(sequences) == 1
            obs_seq, action_seq = sequences[0]
            assert obs_seq.shape[0] == 50  # Truncated to max_length
            assert action_seq.shape[0] == 50
    
    def test_load_sequences_with_windows(self):
        """Test loading sequences with overlapping windows."""
        from passive_walker.bc.dataset import load_sequences_with_windows
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            obs = np.random.randn(100, 11).astype(np.float32)
            actions = np.random.randn(99, 3).astype(np.float32)
            rewards = np.random.randn(99).astype(np.float32)
            done = np.zeros(99, dtype=bool)
            
            npz_path = os.path.join(temp_dir, "episode_000000.npz")
            np.savez(npz_path, obs=obs, act=actions, rew=rewards, done=done)
            
            # Test with windows
            sequences = load_sequences_with_windows([npz_path], "both", "act", 
                                                  window_size=30, stride=15)
            
            assert len(sequences) > 1  # Should have multiple windows
            for obs_seq, action_seq in sequences:
                assert obs_seq.shape[0] == 30  # window_size
                assert action_seq.shape[0] == 30
                assert obs_seq.shape[1] == 11
                assert action_seq.shape[1] == 3


class TestSequenceDataset:
    """Test SequenceDataset class."""
    
    def test_sequence_dataset_creation(self):
        """Test SequenceDataset creation and basic functionality."""
        from passive_walker.bc.dataset import SequenceDataset
        
        # Create dummy sequences
        sequences = []
        for i in range(5):
            seq_len = np.random.randint(20, 100)
            obs_seq = np.random.randn(seq_len, 11).astype(np.float32)
            action_seq = np.random.randn(seq_len, 3).astype(np.float32)
            sequences.append((obs_seq, action_seq))
        
        # Create dataset
        dataset = SequenceDataset(sequences)
        
        assert len(dataset) == 5
        assert dataset.max_length == max(len(seq[0]) for seq in sequences)
        
        # Test __getitem__
        obs_tensor, action_tensor, mask = dataset[0]
        
        assert isinstance(obs_tensor, torch.Tensor)
        assert isinstance(action_tensor, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert obs_tensor.shape[0] == dataset.max_length
        assert action_tensor.shape[0] == dataset.max_length
        assert mask.shape[0] == dataset.max_length
    
    def test_sequence_dataset_padding(self):
        """Test sequence padding strategies."""
        from passive_walker.bc.dataset import SequenceDataset
        
        # Create sequences with different lengths
        sequences = [
            (np.random.randn(20, 11).astype(np.float32), np.random.randn(20, 3).astype(np.float32)),
            (np.random.randn(50, 11).astype(np.float32), np.random.randn(50, 3).astype(np.float32)),
            (np.random.randn(30, 11).astype(np.float32), np.random.randn(30, 3).astype(np.float32)),
        ]
        
        # Test zero padding
        dataset_zero = SequenceDataset(sequences, padding_strategy="zero")
        obs_tensor, action_tensor, mask = dataset_zero[0]
        
        # Check that padding is zeros
        assert torch.all(obs_tensor[20:] == 0)
        assert torch.all(action_tensor[20:] == 0)
        assert mask[20:].sum() == 0  # No valid timesteps after padding
        
        # Test last padding
        dataset_last = SequenceDataset(sequences, padding_strategy="last")
        obs_tensor, action_tensor, mask = dataset_last[0]
        
        # Check that padding repeats last observation
        assert torch.allclose(obs_tensor[20:], obs_tensor[19:20].repeat(30, 1))
        assert torch.allclose(action_tensor[20:], action_tensor[19:20].repeat(30, 1))
    
    def test_sequence_dataset_mask(self):
        """Test sequence masking functionality."""
        from passive_walker.bc.dataset import SequenceDataset
        
        # Create sequences with known lengths
        sequences = [
            (np.random.randn(25, 11).astype(np.float32), np.random.randn(25, 3).astype(np.float32)),
            (np.random.randn(40, 11).astype(np.float32), np.random.randn(40, 3).astype(np.float32)),
        ]
        
        dataset = SequenceDataset(sequences)
        
        # Test masks
        _, _, mask1 = dataset[0]
        _, _, mask2 = dataset[1]
        
        assert mask1.sum() == 25  # First sequence length
        assert mask2.sum() == 40  # Second sequence length
        assert mask1[25:].sum() == 0  # Padding should be masked
        assert mask2[40:].sum() == 0


class TestSequenceDataLoader:
    """Test sequence DataLoader functionality."""
    
    def test_create_sequence_loader(self):
        """Test creating sequence DataLoader."""
        from passive_walker.bc.dataset import create_sequence_loader
        
        # Create dummy sequences
        sequences = []
        for i in range(10):
            seq_len = np.random.randint(20, 80)
            obs_seq = np.random.randn(seq_len, 11).astype(np.float32)
            action_seq = np.random.randn(seq_len, 3).astype(np.float32)
            sequences.append((obs_seq, action_seq))
        
        # Create data loader
        dataloader = create_sequence_loader(sequences, batch_size=4, shuffle=True)
        
        # Test iteration
        batch = next(iter(dataloader))
        obs_batch, action_batch, mask_batch = batch
        
        assert obs_batch.shape[0] == 4  # batch_size
        assert action_batch.shape[0] == 4
        assert mask_batch.shape[0] == 4
        assert obs_batch.shape[2] == 11  # obs_dim
        assert action_batch.shape[2] == 3  # action_dim
    
    def test_create_sequence_loader_from_files(self):
        """Test creating sequence DataLoader from files."""
        from passive_walker.bc.dataset import create_sequence_loader_from_files
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            for i in range(3):
                obs = np.random.randn(100, 11).astype(np.float32)
                actions = np.random.randn(99, 3).astype(np.float32)
                rewards = np.random.randn(99).astype(np.float32)
                done = np.zeros(99, dtype=bool)
                
                npz_path = os.path.join(temp_dir, f"episode_{i:06d}.npz")
                np.savez(npz_path, obs=obs, act=actions, rew=rewards, done=done)
            
            files = [os.path.join(temp_dir, f"episode_{i:06d}.npz") for i in range(3)]
            
            # Create data loader
            dataloader = create_sequence_loader_from_files(
                files, "both", "act", batch_size=2, shuffle=False
            )
            
            # Test iteration
            batch = next(iter(dataloader))
            obs_batch, action_batch, mask_batch = batch
            
            assert obs_batch.shape[0] == 2  # batch_size
            assert obs_batch.shape[2] == 11  # obs_dim
            assert action_batch.shape[2] == 3  # action_dim


class TestTemporalAugmentation:
    """Test temporal data augmentation."""
    
    def test_time_warping(self):
        """Test time warping augmentation."""
        from passive_walker.bc.augmentation import TimeWarping
        
        # Create test sequence
        seq_len = 50
        obs_seq = np.random.randn(seq_len, 11).astype(np.float32)
        action_seq = np.random.randn(seq_len, 3).astype(np.float32)
        
        # Apply time warping
        aug = TimeWarping(warp_range=(0.9, 1.1), probability=1.0)
        warped_obs, warped_action = aug(obs_seq, action_seq)
        
        assert warped_obs.shape == obs_seq.shape
        assert warped_action.shape == action_seq.shape
    
    def test_temporal_jittering(self):
        """Test temporal jittering augmentation."""
        from passive_walker.bc.augmentation import TemporalJittering
        
        # Create test sequence
        seq_len = 50
        obs_seq = np.random.randn(seq_len, 11).astype(np.float32)
        action_seq = np.random.randn(seq_len, 3).astype(np.float32)
        
        # Apply temporal jittering
        aug = TemporalJittering(max_shift=3, probability=1.0)
        jittered_obs, jittered_action = aug(obs_seq, action_seq)
        
        assert jittered_obs.shape == obs_seq.shape
        assert jittered_action.shape == action_seq.shape
    
    def test_subsequence_extraction(self):
        """Test subsequence extraction augmentation."""
        from passive_walker.bc.augmentation import SubsequenceExtraction
        
        # Create test sequence
        seq_len = 100
        obs_seq = np.random.randn(seq_len, 11).astype(np.float32)
        action_seq = np.random.randn(seq_len, 3).astype(np.float32)
        
        # Apply subsequence extraction
        aug = SubsequenceExtraction(min_length_ratio=0.5, max_length_ratio=0.8, probability=1.0)
        sub_obs, sub_action = aug(obs_seq, action_seq)
        
        assert sub_obs.shape[0] < seq_len  # Should be shorter
        assert sub_action.shape[0] == sub_obs.shape[0]
        assert sub_obs.shape[1] == 11
        assert sub_action.shape[1] == 3
    
    def test_frame_dropout(self):
        """Test frame dropout augmentation."""
        from passive_walker.bc.augmentation import FrameDropout
        
        # Create test sequence
        seq_len = 100
        obs_seq = np.random.randn(seq_len, 11).astype(np.float32)
        action_seq = np.random.randn(seq_len, 3).astype(np.float32)
        
        # Apply frame dropout
        aug = FrameDropout(dropout_rate=0.1, probability=1.0)
        dropped_obs, dropped_action = aug(obs_seq, action_seq)
        
        assert dropped_obs.shape[0] < seq_len  # Should be shorter
        assert dropped_action.shape[0] == dropped_obs.shape[0]
        assert dropped_obs.shape[1] == 11
        assert dropped_action.shape[1] == 3
    
    def test_temporal_noise(self):
        """Test temporal noise augmentation."""
        from passive_walker.bc.augmentation import TemporalNoise
        
        # Create test sequence
        seq_len = 50
        obs_seq = np.random.randn(seq_len, 11).astype(np.float32)
        action_seq = np.random.randn(seq_len, 3).astype(np.float32)
        
        # Apply temporal noise
        aug = TemporalNoise(obs_noise_std=0.01, action_noise_std=0.005, probability=1.0)
        noisy_obs, noisy_action = aug(obs_seq, action_seq)
        
        assert noisy_obs.shape == obs_seq.shape
        assert noisy_action.shape == action_seq.shape
        assert np.all(noisy_action >= -1.0) and np.all(noisy_action <= 1.0)  # Clipped
    
    def test_composite_temporal_augmentation(self):
        """Test composite temporal augmentation."""
        from passive_walker.bc.augmentation import CompositeTemporalAugmentation, TemporalJittering, TemporalNoise
        
        # Create test sequence
        seq_len = 50
        obs_seq = np.random.randn(seq_len, 11).astype(np.float32)
        action_seq = np.random.randn(seq_len, 3).astype(np.float32)
        
        # Create composite augmentation
        aug = CompositeTemporalAugmentation([
            TemporalJittering(max_shift=2, probability=1.0),
            TemporalNoise(obs_noise_std=0.01, action_noise_std=0.005, probability=1.0)
        ])
        
        # Apply augmentation
        aug_obs, aug_action = aug(obs_seq, action_seq)
        
        assert aug_obs.shape == obs_seq.shape
        assert aug_action.shape == action_seq.shape
    
    def test_default_temporal_augmentation(self):
        """Test default temporal augmentation pipeline."""
        from passive_walker.bc.augmentation import create_default_temporal_augmentation
        
        # Create test sequence
        seq_len = 50
        obs_seq = np.random.randn(seq_len, 11).astype(np.float32)
        action_seq = np.random.randn(seq_len, 3).astype(np.float32)
        
        # Create default augmentation
        aug = create_default_temporal_augmentation()
        
        # Apply augmentation
        aug_obs, aug_action = aug(obs_seq, action_seq)
        
        assert aug_obs.shape == obs_seq.shape
        assert aug_action.shape == action_seq.shape


class TestSequenceDatasetValidation:
    """Test sequence dataset validation."""
    
    def test_validate_sequence_dataset(self):
        """Test sequence dataset validation."""
        from passive_walker.bc.dataset import validate_sequence_dataset
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test data
            for i in range(3):
                obs = np.random.randn(100, 11).astype(np.float32)
                actions = np.random.randn(99, 3).astype(np.float32)
                rewards = np.random.randn(99).astype(np.float32)
                done = np.zeros(99, dtype=bool)
                
                npz_path = os.path.join(temp_dir, f"episode_{i:06d}.npz")
                np.savez(npz_path, obs=obs, act=actions, rew=rewards, done=done)
            
            files = [os.path.join(temp_dir, f"episode_{i:06d}.npz") for i in range(3)]
            
            # Validate dataset
            stats = validate_sequence_dataset(files, "both", "act")
            
            assert "n_sequences" in stats
            assert "total_timesteps" in stats
            assert "avg_sequence_length" in stats
            assert "obs_dim" in stats
            assert "action_dim" in stats
            assert stats["n_sequences"] == 3
            assert stats["obs_dim"] == 11
            assert stats["action_dim"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
