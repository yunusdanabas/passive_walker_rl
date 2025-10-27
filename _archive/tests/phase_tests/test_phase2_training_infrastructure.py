"""
Tests for Phase 2: Training Infrastructure & Reward Shaping

Tests enhanced rewards, configuration validation, schedulers, and augmentation.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock

from passive_walker.core.reward import compute_reward, RESEARCH_WEIGHTS, RESEARCH_PARAMS
from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.bc.config import TrainingConfig, EvaluationConfig, create_research_config
from passive_walker.bc.schedulers import (
    NoScheduler, PlateauScheduler, CosineScheduler, WarmupCosineScheduler,
    create_scheduler
)
from passive_walker.bc.augmentation import (
    ObservationNoise, ActionNoise, TemporalShift, ScaleAugmentation,
    CompositeAugmentation, create_default_augmentation
)


class TestEnhancedRewards:
    """Test enhanced reward shaping."""
    
    def test_fsm_reward_unchanged(self):
        """Test that FSM reward works as before."""
        signals = {
            'dx': 0.01, 'pitch_abs': 0.1, 'u_abs_sum': 0.5, 'torso_z': 1.2,
            'velocity_x': 0.15, 'left_knee_pos': 0.0, 'right_knee_pos': 0.0,
            'left_foot_z': 0.05, 'right_foot_z': 0.05, 'u_change_sum': 0.1
        }
        
        reward, info = compute_reward(signals, 'fsm')
        
        assert isinstance(reward, float)
        assert reward > 0  # Should be positive for good signals
        assert 'r_dx' in info
        assert 'r_pitch' in info
        assert 'r_ctrl' in info
        assert 'r_alive' in info
        assert 'fell' in info
    
    def test_research_reward_components(self):
        """Test that research reward has all enhanced components."""
        signals = {
            'dx': 0.01, 'pitch_abs': 0.1, 'u_abs_sum': 0.5, 'torso_z': 1.2,
            'velocity_x': 0.15, 'left_knee_pos': 0.0, 'right_knee_pos': 0.0,
            'left_foot_z': 0.05, 'right_foot_z': 0.05, 'u_change_sum': 0.1
        }
        
        reward, info = compute_reward(signals, 'research')
        
        assert isinstance(reward, float)
        assert 'r_dx' in info
        assert 'r_upright' in info
        assert 'r_velocity' in info
        assert 'r_symmetry' in info
        assert 'r_foot_clear' in info
        assert 'r_ctrl' in info
        assert 'r_smooth' in info
        assert 'fell' in info
    
    def test_velocity_tracking_bonus(self):
        """Test velocity tracking bonus."""
        # Perfect velocity tracking
        signals_good = {
            'dx': 0.01, 'pitch_abs': 0.1, 'u_abs_sum': 0.5, 'torso_z': 1.2,
            'velocity_x': RESEARCH_PARAMS['target_velocity'],  # Perfect match
            'left_knee_pos': 0.0, 'right_knee_pos': 0.0,
            'left_foot_z': 0.05, 'right_foot_z': 0.05, 'u_change_sum': 0.1
        }
        
        # Poor velocity tracking
        signals_bad = {
            'dx': 0.01, 'pitch_abs': 0.1, 'u_abs_sum': 0.5, 'torso_z': 1.2,
            'velocity_x': RESEARCH_PARAMS['target_velocity'] + 0.1,  # Off target
            'left_knee_pos': 0.0, 'right_knee_pos': 0.0,
            'left_foot_z': 0.05, 'right_foot_z': 0.05, 'u_change_sum': 0.1
        }
        
        reward_good, info_good = compute_reward(signals_good, 'research')
        reward_bad, info_bad = compute_reward(signals_bad, 'research')
        
        assert info_good['r_velocity'] > info_bad['r_velocity']
    
    def test_symmetry_bonus(self):
        """Test symmetry bonus."""
        # Symmetric knees
        signals_symmetric = {
            'dx': 0.01, 'pitch_abs': 0.1, 'u_abs_sum': 0.5, 'torso_z': 1.2,
            'velocity_x': 0.15, 'left_knee_pos': 0.0, 'right_knee_pos': 0.0,
            'left_foot_z': 0.05, 'right_foot_z': 0.05, 'u_change_sum': 0.1
        }
        
        # Asymmetric knees
        signals_asymmetric = {
            'dx': 0.01, 'pitch_abs': 0.1, 'u_abs_sum': 0.5, 'torso_z': 1.2,
            'velocity_x': 0.15, 'left_knee_pos': 0.1, 'right_knee_pos': -0.1,
            'left_foot_z': 0.05, 'right_foot_z': 0.05, 'u_change_sum': 0.1
        }
        
        reward_sym, info_sym = compute_reward(signals_symmetric, 'research')
        reward_asym, info_asym = compute_reward(signals_asymmetric, 'research')
        
        assert info_sym['r_symmetry'] > info_asym['r_symmetry']
    
    def test_foot_clearance_bonus(self):
        """Test foot clearance bonus."""
        # Good clearance
        signals_good = {
            'dx': 0.01, 'pitch_abs': 0.1, 'u_abs_sum': 0.5, 'torso_z': 1.2,
            'velocity_x': 0.15, 'left_knee_pos': 0.0, 'right_knee_pos': 0.0,
            'left_foot_z': 0.05, 'right_foot_z': 0.05, 'u_change_sum': 0.1
        }
        
        # Poor clearance
        signals_bad = {
            'dx': 0.01, 'pitch_abs': 0.1, 'u_abs_sum': 0.5, 'torso_z': 1.2,
            'velocity_x': 0.15, 'left_knee_pos': 0.0, 'right_knee_pos': 0.0,
            'left_foot_z': 0.01, 'right_foot_z': 0.01, 'u_change_sum': 0.1
        }
        
        reward_good, info_good = compute_reward(signals_good, 'research')
        reward_bad, info_bad = compute_reward(signals_bad, 'research')
        
        assert info_good['r_foot_clear'] > info_bad['r_foot_clear']
    
    def test_environment_reward_modes(self):
        """Test that environment uses correct reward mode."""
        # Test FSM mode
        env_fsm = PassiveWalkerEnv(mode='fsm')
        obs, _ = env_fsm.reset(seed=42)
        action = env_fsm.action_space.sample()
        obs, reward, done, info = env_fsm.step(action)
        
        assert 'r_dx' in info
        assert 'r_pitch' in info
        assert 'r_ctrl' in info
        assert 'r_alive' in info
        
        # Test research mode
        env_research = PassiveWalkerEnv(mode='research')
        obs, _ = env_research.reset(seed=42)
        action = env_research.action_space.sample()
        obs, reward, done, info = env_research.step(action)
        
        assert 'r_dx' in info
        assert 'r_upright' in info
        assert 'r_velocity' in info
        assert 'r_symmetry' in info
        assert 'r_foot_clear' in info
        assert 'r_ctrl' in info
        assert 'r_smooth' in info


class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = TrainingConfig()
        assert config.backend == "torch"
        assert config.section == "both"
        assert config.epochs == 100
        assert config.batch_size == 64
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should work
        config = TrainingConfig(backend="jax", epochs=50)
        assert config.backend == "jax"
        assert config.epochs == 50
        
        # Invalid backend should raise error
        with pytest.raises(ValueError):
            TrainingConfig(backend="invalid")
        
        # Invalid epochs should raise error
        with pytest.raises(ValueError):
            TrainingConfig(epochs=-1)
        
        # Invalid batch size should raise error
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)
    
    def test_research_config(self):
        """Test research configuration."""
        config = create_research_config()
        assert config.use_enhanced_rewards is True
        assert config.randomization_profile == "moderate"
        assert config.scheduler == "cosine"
        assert config.augment is True
    
    def test_config_serialization(self):
        """Test configuration save/load."""
        config = TrainingConfig(backend="jax", epochs=50)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save(f.name)
            
            loaded_config = TrainingConfig.load(f.name)
            assert loaded_config.backend == config.backend
            assert loaded_config.epochs == config.epochs
            
            os.unlink(f.name)
    
    def test_evaluation_config(self):
        """Test evaluation configuration."""
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            f.write(b'dummy checkpoint')
            f.flush()
            
            config = EvaluationConfig(
                checkpoint_path=f.name,
                episodes=5,
                duration_sec=10.0
            )
            
            assert config.episodes == 5
            assert config.duration_sec == 10.0
            
            os.unlink(f.name)


class TestLearningRateSchedulers:
    """Test learning rate schedulers."""
    
    def test_no_scheduler(self):
        """Test no scheduling."""
        scheduler = NoScheduler(initial_lr=0.01)
        assert scheduler.get_lr() == 0.01
        
        lr = scheduler.step(epoch=10)
        assert lr == 0.01
    
    def test_plateau_scheduler(self):
        """Test plateau scheduler."""
        scheduler = PlateauScheduler(initial_lr=0.01, patience=2)
        
        # Good metrics should not reduce LR
        lr1 = scheduler.step(epoch=1, metrics={'val_loss': 0.5})
        lr2 = scheduler.step(epoch=2, metrics={'val_loss': 0.4})
        assert lr1 == lr2 == 0.01
        
        # Bad metrics should reduce LR after patience
        lr3 = scheduler.step(epoch=3, metrics={'val_loss': 0.6})
        lr4 = scheduler.step(epoch=4, metrics={'val_loss': 0.7})
        lr5 = scheduler.step(epoch=5, metrics={'val_loss': 0.8})
        
        assert lr5 < 0.01  # LR should be reduced
    
    def test_cosine_scheduler(self):
        """Test cosine scheduler."""
        scheduler = CosineScheduler(initial_lr=0.01, T_max=10)
        
        lr_start = scheduler.step(epoch=0)
        lr_mid = scheduler.step(epoch=5)
        lr_end = scheduler.step(epoch=10)
        
        assert lr_start == 0.01
        assert lr_mid < lr_start
        assert lr_end == 0.0  # eta_min
    
    def test_warmup_cosine_scheduler(self):
        """Test warmup cosine scheduler."""
        scheduler = WarmupCosineScheduler(initial_lr=0.01, T_max=10, warmup_epochs=3)
        
        lr_warmup = scheduler.step(epoch=1)
        lr_peak = scheduler.step(epoch=3)
        lr_decay = scheduler.step(epoch=7)
        
        assert lr_warmup < lr_peak
        assert lr_peak == 0.01
        assert lr_decay < lr_peak
    
    def test_scheduler_factory(self):
        """Test scheduler factory function."""
        scheduler = create_scheduler("cosine", initial_lr=0.01, T_max=10)
        assert isinstance(scheduler, CosineScheduler)
        
        scheduler = create_scheduler("plateau", initial_lr=0.01, patience=5)
        assert isinstance(scheduler, PlateauScheduler)


class TestDataAugmentation:
    """Test data augmentation."""
    
    def test_observation_noise(self):
        """Test observation noise augmentation."""
        aug = ObservationNoise(position_std=0.01, velocity_std=0.02, probability=1.0)
        
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        action = np.array([0.1, 0.2, 0.3])
        
        noisy_obs, noisy_action = aug(obs, action)
        
        assert noisy_action is action  # Action should be unchanged
        assert not np.array_equal(obs, noisy_obs)  # Obs should be different
        assert np.allclose(obs, noisy_obs, atol=0.1)  # But close
    
    def test_action_noise(self):
        """Test action noise augmentation."""
        aug = ActionNoise(std=0.01, probability=1.0)
        
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        action = np.array([0.1, 0.2, 0.3])
        
        noisy_obs, noisy_action = aug(obs, action)
        
        assert noisy_obs is obs  # Obs should be unchanged
        assert not np.array_equal(action, noisy_action)  # Action should be different
        assert np.allclose(action, noisy_action, atol=0.1)  # But close
        assert np.all(noisy_action >= -1.0) and np.all(noisy_action <= 1.0)  # Clipped
    
    def test_temporal_shift(self):
        """Test temporal shift augmentation."""
        aug = TemporalShift(max_shift=0.1, probability=1.0)
        
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        action = np.array([0.1, 0.2, 0.3])
        
        shifted_obs, shifted_action = aug(obs, action)
        
        assert shifted_action is action  # Action should be unchanged
        # Velocity components should be scaled
        assert not np.array_equal(obs[3:5], shifted_obs[3:5])
        assert not np.array_equal(obs[8:11], shifted_obs[8:11])
    
    def test_scale_augmentation(self):
        """Test scale augmentation."""
        aug = ScaleAugmentation(scale_range=(0.95, 1.05), probability=1.0)
        
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        action = np.array([0.1, 0.2, 0.3])
        
        scaled_obs, scaled_action = aug(obs, action)
        
        assert scaled_action is action  # Action should be unchanged
        # Position components should be scaled
        assert not np.array_equal(obs[0:3], scaled_obs[0:3])
        assert not np.array_equal(obs[5:8], scaled_obs[5:8])
    
    def test_composite_augmentation(self):
        """Test composite augmentation."""
        aug = CompositeAugmentation([
            ObservationNoise(probability=1.0),
            ActionNoise(probability=1.0)
        ])
        
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        action = np.array([0.1, 0.2, 0.3])
        
        aug_obs, aug_action = aug(obs, action)
        
        assert not np.array_equal(obs, aug_obs)
        assert not np.array_equal(action, aug_action)
    
    def test_default_augmentation(self):
        """Test default augmentation pipeline."""
        aug = create_default_augmentation()
        assert isinstance(aug, CompositeAugmentation)
        assert len(aug.augmentations) == 4
    
    def test_augmentation_probability(self):
        """Test that augmentation respects probability."""
        aug = ObservationNoise(probability=0.0)  # Never apply
        
        obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
        action = np.array([0.1, 0.2, 0.3])
        
        # Run multiple times to ensure no augmentation
        for _ in range(10):
            aug_obs, aug_action = aug(obs, action)
            assert np.array_equal(obs, aug_obs)
            assert np.array_equal(action, aug_action)


class TestIntegration:
    """Test integration of Phase 2 components."""
    
    def test_environment_with_research_mode(self):
        """Test environment with research mode."""
        env = PassiveWalkerEnv(mode='research', ctrl_hz=200)
        obs, _ = env.reset(seed=42)
        
        # Run a few steps
        for _ in range(5):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            
            # Check that research reward components are present
            assert 'r_velocity' in info
            assert 'r_symmetry' in info
            assert 'r_foot_clear' in info
            assert 'r_smooth' in info
    
    def test_config_with_enhanced_features(self):
        """Test configuration with enhanced features."""
        config = TrainingConfig(
            use_enhanced_rewards=True,
            randomization_profile="moderate",
            scheduler="cosine",
            augment=True,
            ctrl_hz=200
        )
        
        assert config.use_enhanced_rewards is True
        assert config.randomization_profile == "moderate"
        assert config.scheduler == "cosine"
        assert config.augment is True
        assert config.ctrl_hz == 200
    
    def test_scheduler_with_metrics(self):
        """Test scheduler with realistic metrics."""
        scheduler = PlateauScheduler(initial_lr=0.01, patience=3)
        
        # Simulate training with improving then plateauing metrics
        metrics_sequence = [
            {'val_loss': 1.0},
            {'val_loss': 0.8},
            {'val_loss': 0.6},
            {'val_loss': 0.6},  # Plateau starts
            {'val_loss': 0.6},
            {'val_loss': 0.6},
            {'val_loss': 0.6},  # Should trigger LR reduction
        ]
        
        lrs = []
        for epoch, metrics in enumerate(metrics_sequence):
            lr = scheduler.step(epoch, metrics)
            lrs.append(lr)
        
        # LR should be reduced after patience
        assert lrs[-1] < lrs[0]
    
    def test_augmentation_with_real_data(self):
        """Test augmentation with realistic data shapes."""
        aug = create_default_augmentation()
        
        # Realistic observation and action shapes
        obs = np.random.randn(11)  # 11D observation
        action = np.random.randn(3)  # 3D action
        
        aug_obs, aug_action = aug(obs, action)
        
        assert aug_obs.shape == obs.shape
        assert aug_action.shape == action.shape
        assert np.all(aug_action >= -1.0) and np.all(aug_action <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

