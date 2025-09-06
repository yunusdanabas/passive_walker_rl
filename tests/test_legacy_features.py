"""
Comprehensive tests for legacy feature parity in the unified environment.
Tests domain randomization, FSM parameterization, termination conditions,
RGB rendering, seeding, and rich info metrics.
"""

import pytest
import numpy as np
import tempfile
import os
from passive_walker.core.env import PassiveWalkerEnv
from passive_walker.core.io import load_walker_config
from passive_walker.core.config import WalkerConfig, PhysicsCfg, EnvCfg, ControlCfg, TerminationCfg, RewardCfg, FsmCfg, RenderCfg, DebugCfg


class TestLegacyFeatures:
    """Base class with common test utilities."""
    
    def _create_test_config(self):
        """Create a minimal test configuration."""
        return WalkerConfig(
            mode="research",
            env=EnvCfg(simend=10.0, ctrl_hz=60, xml_path="passive_walker/assets/passiveWalker_model.xml", randomize_physics=False),
            physics=PhysicsCfg(
                ramp_deg_min=10.0, ramp_deg_max=14.0, friction=(0.8, 1.0),
                mass_jitter=0.05, fall_z_min=0.15, fall_pitch_max=1.0, randomize_physics=False
            ),
            control=ControlCfg(
                kp=(5.0, 1000.0, 1000.0), kv=(1.0, 100.0, 100.0),
                umin=(-50.0, -800.0, -800.0), umax=(50.0, 800.0, 800.0),
                joint_ranges=((-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)),
                use_nn_for_hip=False, use_nn_for_knees=False
            ),
            terminations=TerminationCfg(fall_z_min=0.15, fall_pitch_max=1.0),
            reward=RewardCfg(preset="minimal"),
        )


class TestDomainRandomization(TestLegacyFeatures):
    """Test domain randomization features."""
    
    def test_dr_toggle(self):
        """Test that DR can be toggled on/off."""
        # Create config with DR enabled
        cfg = self._create_test_config()
        cfg.physics.randomize_physics = True
        
        env = PassiveWalkerEnv(cfg, use_gui=False)
        
        # Reset multiple times and check that physics parameters change
        env.reset()
        gravity1 = env.model.opt.gravity.copy()
        friction1 = env.model.geom_friction[0, 0].copy()
        mass1 = env.model.body_mass[env.b_torso].copy()
        
        env.reset()
        gravity2 = env.model.opt.gravity.copy()
        friction2 = env.model.geom_friction[0, 0].copy()
        mass2 = env.model.body_mass[env.b_torso].copy()
        
        # Should be different due to randomization
        assert not np.allclose(gravity1, gravity2)
        assert not np.allclose(friction1, friction2)
        assert not np.allclose(mass1, mass2)
        
        env.close()
    
    def test_dr_deterministic_with_seed(self):
        """Test that DR is deterministic with same seed."""
        cfg = self._create_test_config()
        cfg.physics.randomize_physics = True
        
        # Test with same seed
        env1 = PassiveWalkerEnv(cfg, use_gui=False)
        env2 = PassiveWalkerEnv(cfg, use_gui=False)
        
        env1.reset(seed=123)
        env2.reset(seed=123)
        
        # Should be identical
        assert np.allclose(env1.model.opt.gravity, env2.model.opt.gravity)
        assert np.allclose(env1.model.geom_friction[0, 0], env2.model.geom_friction[0, 0])
        assert np.allclose(env1.model.body_mass[env1.b_torso], env2.model.body_mass[env2.b_torso])
        
        env1.close()
        env2.close()
    
    def test_dr_different_with_different_seeds(self):
        """Test that DR produces different results with different seeds."""
        cfg = self._create_test_config()
        cfg.physics.randomize_physics = True
        
        env = PassiveWalkerEnv(cfg, use_gui=False)
        
        env.reset(seed=123)
        gravity1 = env.model.opt.gravity.copy()
        
        env.reset(seed=456)
        gravity2 = env.model.opt.gravity.copy()
        
        # Should be different
        assert not np.allclose(gravity1, gravity2)
        
        env.close()


class TestFSMParameterization(TestLegacyFeatures):
    """Test FSM parameterization features."""
    
    def test_fsm_config_usage(self):
        """Test that FSM uses configuration parameters."""
        cfg = self._create_test_config()
        cfg.fsm = FsmCfg(
            contact_height=0.03,
            knee_release_threshold=0.02,
            hip_swing_pos=0.4,
            hip_swing_neg=-0.4,
            knee_stance=0.1,
            knee_retract=0.3
        )
        
        env = PassiveWalkerEnv(cfg, use_gui=False)
        obs, info = env.reset()
        
        # Run a few steps to let FSM update
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break
        
        # Check that FSM is using config values
        assert env.fsm.cfg.contact_height == 0.03
        assert env.fsm.cfg.knee_release_threshold == 0.02
        assert env.fsm.cfg.hip_swing_pos == 0.4
        assert env.fsm.cfg.hip_swing_neg == -0.4
        assert env.fsm.cfg.knee_stance == 0.1
        assert env.fsm.cfg.knee_retract == 0.3
        
        env.close()
    
    def test_fsm_desired_angles(self):
        """Test that FSM desired angles use config values."""
        cfg = self._create_test_config()
        cfg.fsm = FsmCfg(
            hip_swing_pos=0.5,
            hip_swing_neg=-0.5,
            knee_stance=0.2,
            knee_retract=0.4
        )
        
        env = PassiveWalkerEnv(cfg, use_gui=False)
        obs, info = env.reset()
        
        # Get desired angles
        hip_des = env.fsm.desired_hip()
        lk_des, rk_des = env.fsm.desired_knees()
        
        # Should be using config values
        assert hip_des in [-0.5, 0.5]  # hip_swing_neg or hip_swing_pos
        assert lk_des in [0.2, 0.4]    # knee_stance or knee_retract
        assert rk_des in [0.2, 0.4]    # knee_stance or knee_retract
        
        env.close()


class TestTerminationConditions(TestLegacyFeatures):
    """Test termination condition features."""
    
    def test_stall_termination_toggle(self):
        """Test that stall termination can be toggled."""
        cfg = self._create_test_config()
        cfg.terminations.enable_stall_termination = True
        cfg.terminations.max_idle_speed = 0.05  # Very low threshold
        
        env = PassiveWalkerEnv(cfg, use_gui=False)
        obs, info = env.reset()
        
        # Run with very small actions to trigger stall
        for _ in range(100):
            action = np.array([0.001, 0.001, 0.001])  # Tiny actions
            obs, reward, done, info = env.step(action)
            if done:
                break
        
        # Should terminate due to stall
        assert done
        assert info["stalled"]
        
        env.close()
    
    def test_fall_termination_conditions(self):
        """Test fall termination conditions."""
        cfg = self._create_test_config()
        cfg.terminations.fall_pitch_max = 0.5  # Lower threshold
        cfg.terminations.fall_z_min = 0.5      # Higher threshold
        
        env = PassiveWalkerEnv(cfg, use_gui=False)
        obs, info = env.reset()
        
        # Run with actions that might cause falls
        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break
        
        # Check termination reasons
        if done:
            assert info["fell"] or info["pitch_abs"] > 0.5 or info["torso_z"] < 0.5
        
        env.close()
    
    def test_rich_info_metrics(self):
        """Test rich info metrics."""
        cfg = self._create_test_config()
        cfg.debug.log_quality = True
        cfg.debug.log_fsm = True
        
        env = PassiveWalkerEnv(cfg, use_gui=False)
        obs, info = env.reset()
        
        # Run a few steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break
        
        # Check that rich info is present
        assert "fell" in info
        assert "stalled" in info
        assert "unstable" in info
        assert "quality_score" in info
        assert "fsm_state" in info
        assert "knee_states" in info
        
        # Check quality score is reasonable
        assert 0.0 <= info["quality_score"] <= 3.0
        
        env.close()


class TestRGBRendering(TestLegacyFeatures):
    """Test RGB array rendering features."""
    
    def test_rgb_array_mode(self):
        """Test that rgb_array mode returns image data."""
        cfg = self._create_test_config()
        cfg.render.rgb_array_width = 320
        cfg.render.rgb_array_height = 240
        
        env = PassiveWalkerEnv(cfg, use_gui=True)
        obs, info = env.reset()
        
        # Test rgb_array rendering
        img = env.render(mode="rgb_array")
        
        # Should return numpy array with correct shape
        assert img is not None
        assert isinstance(img, np.ndarray)
        assert img.shape == (240, 320, 3)  # (height, width, channels)
        assert img.dtype == np.uint8
        
        env.close()
    
    def test_render_config_usage(self):
        """Test that render config is used."""
        cfg = self._create_test_config()
        cfg.render.camera_distance = 5.0
        cfg.render.rgb_array_width = 640
        cfg.render.rgb_array_height = 480
        
        env = PassiveWalkerEnv(cfg, use_gui=True)
        obs, info = env.reset()
        
        # Check that config values are used
        assert env.cfg.render.camera_distance == 5.0
        assert env.cfg.render.rgb_array_width == 640
        assert env.cfg.render.rgb_array_height == 480
        
        env.close()


class TestSeedingDeterminism(TestLegacyFeatures):
    """Test seeding and determinism features."""
    
    def test_seed_method(self):
        """Test that seed method works correctly."""
        cfg = self._create_test_config()
        env = PassiveWalkerEnv(cfg, use_gui=False)
        
        # Test seed method
        seed_list = env.seed(123)
        assert seed_list == [123]
        
        # Test that RNG is seeded
        assert env._np_rng is not None
        
        env.close()
    
    def test_deterministic_reset(self):
        """Test that reset with same seed is deterministic."""
        cfg = self._create_test_config()
        cfg.physics.randomize_physics = True
        
        # Create two environments
        env1 = PassiveWalkerEnv(cfg, use_gui=False)
        env2 = PassiveWalkerEnv(cfg, use_gui=False)
        
        # Reset with same seed
        obs1, info1 = env1.reset(seed=123)
        obs2, info2 = env2.reset(seed=123)
        
        # Should be identical
        assert np.allclose(obs1, obs2)
        
        # Run a few steps
        for _ in range(5):
            action = env1.action_space.sample()
            obs1, reward1, done1, info1 = env1.step(action)
            obs2, reward2, done2, info2 = env2.step(action)
            
            assert np.allclose(obs1, obs2)
            assert np.allclose(reward1, reward2)
            assert done1 == done2
        
        env1.close()
        env2.close()


class TestConfigLoading:
    """Test configuration loading with new sections."""
    
    def test_config_with_all_sections(self):
        """Test loading config with all new sections."""
        # Create a temporary YAML file with all sections
        yaml_content = """
mode: "research"
env:
  simend: 20.0
  ctrl_hz: 60
  xml_path: "passive_walker/assets/passiveWalker_model.xml"
  randomize_physics: false
physics:
  ramp_deg_min: 12.0
  ramp_deg_max: 16.0
  friction: [0.9, 1.1]
  mass_jitter: 0.1
  fall_z_min: 0.2
  fall_pitch_max: 0.8
  randomize_physics: true
control:
  kp: [6.0, 1200.0, 1200.0]
  kv: [1.5, 120.0, 120.0]
  umin: [-60.0, -900.0, -900.0]
  umax: [60.0, 900.0, 900.0]
  joint_ranges: [[-0.6, 0.6], [-0.6, 0.6], [-0.6, 0.6]]
  use_nn_for_hip: false
  use_nn_for_knees: false
terminations:
  fall_z_min: 0.2
  fall_pitch_max: 0.8
  max_idle_speed: 0.05
  enable_stall_termination: true
reward:
  preset: "aggressive"
  overrides: {c_ac: 0.001, vx_star: 1.5}
fsm:
  contact_height: 0.025
  knee_release_threshold: 0.015
  hip_swing_pos: 0.35
  hip_swing_neg: -0.35
  knee_stance: 0.05
  knee_retract: 0.25
render:
  camera_distance: 4.0
  rgb_array_width: 800
  rgb_array_height: 600
debug:
  log_quality: true
  log_fsm: true
  verbose_info: true
jax:
  enable: true
  batched: false
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            # Load config
            cfg = load_walker_config(temp_path)
            
            # Check all sections are loaded
            assert cfg.mode == "research"
            assert cfg.physics.randomize_physics == True
            assert cfg.terminations.enable_stall_termination == True
            assert cfg.fsm.contact_height == 0.025
            assert cfg.render.rgb_array_width == 800
            assert cfg.debug.log_quality == True
            assert cfg.jax.enable == True
            
            # Test that environment can be created
            env = PassiveWalkerEnv(cfg, use_gui=False)
            obs, info = env.reset()
            assert obs.shape == (11,)
            env.close()
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__])
