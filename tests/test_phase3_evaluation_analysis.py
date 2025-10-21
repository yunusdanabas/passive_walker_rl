"""
Tests for Phase 3: Evaluation & Analysis

Tests comprehensive evaluation suite, visualization tools, and report generation.
"""

import pytest
import numpy as np
import tempfile
import os
import json
from unittest.mock import Mock, patch
from dataclasses import dataclass

from passive_walker.bc.evaluate import (
    ComprehensiveEvaluator, EvaluationConfig, EvaluationResults, EpisodeMetrics,
    evaluate_model_comprehensive
)
from passive_walker.bc.visualize import (
    TrajectoryVisualizer, create_comprehensive_plots
)
from passive_walker.bc.report import (
    ReportGenerator, generate_evaluation_report
)


class TestComprehensiveEvaluator:
    """Test comprehensive evaluation suite."""
    
    def test_evaluation_config_creation(self):
        """Test evaluation configuration creation."""
        config = EvaluationConfig(
            checkpoint_path="test_model.pt",
            episodes=5,
            duration_sec=10.0,
            physics_conditions=["nominal", "gentle"]
        )
        
        assert config.episodes == 5
        assert config.duration_sec == 10.0
        assert config.physics_conditions == ["nominal", "gentle"]
    
    def test_evaluation_config_validation(self):
        """Test evaluation configuration validation."""
        # Valid config should work
        config = EvaluationConfig(
            checkpoint_path="test_model.pt",
            episodes=5,
            duration_sec=10.0
        )
        assert config.episodes == 5
        
        # Invalid episodes should raise error
        with pytest.raises(ValueError):
            EvaluationConfig(
                checkpoint_path="test_model.pt",
                episodes=0,  # Invalid
                duration_sec=10.0
            )
        
        # Invalid duration should raise error
        with pytest.raises(ValueError):
            EvaluationConfig(
                checkpoint_path="test_model.pt",
                episodes=5,
                duration_sec=0.0  # Invalid
            )
    
    def test_episode_metrics_creation(self):
        """Test episode metrics creation."""
        metrics = EpisodeMetrics(
            episode_id=0,
            duration=25.0,
            steps=2500,
            success=True,
            distance=15.0,
            gait_cycles=8,
            avg_reward=1.5,
            total_reward=3750.0,
            energy_efficiency=0.1,
            fsm_imitation_error=0.05,
            foot_clearance_avg=0.04,
            velocity_tracking_error=0.1,
            symmetry_error=0.05
        )
        
        assert metrics.episode_id == 0
        assert metrics.success is True
        assert metrics.distance == 15.0
        assert metrics.gait_cycles == 8
    
    def test_evaluation_results_creation(self):
        """Test evaluation results creation."""
        episode = EpisodeMetrics(
            episode_id=0, duration=25.0, steps=2500, success=True,
            distance=15.0, gait_cycles=8, avg_reward=1.5, total_reward=3750.0,
            energy_efficiency=0.1, fsm_imitation_error=0.05,
            foot_clearance_avg=0.04, velocity_tracking_error=0.1,
            symmetry_error=0.05
        )
        
        results = EvaluationResults(
            model_path="test_model.pt",
            config={'episodes': 1},
            episodes=[episode],
            summary_stats={'success_rate': 1.0},
            robustness_matrix={'nominal': {'success_rate': 1.0}},
            comparison_with_fsm={'fsm_success_rate': 0.95},
            timestamp="2024-01-01 12:00:00"
        )
        
        assert len(results.episodes) == 1
        assert results.summary_stats['success_rate'] == 1.0
        assert results.model_path == "test_model.pt"
    
    @patch('passive_walker.bc.evaluate.torch.load')
    @patch('passive_walker.bc.evaluate.create_model')
    def test_torch_model_loading(self, mock_create_model, mock_torch_load):
        """Test PyTorch model loading."""
        # Mock checkpoint and model
        mock_checkpoint = {'model_state_dict': {}}
        mock_torch_load.return_value = mock_checkpoint
        
        mock_model = Mock()
        mock_create_model.return_value = mock_model
        
        # Mock metadata file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_meta.json', delete=False) as f:
            metadata = {
                'input_dim': 11,
                'output_dim': 3,
                'hidden_sizes': [256, 256],
                'activation': 'relu',
                'dropout': 0.0
            }
            json.dump(metadata, f)
            meta_path = f.name
        
        # Mock checkpoint file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            evaluator = ComprehensiveEvaluator(EvaluationConfig(checkpoint_path=checkpoint_path))
            model, loaded_metadata = evaluator._load_torch_model(checkpoint_path)
            
            assert model == mock_model
            assert loaded_metadata['input_dim'] == 11
            assert loaded_metadata['output_dim'] == 3
        finally:
            os.unlink(meta_path)
            os.unlink(checkpoint_path)
    
    def test_action_assembly(self):
        """Test action assembly for different control sections."""
        evaluator = ComprehensiveEvaluator(EvaluationConfig(checkpoint_path="test.pt"))
        
        # Test hip section
        action = evaluator._assemble_action("hip", np.array([0.5]))
        expected = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        assert np.array_equal(action, expected)
        
        # Test knees section
        action = evaluator._assemble_action("knees", np.array([0.3, 0.4]))
        expected = np.array([0.0, 0.3, 0.4], dtype=np.float32)
        assert np.array_equal(action, expected)
        
        # Test both section
        action = evaluator._assemble_action("both", np.array([0.1, 0.2, 0.3]))
        expected = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assert np.array_equal(action, expected)
    
    def test_gait_cycle_counting(self):
        """Test gait cycle counting."""
        evaluator = ComprehensiveEvaluator(EvaluationConfig(checkpoint_path="test.pt"))
        
        # Create sample joint position data with oscillations
        joint_positions = []
        for i in range(100):
            hip_angle = 0.5 * np.sin(i * 0.1)  # Oscillating hip
            joint_positions.append([hip_angle, 0.0, 0.0])
        
        cycles = evaluator._count_gait_cycles(joint_positions)
        assert cycles > 0  # Should detect some cycles
    
    def test_summary_stats_computation(self):
        """Test summary statistics computation."""
        episodes = [
            EpisodeMetrics(
                episode_id=0, duration=25.0, steps=2500, success=True,
                distance=15.0, gait_cycles=8, avg_reward=1.5, total_reward=3750.0,
                energy_efficiency=0.1, fsm_imitation_error=0.05,
                foot_clearance_avg=0.04, velocity_tracking_error=0.1,
                symmetry_error=0.05
            ),
            EpisodeMetrics(
                episode_id=1, duration=20.0, steps=2000, success=False,
                distance=10.0, gait_cycles=6, avg_reward=1.0, total_reward=2000.0,
                energy_efficiency=0.08, fsm_imitation_error=0.08,
                foot_clearance_avg=0.03, velocity_tracking_error=0.15,
                symmetry_error=0.08
            )
        ]
        
        evaluator = ComprehensiveEvaluator(EvaluationConfig(checkpoint_path="test.pt"))
        summary_stats = evaluator._compute_summary_stats(episodes)
        
        assert summary_stats['success_rate'] == 0.5  # 1 out of 2 episodes
        assert summary_stats['avg_distance'] == 12.5  # (15 + 10) / 2
        assert summary_stats['avg_reward'] == 1.25  # (1.5 + 1.0) / 2
    
    def test_robustness_matrix_computation(self):
        """Test robustness matrix computation."""
        episodes = [
            EpisodeMetrics(
                episode_id=0, duration=25.0, steps=2500, success=True,
                distance=15.0, gait_cycles=8, avg_reward=1.5, total_reward=3750.0,
                energy_efficiency=0.1, fsm_imitation_error=0.05,
                foot_clearance_avg=0.04, velocity_tracking_error=0.1,
                symmetry_error=0.05
            )
        ]
        
        # Add physics condition attribute
        episodes[0].physics_condition = "nominal"
        
        evaluator = ComprehensiveEvaluator(EvaluationConfig(checkpoint_path="test.pt"))
        robustness_matrix = evaluator._compute_robustness_matrix(episodes)
        
        assert "nominal" in robustness_matrix
        assert robustness_matrix["nominal"]["success_rate"] == 1.0
        assert robustness_matrix["nominal"]["avg_distance"] == 15.0


class TestTrajectoryVisualizer:
    """Test trajectory visualization tools."""
    
    def test_visualizer_creation(self):
        """Test visualizer creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = TrajectoryVisualizer(temp_dir)
            assert visualizer.output_dir == temp_dir
    
    def test_sample_data_creation(self):
        """Test creation of sample trajectory data."""
        sample_data = {
            'observations': [np.random.randn(11) for _ in range(50)],
            'actions': [np.random.randn(3) for _ in range(50)],
            'joint_positions': [np.random.randn(3) for _ in range(50)],
            'joint_velocities': [np.random.randn(3) for _ in range(50)],
            'reward_components': [{'r_dx': 0.1, 'r_pitch': -0.05} for _ in range(50)]
        }
        
        assert len(sample_data['observations']) == 50
        assert len(sample_data['actions']) == 50
        assert len(sample_data['joint_positions']) == 50
    
    def test_joint_trajectory_plotting(self):
        """Test joint trajectory plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = TrajectoryVisualizer(temp_dir)
            
            sample_data = {
                'joint_positions': [np.random.randn(3) for _ in range(50)]
            }
            time = np.linspace(0, 5, 50)
            
            # Should not raise error
            visualizer._plot_joint_trajectories(
                Mock(), sample_data, time, "Test", "blue"
            )
    
    def test_control_signal_plotting(self):
        """Test control signal plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = TrajectoryVisualizer(temp_dir)
            
            sample_data = {
                'actions': [np.random.randn(3) for _ in range(50)]
            }
            time = np.linspace(0, 5, 50)
            
            # Should not raise error
            visualizer._plot_control_signals(
                Mock(), sample_data, time, "Test", "blue"
            )
    
    def test_reward_component_plotting(self):
        """Test reward component plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = TrajectoryVisualizer(temp_dir)
            
            sample_data = {
                'reward_components': [
                    {'r_dx': 0.1, 'r_pitch': -0.05, 'r_velocity': 0.2}
                    for _ in range(50)
                ]
            }
            time = np.linspace(0, 5, 50)
            
            # Should not raise error
            visualizer._plot_reward_components(Mock(), sample_data, time)
    
    def test_phase_portrait_plotting(self):
        """Test phase portrait plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = TrajectoryVisualizer(temp_dir)
            
            sample_data = {
                'joint_positions': [np.random.randn(3) for _ in range(50)],
                'joint_velocities': [np.random.randn(3) for _ in range(50)]
            }
            
            # Should not raise error
            visualizer._plot_phase_portrait(Mock(), sample_data, "Test", "blue")
    
    def test_episode_comparison_plotting(self):
        """Test episode comparison plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = TrajectoryVisualizer(temp_dir)
            
            sample_data = {
                'observations': [np.random.randn(11) for _ in range(50)],
                'actions': [np.random.randn(3) for _ in range(50)],
                'joint_positions': [np.random.randn(3) for _ in range(50)],
                'joint_velocities': [np.random.randn(3) for _ in range(50)],
                'reward_components': [{'r_dx': 0.1} for _ in range(50)]
            }
            
            plot_file = visualizer.plot_episode_comparison(
                sample_data, sample_data, 0, save=True
            )
            
            assert os.path.exists(plot_file)
            assert plot_file.endswith('.png')
    
    def test_reward_analysis_plotting(self):
        """Test reward analysis plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = TrajectoryVisualizer(temp_dir)
            
            episodes = [
                {
                    'total_reward': 100.0,
                    'distance': 15.0,
                    'success': True,
                    'reward_components': {'r_dx': 0.1, 'r_velocity': 0.2}
                },
                {
                    'total_reward': 80.0,
                    'distance': 12.0,
                    'success': False,
                    'reward_components': {'r_dx': 0.08, 'r_velocity': 0.15}
                }
            ]
            
            plot_file = visualizer.plot_reward_analysis(episodes, save=True)
            
            assert os.path.exists(plot_file)
            assert plot_file.endswith('.png')
    
    def test_robustness_matrix_plotting(self):
        """Test robustness matrix plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = TrajectoryVisualizer(temp_dir)
            
            robustness_data = {
                'nominal': {'success_rate': 0.9, 'avg_distance': 15.0, 'avg_reward': 1.5},
                'gentle': {'success_rate': 0.8, 'avg_distance': 12.0, 'avg_reward': 1.2},
                'steep': {'success_rate': 0.6, 'avg_distance': 8.0, 'avg_reward': 0.8}
            }
            
            plot_file = visualizer.plot_robustness_matrix(robustness_data, save=True)
            
            assert os.path.exists(plot_file)
            assert plot_file.endswith('.png')
    
    def test_foot_clearance_analysis_plotting(self):
        """Test foot clearance analysis plotting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = TrajectoryVisualizer(temp_dir)
            
            episodes = [
                {
                    'trajectory_data': {
                        'foot_positions': [[0.05, 0.03], [0.04, 0.06], [0.06, 0.04]]
                    }
                }
            ]
            
            plot_file = visualizer.plot_foot_clearance_analysis(episodes, save=True)
            
            assert os.path.exists(plot_file)
            assert plot_file.endswith('.png')


class TestReportGenerator:
    """Test report generation."""
    
    def test_report_generator_creation(self):
        """Test report generator creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(temp_dir)
            assert generator.output_dir == temp_dir
    
    def test_executive_summary_generation(self):
        """Test executive summary generation."""
        episode = EpisodeMetrics(
            episode_id=0, duration=25.0, steps=2500, success=True,
            distance=15.0, gait_cycles=8, avg_reward=1.5, total_reward=3750.0,
            energy_efficiency=0.1, fsm_imitation_error=0.05,
            foot_clearance_avg=0.04, velocity_tracking_error=0.1,
            symmetry_error=0.05
        )
        
        results = EvaluationResults(
            model_path="test_model.pt",
            config={'episodes': 1},
            episodes=[episode],
            summary_stats={'success_rate': 1.0, 'avg_distance': 15.0},
            robustness_matrix={'nominal': {'success_rate': 1.0}},
            comparison_with_fsm={'fsm_success_rate': 0.95},
            timestamp="2024-01-01 12:00:00"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(temp_dir)
            summary = generator._create_executive_summary(results)
            
            assert "Success Rate: 100.0%" in summary
            assert "Average Distance: 15.00 meters" in summary
            assert "Excellent" in summary  # High success rate
    
    def test_performance_section_generation(self):
        """Test performance section generation."""
        results = EvaluationResults(
            model_path="test_model.pt",
            config={},
            episodes=[],
            summary_stats={
                'success_rate': 0.8,
                'avg_distance': 12.5,
                'avg_reward': 1.2
            },
            robustness_matrix={},
            comparison_with_fsm={},
            timestamp="2024-01-01 12:00:00"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(temp_dir)
            performance_section = generator._create_performance_section(results)
            
            assert "success_rate" in performance_section
            assert "avg_distance" in performance_section
            assert "avg_reward" in performance_section
    
    def test_reward_analysis_section_generation(self):
        """Test reward analysis section generation."""
        episode = EpisodeMetrics(
            episode_id=0, duration=25.0, steps=2500, success=True,
            distance=15.0, gait_cycles=8, avg_reward=1.5, total_reward=3750.0,
            energy_efficiency=0.1, fsm_imitation_error=0.05,
            foot_clearance_avg=0.04, velocity_tracking_error=0.1,
            symmetry_error=0.05,
            reward_components={'r_dx': 0.1, 'r_velocity': 0.2, 'r_symmetry': 0.15}
        )
        
        results = EvaluationResults(
            model_path="test_model.pt",
            config={'use_enhanced_rewards': True},
            episodes=[episode],
            summary_stats={},
            robustness_matrix={},
            comparison_with_fsm={},
            timestamp="2024-01-01 12:00:00"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(temp_dir)
            reward_section = generator._create_reward_analysis_section(results)
            
            assert "r_dx" in reward_section
            assert "r_velocity" in reward_section
            assert "r_symmetry" in reward_section
    
    def test_robustness_section_generation(self):
        """Test robustness section generation."""
        results = EvaluationResults(
            model_path="test_model.pt",
            config={},
            episodes=[],
            summary_stats={},
            robustness_matrix={
                'nominal': {'success_rate': 0.9, 'avg_distance': 15.0},
                'gentle': {'success_rate': 0.8, 'avg_distance': 12.0},
                'steep': {'success_rate': 0.6, 'avg_distance': 8.0}
            },
            comparison_with_fsm={},
            timestamp="2024-01-01 12:00:00"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(temp_dir)
            robustness_section = generator._create_robustness_section(results)
            
            assert "nominal" in robustness_section
            assert "gentle" in robustness_section
            assert "steep" in robustness_section
    
    def test_failure_analysis_section_generation(self):
        """Test failure analysis section generation."""
        # Mix of successful and failed episodes
        episodes = [
            EpisodeMetrics(
                episode_id=0, duration=25.0, steps=2500, success=True,
                distance=15.0, gait_cycles=8, avg_reward=1.5, total_reward=3750.0,
                energy_efficiency=0.1, fsm_imitation_error=0.05,
                foot_clearance_avg=0.04, velocity_tracking_error=0.1,
                symmetry_error=0.05
            ),
            EpisodeMetrics(
                episode_id=1, duration=10.0, steps=1000, success=False,
                distance=5.0, gait_cycles=3, avg_reward=0.8, total_reward=800.0,
                energy_efficiency=0.05, fsm_imitation_error=0.1,
                foot_clearance_avg=0.02, velocity_tracking_error=0.2,
                symmetry_error=0.1
            )
        ]
        
        results = EvaluationResults(
            model_path="test_model.pt",
            config={},
            episodes=episodes,
            summary_stats={'success_rate': 0.5},
            robustness_matrix={},
            comparison_with_fsm={},
            timestamp="2024-01-01 12:00:00"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(temp_dir)
            failure_section = generator._create_failure_analysis_section(results)
            
            assert "Failed Episodes: 1/2" in failure_section
            assert "Failed Episodes" in failure_section
            assert "Successful Episodes" in failure_section
    
    def test_recommendations_section_generation(self):
        """Test recommendations section generation."""
        # Low performance results
        results = EvaluationResults(
            model_path="test_model.pt",
            config={'use_enhanced_rewards': True},
            episodes=[],
            summary_stats={
                'success_rate': 0.3,  # Low success rate
                'avg_distance': 5.0,   # Low distance
                'avg_reward': 0.5     # Low reward
            },
            robustness_matrix={},
            comparison_with_fsm={},
            timestamp="2024-01-01 12:00:00"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = ReportGenerator(temp_dir)
            recommendations = generator._create_recommendations_section(results)
            
            assert "Improve Success Rate" in recommendations
            assert "Improve Distance Performance" in recommendations
            assert "Improve Reward Performance" in recommendations
    
    def test_full_report_generation(self):
        """Test full report generation."""
        episode = EpisodeMetrics(
            episode_id=0, duration=25.0, steps=2500, success=True,
            distance=15.0, gait_cycles=8, avg_reward=1.5, total_reward=3750.0,
            energy_efficiency=0.1, fsm_imitation_error=0.05,
            foot_clearance_avg=0.04, velocity_tracking_error=0.1,
            symmetry_error=0.05,
            reward_components={'r_dx': 0.1, 'r_velocity': 0.2}
        )
        
        results = EvaluationResults(
            model_path="test_model.pt",
            config={'use_enhanced_rewards': True, 'episodes': 1},
            episodes=[episode],
            summary_stats={'success_rate': 1.0, 'avg_distance': 15.0},
            robustness_matrix={'nominal': {'success_rate': 1.0}},
            comparison_with_fsm={'fsm_success_rate': 0.95},
            timestamp="2024-01-01 12:00:00"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = generate_evaluation_report(results, temp_dir, temp_dir)
            
            assert os.path.exists(report_path)
            assert report_path.endswith('.md')
            
            # Check report content
            with open(report_path, 'r') as f:
                content = f.read()
                
            assert "# BC Model Evaluation Report" in content
            assert "Executive Summary" in content
            assert "Performance Metrics" in content
            assert "Enhanced Reward Analysis" in content


class TestIntegration:
    """Test integration of Phase 3 components."""
    
    def test_evaluation_to_visualization_integration(self):
        """Test integration between evaluation and visualization."""
        episode = EpisodeMetrics(
            episode_id=0, duration=25.0, steps=2500, success=True,
            distance=15.0, gait_cycles=8, avg_reward=1.5, total_reward=3750.0,
            energy_efficiency=0.1, fsm_imitation_error=0.05,
            foot_clearance_avg=0.04, velocity_tracking_error=0.1,
            symmetry_error=0.05,
            trajectory_data={
                'observations': [np.random.randn(11) for _ in range(50)],
                'actions': [np.random.randn(3) for _ in range(50)],
                'joint_positions': [np.random.randn(3) for _ in range(50)],
                'joint_velocities': [np.random.randn(3) for _ in range(50)],
                'reward_components': [{'r_dx': 0.1} for _ in range(50)],
                'foot_positions': [[0.05, 0.03] for _ in range(50)]
            }
        )
        
        results = EvaluationResults(
            model_path="test_model.pt",
            config={'use_enhanced_rewards': True},
            episodes=[episode],
            summary_stats={'success_rate': 1.0},
            robustness_matrix={'nominal': {'success_rate': 1.0}},
            comparison_with_fsm={'fsm_success_rate': 0.95},
            timestamp="2024-01-01 12:00:00"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            plot_files = create_comprehensive_plots(results, temp_dir)
            
            assert len(plot_files) > 0
            for plot_file in plot_files:
                assert os.path.exists(plot_file)
                assert plot_file.endswith('.png')
    
    def test_evaluation_to_report_integration(self):
        """Test integration between evaluation and report generation."""
        episode = EpisodeMetrics(
            episode_id=0, duration=25.0, steps=2500, success=True,
            distance=15.0, gait_cycles=8, avg_reward=1.5, total_reward=3750.0,
            energy_efficiency=0.1, fsm_imitation_error=0.05,
            foot_clearance_avg=0.04, velocity_tracking_error=0.1,
            symmetry_error=0.05,
            reward_components={'r_dx': 0.1, 'r_velocity': 0.2}
        )
        
        results = EvaluationResults(
            model_path="test_model.pt",
            config={'use_enhanced_rewards': True},
            episodes=[episode],
            summary_stats={'success_rate': 1.0, 'avg_distance': 15.0},
            robustness_matrix={'nominal': {'success_rate': 1.0}},
            comparison_with_fsm={'fsm_success_rate': 0.95},
            timestamp="2024-01-01 12:00:00"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            report_path = generate_evaluation_report(results, temp_dir, temp_dir)
            
            assert os.path.exists(report_path)
            
            # Check that report contains expected sections
            with open(report_path, 'r') as f:
                content = f.read()
                
            expected_sections = [
                "Executive Summary",
                "Configuration", 
                "Performance Metrics",
                "Enhanced Reward Analysis",
                "Robustness Analysis",
                "Comparison with FSM Baseline",
                "Failure Analysis",
                "Visualizations",
                "Recommendations"
            ]
            
            for section in expected_sections:
                assert section in content
    
    def test_end_to_end_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        # This would test the full pipeline from model loading to report generation
        # For now, we'll test the components work together
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample evaluation results
            episode = EpisodeMetrics(
                episode_id=0, duration=25.0, steps=2500, success=True,
                distance=15.0, gait_cycles=8, avg_reward=1.5, total_reward=3750.0,
                energy_efficiency=0.1, fsm_imitation_error=0.05,
                foot_clearance_avg=0.04, velocity_tracking_error=0.1,
                symmetry_error=0.05,
                reward_components={'r_dx': 0.1, 'r_velocity': 0.2},
                trajectory_data={
                    'observations': [np.random.randn(11) for _ in range(50)],
                    'actions': [np.random.randn(3) for _ in range(50)],
                    'joint_positions': [np.random.randn(3) for _ in range(50)],
                    'joint_velocities': [np.random.randn(3) for _ in range(50)],
                    'reward_components': [{'r_dx': 0.1} for _ in range(50)],
                    'foot_positions': [[0.05, 0.03] for _ in range(50)]
                }
            )
            
            results = EvaluationResults(
                model_path="test_model.pt",
                config={'use_enhanced_rewards': True},
                episodes=[episode],
                summary_stats={'success_rate': 1.0, 'avg_distance': 15.0},
                robustness_matrix={'nominal': {'success_rate': 1.0}},
                comparison_with_fsm={'fsm_success_rate': 0.95},
                timestamp="2024-01-01 12:00:00"
            )
            
            # Generate visualizations
            plot_files = create_comprehensive_plots(results, temp_dir)
            assert len(plot_files) > 0
            
            # Generate report
            report_path = generate_evaluation_report(results, temp_dir, temp_dir)
            assert os.path.exists(report_path)
            
            # Verify report references plots
            with open(report_path, 'r') as f:
                content = f.read()
                assert "Visualizations" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

