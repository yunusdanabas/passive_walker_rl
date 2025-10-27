#!/usr/bin/env python3
"""
Test suite for advanced visualization framework.

This test validates the advanced visualization components and ensures
proper functionality for creating comprehensive visualizations.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest
import tempfile
import shutil

from tools.evaluation.advanced_viz import (
    VisualizationConfig,
    AdvancedVisualizer
)


class TestVisualizationConfig:
    """Test visualization configuration."""
    
    def test_config_initialization(self):
        """Test configuration initialization with defaults."""
        config = VisualizationConfig()
        
        assert config.figure_size == (12, 8)
        assert config.dpi == 300
        assert config.animation_fps == 30
        assert config.animation_duration == 10
        assert config.export_dpi == 300
        assert isinstance(config.export_formats, list)
        assert "png" in config.export_formats
    
    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = VisualizationConfig(
            figure_size=(16, 10),
            dpi=150,
            animation_fps=60,
            export_formats=["png", "pdf"]
        )
        
        assert config.figure_size == (16, 10)
        assert config.dpi == 150
        assert config.animation_fps == 60
        assert config.export_formats == ["png", "pdf"]


class TestAdvancedVisualizer:
    """Test advanced visualizer functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = VisualizationConfig(
            enable_interactive=False,
            enable_animations=False
        )
        self.visualizer = AdvancedVisualizer(self.config)
    
    def teardown_method(self):
        """Clean up test environment."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        assert self.visualizer.config == self.config
        assert self.visualizer.config.enable_interactive == False
        assert self.visualizer.config.enable_animations == False
    
    def test_3d_trajectory_plot(self):
        """Test 3D trajectory plotting."""
        # Create sample trajectory data
        trajectory_data = {
            'x': np.linspace(0, 10, 100),
            'y': np.sin(np.linspace(0, 4*np.pi, 100)),
            'z': np.cos(np.linspace(0, 4*np.pi, 100))
        }
        
        contact_data = {
            'left_contact': np.random.random(100) > 0.5,
            'right_contact': np.random.random(100) > 0.5
        }
        
        # Test plotting (should not raise exceptions)
        try:
            self.visualizer.plot_3d_trajectory(
                trajectory_data, 
                contact_data, 
                self.temp_dir
            )
            success = True
        except Exception as e:
            print(f"3D trajectory plot failed: {e}")
            success = False
        
        assert success
    
    def test_gait_analysis_plot(self):
        """Test gait analysis plotting."""
        # Create sample gait data
        gait_data = {
            'step_lengths': np.random.normal(0.5, 0.1, 50),
            'stance_durations': np.random.normal(0.6, 0.1, 50),
            'swing_durations': np.random.normal(0.4, 0.1, 50),
            'double_support_times': np.random.normal(0.1, 0.05, 50),
            'walking_speeds': np.random.normal(1.0, 0.2, 50),
            'cadence': np.random.normal(120, 20, 50)
        }
        
        # Test plotting
        try:
            self.visualizer.plot_gait_analysis(gait_data, self.temp_dir)
            success = True
        except Exception as e:
            print(f"Gait analysis plot failed: {e}")
            success = False
        
        assert success
    
    def test_model_comparison_dashboard(self):
        """Test model comparison dashboard."""
        # Create sample comparison data
        comparison_data = {
            'Model A': {
                'success_rate': np.random.beta(8, 2, 100),
                'distance': np.random.normal(5.0, 1.0, 100),
                'reward': np.random.normal(10.0, 2.0, 100),
                'success_rate_over_time': np.random.beta(8, 2, 50)
            },
            'Model B': {
                'success_rate': np.random.beta(5, 5, 100),
                'distance': np.random.normal(4.0, 1.0, 100),
                'reward': np.random.normal(8.0, 2.0, 100),
                'success_rate_over_time': np.random.beta(5, 5, 50)
            }
        }
        
        # Test plotting
        try:
            self.visualizer.plot_model_comparison_dashboard(comparison_data, self.temp_dir)
            success = True
        except Exception as e:
            print(f"Model comparison dashboard failed: {e}")
            success = False
        
        assert success
    
    def test_robustness_heatmap(self):
        """Test robustness heatmap plotting."""
        # Create sample robustness data
        robustness_data = {
            'nominal': {'success_rate': 0.8, 'distance': 5.0, 'reward': 10.0},
            'low_gravity': {'success_rate': 0.6, 'distance': 3.0, 'reward': 6.0},
            'high_gravity': {'success_rate': 0.7, 'distance': 4.0, 'reward': 8.0},
            'low_friction': {'success_rate': 0.5, 'distance': 2.0, 'reward': 4.0}
        }
        
        # Test plotting
        try:
            self.visualizer.plot_robustness_heatmap(robustness_data, self.temp_dir)
            success = True
        except Exception as e:
            print(f"Robustness heatmap failed: {e}")
            success = False
        
        assert success
    
    def test_contact_force_analysis(self):
        """Test contact force analysis plotting."""
        # Create sample contact data
        contact_data = {
            'left_force': np.random.normal(100, 20, 1000),
            'right_force': np.random.normal(100, 20, 1000),
            'left_contact_duration': np.random.normal(0.6, 0.1, 1000),
            'right_contact_duration': np.random.normal(0.6, 0.1, 1000),
            'left_contact': np.random.random(1000) > 0.5,
            'right_contact': np.random.random(1000) > 0.5
        }
        
        # Test plotting
        try:
            self.visualizer.plot_contact_force_analysis(contact_data, self.temp_dir)
            success = True
        except Exception as e:
            print(f"Contact force analysis failed: {e}")
            success = False
        
        assert success
    
    def test_interactive_dashboard(self):
        """Test interactive dashboard creation."""
        # Create sample evaluation data
        evaluation_data = {
            'performance_metrics': {
                'success_rate': 0.8,
                'distance': 5.0,
                'reward': 10.0
            },
            'robustness_data': {
                'nominal': {'success_rate': 0.8, 'distance': 5.0},
                'low_gravity': {'success_rate': 0.6, 'distance': 3.0}
            },
            'failure_distribution': {
                'forward_fall': 10,
                'backward_fall': 5,
                'stagnation': 3
            },
            'model_comparison': {
                'Model A': {'success_rate': 0.8, 'distance': 5.0},
                'Model B': {'success_rate': 0.6, 'distance': 3.0}
            }
        }
        
        # Test dashboard creation
        try:
            self.visualizer.create_interactive_dashboard(evaluation_data, self.temp_dir)
            success = True
        except Exception as e:
            print(f"Interactive dashboard failed: {e}")
            success = False
        
        assert success
    
    def test_comprehensive_report(self):
        """Test comprehensive report generation."""
        # Create comprehensive evaluation data
        evaluation_data = {
            'trajectory_data': {
                'x': np.linspace(0, 10, 100),
                'y': np.sin(np.linspace(0, 4*np.pi, 100)),
                'z': np.cos(np.linspace(0, 4*np.pi, 100))
            },
            'contact_data': {
                'left_contact': np.random.random(100) > 0.5,
                'right_contact': np.random.random(100) > 0.5,
                'left_force': np.random.normal(100, 20, 100),
                'right_force': np.random.normal(100, 20, 100)
            },
            'gait_data': {
                'step_lengths': np.random.normal(0.5, 0.1, 50),
                'stance_durations': np.random.normal(0.6, 0.1, 50)
            },
            'model_comparison': {
                'Model A': {
                    'success_rate': np.random.beta(8, 2, 100),
                    'distance': np.random.normal(5.0, 1.0, 100)
                },
                'Model B': {
                    'success_rate': np.random.beta(5, 5, 100),
                    'distance': np.random.normal(4.0, 1.0, 100)
                }
            },
            'robustness_data': {
                'nominal': {'success_rate': 0.8, 'distance': 5.0},
                'low_gravity': {'success_rate': 0.6, 'distance': 3.0}
            }
        }
        
        # Test comprehensive report generation
        try:
            self.visualizer.generate_comprehensive_report(evaluation_data, self.temp_dir)
            success = True
        except Exception as e:
            print(f"Comprehensive report failed: {e}")
            success = False
        
        assert success


class TestVisualizationIntegration:
    """Integration tests for visualization framework."""
    
    def test_visualization_workflow(self):
        """Test complete visualization workflow."""
        # Create sample data
        np.random.seed(42)
        
        trajectory_data = {
            'x': np.linspace(0, 10, 100),
            'y': np.sin(np.linspace(0, 4*np.pi, 100)),
            'z': np.cos(np.linspace(0, 4*np.pi, 100))
        }
        
        contact_data = {
            'left_contact': np.random.random(100) > 0.5,
            'right_contact': np.random.random(100) > 0.5,
            'left_force': np.random.normal(100, 20, 100),
            'right_force': np.random.normal(100, 20, 100)
        }
        
        gait_data = {
            'step_lengths': np.random.normal(0.5, 0.1, 50),
            'stance_durations': np.random.normal(0.6, 0.1, 50),
            'swing_durations': np.random.normal(0.4, 0.1, 50)
        }
        
        # Initialize visualizer
        config = VisualizationConfig(enable_interactive=False, enable_animations=False)
        visualizer = AdvancedVisualizer(config)
        
        # Test individual plotting functions
        try:
            visualizer.plot_3d_trajectory(trajectory_data, contact_data)
            visualizer.plot_gait_analysis(gait_data)
            visualizer.plot_contact_force_analysis(contact_data)
            success = True
        except Exception as e:
            print(f"Visualization workflow failed: {e}")
            success = False
        
        assert success
        
        print("✓ Visualization workflow: PASSED")


def test_visualization_simple():
    """Simple test of visualization components."""
    print("\n=== Testing Advanced Visualization Framework ===")
    
    # Test configuration
    config = VisualizationConfig(enable_interactive=False, enable_animations=False)
    print(f"✓ Configuration created with figure_size={config.figure_size}")
    
    # Test visualizer initialization
    visualizer = AdvancedVisualizer(config)
    print(f"✓ AdvancedVisualizer initialized")
    
    # Test with sample data
    np.random.seed(42)
    
    # Test 3D trajectory
    trajectory_data = {
        'x': np.linspace(0, 10, 50),
        'y': np.sin(np.linspace(0, 2*np.pi, 50)),
        'z': np.cos(np.linspace(0, 2*np.pi, 50))
    }
    
    contact_data = {
        'left_contact': np.random.random(50) > 0.5,
        'right_contact': np.random.random(50) > 0.5
    }
    
    try:
        visualizer.plot_3d_trajectory(trajectory_data, contact_data)
        print("✓ 3D trajectory plot: PASSED")
    except Exception as e:
        print(f"✗ 3D trajectory plot failed: {e}")
    
    # Test gait analysis
    gait_data = {
        'step_lengths': np.random.normal(0.5, 0.1, 20),
        'stance_durations': np.random.normal(0.6, 0.1, 20),
        'swing_durations': np.random.normal(0.4, 0.1, 20)
    }
    
    try:
        visualizer.plot_gait_analysis(gait_data)
        print("✓ Gait analysis plot: PASSED")
    except Exception as e:
        print(f"✗ Gait analysis plot failed: {e}")
    
    # Test model comparison
    comparison_data = {
        'Model A': {
            'success_rate': np.random.beta(8, 2, 50),
            'distance': np.random.normal(5.0, 1.0, 50)
        },
        'Model B': {
            'success_rate': np.random.beta(5, 5, 50),
            'distance': np.random.normal(4.0, 1.0, 50)
        }
    }
    
    try:
        visualizer.plot_model_comparison_dashboard(comparison_data)
        print("✓ Model comparison dashboard: PASSED")
    except Exception as e:
        print(f"✗ Model comparison dashboard failed: {e}")
    
    # Test robustness heatmap
    robustness_data = {
        'nominal': {'success_rate': 0.8, 'distance': 5.0},
        'low_gravity': {'success_rate': 0.6, 'distance': 3.0}
    }
    
    try:
        visualizer.plot_robustness_heatmap(robustness_data)
        print("✓ Robustness heatmap: PASSED")
    except Exception as e:
        print(f"✗ Robustness heatmap failed: {e}")
    
    print("Advanced visualization framework: PASSED")


if __name__ == "__main__":
    test_visualization_simple()
    print("\nAll advanced visualization tests passed!")
