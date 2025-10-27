#!/usr/bin/env python3
"""
Failure Mode Analysis for Passive Walker Models

This module implements systematic failure detection and analysis:
- Forward fall, backward fall, lateral collapse detection
- Stagnation and oscillation detection
- Failure pattern clustering using TSNE/k-means
- Failure prediction classifier
- High-risk state identification
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
try:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from passive_walker.core.env import PassiveWalkerEnv


@dataclass
class FailureAnalysisConfig:
    """Configuration for failure analysis."""
    # Failure detection thresholds
    pitch_fall_threshold: float = 0.5  # radians
    roll_collapse_threshold: float = 0.3  # radians
    velocity_stagnation_threshold: float = 0.1  # m/s
    oscillation_threshold: float = 0.5  # joint velocity variance
    
    # Analysis parameters
    failure_window_size: int = 50  # steps before failure to analyze
    min_episode_length: int = 100  # minimum episode length for analysis
    
    # Clustering parameters
    n_clusters: int = 5
    tsne_perplexity: float = 30.0
    tsne_n_components: int = 2
    
    # Prediction parameters
    prediction_horizon: int = 50  # steps ahead to predict failure
    test_size: float = 0.2  # fraction for test set


@dataclass
class FailureEvent:
    """Represents a single failure event."""
    episode_idx: int
    step_idx: int
    failure_type: str
    failure_state: np.ndarray
    pre_failure_sequence: np.ndarray
    failure_context: Dict


@dataclass
class FailureAnalysisResult:
    """Results from failure analysis."""
    total_episodes: int
    total_failures: int
    failure_rate: float
    failure_distribution: Dict[str, int]
    failure_clusters: Dict[str, List[FailureEvent]]
    high_risk_states: List[np.ndarray]
    prediction_accuracy: float
    failure_patterns: Dict[str, Dict]


class FailureAnalyzer:
    """Systematic failure detection and analysis for passive walker models."""
    
    def __init__(self, config: Optional[FailureAnalysisConfig] = None):
        """Initialize failure analyzer.
        
        Args:
            config: Failure analysis configuration
        """
        self.config = config or FailureAnalysisConfig()
        self.failure_events: List[FailureEvent] = []
        self.failure_classifier = None
        
    def analyze_failures(self, episodes: List[Dict]) -> FailureAnalysisResult:
        """Analyze failures in episode data.
        
        Args:
            episodes: List of episode dictionaries with observations and actions
            
        Returns:
            Failure analysis results
        """
        print("Starting failure analysis...")
        
        # Detect failures
        self._detect_failures(episodes)
        
        # Cluster failure patterns
        failure_clusters = self._cluster_failure_patterns()
        
        # Identify high-risk states
        high_risk_states = self._identify_high_risk_states(episodes)
        
        # Train failure prediction classifier
        prediction_accuracy = self._train_failure_predictor(episodes)
        
        # Analyze failure patterns
        failure_patterns = self._analyze_failure_patterns()
        
        # Compute overall statistics
        total_episodes = len(episodes)
        total_failures = len(self.failure_events)
        failure_rate = total_failures / total_episodes if total_episodes > 0 else 0.0
        
        # Compute failure distribution
        failure_distribution = {}
        for event in self.failure_events:
            failure_distribution[event.failure_type] = failure_distribution.get(event.failure_type, 0) + 1
        
        result = FailureAnalysisResult(
            total_episodes=total_episodes,
            total_failures=total_failures,
            failure_rate=failure_rate,
            failure_distribution=failure_distribution,
            failure_clusters=failure_clusters,
            high_risk_states=high_risk_states,
            prediction_accuracy=prediction_accuracy,
            failure_patterns=failure_patterns
        )
        
        print(f"Failure analysis completed: {total_failures} failures detected in {total_episodes} episodes")
        return result
    
    def _detect_failures(self, episodes: List[Dict]):
        """Detect failures in episode data."""
        self.failure_events = []
        
        for episode_idx, episode in enumerate(episodes):
            observations = episode.get("observations", [])
            actions = episode.get("actions", [])
            
            if len(observations) < self.config.min_episode_length:
                continue
            
            # Analyze each step for failure conditions
            for step_idx in range(len(observations)):
                obs = observations[step_idx]
                
                # Check for different failure types
                failure_type = self._classify_failure_type(obs, step_idx, observations, actions)
                
                if failure_type != "none":
                    # Extract pre-failure sequence
                    start_idx = max(0, step_idx - self.config.failure_window_size)
                    pre_failure_sequence = np.array(observations[start_idx:step_idx])
                    
                    # Create failure event
                    failure_event = FailureEvent(
                        episode_idx=episode_idx,
                        step_idx=step_idx,
                        failure_type=failure_type,
                        failure_state=obs.copy(),
                        pre_failure_sequence=pre_failure_sequence,
                        failure_context={
                            "pitch": obs[2],
                            "roll": obs[1] if len(obs) > 1 else 0.0,
                            "velocity": np.sqrt(obs[3]**2 + obs[4]**2) if len(obs) > 4 else 0.0,
                            "position": obs[0] if len(obs) > 0 else 0.0
                        }
                    )
                    
                    self.failure_events.append(failure_event)
                    break  # Only record first failure per episode
    
    def _classify_failure_type(self, obs: np.ndarray, step_idx: int, 
                              observations: List[np.ndarray], actions: List[np.ndarray]) -> str:
        """Classify the type of failure at a given step."""
        # Forward fall (pitch too high)
        if obs[2] > self.config.pitch_fall_threshold:
            return "forward_fall"
        
        # Backward fall (pitch too low)
        if obs[2] < -self.config.pitch_fall_threshold:
            return "backward_fall"
        
        # Lateral collapse (roll too high)
        if len(obs) > 1 and abs(obs[1]) > self.config.roll_collapse_threshold:
            return "lateral_collapse"
        
        # Stagnation (low velocity)
        if len(obs) > 4:
            velocity = np.sqrt(obs[3]**2 + obs[4]**2)
            if velocity < self.config.velocity_stagnation_threshold:
                return "stagnation"
        
        # Oscillation (high joint velocity variance)
        if step_idx > 10:  # Need some history
            recent_actions = actions[max(0, step_idx-10):step_idx]
            if recent_actions:
                action_variance = np.var(recent_actions, axis=0)
                if np.mean(action_variance) > self.config.oscillation_threshold:
                    return "oscillation"
        
        return "none"
    
    def _cluster_failure_patterns(self) -> Dict[str, List[FailureEvent]]:
        """Cluster failure events by pattern similarity."""
        if len(self.failure_events) < 2:
            return {"cluster_0": self.failure_events}
        
        # Extract features from pre-failure sequences
        features = []
        for event in self.failure_events:
            if len(event.pre_failure_sequence) > 0:
                # Use mean and std of pre-failure sequence as features
                feature_vector = np.concatenate([
                    np.mean(event.pre_failure_sequence, axis=0),
                    np.std(event.pre_failure_sequence, axis=0)
                ])
                features.append(feature_vector)
            else:
                features.append(np.zeros(34))  # 17 obs dims * 2 (mean + std)
        
        features = np.array(features)
        
        if HAS_SKLEARN:
            # Apply t-SNE for visualization
            if len(features) > 1:
                tsne = TSNE(n_components=self.config.tsne_n_components, 
                           perplexity=min(self.config.tsne_perplexity, len(features)-1),
                           random_state=42)
                features_tsne = tsne.fit_transform(features)
            else:
                features_tsne = features
            
            # Cluster using K-means
            n_clusters = min(self.config.n_clusters, len(self.failure_events))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(features_tsne)
            else:
                cluster_labels = [0] * len(self.failure_events)
        else:
            # Fallback: simple clustering based on failure type
            failure_types = [event.failure_type for event in self.failure_events]
            unique_types = list(set(failure_types))
            cluster_labels = [unique_types.index(ft) for ft in failure_types]
        
        # Group failures by cluster
        failure_clusters = {}
        for i, event in enumerate(self.failure_events):
            cluster_id = f"cluster_{cluster_labels[i]}"
            if cluster_id not in failure_clusters:
                failure_clusters[cluster_id] = []
            failure_clusters[cluster_id].append(event)
        
        return failure_clusters
    
    def _identify_high_risk_states(self, episodes: List[Dict]) -> List[np.ndarray]:
        """Identify high-risk states that often lead to failures."""
        if not self.failure_events:
            return []
        
        # Collect states from failure events
        failure_states = [event.failure_state for event in self.failure_events]
        
        # Also collect states from pre-failure sequences
        pre_failure_states = []
        for event in self.failure_events:
            if len(event.pre_failure_sequence) > 0:
                # Take last few states from pre-failure sequence
                last_states = event.pre_failure_sequence[-5:]
                pre_failure_states.extend(last_states)
        
        # Combine and find common patterns
        all_risk_states = failure_states + pre_failure_states
        
        if len(all_risk_states) == 0:
            return []
        
        # Use clustering to identify representative high-risk states
        n_representatives = min(10, len(all_risk_states))
        if HAS_SKLEARN and n_representatives > 1:
            kmeans = KMeans(n_clusters=n_representatives, random_state=42)
            kmeans.fit(all_risk_states)
            high_risk_states = kmeans.cluster_centers_
        else:
            # Fallback: take every nth state
            step = max(1, len(all_risk_states) // n_representatives)
            high_risk_states = all_risk_states[::step][:n_representatives]
        
        return high_risk_states
    
    def _train_failure_predictor(self, episodes: List[Dict]) -> float:
        """Train a classifier to predict failures."""
        if len(self.failure_events) < 10:  # Need sufficient data
            return 0.0
        
        # Prepare training data
        X = []  # Features (observations)
        y = []  # Labels (failure in next N steps)
        
        for episode_idx, episode in enumerate(episodes):
            observations = episode.get("observations", [])
            
            for step_idx in range(len(observations) - self.config.prediction_horizon):
                obs = observations[step_idx]
                
                # Check if failure occurs in next N steps
                failure_in_horizon = False
                for future_step in range(step_idx + 1, min(step_idx + self.config.prediction_horizon + 1, len(observations))):
                    future_obs = observations[future_step]
                    if self._classify_failure_type(future_obs, future_step, observations, []) != "none":
                        failure_in_horizon = True
                        break
                
                X.append(obs)
                y.append(1 if failure_in_horizon else 0)
        
        if len(X) < 20:  # Need sufficient training data
            return 0.0
        
        X = np.array(X)
        y = np.array(y)
        
        if HAS_SKLEARN:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=42, stratify=y
            )
            
            # Train classifier
            self.failure_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight='balanced'
            )
            self.failure_classifier.fit(X_train, y_train)
            
            # Evaluate
            accuracy = self.failure_classifier.score(X_test, y_test)
        else:
            # Fallback: simple accuracy based on failure rate
            accuracy = np.mean(y)
        
        return accuracy
    
    def _analyze_failure_patterns(self) -> Dict[str, Dict]:
        """Analyze patterns in different failure types."""
        patterns = {}
        
        for failure_type in set(event.failure_type for event in self.failure_events):
            type_events = [event for event in self.failure_events if event.failure_type == failure_type]
            
            if not type_events:
                continue
            
            # Analyze context statistics
            contexts = [event.failure_context for event in type_events]
            
            pattern = {
                "count": len(type_events),
                "avg_pitch": np.mean([ctx.get("pitch", 0) for ctx in contexts]),
                "avg_velocity": np.mean([ctx.get("velocity", 0) for ctx in contexts]),
                "avg_position": np.mean([ctx.get("position", 0) for ctx in contexts]),
                "pitch_std": np.std([ctx.get("pitch", 0) for ctx in contexts]),
                "velocity_std": np.std([ctx.get("velocity", 0) for ctx in contexts]),
                "position_std": np.std([ctx.get("position", 0) for ctx in contexts])
            }
            
            patterns[failure_type] = pattern
        
        return patterns
    
    def generate_failure_report(self, result: FailureAnalysisResult, 
                              output_dir: str = "experiments/outputs/failure_analysis"):
        """Generate comprehensive failure analysis report."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate failure distribution pie chart
        self._plot_failure_distribution(result, output_path)
        
        # Generate failure pattern visualization
        self._plot_failure_patterns(result, output_path)
        
        # Generate high-risk state visualization
        self._plot_high_risk_states(result, output_path)
        
        # Generate summary report
        self._generate_summary_report(result, output_path)
        
        print(f"Failure analysis report generated: {output_path}")
    
    def _plot_failure_distribution(self, result: FailureAnalysisResult, output_path: Path):
        """Generate failure distribution pie chart."""
        if not result.failure_distribution:
            print("No failures to plot.")
            return
        
        plt.figure(figsize=(10, 8))
        labels = list(result.failure_distribution.keys())
        sizes = list(result.failure_distribution.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title("Failure Mode Distribution")
        plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig(output_path / "failure_distribution.png", dpi=300)
        plt.close()
    
    def _plot_failure_patterns(self, result: FailureAnalysisResult, output_path: Path):
        """Generate failure pattern visualization."""
        if not result.failure_clusters:
            print("No failure clusters to plot.")
            return
        
        # Create scatter plot of failure patterns
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        failure_types = list(result.failure_patterns.keys())
        
        for idx, failure_type in enumerate(failure_types[:4]):  # Plot up to 4 types
            if idx >= 4:
                break
            
            ax = axes[idx]
            pattern = result.failure_patterns[failure_type]
            
            # Create bar plot of pattern statistics
            stats = ["avg_pitch", "avg_velocity", "avg_position"]
            values = [pattern[stat] for stat in stats]
            
            bars = ax.bar(stats, values)
            ax.set_title(f"{failure_type.replace('_', ' ').title()} Pattern")
            ax.set_ylabel("Value")
            
            # Add count annotation
            ax.text(0.5, 0.95, f"Count: {pattern['count']}", 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for idx in range(len(failure_types), 4):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_path / "failure_patterns.png", dpi=300)
        plt.close()
    
    def _plot_high_risk_states(self, result: FailureAnalysisResult, output_path: Path):
        """Generate high-risk state visualization."""
        if not result.high_risk_states:
            print("No high-risk states to plot.")
            return
        
        # Create heatmap of high-risk states
        plt.figure(figsize=(12, 8))
        
        # Use first 10 dimensions for visualization
        n_dims = min(10, result.high_risk_states[0].shape[0])
        risk_matrix = np.array([state[:n_dims] for state in result.high_risk_states])
        
        plt.imshow(risk_matrix, cmap='Reds', aspect='auto')
        plt.colorbar(label='State Value')
        plt.xlabel('Observation Dimension')
        plt.ylabel('High-Risk State')
        plt.title('High-Risk States Heatmap')
        
        # Add dimension labels
        plt.xticks(range(n_dims), [f'Dim {i}' for i in range(n_dims)])
        
        plt.tight_layout()
        plt.savefig(output_path / "high_risk_states.png", dpi=300)
        plt.close()
    
    def _generate_summary_report(self, result: FailureAnalysisResult, output_path: Path):
        """Generate text summary report."""
        report_path = output_path / "failure_analysis_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("FAILURE ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total episodes analyzed: {result.total_episodes}\n")
            f.write(f"Total failures detected: {result.total_failures}\n")
            f.write(f"Overall failure rate: {result.failure_rate:.2%}\n\n")
            
            f.write("FAILURE DISTRIBUTION\n")
            f.write("-" * 20 + "\n")
            for failure_type, count in result.failure_distribution.items():
                percentage = count / result.total_failures * 100 if result.total_failures > 0 else 0
                f.write(f"{failure_type}: {count} ({percentage:.1f}%)\n")
            
            f.write(f"\nFAILURE PREDICTION ACCURACY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Accuracy: {result.prediction_accuracy:.2%}\n")
            
            f.write(f"\nFAILURE PATTERNS\n")
            f.write("-" * 20 + "\n")
            for failure_type, pattern in result.failure_patterns.items():
                f.write(f"\n{failure_type}:\n")
                f.write(f"  Count: {pattern['count']}\n")
                f.write(f"  Average pitch: {pattern['avg_pitch']:.3f}\n")
                f.write(f"  Average velocity: {pattern['avg_velocity']:.3f}\n")
                f.write(f"  Average position: {pattern['avg_position']:.3f}\n")
            
            f.write(f"\nHIGH-RISK STATES\n")
            f.write("-" * 20 + "\n")
            f.write(f"Number of high-risk states identified: {len(result.high_risk_states)}\n")
            
            if result.high_risk_states:
                f.write("High-risk state characteristics:\n")
                for i, state in enumerate(result.high_risk_states[:5]):  # Show first 5
                    f.write(f"  State {i+1}: pitch={state[2]:.3f}, velocity={np.sqrt(state[3]**2 + state[4]**2):.3f}\n")
        
        print(f"Summary report saved: {report_path}")


def main():
    """Main function for failure analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Failure Analysis for Passive Walker Models")
    parser.add_argument("--episodes-file", type=str, required=True,
                      help="Path to episodes data file (NPZ format)")
    parser.add_argument("--output-dir", type=str, default="experiments/outputs/failure_analysis",
                      help="Output directory for results")
    
    args = parser.parse_args()
    
    # Load episodes data
    episodes_data = np.load(args.episodes_file)
    
    # Convert to episode format
    episodes = []
    for i in range(len(episodes_data['obs'])):
        episode = {
            "observations": episodes_data['obs'][i],
            "actions": episodes_data['act'][i],
            "rewards": episodes_data['rew'][i] if 'rew' in episodes_data else [],
            "done": episodes_data['done'][i] if 'done' in episodes_data else False
        }
        episodes.append(episode)
    
    # Initialize analyzer
    analyzer = FailureAnalyzer()
    
    # Run analysis
    result = analyzer.analyze_failures(episodes)
    
    # Generate report
    analyzer.generate_failure_report(result, args.output_dir)
    
    print("Failure analysis completed!")


if __name__ == "__main__":
    main()
