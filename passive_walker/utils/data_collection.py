"""
Data collection utilities for FSM environment.
Focuses on generating high-quality expert trajectories for imitation learning.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from passive_walker.envs.mujoco_fsm_env import PassiveWalkerEnv
from passive_walker.constants import XML_PATH


class DataCollector:
    """
    Collects high-quality expert trajectories from FSM environment.
    Includes filtering and quality assessment for imitation learning.
    """
    
    def __init__(self, 
                 xml_path: str = str(XML_PATH),
                 quality_threshold: float = 2.0,
                 min_episode_length: int = 100,
                 max_episode_length: int = 2000):
        """
        Initialize data collector.
        
        Args:
            xml_path: Path to MuJoCo XML model
            quality_threshold: Minimum quality score to keep episode
            min_episode_length: Minimum steps for valid episode
            max_episode_length: Maximum steps before truncation
        """
        self.xml_path = xml_path
        self.quality_threshold = quality_threshold
        self.min_episode_length = min_episode_length
        self.max_episode_length = max_episode_length
        
        # Statistics
        self.episode_count = 0
        self.valid_episodes = 0
        self.total_steps = 0
        self.quality_scores = []
        
    def collect_episode(self, 
                       use_nn_for_hip: bool = False,
                       use_nn_for_knees: bool = True,
                       use_gui: bool = False,
                       simend: float = 10.0) -> Optional[Dict[str, Any]]:
        """
        Collect a single episode from FSM environment.
        
        Returns:
            Episode data dict if quality is sufficient, None otherwise
        """
        # Create environment
        env = PassiveWalkerEnv(
            xml_path=self.xml_path,
            simend=simend,
            use_nn_for_hip=use_nn_for_hip,
            use_nn_for_knees=use_nn_for_knees,
            use_gui=use_gui
        )
        
        # Reset environment
        obs = env.reset()
        
        # Collect episode data
        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'infos': [],
            'quality_metrics': {
                'avg_quality_score': 0.0,
                'stability_score': 0.0,
                'motion_score': 0.0,
                'clearance_score': 0.0,
                'episode_length': 0,
                'total_distance': 0.0,
                'fall_rate': 0.0,
                'stall_rate': 0.0
            }
        }
        
        step_count = 0
        quality_scores = []
        stability_scores = []
        motion_scores = []
        clearance_scores = []
        fall_count = 0
        stall_count = 0
        total_distance = 0.0
        
        try:
            while step_count < self.max_episode_length:
                # FSM environment uses internal control, so action is dummy
                action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                
                # Step environment
                obs, reward, done, info = env.step(action)
                
                # Store data
                episode_data['observations'].append(obs.copy())
                episode_data['actions'].append(action.copy())
                episode_data['rewards'].append(reward)
                episode_data['dones'].append(done)
                episode_data['infos'].append(info.copy())
                
                # Extract quality metrics
                quality_score = info.get('quality_score', 0.0)
                quality_scores.append(quality_score)
                
                # Compute individual scores
                pitch_abs = info.get('pitch_abs', 0.0)
                vx = info.get('vx', 0.0)
                left_foot_z = info.get('left_foot_z', 0.0)
                right_foot_z = info.get('right_foot_z', 0.0)
                
                stability_score = max(0.0, 1.0 - pitch_abs / 0.5)
                motion_score = 1.0 if 0.1 <= abs(vx) <= 2.0 else 0.5
                foot_clearance = min(left_foot_z, right_foot_z)
                clearance_score = 1.0 if foot_clearance > 0.02 else 0.0
                
                stability_scores.append(stability_score)
                motion_scores.append(motion_score)
                clearance_scores.append(clearance_score)
                
                # Track statistics
                if info.get('fell', False):
                    fall_count += 1
                if info.get('stalled', False):
                    stall_count += 1
                
                total_distance += abs(info.get('dx', 0.0))
                step_count += 1
                
                if done:
                    break
                    
        except Exception as e:
            print(f"Error during episode collection: {e}")
            env.close()
            return None
        finally:
            env.close()
        
        # Check episode quality
        if step_count < self.min_episode_length:
            print(f"Episode too short: {step_count} steps")
            return None
        
        # Compute episode quality metrics
        avg_quality_score = np.mean(quality_scores)
        episode_data['quality_metrics'].update({
            'avg_quality_score': float(avg_quality_score),
            'stability_score': float(np.mean(stability_scores)),
            'motion_score': float(np.mean(motion_scores)),
            'clearance_score': float(np.mean(clearance_scores)),
            'episode_length': step_count,
            'total_distance': float(total_distance),
            'fall_rate': float(fall_count / step_count),
            'stall_rate': float(stall_count / step_count)
        })
        
        # Filter by quality
        if avg_quality_score < self.quality_threshold:
            print(f"Episode quality too low: {avg_quality_score:.2f} < {self.quality_threshold}")
            return None
        
        # Convert lists to numpy arrays for efficiency
        episode_data['observations'] = np.array(episode_data['observations'], dtype=np.float32)
        episode_data['actions'] = np.array(episode_data['actions'], dtype=np.float32)
        episode_data['rewards'] = np.array(episode_data['rewards'], dtype=np.float32)
        episode_data['dones'] = np.array(episode_data['dones'], dtype=bool)
        
        # Update statistics
        self.episode_count += 1
        self.valid_episodes += 1
        self.total_steps += step_count
        self.quality_scores.append(avg_quality_score)
        
        print(f"Collected episode {self.episode_count}: "
              f"length={step_count}, quality={avg_quality_score:.2f}, "
              f"distance={total_distance:.2f}m")
        
        return episode_data
    
    def collect_dataset(self, 
                       num_episodes: int = 100,
                       save_path: Optional[str] = None,
                       **episode_kwargs) -> List[Dict[str, Any]]:
        """
        Collect a dataset of high-quality episodes.
        
        Args:
            num_episodes: Number of episodes to collect
            save_path: Optional path to save dataset
            **episode_kwargs: Additional arguments for collect_episode
            
        Returns:
            List of episode data dictionaries
        """
        print(f"Collecting {num_episodes} episodes...")
        print(f"Quality threshold: {self.quality_threshold}")
        print(f"Min episode length: {self.min_episode_length}")
        
        dataset = []
        start_time = time.time()
        
        while len(dataset) < num_episodes:
            episode_data = self.collect_episode(**episode_kwargs)
            if episode_data is not None:
                dataset.append(episode_data)
                
                # Progress update
                if len(dataset) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = len(dataset) / elapsed
                    print(f"Progress: {len(dataset)}/{num_episodes} episodes "
                          f"({rate:.1f} episodes/sec)")
        
        # Save dataset if requested
        if save_path:
            self.save_dataset(dataset, save_path)
        
        # Print final statistics
        self.print_statistics()
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, Any]], save_path: str):
        """Save dataset to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save episode data
        np.savez_compressed(
            save_path.with_suffix('.npz'),
            episodes=[ep['observations'] for ep in dataset],
            actions=[ep['actions'] for ep in dataset],
            rewards=[ep['rewards'] for ep in dataset],
            dones=[ep['dones'] for ep in dataset],
            infos=[ep['infos'] for ep in dataset]
        )
        
        # Save metadata
        metadata = {
            'num_episodes': len(dataset),
            'quality_threshold': self.quality_threshold,
            'min_episode_length': self.min_episode_length,
            'max_episode_length': self.max_episode_length,
            'collection_stats': {
                'episode_count': self.episode_count,
                'valid_episodes': self.valid_episodes,
                'total_steps': self.total_steps,
                'avg_quality_score': float(np.mean(self.quality_scores)),
                'quality_std': float(np.std(self.quality_scores))
            }
        }
        
        with open(save_path.with_suffix('.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {save_path}")
    
    def print_statistics(self):
        """Print collection statistics."""
        print("\n" + "="*50)
        print("DATA COLLECTION STATISTICS")
        print("="*50)
        print(f"Total episodes attempted: {self.episode_count}")
        print(f"Valid episodes collected: {self.valid_episodes}")
        print(f"Success rate: {self.valid_episodes/self.episode_count*100:.1f}%")
        print(f"Total steps collected: {self.total_steps}")
        print(f"Average episode length: {self.total_steps/self.valid_episodes:.1f}")
        print(f"Average quality score: {np.mean(self.quality_scores):.2f} ± {np.std(self.quality_scores):.2f}")
        print("="*50)


def collect_fsm_dataset(num_episodes: int = 100,
                       quality_threshold: float = 2.0,
                       save_path: str = "fsm_dataset.npz",
                       use_gui: bool = False):
    """
    Convenience function to collect FSM dataset.
    
    Args:
        num_episodes: Number of episodes to collect
        quality_threshold: Minimum quality score to keep episode
        save_path: Path to save dataset
        use_gui: Whether to show GUI during collection
    """
    collector = DataCollector(
        quality_threshold=quality_threshold,
        min_episode_length=100,
        max_episode_length=2000
    )
    
    dataset = collector.collect_dataset(
        num_episodes=num_episodes,
        save_path=save_path,
        use_gui=use_gui,
        simend=10.0
    )
    
    return dataset


if __name__ == "__main__":
    # Example usage
    print("Collecting FSM dataset for imitation learning...")
    dataset = collect_fsm_dataset(
        num_episodes=50,
        quality_threshold=2.0,
        save_path="data/fsm_expert_trajectories.npz",
        use_gui=False
    )
    print(f"Collected {len(dataset)} high-quality episodes!")
