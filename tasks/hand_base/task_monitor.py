"""
Task Monitor for tracking and logging task performance.
"""

import torch
import time
from typing import Dict, List, Optional, Any
from collections import deque
import numpy as np


class TaskMonitor:
    """Monitors task performance and logs metrics."""
    
    def __init__(
        self,
        window_size: int = 100,
        log_interval: int = 100,
        track_metrics: Optional[List[str]] = None
    ):
        """
        Initialize task monitor.
        
        Args:
            window_size: Size of sliding window for metrics
            log_interval: Interval for logging (in steps)
            track_metrics: List of metric names to track
        """
        self.window_size = window_size
        self.log_interval = log_interval
        self.step_count = 0
        
        # Default metrics to track
        if track_metrics is None:
            track_metrics = [
                'reward', 'episode_length', 'success_rate',
                'value_loss', 'policy_loss', 'entropy'
            ]
        
        self.track_metrics = track_metrics
        
        # Metric buffers (sliding windows)
        self.metric_buffers = {
            metric: deque(maxlen=window_size)
            for metric in track_metrics
        }
        
        # Episode tracking
        self.episode_rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.episode_successes = deque(maxlen=window_size)
        
        # Timing
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        # Performance statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.total_reward = 0.0
    
    def update(
        self,
        rewards: torch.Tensor,
        resets: torch.Tensor,
        extras: Optional[Dict[str, Any]] = None,
        losses: Optional[Dict[str, float]] = None
    ):
        """
        Update monitor with new data.
        
        Args:
            rewards: Reward tensor
            resets: Reset buffer tensor
            extras: Optional extra information dictionary
            losses: Optional loss dictionary
        """
        self.step_count += 1
        self.total_steps += rewards.shape[0]
        self.total_reward += rewards.sum().item()
        
        # Update reward buffer
        self.metric_buffers['reward'].append(rewards.mean().item())
        
        # Handle episode completions
        reset_indices = resets.nonzero(as_tuple=False).squeeze(-1)
        if reset_indices.numel() > 0:
            self.total_episodes += reset_indices.numel()
            
            # Track episode rewards
            if 'episode_rewards' in (extras or {}):
                episode_rewards = extras['episode_rewards'][reset_indices]
                self.episode_rewards.extend(episode_rewards.cpu().numpy().tolist())
            
            # Track episode lengths
            if 'episode_lengths' in (extras or {}):
                episode_lengths = extras['episode_lengths'][reset_indices]
                self.episode_lengths.extend(episode_lengths.cpu().numpy().tolist())
                self.metric_buffers['episode_length'].extend(
                    episode_lengths.cpu().numpy().tolist()
                )
            
            # Track success rate
            if 'successes' in (extras or {}):
                successes = extras['successes'][reset_indices]
                success_rate = successes.float().mean().item()
                self.episode_successes.append(success_rate)
                if 'success_rate' in self.metric_buffers:
                    self.metric_buffers['success_rate'].append(success_rate)
        
        # Update loss metrics
        if losses is not None:
            for loss_name, loss_value in losses.items():
                if loss_name in self.metric_buffers:
                    self.metric_buffers[loss_name].append(loss_value)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current metric statistics.
        
        Returns:
            Dictionary of metric statistics
        """
        metrics = {}
        
        for metric_name, buffer in self.metric_buffers.items():
            if len(buffer) > 0:
                metrics[f'{metric_name}_mean'] = np.mean(buffer)
                metrics[f'{metric_name}_std'] = np.std(buffer)
                metrics[f'{metric_name}_min'] = np.min(buffer)
                metrics[f'{metric_name}_max'] = np.max(buffer)
            else:
                metrics[f'{metric_name}_mean'] = 0.0
                metrics[f'{metric_name}_std'] = 0.0
                metrics[f'{metric_name}_min'] = 0.0
                metrics[f'{metric_name}_max'] = 0.0
        
        # Episode statistics
        if len(self.episode_rewards) > 0:
            metrics['episode_reward_mean'] = np.mean(self.episode_rewards)
            metrics['episode_reward_std'] = np.std(self.episode_rewards)
        
        if len(self.episode_lengths) > 0:
            metrics['episode_length_mean'] = np.mean(self.episode_lengths)
            metrics['episode_length_std'] = np.std(self.episode_lengths)
        
        if len(self.episode_successes) > 0:
            metrics['success_rate'] = np.mean(self.episode_successes)
        
        # Performance metrics
        elapsed_time = time.time() - self.start_time
        metrics['steps_per_second'] = self.total_steps / elapsed_time if elapsed_time > 0 else 0.0
        metrics['episodes_per_second'] = self.total_episodes / elapsed_time if elapsed_time > 0 else 0.0
        metrics['total_steps'] = self.total_steps
        metrics['total_episodes'] = self.total_episodes
        metrics['total_reward'] = self.total_reward
        
        return metrics
    
    def should_log(self) -> bool:
        """Check if it's time to log metrics."""
        return self.step_count % self.log_interval == 0
    
    def log_metrics(self, prefix: str = ""):
        """
        Log current metrics.
        
        Args:
            prefix: Optional prefix for log messages
        """
        if not self.should_log():
            return
        
        metrics = self.get_metrics()
        elapsed_time = time.time() - self.last_log_time
        
        print(f"\n{'=' * 60}")
        if prefix:
            print(f"{prefix} - Step {self.step_count}")
        else:
            print(f"Step {self.step_count}")
        print(f"{'=' * 60}")
        
        print(f"Performance:")
        print(f"  Steps/sec: {metrics['steps_per_second']:.2f}")
        print(f"  Episodes/sec: {metrics['episodes_per_second']:.2f}")
        print(f"  Total steps: {metrics['total_steps']:,}")
        print(f"  Total episodes: {metrics['total_episodes']:,}")
        
        if 'reward_mean' in metrics:
            print(f"\nRewards:")
            print(f"  Mean: {metrics['reward_mean']:.4f} ± {metrics['reward_std']:.4f}")
            print(f"  Range: [{metrics['reward_min']:.4f}, {metrics['reward_max']:.4f}]")
        
        if 'episode_reward_mean' in metrics:
            print(f"  Episode mean: {metrics['episode_reward_mean']:.4f} ± {metrics['episode_reward_std']:.4f}")
        
        if 'episode_length_mean' in metrics:
            print(f"\nEpisode Length:")
            print(f"  Mean: {metrics['episode_length_mean']:.2f} ± {metrics['episode_length_std']:.2f}")
        
        if 'success_rate' in metrics:
            print(f"\nSuccess Rate: {metrics['success_rate']:.4f}")
        
        if 'value_loss_mean' in metrics:
            print(f"\nLosses:")
            print(f"  Value loss: {metrics['value_loss_mean']:.6f}")
        
        if 'policy_loss_mean' in metrics:
            print(f"  Policy loss: {metrics['policy_loss_mean']:.6f}")
        
        if 'entropy_mean' in metrics:
            print(f"  Entropy: {metrics['entropy_mean']:.6f}")
        
        print(f"{'=' * 60}\n")
        
        self.last_log_time = time.time()
    
    def reset(self):
        """Reset monitor statistics."""
        self.step_count = 0
        self.total_steps = 0
        self.total_episodes = 0
        self.total_reward = 0.0
        self.start_time = time.time()
        self.last_log_time = time.time()
        
        for buffer in self.metric_buffers.values():
            buffer.clear()
        
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_successes.clear()


class PerformanceTracker:
    """Tracks detailed performance metrics over time."""
    
    def __init__(self, max_history: int = 10000):
        """
        Initialize performance tracker.
        
        Args:
            max_history: Maximum number of data points to keep
        """
        self.max_history = max_history
        self.history = {
            'step': deque(maxlen=max_history),
            'reward': deque(maxlen=max_history),
            'episode_reward': deque(maxlen=max_history),
            'episode_length': deque(maxlen=max_history),
            'success': deque(maxlen=max_history),
        }
    
    def record(
        self,
        step: int,
        reward: float,
        episode_reward: Optional[float] = None,
        episode_length: Optional[int] = None,
        success: Optional[bool] = None
    ):
        """Record a performance data point."""
        self.history['step'].append(step)
        self.history['reward'].append(reward)
        
        if episode_reward is not None:
            self.history['episode_reward'].append(episode_reward)
        if episode_length is not None:
            self.history['episode_length'].append(episode_length)
        if success is not None:
            self.history['success'].append(1.0 if success else 0.0)
    
    def get_recent_performance(self, window: int = 100) -> Dict[str, float]:
        """
        Get performance statistics for recent window.
        
        Args:
            window: Number of recent data points to analyze
            
        Returns:
            Dictionary of performance statistics
        """
        if len(self.history['reward']) == 0:
            return {}
        
        recent_rewards = list(self.history['reward'])[-window:]
        recent_episode_rewards = list(self.history['episode_reward'])[-window:]
        recent_episode_lengths = list(self.history['episode_length'])[-window:]
        recent_successes = list(self.history['success'])[-window:]
        
        stats = {
            'mean_reward': np.mean(recent_rewards) if recent_rewards else 0.0,
            'std_reward': np.std(recent_rewards) if recent_rewards else 0.0,
        }
        
        if recent_episode_rewards:
            stats['mean_episode_reward'] = np.mean(recent_episode_rewards)
            stats['std_episode_reward'] = np.std(recent_episode_rewards)
        
        if recent_episode_lengths:
            stats['mean_episode_length'] = np.mean(recent_episode_lengths)
            stats['std_episode_length'] = np.std(recent_episode_lengths)
        
        if recent_successes:
            stats['success_rate'] = np.mean(recent_successes)
        
        return stats
    
    def clear(self):
        """Clear all history."""
        for buffer in self.history.values():
            buffer.clear()

