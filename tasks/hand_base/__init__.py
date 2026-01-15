"""
Task handling utilities and base classes.
"""

from .base_task import BaseTask
from .vec_task import VecTask, VecTaskPython
from .task_manager import TaskManager, TaskScheduler
from .task_factory import TaskFactory, TaskConfigValidator
from .task_monitor import TaskMonitor, PerformanceTracker
from .task_utils import (
    compute_episode_statistics,
    normalize_observations,
    update_running_stats,
    compute_success_metrics,
    compute_reward_statistics,
    clip_observations,
    clip_actions,
    compute_gae,
    compute_discounted_returns,
    batch_episode_data,
    split_episodes,
    compute_trajectory_statistics
)
from .task_evaluator import TaskEvaluator, ComparativeEvaluator, PerformanceBenchmark

__all__ = [
    # Base classes
    'BaseTask',
    'VecTask',
    'VecTaskPython',
    
    # Task management
    'TaskManager',
    'TaskScheduler',
    'TaskFactory',
    'TaskConfigValidator',
    
    # Monitoring
    'TaskMonitor',
    'PerformanceTracker',
    
    # Evaluation
    'TaskEvaluator',
    'ComparativeEvaluator',
    'PerformanceBenchmark',
    
    # Utilities
    'compute_episode_statistics',
    'normalize_observations',
    'update_running_stats',
    'compute_success_metrics',
    'compute_reward_statistics',
    'clip_observations',
    'clip_actions',
    'compute_gae',
    'compute_discounted_returns',
    'batch_episode_data',
    'split_episodes',
    'compute_trajectory_statistics',
]

