"""
Example usage of task handling tools.
Demonstrates TaskManager, TaskMonitor, TaskEvaluator, etc.
"""

import torch
from task_manager import TaskManager, TaskScheduler
from task_monitor import TaskMonitor, PerformanceTracker
from task_evaluator import TaskEvaluator, ComparativeEvaluator
from task_utils import (
    compute_episode_statistics,
    compute_success_metrics,
    compute_reward_statistics
)


def example_task_manager():
    """Example of using TaskManager."""
    print("=" * 60)
    print("Task Manager Example")
    print("=" * 60)
    
    # Assume we have multiple tasks
    # tasks = {
    #     'task1': task1_instance,
    #     'task2': task2_instance,
    # }
    # manager = TaskManager(tasks, default_task='task1')
    
    # Switch between tasks
    # manager.switch_task('task2')
    # current_task = manager.get_current_task()
    
    # Update statistics
    # rewards = torch.randn(1000)
    # resets = torch.zeros(1000)
    # manager.update_stats(rewards, resets)
    
    # Get statistics
    # stats = manager.get_stats()
    # manager.print_stats()
    
    print("TaskManager usage demonstrated (commented out - requires actual task instances)")


def example_task_monitor():
    """Example of using TaskMonitor."""
    print("\n" + "=" * 60)
    print("Task Monitor Example")
    print("=" * 60)
    
    # Create monitor
    monitor = TaskMonitor(window_size=100, log_interval=50)
    
    # Simulate updates
    for step in range(200):
        rewards = torch.randn(1000) * 0.1
        resets = torch.zeros(1000)
        resets[torch.randint(0, 1000, (10,))] = 1  # Random resets
        
        extras = {
            'successes': torch.rand(1000) > 0.7,  # 30% success rate
        }
        
        losses = {
            'value_loss': 0.01,
            'policy_loss': 0.02,
            'entropy': 0.5,
        }
        
        monitor.update(rewards, resets, extras, losses)
        
        # Log periodically
        if monitor.should_log():
            monitor.log_metrics(prefix=f"Step {step}")
    
    # Get final metrics
    final_metrics = monitor.get_metrics()
    print(f"\nFinal Metrics:")
    for key, value in final_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")


def example_task_evaluator():
    """Example of using TaskEvaluator."""
    print("\n" + "=" * 60)
    print("Task Evaluator Example")
    print("=" * 60)
    
    # This would require actual task and policy instances
    # evaluator = TaskEvaluator(task, num_eval_episodes=10)
    # results = evaluator.evaluate(policy, deterministic=True)
    # report = evaluator.generate_report(results)
    # print(report)
    
    print("TaskEvaluator usage demonstrated (commented out - requires actual task and policy)")


def example_task_utils():
    """Example of using task utility functions."""
    print("\n" + "=" * 60)
    print("Task Utils Example")
    print("=" * 60)
    
    # Simulate episode data
    num_envs = 1000
    num_steps = 100
    
    rewards = torch.randn(num_steps, num_envs) * 0.1
    resets = torch.zeros(num_steps, num_envs)
    resets[torch.randint(0, num_steps, (10,)), torch.randint(0, num_envs, (10,))] = 1
    
    # Compute episode statistics
    episode_stats = compute_episode_statistics(
        rewards.flatten(),
        resets.flatten()
    )
    print("Episode Statistics:")
    for key, value in episode_stats.items():
        print(f"  {key}: {value}")
    
    # Compute reward statistics
    reward_stats = compute_reward_statistics(rewards.flatten())
    print("\nReward Statistics:")
    for key, value in reward_stats.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
    
    # Compute success metrics
    successes = torch.rand(num_envs) > 0.7
    success_metrics = compute_success_metrics(
        successes,
        resets[-1]  # Last step resets
    )
    print("\nSuccess Metrics:")
    for key, value in success_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")


def example_performance_tracker():
    """Example of using PerformanceTracker."""
    print("\n" + "=" * 60)
    print("Performance Tracker Example")
    print("=" * 60)
    
    tracker = PerformanceTracker(max_history=1000)
    
    # Simulate recording performance
    for step in range(100):
        reward = torch.randn(1).item() * 0.1
        episode_reward = reward * 10
        episode_length = 50
        success = torch.rand(1).item() > 0.7
        
        tracker.record(
            step=step,
            reward=reward,
            episode_reward=episode_reward,
            episode_length=episode_length,
            success=success
        )
    
    # Get recent performance
    recent_perf = tracker.get_recent_performance(window=50)
    print("Recent Performance (last 50 steps):")
    for key, value in recent_perf.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    example_task_manager()
    example_task_monitor()
    example_task_evaluator()
    example_task_utils()
    example_performance_tracker()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

