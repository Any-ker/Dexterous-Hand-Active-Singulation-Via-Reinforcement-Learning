"""
Task Manager for handling multiple tasks and task switching.
"""

import torch
from typing import Dict, List, Optional, Any
from collections import defaultdict
import time


class TaskManager:
    """Manages multiple tasks and provides task switching capabilities."""
    
    def __init__(self, tasks: Dict[str, Any], default_task: str = None):
        """
        Initialize task manager.
        
        Args:
            tasks: Dictionary mapping task names to task instances
            default_task: Name of the default task to use
        """
        self.tasks = tasks
        self.task_names = list(tasks.keys())
        
        if default_task is None:
            default_task = self.task_names[0] if self.task_names else None
        
        if default_task not in self.tasks:
            raise ValueError(f"Default task '{default_task}' not found in tasks")
        
        self.current_task_name = default_task
        self.current_task = self.tasks[self.current_task_name]
        
        # Task statistics
        self.task_stats = defaultdict(lambda: {
            'switches': 0,
            'steps': 0,
            'resets': 0,
            'total_reward': 0.0,
            'episode_rewards': [],
            'episode_lengths': []
        })
        
        # Task switching history
        self.switch_history = []
    
    def switch_task(self, task_name: str) -> bool:
        """
        Switch to a different task.
        
        Args:
            task_name: Name of the task to switch to
            
        Returns:
            True if switch was successful, False otherwise
        """
        if task_name not in self.tasks:
            print(f"Warning: Task '{task_name}' not found. Available tasks: {self.task_names}")
            return False
        
        if task_name == self.current_task_name:
            return True
        
        # Record switch
        self.switch_history.append({
            'from': self.current_task_name,
            'to': task_name,
            'time': time.time()
        })
        
        # Update statistics
        self.task_stats[self.current_task_name]['switches'] += 1
        
        # Switch task
        self.current_task_name = task_name
        self.current_task = self.tasks[task_name]
        
        return True
    
    def get_current_task(self) -> Any:
        """Get the current active task."""
        return self.current_task
    
    def get_task(self, task_name: str) -> Optional[Any]:
        """Get a specific task by name."""
        return self.tasks.get(task_name)
    
    def update_stats(self, rewards: torch.Tensor, resets: torch.Tensor, episode_lengths: torch.Tensor = None):
        """
        Update task statistics.
        
        Args:
            rewards: Reward tensor
            resets: Reset buffer tensor
            episode_lengths: Optional episode length tensor
        """
        stats = self.task_stats[self.current_task_name]
        stats['steps'] += rewards.shape[0]
        stats['total_reward'] += rewards.sum().item()
        
        reset_indices = resets.nonzero(as_tuple=False).squeeze(-1)
        if reset_indices.numel() > 0:
            stats['resets'] += reset_indices.numel()
            
            if episode_lengths is not None:
                episode_lens = episode_lengths[reset_indices].cpu().numpy().tolist()
                stats['episode_lengths'].extend(episode_lens)
            
            # Calculate episode rewards (sum of rewards until reset)
            # This is a simplified version - in practice, you'd track per-episode rewards
            if len(reset_indices) > 0:
                # For simplicity, use mean reward as episode reward
                mean_reward = rewards[reset_indices].mean().item()
                stats['episode_rewards'].append(mean_reward)
    
    def get_stats(self, task_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for a task.
        
        Args:
            task_name: Name of the task. If None, returns current task stats.
            
        Returns:
            Dictionary of task statistics
        """
        if task_name is None:
            task_name = self.current_task_name
        
        stats = self.task_stats[task_name].copy()
        
        # Calculate additional metrics
        if len(stats['episode_rewards']) > 0:
            stats['mean_episode_reward'] = sum(stats['episode_rewards']) / len(stats['episode_rewards'])
            stats['max_episode_reward'] = max(stats['episode_rewards'])
            stats['min_episode_reward'] = min(stats['episode_rewards'])
        else:
            stats['mean_episode_reward'] = 0.0
            stats['max_episode_reward'] = 0.0
            stats['min_episode_reward'] = 0.0
        
        if len(stats['episode_lengths']) > 0:
            stats['mean_episode_length'] = sum(stats['episode_lengths']) / len(stats['episode_lengths'])
            stats['max_episode_length'] = max(stats['episode_lengths'])
            stats['min_episode_length'] = min(stats['episode_lengths'])
        else:
            stats['mean_episode_length'] = 0.0
            stats['max_episode_length'] = 0.0
            stats['min_episode_length'] = 0.0
        
        if stats['steps'] > 0:
            stats['mean_reward_per_step'] = stats['total_reward'] / stats['steps']
        else:
            stats['mean_reward_per_step'] = 0.0
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tasks."""
        return {name: self.get_stats(name) for name in self.task_names}
    
    def reset_stats(self, task_name: Optional[str] = None):
        """Reset statistics for a task or all tasks."""
        if task_name is None:
            self.task_stats.clear()
        elif task_name in self.task_stats:
            self.task_stats[task_name] = {
                'switches': 0,
                'steps': 0,
                'resets': 0,
                'total_reward': 0.0,
                'episode_rewards': [],
                'episode_lengths': []
            }
    
    def print_stats(self, task_name: Optional[str] = None):
        """Print statistics for a task."""
        stats = self.get_stats(task_name)
        name = task_name if task_name else self.current_task_name
        
        print(f"\n=== Task Statistics: {name} ===")
        print(f"  Steps: {stats['steps']:,}")
        print(f"  Resets: {stats['resets']:,}")
        print(f"  Total Reward: {stats['total_reward']:.2f}")
        print(f"  Mean Reward/Step: {stats['mean_reward_per_step']:.4f}")
        print(f"  Mean Episode Reward: {stats['mean_episode_reward']:.4f}")
        print(f"  Mean Episode Length: {stats['mean_episode_length']:.2f}")
        print(f"  Task Switches: {stats['switches']}")


class TaskScheduler:
    """Schedules task switching based on various criteria."""
    
    def __init__(self, task_manager: TaskManager, schedule_type: str = "round_robin"):
        """
        Initialize task scheduler.
        
        Args:
            task_manager: TaskManager instance
            schedule_type: Type of scheduling ("round_robin", "random", "performance_based")
        """
        self.task_manager = task_manager
        self.schedule_type = schedule_type
        self.current_index = 0
        self.switch_counter = 0
        self.switch_interval = 1000  # Switch every N steps
    
    def should_switch(self, step: int) -> bool:
        """
        Determine if task should be switched.
        
        Args:
            step: Current training step
            
        Returns:
            True if task should be switched
        """
        if step % self.switch_interval == 0 and step > 0:
            return True
        return False
    
    def get_next_task(self) -> str:
        """Get the next task name based on scheduling strategy."""
        if self.schedule_type == "round_robin":
            self.current_index = (self.current_index + 1) % len(self.task_manager.task_names)
            return self.task_manager.task_names[self.current_index]
        
        elif self.schedule_type == "random":
            import random
            available_tasks = [t for t in self.task_manager.task_names 
                             if t != self.task_manager.current_task_name]
            if available_tasks:
                return random.choice(available_tasks)
            return self.task_manager.current_task_name
        
        elif self.schedule_type == "performance_based":
            # Switch to task with lowest performance
            all_stats = self.task_manager.get_all_stats()
            task_performances = {
                name: stats.get('mean_episode_reward', 0.0)
                for name, stats in all_stats.items()
            }
            
            # Find task with lowest performance
            min_performance = min(task_performances.values())
            worst_tasks = [name for name, perf in task_performances.items() 
                          if perf == min_performance]
            
            if worst_tasks:
                import random
                return random.choice(worst_tasks)
            return self.task_manager.current_task_name
        
        else:
            return self.task_manager.current_task_name
    
    def schedule(self, step: int) -> bool:
        """
        Schedule task switching.
        
        Args:
            step: Current training step
            
        Returns:
            True if task was switched
        """
        if self.should_switch(step):
            next_task = self.get_next_task()
            if next_task != self.task_manager.current_task_name:
                return self.task_manager.switch_task(next_task)
        return False
    
    def set_switch_interval(self, interval: int):
        """Set the interval for task switching."""
        self.switch_interval = interval

