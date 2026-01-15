"""
Task Evaluator for evaluating task performance and generating reports.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import time


class TaskEvaluator:
    """Evaluates task performance and generates evaluation reports."""
    
    def __init__(self, task: Any, num_eval_episodes: int = 10):
        """
        Initialize task evaluator.
        
        Args:
            task: Task instance to evaluate
            num_eval_episodes: Number of episodes to run for evaluation
        """
        self.task = task
        self.num_eval_episodes = num_eval_episodes
        self.device = task.device
    
    def evaluate(
        self,
        policy: Any,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate policy on the task.
        
        Args:
            policy: Policy network (must have act_inference or act method)
            deterministic: Whether to use deterministic actions
            render: Whether to render during evaluation
            
        Returns:
            Dictionary of evaluation results
        """
        policy.eval()
        
        episode_rewards = []
        episode_lengths = []
        episode_successes = []
        episode_data = []
        
        # Reset environment
        obs = self.task.reset()
        
        for episode in range(self.num_eval_episodes):
            episode_reward = 0.0
            episode_length = 0
            episode_obs = []
            episode_actions = []
            episode_rewards_list = []
            
            done = False
            step = 0
            
            while not done and step < self.task.max_episode_length:
                # Get action from policy
                with torch.no_grad():
                    if hasattr(policy, 'act_inference'):
                        actions, _ = policy.act_inference(obs, act_value=False)
                    elif hasattr(policy, 'act'):
                        actions, _, _, _, _ = policy.act(obs, deterministic=deterministic)
                    else:
                        raise ValueError("Policy must have 'act_inference' or 'act' method")
                
                # Step environment
                next_obs, rewards, dones, infos = self.task.step(actions, step)
                
                # Store data
                episode_obs.append(obs.cpu())
                episode_actions.append(actions.cpu())
                episode_rewards_list.append(rewards.cpu())
                
                episode_reward += rewards.mean().item()
                episode_length += 1
                step += 1
                
                # Check if done
                if dones.any():
                    done = True
                    if 'successes' in infos:
                        success = infos['successes'].float().mean().item()
                        episode_successes.append(success)
                
                obs = next_obs
                
                if render and hasattr(self.task, 'render'):
                    self.task.render()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            episode_data.append({
                'observations': torch.stack(episode_obs) if episode_obs else None,
                'actions': torch.stack(episode_actions) if episode_actions else None,
                'rewards': torch.stack(episode_rewards_list) if episode_rewards_list else None,
            })
            
            # Reset for next episode
            obs = self.task.reset()
        
        policy.train()
        
        # Compute statistics
        results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
        }
        
        if episode_successes:
            results['success_rate'] = np.mean(episode_successes)
            results['episode_successes'] = episode_successes
        
        results['episode_data'] = episode_data
        
        return results
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a text report from evaluation results.
        
        Args:
            results: Evaluation results dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("Task Evaluation Report")
        report.append("=" * 60)
        report.append(f"Number of episodes: {self.num_eval_episodes}")
        report.append("")
        
        report.append("Reward Statistics:")
        report.append(f"  Mean: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
        report.append(f"  Range: [{results['min_reward']:.4f}, {results['max_reward']:.4f}]")
        report.append("")
        
        report.append("Episode Length Statistics:")
        report.append(f"  Mean: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
        report.append("")
        
        if 'success_rate' in results:
            report.append(f"Success Rate: {results['success_rate']:.4f}")
            report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)


class ComparativeEvaluator:
    """Compares performance of multiple policies on the same task."""
    
    def __init__(self, task: Any, num_eval_episodes: int = 10):
        """
        Initialize comparative evaluator.
        
        Args:
            task: Task instance
            num_eval_episodes: Number of episodes per policy
        """
        self.task = task
        self.num_eval_episodes = num_eval_episodes
        self.evaluator = TaskEvaluator(task, num_eval_episodes)
    
    def compare_policies(
        self,
        policies: Dict[str, Any],
        deterministic: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple policies.
        
        Args:
            policies: Dictionary mapping policy names to policy instances
            deterministic: Whether to use deterministic actions
            
        Returns:
            Dictionary mapping policy names to evaluation results
        """
        results = {}
        
        for policy_name, policy in policies.items():
            print(f"Evaluating policy: {policy_name}")
            results[policy_name] = self.evaluator.evaluate(
                policy,
                deterministic=deterministic,
                render=False
            )
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a comparison report.
        
        Args:
            results: Dictionary of evaluation results for each policy
            
        Returns:
            Formatted comparison report
        """
        report = []
        report.append("=" * 60)
        report.append("Policy Comparison Report")
        report.append("=" * 60)
        report.append("")
        
        # Create comparison table
        report.append(f"{'Policy':<20} {'Mean Reward':<15} {'Std Reward':<15} {'Success Rate':<15}")
        report.append("-" * 60)
        
        for policy_name, result in results.items():
            mean_reward = result['mean_reward']
            std_reward = result['std_reward']
            success_rate = result.get('success_rate', 0.0)
            
            report.append(
                f"{policy_name:<20} {mean_reward:>10.4f} ± {std_reward:>8.4f}   {success_rate:>10.4f}"
            )
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


class PerformanceBenchmark:
    """Benchmarks task performance and tracks improvements over time."""
    
    def __init__(self, task: Any):
        """
        Initialize performance benchmark.
        
        Args:
            task: Task instance
        """
        self.task = task
        self.benchmarks = defaultdict(list)
        self.best_performance = {}
    
    def record_performance(
        self,
        iteration: int,
        metrics: Dict[str, float]
    ):
        """
        Record performance metrics.
        
        Args:
            iteration: Training iteration
            metrics: Dictionary of metric names and values
        """
        for metric_name, value in metrics.items():
            self.benchmarks[metric_name].append({
                'iteration': iteration,
                'value': value,
                'timestamp': time.time()
            })
            
            # Update best performance
            if metric_name not in self.best_performance:
                self.best_performance[metric_name] = {
                    'value': value,
                    'iteration': iteration
                }
            else:
                # Assume higher is better (can be customized)
                if value > self.best_performance[metric_name]['value']:
                    self.best_performance[metric_name] = {
                        'value': value,
                        'iteration': iteration
                    }
    
    def get_best_performance(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get best performance for a metric."""
        return self.best_performance.get(metric_name)
    
    def get_performance_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """Get performance history for a metric."""
        return self.benchmarks.get(metric_name, [])
    
    def get_improvement_rate(self, metric_name: str, window: int = 100) -> float:
        """
        Calculate improvement rate over recent window.
        
        Args:
            metric_name: Name of the metric
            window: Number of recent points to consider
            
        Returns:
            Improvement rate (positive means improving)
        """
        history = self.benchmarks.get(metric_name, [])
        if len(history) < 2:
            return 0.0
        
        recent = history[-window:]
        if len(recent) < 2:
            return 0.0
        
        values = [h['value'] for h in recent]
        improvement = (values[-1] - values[0]) / len(values)
        
        return improvement

