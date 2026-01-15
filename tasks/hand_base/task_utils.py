"""
Utility functions for task handling and processing.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict


def compute_episode_statistics(
    rewards: torch.Tensor,
    resets: torch.Tensor,
    episode_rewards: Optional[torch.Tensor] = None,
    episode_lengths: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Compute episode-level statistics.
    
    Args:
        rewards: Reward tensor
        resets: Reset buffer tensor
        episode_rewards: Optional pre-computed episode rewards
        episode_lengths: Optional pre-computed episode lengths
        
    Returns:
        Dictionary of episode statistics
    """
    reset_indices = resets.nonzero(as_tuple=False).squeeze(-1)
    
    stats = {
        'num_episodes': reset_indices.numel(),
        'total_steps': rewards.shape[0],
    }
    
    if reset_indices.numel() > 0:
        if episode_rewards is not None:
            completed_rewards = episode_rewards[reset_indices]
            stats['mean_episode_reward'] = completed_rewards.mean().item()
            stats['std_episode_reward'] = completed_rewards.std().item()
            stats['min_episode_reward'] = completed_rewards.min().item()
            stats['max_episode_reward'] = completed_rewards.max().item()
        
        if episode_lengths is not None:
            completed_lengths = episode_lengths[reset_indices]
            stats['mean_episode_length'] = completed_lengths.mean().item()
            stats['std_episode_length'] = completed_lengths.std().item()
            stats['min_episode_length'] = completed_lengths.min().item()
            stats['max_episode_length'] = completed_lengths.max().item()
    
    return stats


def normalize_observations(obs: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize observations using running statistics.
    
    Args:
        obs: Observation tensor
        running_mean: Running mean tensor
        running_var: Running variance tensor
        eps: Small epsilon for numerical stability
        
    Returns:
        Normalized observations
    """
    return (obs - running_mean) / torch.sqrt(running_var + eps)


def update_running_stats(
    obs: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    count: int,
    momentum: float = 0.99
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Update running statistics for observation normalization.
    
    Args:
        obs: New observations
        running_mean: Current running mean
        running_var: Current running variance
        count: Current count of observations
        momentum: Momentum for exponential moving average
        
    Returns:
        Updated (running_mean, running_var, count)
    """
    batch_mean = obs.mean(dim=0)
    batch_var = obs.var(dim=0, unbiased=False)
    batch_count = obs.shape[0]
    
    # Update running statistics
    new_count = count + batch_count
    delta = batch_mean - running_mean
    new_mean = running_mean + delta * (batch_count / new_count)
    
    # Update variance using Welford's algorithm
    m_a = running_var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + delta ** 2 * count * batch_count / new_count
    new_var = M2 / new_count
    
    # Apply momentum
    running_mean = momentum * running_mean + (1 - momentum) * new_mean
    running_var = momentum * running_var + (1 - momentum) * new_var
    
    return running_mean, running_var, new_count


def compute_success_metrics(
    successes: torch.Tensor,
    resets: torch.Tensor,
    window_size: int = 100
) -> Dict[str, float]:
    """
    Compute success-related metrics.
    
    Args:
        successes: Success tensor (1 for success, 0 for failure)
        resets: Reset buffer tensor
        window_size: Window size for rolling statistics
        
    Returns:
        Dictionary of success metrics
    """
    reset_indices = resets.nonzero(as_tuple=False).squeeze(-1)
    
    metrics = {
        'current_success_rate': successes.float().mean().item(),
        'total_successes': successes.sum().item(),
        'total_resets': reset_indices.numel(),
    }
    
    if reset_indices.numel() > 0:
        completed_successes = successes[reset_indices]
        metrics['episode_success_rate'] = completed_successes.float().mean().item()
        metrics['successful_episodes'] = completed_successes.sum().item()
    
    # Rolling window statistics
    if len(successes) >= window_size:
        recent_successes = successes[-window_size:]
        metrics['recent_success_rate'] = recent_successes.float().mean().item()
    
    return metrics


def compute_reward_statistics(
    rewards: torch.Tensor,
    window_size: int = 100
) -> Dict[str, float]:
    """
    Compute reward statistics.
    
    Args:
        rewards: Reward tensor
        window_size: Window size for rolling statistics
        
    Returns:
        Dictionary of reward statistics
    """
    stats = {
        'mean': rewards.mean().item(),
        'std': rewards.std().item(),
        'min': rewards.min().item(),
        'max': rewards.max().item(),
        'sum': rewards.sum().item(),
    }
    
    # Rolling window statistics
    if len(rewards) >= window_size:
        recent_rewards = rewards[-window_size:]
        stats['recent_mean'] = recent_rewards.mean().item()
        stats['recent_std'] = recent_rewards.std().item()
    
    return stats


def clip_observations(obs: torch.Tensor, clip_value: float = 5.0) -> torch.Tensor:
    """
    Clip observations to a specified range.
    
    Args:
        obs: Observation tensor
        clip_value: Maximum absolute value
        
    Returns:
        Clipped observations
    """
    return torch.clamp(obs, -clip_value, clip_value)


def clip_actions(actions: torch.Tensor, clip_value: float = 1.0) -> torch.Tensor:
    """
    Clip actions to a specified range.
    
    Args:
        actions: Action tensor
        clip_value: Maximum absolute value
        
    Returns:
        Clipped actions
    """
    return torch.clamp(actions, -clip_value, clip_value)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Reward tensor [T, N]
        values: Value estimates [T, N]
        dones: Done flags [T, N]
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    # Compute advantages backwards
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    
    # Compute returns
    returns = advantages + values
    
    return advantages, returns


def compute_discounted_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    last_value: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute discounted returns.
    
    Args:
        rewards: Reward tensor [T, N]
        dones: Done flags [T, N]
        gamma: Discount factor
        last_value: Optional value estimate for last state
        
    Returns:
        Discounted returns tensor
    """
    T, N = rewards.shape
    returns = torch.zeros_like(rewards)
    
    if last_value is not None:
        returns[-1] = rewards[-1] + gamma * last_value * (1 - dones[-1])
    else:
        returns[-1] = rewards[-1]
    
    for t in reversed(range(T - 1)):
        returns[t] = rewards[t] + gamma * returns[t + 1] * (1 - dones[t])
    
    return returns


def batch_episode_data(
    observations: List[torch.Tensor],
    actions: List[torch.Tensor],
    rewards: List[torch.Tensor],
    dones: List[torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Batch episode data into tensors.
    
    Args:
        observations: List of observation tensors
        actions: List of action tensors
        rewards: List of reward tensors
        dones: List of done tensors
        
    Returns:
        Dictionary of batched tensors
    """
    return {
        'observations': torch.stack(observations),
        'actions': torch.stack(actions),
        'rewards': torch.stack(rewards),
        'dones': torch.stack(dones),
    }


def split_episodes(
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor
) -> List[Dict[str, torch.Tensor]]:
    """
    Split batched data into individual episodes.
    
    Args:
        observations: Observation tensor [T, N, ...]
        actions: Action tensor [T, N, ...]
        rewards: Reward tensor [T, N]
        dones: Done tensor [T, N]
        
    Returns:
        List of episode dictionaries
    """
    T, N = rewards.shape
    episodes = []
    
    for n in range(N):
        episode_dones = dones[:, n].nonzero(as_tuple=False).squeeze(-1)
        
        if len(episode_dones) == 0:
            # Single episode
            episodes.append({
                'observations': observations[:, n],
                'actions': actions[:, n],
                'rewards': rewards[:, n],
                'dones': dones[:, n],
            })
        else:
            # Multiple episodes
            start_idx = 0
            for end_idx in episode_dones:
                episodes.append({
                    'observations': observations[start_idx:end_idx+1, n],
                    'actions': actions[start_idx:end_idx+1, n],
                    'rewards': rewards[start_idx:end_idx+1, n],
                    'dones': dones[start_idx:end_idx+1, n],
                })
                start_idx = end_idx + 1
            
            # Last episode if not ended
            if start_idx < T:
                episodes.append({
                    'observations': observations[start_idx:, n],
                    'actions': actions[start_idx:, n],
                    'rewards': rewards[start_idx:, n],
                    'dones': dones[start_idx:, n],
                })
    
    return episodes


def compute_trajectory_statistics(trajectory: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Compute statistics for a single trajectory.
    
    Args:
        trajectory: Dictionary containing 'observations', 'actions', 'rewards', 'dones'
        
    Returns:
        Dictionary of trajectory statistics
    """
    rewards = trajectory['rewards']
    dones = trajectory['dones']
    
    stats = {
        'length': len(rewards),
        'total_reward': rewards.sum().item(),
        'mean_reward': rewards.mean().item(),
        'final_reward': rewards[-1].item() if len(rewards) > 0 else 0.0,
        'success': dones[-1].item() if len(dones) > 0 and dones.sum() > 0 else 0.0,
    }
    
    # Compute discounted return
    if len(rewards) > 0:
        discounted_return = 0.0
        gamma = 0.99
        for i, r in enumerate(rewards):
            discounted_return += (gamma ** i) * r.item()
        stats['discounted_return'] = discounted_return
    
    return stats

