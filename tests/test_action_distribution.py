#!/usr/bin/env python3
"""
Action distribution test
- Tests action sampling from different networks
- Verifies distribution properties
- Checks action bounds and statistics
- Logs distribution metrics
"""

import os
import sys
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from algorithms.rl.ppo.module import ActorCritic as OriginalActorCritic
from algorithms.rl.ppo.new_network import ModernActorCritic
from algorithms.rl.ppo.advanced_network import AdvancedActorCritic


def test_action_distribution(net, name, observations, states, num_samples=10000):
    """Test action distribution properties."""
    device = observations.device
    net.eval()
    
    all_actions = []
    all_log_probs = []
    all_entropies = []
    all_values = []
    
    with torch.no_grad():
        for _ in range(num_samples // observations.shape[0] + 1):
            actions, log_probs, values, means, stds = net.act(
                observations, states, deterministic=False
            )
            
            log_probs_eval, entropy, values_eval, _, _ = net.evaluate(
                observations, None, actions
            )
            
            all_actions.append(actions.cpu())
            all_log_probs.append(log_probs_eval.cpu())
            all_entropies.append(entropy.cpu())
            all_values.append(values_eval.cpu())
    
    all_actions = torch.cat(all_actions, dim=0)[:num_samples]
    all_log_probs = torch.cat(all_log_probs, dim=0)[:num_samples]
    all_entropies = torch.cat(all_entropies, dim=0)[:num_samples]
    all_values = torch.cat(all_values, dim=0)[:num_samples]
    
    # Compute statistics
    action_stats = {
        'mean': all_actions.mean(dim=0).numpy().tolist(),
        'std': all_actions.std(dim=0).numpy().tolist(),
        'min': all_actions.min(dim=0)[0].numpy().tolist(),
        'max': all_actions.max(dim=0)[0].numpy().tolist(),
        'overall_mean': float(all_actions.mean().item()),
        'overall_std': float(all_actions.std().item()),
        'overall_min': float(all_actions.min().item()),
        'overall_max': float(all_actions.max().item()),
    }
    
    log_prob_stats = {
        'mean': float(all_log_probs.mean().item()),
        'std': float(all_log_probs.std().item()),
        'min': float(all_log_probs.min().item()),
        'max': float(all_log_probs.max().item()),
    }
    
    entropy_stats = {
        'mean': float(all_entropies.mean().item()),
        'std': float(all_entropies.std().item()),
        'min': float(all_entropies.min().item()),
        'max': float(all_entropies.max().item()),
    }
    
    value_stats = {
        'mean': float(all_values.mean().item()),
        'std': float(all_values.std().item()),
        'min': float(all_values.min().item()),
        'max': float(all_values.max().item()),
    }
    
    # Check for NaN/Inf
    has_nan = torch.isnan(all_actions).any().item()
    has_inf = torch.isinf(all_actions).any().item()
    
    return {
        'name': name,
        'num_samples': num_samples,
        'action_stats': action_stats,
        'log_prob_stats': log_prob_stats,
        'entropy_stats': entropy_stats,
        'value_stats': value_stats,
        'has_nan': bool(has_nan),
        'has_inf': bool(has_inf),
        'action_samples': all_actions.numpy().tolist()[:100],  # Sample for visualization
    }


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cpu')

    num_obs = 64
    num_actions = 22
    batch_size = 32

    obs_shape = (num_obs,)
    states_shape = (num_obs,)
    action_shape = (num_actions,)

    observations = torch.randn(batch_size, num_obs, device=device)
    states = torch.zeros(batch_size, num_obs, device=device)

    print("=" * 60)
    print("Action Distribution Test")
    print("=" * 60)
    print(f"Number of samples: 10,000")

    results = []

    # Test Original Network
    print("\nTesting Original Network action distribution...")
    try:
        model_cfg = {
            'pi_hid_sizes': [128, 128],
            'vf_hid_sizes': [128, 128],
            'activation': 'elu',
            'sigmoid_actions': False,
        }
        original_net = OriginalActorCritic(
            obs_shape=obs_shape,
            states_shape=states_shape,
            actions_shape=action_shape,
            initial_std=0.5,
            model_cfg=model_cfg,
            asymmetric=False,
        ).to(device)
        
        result = test_action_distribution(original_net, 'Original', observations, states)
        results.append(result)
        print(f"  Action mean: {result['action_stats']['overall_mean']:.4f}")
        print(f"  Action std: {result['action_stats']['overall_std']:.4f}")
        print(f"  Entropy: {result['entropy_stats']['mean']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test New Network
    print("\nTesting Modern Network action distribution...")
    try:
        new_net = ModernActorCritic(
            obs_shape=obs_shape,
            state_shape=states_shape,
            action_shape=action_shape,
            initial_std=0.5,
            actor_hidden_dims=[128, 128],
            critic_hidden_dims=[128, 128],
            activation="gelu",
            use_residual=True,
        ).to(device)
        
        result = test_action_distribution(new_net, 'Modern', observations, states)
        results.append(result)
        print(f"  Action mean: {result['action_stats']['overall_mean']:.4f}")
        print(f"  Action std: {result['action_stats']['overall_std']:.4f}")
        print(f"  Entropy: {result['entropy_stats']['mean']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test Advanced Network
    print("\nTesting Advanced Network action distribution...")
    try:
        advanced_net = AdvancedActorCritic(
            obs_shape=obs_shape,
            state_shape=states_shape,
            action_shape=action_shape,
            initial_std=0.5,
            actor_hidden_dims=[128, 128],
            critic_hidden_dims=[128, 128],
            num_attention_heads=4,
            use_attention=True,
            use_gating=True,
            dropout=0.1,
        ).to(device)
        
        result = test_action_distribution(advanced_net, 'Advanced', observations, states)
        results.append(result)
        print(f"  Action mean: {result['action_stats']['overall_mean']:.4f}")
        print(f"  Action std: {result['action_stats']['overall_std']:.4f}")
        print(f"  Entropy: {result['entropy_stats']['mean']:.4f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Save results
    out_dir = ROOT / 'tests'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Simplify for JSON (remove large arrays)
    json_results = []
    for r in results:
        json_r = r.copy()
        json_r.pop('action_samples', None)  # Remove large sample array
        json_results.append(json_r)
    
    with open(out_dir / 'action_distribution_log.txt', 'w') as f:
        f.write(json.dumps(json_results, indent=2))

    # Plot action distributions
    if len(results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = ['blue', 'green', 'red', 'purple']
        
        # Action mean distribution
        for i, result in enumerate(results):
            action_means = result['action_stats']['mean']
            axes[0, 0].hist(action_means, bins=20, alpha=0.6, 
                          label=result['name'], color=colors[i % len(colors)])
        axes[0, 0].set_xlabel('Action Mean')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Action Means')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Action std distribution
        for i, result in enumerate(results):
            action_stds = result['action_stats']['std']
            axes[0, 1].hist(action_stds, bins=20, alpha=0.6,
                          label=result['name'], color=colors[i % len(colors)])
        axes[0, 1].set_xlabel('Action Std')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Action Stds')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Entropy comparison
        entropies = [r['entropy_stats']['mean'] for r in results]
        names = [r['name'] for r in results]
        axes[1, 0].bar(names, entropies, color=colors[:len(results)])
        axes[1, 0].set_ylabel('Mean Entropy')
        axes[1, 0].set_title('Policy Entropy Comparison')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Log prob comparison
        log_probs = [r['log_prob_stats']['mean'] for r in results]
        axes[1, 1].bar(names, log_probs, color=colors[:len(results)])
        axes[1, 1].set_ylabel('Mean Log Probability')
        axes[1, 1].set_title('Log Probability Comparison')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig_path = out_dir / 'action_distribution_plots.png'
        plt.savefig(fig_path)
        print(f"\nSaved distribution plots to {fig_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Action Distribution Summary")
    print("=" * 60)
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Action range: [{result['action_stats']['overall_min']:.4f}, "
              f"{result['action_stats']['overall_max']:.4f}]")
        print(f"  Action mean: {result['action_stats']['overall_mean']:.4f}")
        print(f"  Action std: {result['action_stats']['overall_std']:.4f}")
        print(f"  Log prob mean: {result['log_prob_stats']['mean']:.4f}")
        print(f"  Entropy mean: {result['entropy_stats']['mean']:.4f}")
        if result['has_nan']:
            print(f"  ⚠️  Warning: Contains NaN values")
        if result['has_inf']:
            print(f"  ⚠️  Warning: Contains Inf values")

    print("\n✅ Action distribution test completed!")


if __name__ == '__main__':
    main()

