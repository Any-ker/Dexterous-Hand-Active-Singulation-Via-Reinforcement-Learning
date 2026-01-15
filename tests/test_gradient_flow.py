#!/usr/bin/env python3
"""
Gradient flow analysis test
- Tests gradient flow through different network architectures
- Measures gradient norms at different layers
- Detects vanishing/exploding gradients
- Logs gradient statistics
"""

import os
import sys
import json
import math
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from algorithms.rl.ppo.module import ActorCritic as OriginalActorCritic
from algorithms.rl.ppo.new_network import ModernActorCritic
from algorithms.rl.ppo.advanced_network import AdvancedActorCritic


def register_gradient_hooks(net, hook_dict):
    """Register hooks to capture gradients."""
    def make_hook(name):
        def hook(grad):
            if grad is not None:
                grad_norm = grad.norm().item()
                grad_max = grad.abs().max().item()
                hook_dict[name].append({
                    'norm': grad_norm,
                    'max': grad_max,
                    'mean': grad.abs().mean().item(),
                })
            return grad
        return hook
    
    for name, param in net.named_parameters():
        if param.requires_grad:
            param.register_hook(make_hook(name))
            hook_dict[name] = []


def test_gradient_flow(net, name, observations, states, num_steps=100):
    """Test gradient flow through network."""
    device = observations.device
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    
    gradient_stats = {}
    register_gradient_hooks(net, gradient_stats)
    
    step_grad_norms = []
    step_grad_maxs = []
    
    for step in range(num_steps):
        with torch.no_grad():
            actions, _, _, _, _ = net.act(observations, states, deterministic=False)
        
        log_probs, entropy, values, mu, stds = net.evaluate(observations, None, actions)
        
        target_returns = torch.zeros_like(values)
        policy_loss = -log_probs.mean()
        value_loss = nn.MSELoss()(values, target_returns)
        loss = policy_loss + value_loss
        
        optimizer.zero_grad()
        loss.backward()
        
        # Collect gradient statistics
        all_grad_norms = []
        all_grad_maxs = []
        for param_name, param in net.named_parameters():
            if param.grad is not None:
                all_grad_norms.append(param.grad.norm().item())
                all_grad_maxs.append(param.grad.abs().max().item())
        
        if all_grad_norms:
            step_grad_norms.append(np.mean(all_grad_norms))
            step_grad_maxs.append(np.max(all_grad_maxs))
        
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
    
    # Aggregate statistics
    layer_stats = {}
    for param_name, grad_list in gradient_stats.items():
        if grad_list:
            norms = [g['norm'] for g in grad_list]
            maxs = [g['max'] for g in grad_list]
            means = [g['mean'] for g in grad_list]
            
            layer_stats[param_name] = {
                'mean_norm': np.mean(norms),
                'std_norm': np.std(norms),
                'max_norm': np.max(norms),
                'min_norm': np.min(norms),
                'mean_max': np.mean(maxs),
                'mean_mean': np.mean(means),
            }
    
    return {
        'name': name,
        'layer_stats': layer_stats,
        'step_grad_norms': step_grad_norms,
        'step_grad_maxs': step_grad_maxs,
        'mean_grad_norm': np.mean(step_grad_norms) if step_grad_norms else 0.0,
        'max_grad_norm': np.max(step_grad_maxs) if step_grad_maxs else 0.0,
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
    print("Gradient Flow Analysis")
    print("=" * 60)

    results = []

    # Test Original Network
    print("\nTesting Original Network gradient flow...")
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
        
        result = test_gradient_flow(original_net, 'Original', observations, states)
        results.append(result)
        print(f"  Mean grad norm: {result['mean_grad_norm']:.6f}")
        print(f"  Max grad norm: {result['max_grad_norm']:.6f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test New Network
    print("\nTesting Modern Network gradient flow...")
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
        
        result = test_gradient_flow(new_net, 'Modern', observations, states)
        results.append(result)
        print(f"  Mean grad norm: {result['mean_grad_norm']:.6f}")
        print(f"  Max grad norm: {result['max_grad_norm']:.6f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test Advanced Network
    print("\nTesting Advanced Network gradient flow...")
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
        
        result = test_gradient_flow(advanced_net, 'Advanced', observations, states)
        results.append(result)
        print(f"  Mean grad norm: {result['mean_grad_norm']:.6f}")
        print(f"  Max grad norm: {result['max_grad_norm']:.6f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Save results
    out_dir = ROOT / 'tests'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    json_results = []
    for r in results:
        json_r = {
            'name': r['name'],
            'mean_grad_norm': float(r['mean_grad_norm']),
            'max_grad_norm': float(r['max_grad_norm']),
            'step_grad_norms': [float(x) for x in r['step_grad_norms']],
            'step_grad_maxs': [float(x) for x in r['step_grad_maxs']],
        }
        # Simplify layer stats for JSON
        json_r['layer_stats'] = {
            k: {k2: float(v2) for k2, v2 in v.items()}
            for k, v in r['layer_stats'].items()
        }
        json_results.append(json_r)
    
    with open(out_dir / 'gradient_flow_log.txt', 'w') as f:
        f.write(json.dumps(json_results, indent=2))

    # Plot gradient flow
    if len(results) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        colors = ['blue', 'green', 'red', 'purple']
        
        # Gradient norms over steps
        for i, result in enumerate(results):
            if result['step_grad_norms']:
                axes[0].plot(result['step_grad_norms'], 
                           label=f"{result['name']} (mean={result['mean_grad_norm']:.4f})",
                           color=colors[i % len(colors)], alpha=0.7)
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Mean Gradient Norm')
        axes[0].set_title('Gradient Norm Over Training Steps')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_yscale('log')
        
        # Gradient max over steps
        for i, result in enumerate(results):
            if result['step_grad_maxs']:
                axes[1].plot(result['step_grad_maxs'],
                           label=f"{result['name']} (max={result['max_grad_norm']:.4f})",
                           color=colors[i % len(colors)], alpha=0.7)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Max Gradient Value')
        axes[1].set_title('Max Gradient Value Over Training Steps')
        axes[1].legend()
        axes[1].grid(True)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        fig_path = out_dir / 'gradient_flow_curves.png'
        plt.savefig(fig_path)
        print(f"\nSaved gradient flow plot to {fig_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Gradient Flow Summary")
    print("=" * 60)
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Mean gradient norm: {result['mean_grad_norm']:.6f}")
        print(f"  Max gradient norm: {result['max_grad_norm']:.6f}")
        
        # Check for vanishing/exploding gradients
        if result['mean_grad_norm'] < 1e-6:
            print(f"  ⚠️  Warning: Very small gradients (possible vanishing gradient)")
        elif result['max_grad_norm'] > 100:
            print(f"  ⚠️  Warning: Very large gradients (possible exploding gradient)")
        else:
            print(f"  ✅ Gradient flow looks healthy")

    print("\n✅ Gradient flow analysis completed!")


if __name__ == '__main__':
    main()

