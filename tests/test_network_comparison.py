#!/usr/bin/env python3
"""
Network comparison test
- Compares original, new, and advanced networks
- Tests convergence speed and final performance
- Logs comparison metrics
- Generates comparison plots
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


def tensor_stats(t: torch.Tensor):
    return {
        'mean': float(torch.nan_to_num(t.mean()).item()),
        'std': float(torch.nan_to_num(t.std()).item()),
        'min': float(torch.nan_to_num(t.min()).item()),
        'max': float(torch.nan_to_num(t.max()).item()),
    }


def test_network_convergence(net, name, observations, states, num_steps=300):
    """Test network convergence on fixed inputs."""
    device = observations.device
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    
    losses = []
    policy_losses = []
    value_losses = []
    
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
        nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(float(loss.item()))
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))
    
    return {
        'name': name,
        'losses': losses,
        'policy_losses': policy_losses,
        'value_losses': value_losses,
        'final_loss': losses[-1],
        'final_policy_loss': policy_losses[-1],
        'final_value_loss': value_losses[-1],
        'convergence_step': np.argmin(losses),
        'num_parameters': sum(p.numel() for p in net.parameters()),
    }


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device('cpu')

    # Synthetic problem sizes
    num_obs = 64
    num_actions = 22
    batch_size = 64

    obs_shape = (num_obs,)
    states_shape = (num_obs,)
    action_shape = (num_actions,)

    # Fixed inputs for fair comparison
    observations = torch.randn(batch_size, num_obs, device=device)
    states = torch.zeros(batch_size, num_obs, device=device)

    print("=" * 60)
    print("Network Comparison Test")
    print("=" * 60)

    results = []

    # Test Original Network
    print("\nTesting Original ActorCritic...")
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
        
        result = test_network_convergence(original_net, 'Original', observations, states)
        results.append(result)
        print(f"  Parameters: {result['num_parameters']:,}")
        print(f"  Final loss: {result['final_loss']:.6f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test New Network
    print("\nTesting Modern ActorCritic...")
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
            asymmetric=False,
        ).to(device)
        
        result = test_network_convergence(new_net, 'Modern', observations, states)
        results.append(result)
        print(f"  Parameters: {result['num_parameters']:,}")
        print(f"  Final loss: {result['final_loss']:.6f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Test Advanced Network
    print("\nTesting Advanced ActorCritic...")
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
            asymmetric=False,
        ).to(device)
        
        result = test_network_convergence(advanced_net, 'Advanced', observations, states)
        results.append(result)
        print(f"  Parameters: {result['num_parameters']:,}")
        print(f"  Final loss: {result['final_loss']:.6f}")
    except Exception as e:
        print(f"  Error: {e}")

    # Save results
    out_dir = ROOT / 'tests'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'network_comparison_log.txt', 'w') as f:
        f.write(json.dumps(results, indent=2))

    # Plot comparison
    if len(results) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        # Total loss
        for i, result in enumerate(results):
            axes[0, 0].plot(result['losses'], label=result['name'], color=colors[i % len(colors)])
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Policy loss
        for i, result in enumerate(results):
            axes[0, 1].plot(result['policy_losses'], label=result['name'], color=colors[i % len(colors)])
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Policy Loss')
        axes[0, 1].set_title('Policy Loss Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Value loss
        for i, result in enumerate(results):
            axes[1, 0].plot(result['value_losses'], label=result['name'], color=colors[i % len(colors)])
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Value Loss')
        axes[1, 0].set_title('Value Loss Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Summary bar chart
        names = [r['name'] for r in results]
        final_losses = [r['final_loss'] for r in results]
        num_params = [r['num_parameters'] for r in results]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax2 = axes[1, 1]
        bars1 = ax2.bar(x - width/2, final_losses, width, label='Final Loss', color='skyblue')
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x + width/2, [p/1000 for p in num_params], width, label='Parameters (K)', color='lightcoral')
        
        ax2.set_xlabel('Network')
        ax2.set_ylabel('Final Loss', color='skyblue')
        ax2_twin.set_ylabel('Parameters (K)', color='lightcoral')
        ax2.set_title('Final Performance & Model Size')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path = out_dir / 'network_comparison_curves.png'
        plt.savefig(fig_path)
        print(f"\nSaved comparison plot to {fig_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Comparison Summary")
    print("=" * 60)
    for result in results:
        print(f"\n{result['name']}:")
        print(f"  Parameters: {result['num_parameters']:,}")
        print(f"  Final Loss: {result['final_loss']:.6f}")
        print(f"  Final Policy Loss: {result['final_policy_loss']:.6f}")
        print(f"  Final Value Loss: {result['final_value_loss']:.6f}")
        print(f"  Convergence Step: {result['convergence_step']}")

    print("\n✅ Network comparison test completed!")


if __name__ == '__main__':
    main()

