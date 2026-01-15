#!/usr/bin/env python3
"""
New Network (ModernActorCritic) convergence probe
- Tests the new network with residual connections
- Uses fixed synthetic inputs
- Logs convergence metrics
- Plots loss curves
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

from algorithms.rl.ppo.new_network import ModernActorCritic


def tensor_stats(t: torch.Tensor):
    return {
        'shape': list(t.shape),
        'dtype': str(t.dtype),
        'min': float(torch.nan_to_num(t.min()).item()),
        'max': float(torch.nan_to_num(t.max()).item()),
        'mean': float(torch.nan_to_num(t.mean()).item()),
        'std': float(torch.nan_to_num(t.std()).item()),
        'nan': int(torch.isnan(t).any().item()),
        'inf': int(torch.isinf(t).any().item()),
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

    net = ModernActorCritic(
        obs_shape=obs_shape,
        state_shape=states_shape,
        action_shape=action_shape,
        initial_std=0.5,
        actor_hidden_dims=[128, 128, 64],
        critic_hidden_dims=[128, 128, 64],
        activation="gelu",
        use_residual=True,
        asymmetric=False,
        learnable_std=True,
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    # Fixed synthetic batch
    observations = torch.randn(batch_size, num_obs, device=device)
    states = torch.zeros(batch_size, num_obs, device=device)

    # Logging setup
    out_dir = ROOT / 'tests'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'new_network_convergence_log.txt'
    with open(log_path, 'w') as f:
        f.write('')

    num_steps = 500
    losses, policy_losses, value_losses, entropies = [], [], [], []

    print("Testing ModernActorCritic network convergence...")
    print(f"Network parameters: {sum(p.numel() for p in net.parameters()):,}")

    # Training loop
    for step in range(num_steps):
        with torch.no_grad():
            params_before = [p.clone() for p in net.parameters()]

        with torch.no_grad():
            sampled_actions, _, _, _, _ = net.act(observations, states, deterministic=False)

        log_probs, entropy, values, mu, stds = net.evaluate(observations, None, sampled_actions)

        # Fixed targets
        target_returns = torch.zeros_like(values)

        policy_loss = -log_probs.mean()
        value_loss = nn.MSELoss()(values, target_returns)
        entropy_bonus = -entropy.mean() * 0.01
        loss = policy_loss + value_loss + entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        # Parameter change
        with torch.no_grad():
            total_param_delta = 0.0
            total_param_norm = 0.0
            for p0, p1 in zip(params_before, net.parameters()):
                total_param_delta += (p1 - p0).abs().sum().item()
                total_param_norm += p1.abs().sum().item()

        # Track metrics
        losses.append(float(loss.item()))
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))
        entropies.append(float(entropy.mean().item()))

        # Log JSONL
        entry = {
            'step': step,
            'loss': float(loss.item()),
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'entropy': float(entropy.mean().item()),
            'grad_norm': float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            'param_total_abs_change': float(total_param_delta),
            'param_total_abs_norm': float(total_param_norm),
            'actions_stats': tensor_stats(sampled_actions),
            'values_stats': tensor_stats(values),
            'mu_stats': tensor_stats(mu),
            'stds_stats': tensor_stats(stds),
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        if (step + 1) % 50 == 0:
            print(f"Step {step+1}/{num_steps} loss={entry['loss']:.4f} "
                  f"policy={entry['policy_loss']:.4f} value={entry['value_loss']:.4f} "
                  f"entropy={entry['entropy']:.4f}")

    # Plot curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    x = np.arange(len(losses))
    
    axes[0, 0].plot(x, losses, label='Total Loss', color='blue')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(x, policy_losses, label='Policy Loss', color='red')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Policy Loss')
    axes[0, 1].set_title('Policy Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    axes[1, 0].plot(x, value_losses, label='Value Loss', color='green')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Value Loss')
    axes[1, 0].set_title('Value Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(x, entropies, label='Entropy', color='purple')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Entropy')
    axes[1, 1].set_title('Policy Entropy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    fig_path = out_dir / 'new_network_convergence_curves.png'
    plt.savefig(fig_path)
    print(f"\nSaved plot to {fig_path}")
    print("✅ New network convergence test completed!")


if __name__ == '__main__':
    main()

