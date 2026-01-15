#!/usr/bin/env python3
"""
Advanced Network (AdvancedActorCritic) sanity test
- Tests the advanced network with attention and gating
- Verifies forward/backward pass
- Checks for numerical stability
- Logs detailed metrics
"""

import os
import sys
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from algorithms.rl.ppo.advanced_network import AdvancedActorCritic


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
    device = torch.device('cpu')

    # Synthetic problem sizes
    num_obs = 64
    num_actions = 22
    batch_size = 32

    obs_shape = (num_obs,)
    states_shape = (num_obs,)
    action_shape = (num_actions,)

    print("Testing AdvancedActorCritic network...")
    
    net = AdvancedActorCritic(
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
        learnable_std=True
    ).to(device)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    # Random synthetic batch
    observations = torch.randn(batch_size, num_obs, device=device)
    states = torch.zeros(batch_size, num_obs, device=device)

    # Snapshot parameters
    with torch.no_grad():
        params_before = [p.clone() for p in net.parameters()]

    # Forward pass - sampling
    with torch.no_grad():
        sampled_actions, log_probs1, values1, means1, stds1 = net.act(
            observations, states, deterministic=False
        )

    # Forward pass - deterministic
    with torch.no_grad():
        det_actions, log_probs2, values2, means2, stds2 = net.act(
            observations, states, deterministic=True
        )

    # Evaluate
    log_probs, entropy, values, mu, stds = net.evaluate(observations, None, sampled_actions)

    # Sanity checks
    assert torch.isfinite(sampled_actions).all(), 'Actions contain non-finite values'
    assert torch.isfinite(log_probs).all(), 'Log-prob contains non-finite values'
    assert torch.isfinite(values).all(), 'Values contain non-finite values'
    assert torch.isfinite(entropy).all(), 'Entropy contains non-finite values'

    # Loss computation
    target_returns = torch.randn_like(values)
    policy_loss = -log_probs.mean()
    value_loss = nn.MSELoss()(values, target_returns)
    entropy_bonus = -entropy.mean() * 0.01
    loss = policy_loss + value_loss + entropy_bonus

    optimizer.zero_grad()
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
    optimizer.step()

    # Check parameter change
    with torch.no_grad():
        total_param_delta = 0.0
        total_param_norm = 0.0
        for p0, p1 in zip(params_before, net.parameters()):
            total_param_delta += (p1 - p0).abs().sum().item()
            total_param_norm += p1.abs().sum().item()

    # Forward again after update
    with torch.no_grad():
        actions_after, log_probs_after, values_after, means_after, stds_after = net.act(
            observations, states, deterministic=False
        )

    log = {
        'loss': float(loss.item()),
        'policy_loss': float(policy_loss.item()),
        'value_loss': float(value_loss.item()),
        'entropy': float(entropy.mean().item()),
        'grad_norm': float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        'param_total_abs_change': float(total_param_delta),
        'param_total_abs_norm': float(total_param_norm),
        'total_parameters': total_params,
        'actions_stats_before': tensor_stats(sampled_actions),
        'actions_stats_deterministic': tensor_stats(det_actions),
        'actions_stats_after': tensor_stats(actions_after),
        'values_stats_before': tensor_stats(values),
        'values_stats_after': tensor_stats(values_after),
        'log_probs_stats': tensor_stats(log_probs),
        'entropy_stats': tensor_stats(entropy),
        'mu_stats': tensor_stats(mu),
        'stds_stats': tensor_stats(stds),
    }

    # Print results
    print("\n==== Advanced Network Sanity Test ====")
    for k, v in log.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for k2, v2 in v.items():
                print(f"  {k2}: {v2}")
        else:
            print(f"{k}: {v}")

    # Save log
    out_dir = ROOT / 'tests'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'advanced_network_sanity_log.txt', 'w') as f:
        f.write(json.dumps(log, indent=2))

    # Assertions
    assert math.isfinite(log['loss']), 'Non-finite loss'
    assert log['param_total_abs_change'] > 0.0, 'Parameters did not change'
    assert log['actions_stats_before']['nan'] == 0, 'NaN in actions'
    assert log['values_stats_before']['nan'] == 0, 'NaN in values'
    
    print("\n✅ Advanced network sanity test passed!")


if __name__ == '__main__':
    main()

