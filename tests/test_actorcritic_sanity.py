#!/usr/bin/env python3
"""
ActorCritic forward/backward sanity test
 - Builds a small ActorCritic network from algorithms/rl/ppo/module.py
 - Runs a forward pass with synthetic inputs
 - Computes simple policy/value losses and runs backward + optimizer step
 - Verifies parameters changed and outputs remain finite/reasonable
 - Logs key metrics to stdout and to tests/nn_sanity_log.txt
"""

import os
import sys
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Direct import to avoid triggering __init__.py which imports ppo.py (requires tqdm)
import importlib.util
module_path = ROOT / 'algorithms' / 'rl' / 'ppo' / 'module.py'
spec = importlib.util.spec_from_file_location("module", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ActorCritic = module.ActorCritic


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

    # Synthetic problem sizes (keep small to be fast/stable)
    num_obs = 64
    num_actions = 22
    batch_size = 32

    obs_shape = (num_obs,)
    states_shape = (num_obs,)  # unused when asymmetric=False
    actions_shape = (num_actions,)

    model_cfg = {
        'pi_hid_sizes': [128, 128],
        'vf_hid_sizes': [128, 128],
        'activation': 'elu',
        'sigmoid_actions': False,
    }

    net = ActorCritic(
        obs_shape=obs_shape,
        states_shape=states_shape,
        actions_shape=actions_shape,
        initial_std=0.5,
        model_cfg=model_cfg,
        asymmetric=False,
    ).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)

    # Random synthetic batch
    observations = torch.randn(batch_size, num_obs, device=device)
    states = torch.zeros(batch_size, num_obs, device=device)

    # Keep a snapshot of parameters before step
    with torch.no_grad():
        params_before = [p.clone() for p in net.parameters()]

    # Forward to get actions (sampling), then evaluate to get differentiable tensors
    with torch.no_grad():
        sampled_actions, _, _, _, _ = net.act(observations, states)

    actions_log_prob, entropy, values, mu, log_std = net.evaluate(observations, None, sampled_actions)

    # Sanity: all finite
    assert torch.isfinite(sampled_actions).all(), 'Actions contain non-finite values'
    assert torch.isfinite(actions_log_prob).all(), 'Log-prob contains non-finite values'
    assert torch.isfinite(values).all(), 'Values contain non-finite values'

    # Simple synthetic targets for value learning
    target_returns = torch.randn_like(values)

    # Losses: encourage high log-prob and fit value to target
    policy_loss = -actions_log_prob.mean()
    value_loss = nn.MSELoss()(values, target_returns)
    loss = policy_loss + value_loss

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

    # Forward again to ensure produces reasonable outputs post-update
    with torch.no_grad():
        actions2, actions_log_prob2, values2, mu2, log_std2 = net.act(observations, states)

    log = {
        'loss': float(loss.item()),
        'policy_loss': float(policy_loss.item()),
        'value_loss': float(value_loss.item()),
        'grad_norm': float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
        'param_total_abs_change': float(total_param_delta),
        'param_total_abs_norm': float(total_param_norm),
        'actions_stats_before': tensor_stats(sampled_actions),
        'values_stats_before': tensor_stats(values),
        'actions_stats_after': tensor_stats(actions2),
        'values_stats_after': tensor_stats(values2),
        'log_std_stats': tensor_stats(log_std.mean(dim=0)),
    }

    # Print and save
    print("==== ActorCritic Sanity Test ====")
    for k, v in log.items():
        print(f"{k}: {v}")

    out_dir = ROOT / 'tests'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'nn_sanity_log.txt', 'w') as f:
        f.write(json.dumps(log, indent=2))

    # Basic pass criteria
    assert math.isfinite(log['loss']), 'Non-finite loss'
    assert log['param_total_abs_change'] > 0.0, 'Parameters did not change after backward/step'
    print("✅ Sanity test passed: network forward/backward working and parameters updated.")


if __name__ == '__main__':
    main()


