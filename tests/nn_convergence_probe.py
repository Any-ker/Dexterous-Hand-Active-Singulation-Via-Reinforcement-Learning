#!/usr/bin/env python3
"""
Fixed-input ActorCritic convergence probe
- Uses the same synthetic inputs every step
- Repeats forward/backward for many steps
- Logs per-step metrics to tests/nn_convergence_log.txt (JSONL)
- Plots loss curve to tests/nn_convergence_curve.png
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

from algorithms.rl.ppo.module import ActorCritic


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

    # Fixed synthetic batch (same every step)
    observations = torch.randn(batch_size, num_obs, device=device)
    states = torch.zeros(batch_size, num_obs, device=device)

    # Logging setup
    out_dir = ROOT / 'tests'
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / 'nn_convergence_log.txt'
    with open(log_path, 'w') as f:
        f.write('')

    num_steps = 500
    losses, policy_losses, value_losses = [], [], []

    # Training loop on fixed inputs
    for step in range(num_steps):
        # Snapshot params
        with torch.no_grad():
            params_before = [p.clone() for p in net.parameters()]

        with torch.no_grad():
            sampled_actions, _, _, _, _ = net.act(observations, states)

        actions_log_prob, entropy, values, mu, log_std = net.evaluate(observations, None, sampled_actions)

        # Targets fixed per step to keep problem stationary
        target_returns = torch.zeros_like(values)

        policy_loss = -actions_log_prob.mean()
        value_loss = nn.MSELoss()(values, target_returns)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()

        # Param change
        with torch.no_grad():
            total_param_delta = 0.0
            total_param_norm = 0.0
            for p0, p1 in zip(params_before, net.parameters()):
                total_param_delta += (p1 - p0).abs().sum().item()
                total_param_norm += p1.abs().sum().item()

        # Track series
        losses.append(float(loss.item()))
        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))

        # Log JSONL
        entry = {
            'step': step,
            'loss': float(loss.item()),
            'policy_loss': float(policy_loss.item()),
            'value_loss': float(value_loss.item()),
            'grad_norm': float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            'param_total_abs_change': float(total_param_delta),
            'param_total_abs_norm': float(total_param_norm),
            'actions_stats': tensor_stats(sampled_actions),
            'values_stats': tensor_stats(values),
            'log_std_stats': tensor_stats(log_std.mean(dim=0)),
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

        if (step + 1) % 50 == 0:
            print(f"Step {step+1}/{num_steps} loss={entry['loss']:.4f} policy={entry['policy_loss']:.4f} value={entry['value_loss']:.4f}")

    # Plot (only value loss)
    plt.figure(figsize=(10, 6))
    x = np.arange(len(value_losses))
    plt.plot(x, value_losses, label='Value Loss', color='green')
    plt.xlabel('Step')
    plt.ylabel('Value Loss')
    plt.title('Fixed-input Convergence Probe (Value Loss)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_path = out_dir / 'nn_convergence_value_curve.png'
    plt.savefig(fig_path)
    print(f"Saved plot to {fig_path}")


if __name__ == '__main__':
    main()


