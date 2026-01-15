#!/usr/bin/env python3
"""
Network performance benchmark test
- Measures forward/backward pass speed
- Memory usage analysis
- Throughput comparison
- Logs performance metrics
"""

import os
import sys
import json
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from algorithms.rl.ppo.module import ActorCritic as OriginalActorCritic
from algorithms.rl.ppo.new_network import ModernActorCritic
from algorithms.rl.ppo.advanced_network import AdvancedActorCritic


def benchmark_network(net, name, observations, states, num_iterations=100, warmup=10):
    """Benchmark network performance."""
    device = observations.device
    net.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = net.act(observations, states, deterministic=True)
            _ = net.evaluate(observations, None, torch.randn_like(observations[:, :22]))
    
    # Benchmark forward pass (inference)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            actions, _, values, _, _ = net.act(observations, states, deterministic=True)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    forward_time = (time.time() - start_time) / num_iterations
    
    # Benchmark forward pass (sampling)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            actions, _, values, _, _ = net.act(observations, states, deterministic=False)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    forward_sampling_time = (time.time() - start_time) / num_iterations
    
    # Benchmark evaluate
    actions = torch.randn(observations.shape[0], 22, device=device)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = net.evaluate(observations, None, actions)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    evaluate_time = (time.time() - start_time) / num_iterations
    
    # Benchmark backward pass
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4)
    actions = torch.randn(observations.shape[0], 22, device=device)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(num_iterations):
        log_probs, entropy, values, _, _ = net.evaluate(observations, None, actions)
        loss = -log_probs.mean() + nn.MSELoss()(values, torch.zeros_like(values))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if device.type == 'cuda' else None
    backward_time = (time.time() - start_time) / num_iterations
    
    # Memory usage (approximate)
    num_params = sum(p.numel() for p in net.parameters())
    param_memory_mb = num_params * 4 / (1024 * 1024)  # Assuming float32
    
    # Throughput
    batch_size = observations.shape[0]
    forward_throughput = batch_size / forward_time
    backward_throughput = batch_size / backward_time
    
    return {
        'name': name,
        'num_parameters': num_params,
        'param_memory_mb': param_memory_mb,
        'forward_time_ms': forward_time * 1000,
        'forward_sampling_time_ms': forward_sampling_time * 1000,
        'evaluate_time_ms': evaluate_time * 1000,
        'backward_time_ms': backward_time * 1000,
        'forward_throughput': forward_throughput,
        'backward_throughput': backward_throughput,
        'total_time_ms': (forward_time + backward_time) * 1000,
    }


def main():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_obs = 64
    num_actions = 22
    batch_size = 64

    obs_shape = (num_obs,)
    states_shape = (num_obs,)
    action_shape = (num_actions,)

    observations = torch.randn(batch_size, num_obs, device=device)
    states = torch.zeros(batch_size, num_obs, device=device)

    print("=" * 60)
    print("Network Performance Benchmark")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Warmup iterations: 10")
    print(f"Benchmark iterations: 100")

    results = []

    # Benchmark Original Network
    print("\nBenchmarking Original Network...")
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
        
        result = benchmark_network(original_net, 'Original', observations, states)
        results.append(result)
        print(f"  Forward: {result['forward_time_ms']:.3f} ms")
        print(f"  Backward: {result['backward_time_ms']:.3f} ms")
        print(f"  Parameters: {result['num_parameters']:,}")
    except Exception as e:
        print(f"  Error: {e}")

    # Benchmark New Network
    print("\nBenchmarking Modern Network...")
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
        
        result = benchmark_network(new_net, 'Modern', observations, states)
        results.append(result)
        print(f"  Forward: {result['forward_time_ms']:.3f} ms")
        print(f"  Backward: {result['backward_time_ms']:.3f} ms")
        print(f"  Parameters: {result['num_parameters']:,}")
    except Exception as e:
        print(f"  Error: {e}")

    # Benchmark Advanced Network
    print("\nBenchmarking Advanced Network...")
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
        
        result = benchmark_network(advanced_net, 'Advanced', observations, states)
        results.append(result)
        print(f"  Forward: {result['forward_time_ms']:.3f} ms")
        print(f"  Backward: {result['backward_time_ms']:.3f} ms")
        print(f"  Parameters: {result['num_parameters']:,}")
    except Exception as e:
        print(f"  Error: {e}")

    # Save results
    out_dir = ROOT / 'tests'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'network_performance_log.txt', 'w') as f:
        f.write(json.dumps(results, indent=2))

    # Print summary table
    print("\n" + "=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print(f"{'Network':<15} {'Params':<12} {'Forward(ms)':<12} {'Backward(ms)':<12} "
          f"{'Total(ms)':<12} {'Throughput':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<15} {result['num_parameters']:>11,} "
              f"{result['forward_time_ms']:>11.3f} {result['backward_time_ms']:>11.3f} "
              f"{result['total_time_ms']:>11.3f} {result['forward_throughput']:>11.1f}")
    
    print("=" * 80)
    
    # Speedup comparison
    if len(results) > 1:
        baseline = results[0]
        print("\nSpeedup relative to baseline:")
        for result in results[1:]:
            speedup = baseline['total_time_ms'] / result['total_time_ms']
            print(f"  {result['name']}: {speedup:.2f}x")

    print("\n✅ Performance benchmark completed!")


if __name__ == '__main__':
    main()

