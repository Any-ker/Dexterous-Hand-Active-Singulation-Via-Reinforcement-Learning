import torch
from advanced_network import AdvancedActorCritic


def example_usage():
    """Example of how to use the AdvancedActorCritic network."""
    
    # Network configuration
    obs_shape = (300,)  # Observation dimension
    state_shape = (300,)  # State dimension
    action_shape = (22,)  # Action dimension
    
    print("=" * 60)
    print("Advanced Actor-Critic Network Example")
    print("=" * 60)
    
    # Create network with advanced features
    network = AdvancedActorCritic(
        obs_shape=obs_shape,
        state_shape=state_shape,
        action_shape=action_shape,
        initial_std=0.8,
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[512, 512, 256],
        num_attention_heads=8,
        use_attention=True,
        use_gating=True,
        dropout=0.1,
        asymmetric=False,
        learnable_std=True
    )
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    network.train()
    
    print(f"\nDevice: {device}")
    print(f"Network created with:")
    print(f"  - Multi-head self-attention: Enabled (8 heads)")
    print(f"  - Adaptive gating units: Enabled")
    print(f"  - Custom layers: All implemented from scratch")
    
    # Example: Training mode (sampling actions)
    batch_size = 1000
    observations = torch.randn(batch_size, obs_shape[0]).to(device)
    
    print(f"\n{'=' * 60}")
    print("Training Mode - Action Sampling")
    print(f"{'=' * 60}")
    
    actions, log_probs, values, means, stds = network.act(
        observations,
        deterministic=False
    )
    
    print(f"Input observations shape: {observations.shape}")
    print(f"Output actions shape: {actions.shape}")
    print(f"Log probabilities shape: {log_probs.shape}")
    print(f"State values shape: {values.shape}")
    print(f"Action means shape: {means.shape}")
    print(f"Action stds shape: {stds.shape}")
    print(f"\nAction statistics:")
    print(f"  Mean: {actions.mean().item():.4f}")
    print(f"  Std: {actions.std().item():.4f}")
    print(f"  Min: {actions.min().item():.4f}")
    print(f"  Max: {actions.max().item():.4f}")
    
    # Example: Evaluation mode (deterministic actions)
    print(f"\n{'=' * 60}")
    print("Evaluation Mode - Deterministic Actions")
    print(f"{'=' * 60}")
    
    actions_det, log_probs_det, values_det, means_det, stds_det = network.act(
        observations,
        deterministic=True
    )
    
    print(f"Deterministic actions shape: {actions_det.shape}")
    print(f"Deterministic action statistics:")
    print(f"  Mean: {actions_det.mean().item():.4f}")
    print(f"  Std: {actions_det.std().item():.4f}")
    
    # Example: Policy evaluation (for PPO update)
    print(f"\n{'=' * 60}")
    print("Policy Evaluation - For PPO Update")
    print(f"{'=' * 60}")
    
    log_probs_eval, entropy, values_eval, means_eval, stds_eval = network.evaluate(
        observations=observations,
        states=None,
        actions=actions
    )
    
    print(f"Evaluation log probs shape: {log_probs_eval.shape}")
    print(f"Entropy shape: {entropy.shape}")
    print(f"Evaluation values shape: {values_eval.shape}")
    print(f"\nPolicy statistics:")
    print(f"  Mean log prob: {log_probs_eval.mean().item():.4f}")
    print(f"  Mean entropy: {entropy.mean().item():.4f}")
    print(f"  Mean value: {values_eval.mean().item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    print(f"\n{'=' * 60}")
    print("Network Statistics")
    print(f"{'=' * 60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size (MB): {total_params * 4 / (1024 * 1024):.2f}")  # Assuming float32
    
    # Test individual components
    print(f"\n{'=' * 60}")
    print("Individual Component Forward Pass")
    print(f"{'=' * 60}")
    
    action_mean = network.get_action_mean(observations)
    value_estimate = network.get_value(observations)
    
    print(f"Action mean shape: {action_mean.shape}")
    print(f"Value estimate shape: {value_estimate.shape}")
    
    # Test gradient flow
    print(f"\n{'=' * 60}")
    print("Gradient Flow Test")
    print(f"{'=' * 60}")
    
    loss = (actions ** 2).mean()
    loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None for p in network.parameters() if p.requires_grad)
    print(f"Gradients computed: {has_grad}")
    
    if has_grad:
        grad_norms = []
        for name, param in network.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
        
        if grad_norms:
            print(f"Gradient statistics:")
            print(f"  Mean norm: {sum(grad_norms) / len(grad_norms):.6f}")
            print(f"  Max norm: {max(grad_norms):.6f}")
            print(f"  Min norm: {min(grad_norms):.6f}")
    
    print(f"\n{'=' * 60}")
    print("Example completed successfully!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    example_usage()

