
import torch
from new_network import ModernActorCritic


def example_usage():
    """Example of how to use the ModernActorCritic network."""
    
    # Network configuration
    obs_shape = (300,)  # Observation dimension
    state_shape = (300,)  # State dimension (same as obs for symmetric)
    action_shape = (22,)  # Action dimension
    
    # Create network
    network = ModernActorCritic(
        obs_shape=obs_shape,
        state_shape=state_shape,
        action_shape=action_shape,
        initial_std=0.8,
        actor_hidden_dims=[512, 512, 256],
        critic_hidden_dims=[512, 512, 256],
        activation="gelu",
        use_residual=True,
        asymmetric=False,
        learnable_std=True,
        output_activation=None
    )
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = network.to(device)
    network.train()
    
    # Example: Training mode (sampling actions)
    batch_size = 1000
    observations = torch.randn(batch_size, obs_shape[0]).to(device)
    
    print("=== Training Mode ===")
    actions, log_probs, values, means, stds = network.act(
        observations,
        deterministic=False
    )
    
    print(f"Observations shape: {observations.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Log probs shape: {log_probs.shape}")
    print(f"Values shape: {values.shape}")
    print(f"Action means shape: {means.shape}")
    print(f"Action stds shape: {stds.shape}")
    
    # Example: Evaluation mode (deterministic actions)
    print("\n=== Evaluation Mode ===")
    actions_det, log_probs_det, values_det, means_det, stds_det = network.act(
        observations,
        deterministic=True
    )
    print(f"Deterministic actions shape: {actions_det.shape}")
    
    # Example: Policy evaluation (for PPO update)
    print("\n=== Policy Evaluation ===")
    log_probs_eval, entropy, values_eval, means_eval, stds_eval = network.evaluate(
        observations=observations,
        states=None,
        actions=actions
    )
    print(f"Evaluation log probs shape: {log_probs_eval.shape}")
    print(f"Entropy shape: {entropy.shape}")
    print(f"Evaluation values shape: {values_eval.shape}")
    
    # Example: Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"\n=== Network Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Example: Forward pass through individual networks
    print("\n=== Individual Network Forward Pass ===")
    action_mean = network.get_action_mean(observations)
    value_estimate = network.get_value(observations)
    print(f"Action mean shape: {action_mean.shape}")
    print(f"Value estimate shape: {value_estimate.shape}")


if __name__ == "__main__":
    example_usage()

