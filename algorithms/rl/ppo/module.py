import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from typing import Iterable, Optional, Tuple


def get_activation_function(activation_type: str) -> nn.Module:
    """Create activation function module based on name."""
    activation_type = activation_type.lower()
    if activation_type == "elu":
        return nn.ELU()
    elif activation_type == "selu":
        return nn.SELU()
    elif activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "lrelu":
        return nn.LeakyReLU()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unknown activation function: {activation_type}")


class PolicyNetwork(nn.Module):
    """Policy network (actor) that outputs action means."""
    
    def __init__(self, input_size: int, hidden_sizes: Iterable[int], output_size: int, 
                 activation_name: str, use_tanh_output: bool):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.use_tanh = use_tanh_output
        
        # Build hidden layers
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.activations.append(get_activation_function(activation_name))
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = nn.Linear(prev_size, output_size)
        if use_tanh_output:
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.output_layer.weight, gain=1.0)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through policy network."""
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        x = self.output_layer(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x


class ValueNetwork(nn.Module):
    """Value network (critic) that estimates state values."""
    
    def __init__(self, input_size: int, hidden_sizes: Iterable[int], activation_name: str):
        super().__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        
        # Build hidden layers
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.activations.append(get_activation_function(activation_name))
            prev_size = hidden_size
        
        # Output layer (single value)
        self.output_layer = nn.Linear(prev_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=1.0)
                nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.output_layer.weight, gain=1.0)
        nn.init.constant_(self.output_layer.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through value network."""
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x))
        x = self.output_layer(x)
        return x


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO algorithm.
    
    This network consists of two separate networks:
    - Actor (policy): outputs action means
    - Critic (value): estimates state values
    """

    def __init__(
        self,
        obs_shape,
        states_shape,
        actions_shape,
        initial_std,
        model_cfg,
        asymmetric: bool = False,
    ):
        super().__init__()
        self.use_asymmetric = asymmetric
        self.action_dim = actions_shape[0]
        self.obs_dim = obs_shape[0]
        
        # Extract configuration with defaults
        if model_cfg is not None:
            actor_hidden = model_cfg.get("pi_hid_sizes", [256, 256, 256])
            critic_hidden = model_cfg.get("vf_hid_sizes", [256, 256, 256])
            activation = model_cfg.get("activation", "selu")
            use_tanh = model_cfg.get("sigmoid_actions", False)
        else:
            actor_hidden = [256, 256, 256]
            critic_hidden = [256, 256, 256]
            activation = "selu"
            use_tanh = False

        # Build policy network
        self.policy_net = PolicyNetwork(
            input_size=self.obs_dim,
            hidden_sizes=actor_hidden,
            output_size=self.action_dim,
            activation_name=activation,
            use_tanh_output=use_tanh
        )
        
        # Build value network
        value_input_size = states_shape[0] if asymmetric else self.obs_dim
        self.value_net = ValueNetwork(
            input_size=value_input_size,
            hidden_sizes=critic_hidden,
            activation_name=activation
        )

        # Initialize log standard deviation parameter
        std_init = float(initial_std)
        log_std_init = torch.log(torch.tensor(std_init))
        self.log_std = nn.Parameter(torch.full((self.action_dim,), log_std_init))

    def forward(self, *args, **kwargs):
        """Direct forward pass is not used."""
        raise NotImplementedError("Use act(), act_inference(), or evaluate() methods instead.")

    def _compute_action_mean(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute action mean from observations."""
        return self.policy_net(observations)

    def _compute_state_value(self, observations: torch.Tensor, states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute state value estimate."""
        if self.use_asymmetric and states is not None:
            return self.value_net(states)
        return self.value_net(observations)

    def _create_action_distribution(self, mean: torch.Tensor) -> MultivariateNormal:
        """Create multivariate normal distribution for action sampling."""
        # Compute standard deviation
        std = torch.exp(self.log_std)
        
        # Create diagonal covariance matrix
        # For MultivariateNormal with diagonal covariance, we use scale_tril
        # which is the lower triangular Cholesky factor of the covariance matrix
        # For diagonal case, this is just a diagonal matrix with std values
        scale_tril = torch.diag(std)
        
        return MultivariateNormal(mean, scale_tril=scale_tril)

    def act(self, observations: torch.Tensor, states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from policy during training.
        
        Returns:
            actions: Sampled actions
            log_probs: Log probabilities of actions
            values: State value estimates
            means: Action means
            stds: Action standard deviations
        """
        # Compute action mean
        action_mean = self._compute_action_mean(observations)
        
        # Create distribution and sample
        action_dist = self._create_action_distribution(action_mean)
        sampled_actions = action_dist.sample()
        action_log_probs = action_dist.log_prob(sampled_actions)
        
        # Compute state values
        state_values = self._compute_state_value(observations, states)
        
        # Get standard deviation for return
        std_values = torch.exp(self.log_std)
        std_expanded = std_values.unsqueeze(0).expand(action_mean.shape[0], -1)
        
        # Detach all outputs for training stability
        return (
            sampled_actions.detach(),
            action_log_probs.detach(),
            state_values.detach(),
            action_mean.detach(),
            std_expanded.detach()
        )

    def act_inference(self, observations: torch.Tensor, act_value: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Get deterministic actions for evaluation/testing.
        
        Args:
            observations: Input observations
            act_value: Whether to also return value estimates
            
        Returns:
            actions: Deterministic actions (means)
            values: State value estimates (if act_value=True, else None)
        """
        # Use mean as deterministic action
        deterministic_actions = self._compute_action_mean(observations)
        
        if act_value:
            state_values = self._compute_state_value(observations)
            return deterministic_actions.detach(), state_values.detach()
        else:
            return deterministic_actions.detach(), None

    def evaluate(self, observations: torch.Tensor, states: Optional[torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate policy and value function on given observations and actions.
        
        Args:
            observations: Input observations
            states: Optional state information (for asymmetric case)
            actions: Actions to evaluate
            
        Returns:
            log_probs: Log probabilities of actions
            entropy: Entropy of action distribution
            values: State value estimates
            means: Action means
            stds: Action standard deviations
        """
        # Compute action mean
        action_mean = self._compute_action_mean(observations)
        
        # Create distribution
        action_dist = self._create_action_distribution(action_mean)
        
        # Evaluate given actions
        action_log_probs = action_dist.log_prob(actions)
        action_entropy = action_dist.entropy()
        
        # Compute state values
        state_values = self._compute_state_value(observations, states)
        
        # Get standard deviation
        std_values = torch.exp(self.log_std)
        std_expanded = std_values.unsqueeze(0).expand(action_mean.shape[0], -1)
        
        return action_log_probs, action_entropy, state_values, action_mean, std_expanded
    
    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict with backward compatibility for old model formats.
        
        This method handles conversion from old model format (using 'actor' and 'critic')
        to new format (using 'policy_net' and 'value_net').
        """
        # Check if this is an old format model
        old_format = any(key.startswith('actor.') or key.startswith('critic.') for key in state_dict.keys())
        
        if old_format:
            # Convert old format to new format
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('actor.'):
                    # Convert actor.X.weight/bias to policy_net.layers.Y.weight/bias or policy_net.output_layer.weight/bias
                    new_key = self._convert_actor_key(key)
                    new_state_dict[new_key] = value
                elif key.startswith('critic.'):
                    # Convert critic.X.weight/bias to value_net.layers.Y.weight/bias or value_net.output_layer.weight/bias
                    new_key = self._convert_critic_key(key)
                    new_state_dict[new_key] = value
                else:
                    # Keep other keys as-is (e.g., log_std)
                    new_state_dict[key] = value
            
            # Load the converted state dict
            return super().load_state_dict(new_state_dict, strict=strict)
        else:
            # New format, load directly
            return super().load_state_dict(state_dict, strict=strict)
    
    def _convert_actor_key(self, old_key: str) -> str:
        """Convert old actor key to new policy_net key format.
        
        Old format uses Sequential with alternating Linear and Activation layers:
        - actor.0 = first Linear layer
        - actor.2 = second Linear layer
        - actor.4 = third Linear layer
        - actor.6 = fourth Linear layer (if exists)
        - actor.8 = output Linear layer
        """
        parts = old_key.split('.')
        layer_idx = int(parts[1])
        param_type = parts[2]  # 'weight' or 'bias'
        
        # Old format: indices are 0, 2, 4, 6, 8 (even numbers, skipping activation layers)
        # Convert to actual layer index: 0->0, 2->1, 4->2, 6->3, 8->output
        actual_layer_idx = layer_idx // 2
        
        num_hidden_layers = len(self.policy_net.layers)
        
        if actual_layer_idx < num_hidden_layers:
            # Hidden layer
            return f'policy_net.layers.{actual_layer_idx}.{param_type}'
        else:
            # Output layer
            return f'policy_net.output_layer.{param_type}'
    
    def _convert_critic_key(self, old_key: str) -> str:
        """Convert old critic key to new value_net key format.
        
        Old format uses Sequential with alternating Linear and Activation layers:
        - critic.0 = first Linear layer
        - critic.2 = second Linear layer
        - critic.4 = third Linear layer
        - critic.6 = fourth Linear layer (if exists)
        - critic.8 = output Linear layer
        """
        parts = old_key.split('.')
        layer_idx = int(parts[1])
        param_type = parts[2]  # 'weight' or 'bias'
        
        # Old format: indices are 0, 2, 4, 6, 8 (even numbers, skipping activation layers)
        # Convert to actual layer index: 0->0, 2->1, 4->2, 6->3, 8->output
        actual_layer_idx = layer_idx // 2
        
        num_hidden_layers = len(self.value_net.layers)
        
        if actual_layer_idx < num_hidden_layers:
            # Hidden layer
            return f'value_net.layers.{actual_layer_idx}.{param_type}'
        else:
            # Output layer
            return f'value_net.output_layer.{param_type}'
