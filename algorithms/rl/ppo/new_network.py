

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional


class ResidualBlock(nn.Module):
    """Residual block with layer normalization for stable training."""
    
    def __init__(self, hidden_dim: int, activation: str = "relu"):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Activation function
        if activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "swish":
            self.activation = nn.SiLU()  # SiLU is Swish
        else:
            self.activation = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier uniform initialization for better gradient flow."""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.norm1(x)
        x = self.activation(self.linear1(x))
        x = self.norm2(x)
        x = self.linear2(x)
        return self.activation(x + residual)  # Residual connection


class ActorNetwork(nn.Module):
    """Actor network with residual blocks for policy learning."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list = [512, 512, 256],
        activation: str = "gelu",
        use_residual: bool = True,
        output_activation: Optional[str] = None
    ):
        super().__init__()
        self.use_residual = use_residual
        self.hidden_dims = hidden_dims
        
        # Input projection
        self.input_proj = nn.Linear(obs_dim, hidden_dims[0])
        self.input_norm = nn.LayerNorm(hidden_dims[0])
        
        # Hidden layers with residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if use_residual:
                self.residual_blocks.append(
                    ResidualBlock(hidden_dims[i], activation)
                )
            else:
                # Simple feedforward if not using residual
                block = nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.LayerNorm(hidden_dims[i+1]),
                    nn.GELU() if activation == "gelu" else nn.ReLU()
                )
                self.residual_blocks.append(block)
        
        # Output layer for action mean
        self.output_layer = nn.Linear(hidden_dims[-1], action_dim)
        
        # Output activation
        if output_activation == "tanh":
            self.output_act = nn.Tanh()
        elif output_activation == "sigmoid":
            self.output_act = nn.Sigmoid()
        else:
            self.output_act = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        nn.init.zeros_(self.input_proj.bias)
        
        # Output layer with smaller initialization for stability
        nn.init.orthogonal_(self.output_layer.weight, gain=0.01)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network."""
        x = self.input_proj(obs)
        x = self.input_norm(x)
        x = F.gelu(x)  # GELU activation
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            if self.use_residual:
                x = block(x)
            else:
                x = block(x)
        
        # Output layer
        x = self.output_layer(x)
        if self.output_act is not None:
            x = self.output_act(x)
        
        return x


class CriticNetwork(nn.Module):
    """Critic network with residual blocks for value estimation."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 512, 256],
        activation: str = "gelu",
        use_residual: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        self.input_norm = nn.LayerNorm(hidden_dims[0])
        
        # Hidden layers with residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if use_residual:
                self.residual_blocks.append(
                    ResidualBlock(hidden_dims[i], activation)
                )
            else:
                block = nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.LayerNorm(hidden_dims[i+1]),
                    nn.GELU() if activation == "gelu" else nn.ReLU()
                )
                self.residual_blocks.append(block)
        
        # Output layer (single value)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        nn.init.xavier_uniform_(self.input_proj.weight, gain=0.5)
        nn.init.zeros_(self.input_proj.bias)
        
        # Output layer with small initialization
        nn.init.orthogonal_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network."""
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        # Pass through residual blocks
        for block in self.residual_blocks:
            if self.use_residual:
                x = block(x)
            else:
                x = block(x)
        
        # Output value
        return self.output_layer(x)


class ModernActorCritic(nn.Module):
    """
    Modern Actor-Critic network with residual connections and layer normalization.
    
    Features:
    - Residual blocks for deeper networks
    - Layer normalization for training stability
    - Independent action std parameters
    - Support for both symmetric and asymmetric architectures
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        state_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        initial_std: float = 0.5,
        actor_hidden_dims: list = [512, 512, 256],
        critic_hidden_dims: list = [512, 512, 256],
        activation: str = "gelu",
        use_residual: bool = True,
        asymmetric: bool = False,
        learnable_std: bool = True,
        output_activation: Optional[str] = None
    ):
        super().__init__()
        
        self.obs_dim = obs_shape[0]
        self.state_dim = state_shape[0] if state_shape else obs_shape[0]
        self.action_dim = action_shape[0]
        self.asymmetric = asymmetric
        self.learnable_std = learnable_std
        
        # Build actor network
        self.actor = ActorNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=actor_hidden_dims,
            activation=activation,
            use_residual=use_residual,
            output_activation=output_activation
        )
        
        # Build critic network
        critic_input_dim = self.state_dim if asymmetric else self.obs_dim
        self.critic = CriticNetwork(
            input_dim=critic_input_dim,
            hidden_dims=critic_hidden_dims,
            activation=activation,
            use_residual=use_residual
        )
        
        # Action standard deviation (learnable or fixed)
        if learnable_std:
            # Learnable log std as parameter
            log_std_init = torch.log(torch.tensor(initial_std, dtype=torch.float32))
            self.log_std = nn.Parameter(
                torch.full((self.action_dim,), log_std_init.item())
            )
        else:
            # Fixed std
            self.register_buffer(
                'log_std',
                torch.full((self.action_dim,), torch.log(torch.tensor(initial_std)))
            )
    
    def get_action_mean(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute action mean from observations."""
        return self.actor(observations)
    
    def get_value(self, observations: torch.Tensor, states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute state value estimate."""
        if self.asymmetric and states is not None:
            return self.critic(states)
        return self.critic(observations)
    
    def get_action_distribution(self, action_mean: torch.Tensor) -> Normal:
        """Create action distribution from mean."""
        std = torch.exp(self.log_std).clamp(min=1e-6, max=2.0)  # Clamp for stability
        # Expand std to match batch size
        if std.dim() == 1:
            std = std.unsqueeze(0).expand(action_mean.shape[0], -1)
        return Normal(action_mean, std)
    
    def act(
        self,
        observations: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample actions from policy.
        
        Args:
            observations: Input observations
            states: Optional state information (for asymmetric case)
            deterministic: If True, return mean action; if False, sample from distribution
            
        Returns:
            actions: Sampled or deterministic actions
            log_probs: Log probabilities of actions
            values: State value estimates
            means: Action means
            stds: Action standard deviations
        """
        # Compute action mean
        action_mean = self.get_action_mean(observations)
        
        # Create distribution
        action_dist = self.get_action_distribution(action_mean)
        
        # Sample or use mean
        if deterministic:
            actions = action_mean
            log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        else:
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # Compute values
        values = self.get_value(observations, states)
        
        # Get std
        stds = torch.exp(self.log_std).unsqueeze(0).expand(actions.shape[0], -1)
        
        return actions, log_probs, values, action_mean, stds
    
    def evaluate(
        self,
        observations: torch.Tensor,
        states: Optional[torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate policy and value function on given observations and actions.
        
        Args:
            observations: Input observations
            states: Optional state information
            actions: Actions to evaluate
            
        Returns:
            log_probs: Log probabilities of actions
            entropy: Entropy of action distribution
            values: State value estimates
            means: Action means
            stds: Action standard deviations
        """
        # Compute action mean
        action_mean = self.get_action_mean(observations)
        
        # Create distribution
        action_dist = self.get_action_distribution(action_mean)
        
        # Evaluate given actions
        log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = action_dist.entropy().sum(dim=-1, keepdim=True)
        
        # Compute values
        values = self.get_value(observations, states)
        
        # Get std
        stds = torch.exp(self.log_std).unsqueeze(0).expand(actions.shape[0], -1)
        
        return log_probs, entropy, values, action_mean, stds
    
    def forward(self, *args, **kwargs):
        """Direct forward is not used. Use act() or evaluate() instead."""
        raise NotImplementedError(
            "Use act() for action sampling or evaluate() for policy evaluation."
        )

