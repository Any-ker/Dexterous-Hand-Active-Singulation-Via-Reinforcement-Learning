

import torch
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional, List
import math


class CustomLinear:
    """Custom linear layer implementation without nn.Linear."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using Xavier uniform
        bound = math.sqrt(6.0 / (in_features + out_features))
        self.weight = torch.empty(out_features, in_features).uniform_(-bound, bound)
        self.weight = torch.nn.Parameter(self.weight)
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: y = xW^T + b"""
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output = output + self.bias
        return output


class CustomLayerNorm:
    """Custom layer normalization implementation."""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Layer normalization: (x - mean) / sqrt(var + eps) * weight + bias"""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class MultiHeadSelfAttention:
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = dropout
        
        # Query, Key, Value projections
        self.W_q = CustomLinear(d_model, d_model)
        self.W_k = CustomLinear(d_model, d_model)
        self.W_v = CustomLinear(d_model, d_model)
        self.W_o = CustomLinear(d_model, d_model)
        
        # Dropout
        self.dropout_layer = torch.nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Self-attention forward pass.
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        Q = self.W_q.forward(x)  # [batch, seq_len, d_model]
        K = self.W_k.forward(x)
        V = self.W_v.forward(x)
        
        # Reshape for multi-head: [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, d_k]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Output projection
        output = self.W_o.forward(attn_output)
        return output


class GatedFeedForward:
    """Gated feedforward network with GLU (Gated Linear Unit) mechanism."""
    
    def __init__(self, d_model: int, d_ff: int, activation: str = "gelu"):
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Two parallel linear layers for gating
        self.linear1 = CustomLinear(d_model, d_ff)
        self.linear2 = CustomLinear(d_model, d_ff)
        self.linear3 = CustomLinear(d_ff, d_model)
        
        if activation == "gelu":
            self.activation = lambda x: 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
        elif activation == "swish":
            self.activation = lambda x: x * torch.sigmoid(x)
        else:
            self.activation = lambda x: torch.maximum(torch.zeros_like(x), x)  # ReLU
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Gated feedforward: gate(x) * activation(linear(x))"""
        gate = self.linear1.forward(x)
        value = self.linear2.forward(x)
        gated = gate * self.activation(value)
        return self.linear3.forward(gated)


class TransformerBlock:
    """Transformer block with self-attention and gated feedforward."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.ffn = GatedFeedForward(d_model, d_ff)
        self.norm1 = CustomLayerNorm(d_model)
        self.norm2 = CustomLayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Transformer block with residual connections."""
        # Self-attention with residual
        attn_output = self.attention.forward(x, mask)
        if self.dropout is not None:
            attn_output = self.dropout(attn_output)
        x = self.norm1.forward(x + attn_output)
        
        # Feedforward with residual
        ffn_output = self.ffn.forward(x)
        if self.dropout is not None:
            ffn_output = self.dropout(ffn_output)
        x = self.norm2.forward(x + ffn_output)
        
        return x


class AdaptiveGatingUnit:
    """Adaptive gating unit for feature selection."""
    
    def __init__(self, d_model: int):
        self.gate_linear = CustomLinear(d_model, d_model)
        self.transform_linear = CustomLinear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adaptive gating: gate * transform(x) + (1 - gate) * x"""
        gate = torch.sigmoid(self.gate_linear.forward(x))
        transformed = self.transform_linear.forward(x)
        return gate * transformed + (1 - gate) * x


class AdvancedActorNetwork:
    """Advanced actor network with attention and gating mechanisms."""
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 512, 256],
        num_attention_heads: int = 8,
        use_attention: bool = True,
        use_gating: bool = True,
        dropout: float = 0.1
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.use_attention = use_attention
        self.use_gating = use_gating
        
        # Input projection
        self.input_proj = CustomLinear(obs_dim, hidden_dims[0])
        self.input_norm = CustomLayerNorm(hidden_dims[0])
        
        # Transformer blocks for attention
        self.transformer_blocks = torch.nn.ModuleList()
        if use_attention:
            for _ in range(2):  # 2 transformer blocks
                self.transformer_blocks.append(
                    TransformerBlock(
                        d_model=hidden_dims[0],
                        num_heads=num_attention_heads,
                        d_ff=hidden_dims[0] * 4,
                        dropout=dropout
                    )
                )
        
        # Hidden layers with adaptive gating
        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_norms = torch.nn.ModuleList()
        self.gating_units = torch.nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                CustomLinear(hidden_dims[i], hidden_dims[i + 1])
            )
            self.hidden_norms.append(CustomLayerNorm(hidden_dims[i + 1]))
            if use_gating:
                self.gating_units.append(AdaptiveGatingUnit(hidden_dims[i + 1]))
        
        # Output layer
        self.output_layer = CustomLinear(hidden_dims[-1], action_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Custom weight initialization."""
        # Output layer with small initialization
        with torch.no_grad():
            bound = 0.01
            self.output_layer.weight.uniform_(-bound, bound)
            if self.output_layer.bias is not None:
                self.output_layer.bias.zero_()
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass through actor network."""
        # Input projection
        x = self.input_proj.forward(obs)
        x = self.input_norm.forward(x)
        
        # Add sequence dimension for attention (treat batch as sequence)
        if self.use_attention and len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch, 1, hidden]
        
        # Transformer blocks
        if self.use_attention:
            for transformer in self.transformer_blocks:
                x = transformer.forward(x)
            x = x.squeeze(1) if x.shape[1] == 1 else x.mean(dim=1)  # Remove seq dim
        
        # Hidden layers with gating
        for i, (layer, norm) in enumerate(zip(self.hidden_layers, self.hidden_norms)):
            residual = x if i == 0 and x.shape[-1] == layer.out_features else None
            x = layer.forward(x)
            x = norm.forward(x)
            x = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))  # GELU
            
            # Adaptive gating
            if self.use_gating and i < len(self.gating_units):
                x = self.gating_units[i].forward(x)
            
            # Residual connection if dimensions match
            if residual is not None and x.shape == residual.shape:
                x = x + residual
        
        # Output layer
        x = self.output_layer.forward(x)
        return x


class AdvancedCriticNetwork:
    """Advanced critic network with attention and gating mechanisms."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 512, 256],
        num_attention_heads: int = 8,
        use_attention: bool = True,
        use_gating: bool = True,
        dropout: float = 0.1
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_attention = use_attention
        self.use_gating = use_gating
        
        # Input projection
        self.input_proj = CustomLinear(input_dim, hidden_dims[0])
        self.input_norm = CustomLayerNorm(hidden_dims[0])
        
        # Transformer blocks
        self.transformer_blocks = torch.nn.ModuleList()
        if use_attention:
            for _ in range(2):
                self.transformer_blocks.append(
                    TransformerBlock(
                        d_model=hidden_dims[0],
                        num_heads=num_attention_heads,
                        d_ff=hidden_dims[0] * 4,
                        dropout=dropout
                    )
                )
        
        # Hidden layers
        self.hidden_layers = torch.nn.ModuleList()
        self.hidden_norms = torch.nn.ModuleList()
        self.gating_units = torch.nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            self.hidden_layers.append(
                CustomLinear(hidden_dims[i], hidden_dims[i + 1])
            )
            self.hidden_norms.append(CustomLayerNorm(hidden_dims[i + 1]))
            if use_gating:
                self.gating_units.append(AdaptiveGatingUnit(hidden_dims[i + 1]))
        
        # Output layer (single value)
        self.output_layer = CustomLinear(hidden_dims[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Custom weight initialization."""
        with torch.no_grad():
            bound = 0.1
            self.output_layer.weight.uniform_(-bound, bound)
            if self.output_layer.bias is not None:
                self.output_layer.bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through critic network."""
        # Input projection
        x = self.input_proj.forward(x)
        x = self.input_norm.forward(x)
        
        # Add sequence dimension for attention
        if self.use_attention and len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        # Transformer blocks
        if self.use_attention:
            for transformer in self.transformer_blocks:
                x = transformer.forward(x)
            x = x.squeeze(1) if x.shape[1] == 1 else x.mean(dim=1)
        
        # Hidden layers
        for i, (layer, norm) in enumerate(zip(self.hidden_layers, self.hidden_norms)):
            residual = x if i == 0 and x.shape[-1] == layer.out_features else None
            x = layer.forward(x)
            x = norm.forward(x)
            x = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))  # GELU
            
            if self.use_gating and i < len(self.gating_units):
                x = self.gating_units[i].forward(x)
            
            if residual is not None and x.shape == residual.shape:
                x = x + residual
        
        # Output value
        return self.output_layer.forward(x)


class AdvancedActorCritic(torch.nn.Module):
    """
    Advanced Actor-Critic network with custom implementations.
    
    Features:
    - Multi-head self-attention mechanism
    - Gated feedforward networks (GLU)
    - Adaptive gating units
    - Custom layer normalization
    - All layers implemented from scratch (no torch.nn native modules)
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        state_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        initial_std: float = 0.5,
        actor_hidden_dims: List[int] = [512, 512, 256],
        critic_hidden_dims: List[int] = [512, 512, 256],
        num_attention_heads: int = 8,
        use_attention: bool = True,
        use_gating: bool = True,
        dropout: float = 0.1,
        asymmetric: bool = False,
        learnable_std: bool = True
    ):
        super().__init__()
        
        self.obs_dim = obs_shape[0]
        self.state_dim = state_shape[0] if state_shape else obs_shape[0]
        self.action_dim = action_shape[0]
        self.asymmetric = asymmetric
        self.learnable_std = learnable_std
        
        # Build actor network
        self.actor = AdvancedActorNetwork(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=actor_hidden_dims,
            num_attention_heads=num_attention_heads,
            use_attention=use_attention,
            use_gating=use_gating,
            dropout=dropout
        )
        
        # Build critic network
        critic_input_dim = self.state_dim if asymmetric else self.obs_dim
        self.critic = AdvancedCriticNetwork(
            input_dim=critic_input_dim,
            hidden_dims=critic_hidden_dims,
            num_attention_heads=num_attention_heads,
            use_attention=use_attention,
            use_gating=use_gating,
            dropout=dropout
        )
        
        # Action standard deviation
        if learnable_std:
            log_std_init = math.log(initial_std)
            self.log_std = torch.nn.Parameter(
                torch.full((self.action_dim,), log_std_init)
            )
        else:
            self.register_buffer(
                'log_std',
                torch.full((self.action_dim,), math.log(initial_std))
            )
    
    def get_action_mean(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute action mean from observations."""
        return self.actor.forward(observations)
    
    def get_value(self, observations: torch.Tensor, states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute state value estimate."""
        if self.asymmetric and states is not None:
            return self.critic.forward(states)
        return self.critic.forward(observations)
    
    def get_action_distribution(self, action_mean: torch.Tensor) -> Normal:
        """Create action distribution from mean."""
        std = torch.exp(self.log_std).clamp(min=1e-6, max=2.0)
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
        
        Returns:
            actions, log_probs, values, means, stds
        """
        action_mean = self.get_action_mean(observations)
        action_dist = self.get_action_distribution(action_mean)
        
        if deterministic:
            actions = action_mean
            log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        else:
            actions = action_dist.sample()
            log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        values = self.get_value(observations, states)
        stds = torch.exp(self.log_std).unsqueeze(0).expand(actions.shape[0], -1)
        
        return actions, log_probs, values, action_mean, stds
    
    def evaluate(
        self,
        observations: torch.Tensor,
        states: Optional[torch.Tensor],
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate policy and value function.
        
        Returns:
            log_probs, entropy, values, means, stds
        """
        action_mean = self.get_action_mean(observations)
        action_dist = self.get_action_distribution(action_mean)
        
        log_probs = action_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = action_dist.entropy().sum(dim=-1, keepdim=True)
        values = self.get_value(observations, states)
        stds = torch.exp(self.log_std).unsqueeze(0).expand(actions.shape[0], -1)
        
        return log_probs, entropy, values, action_mean, stds
    
    def forward(self, *args, **kwargs):
        """Direct forward is not used."""
        raise NotImplementedError("Use act() or evaluate() instead.")

