import torch.nn as nn
from .util import simple_linear


class MLPLayer(nn.Module):
    """Tiny fully connected block used by the policy/value networks."""

    def __init__(self, input_dim, hidden_size, num_layers, use_orthogonal=True, use_relu=True):
        super().__init__()
        activation = nn.ReLU() if use_relu else nn.Tanh()
        layers = [simple_linear(input_dim, hidden_size, use_orthogonal), activation]
        for _ in range(num_layers):
            layers.append(simple_linear(hidden_size, hidden_size, use_orthogonal))
            layers.append(activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPBase(nn.Module):
    def __init__(self, config, obs_shape):
        super().__init__()
        self.normalize = config.get("use_feature_normalization", False)
        obs_dim = obs_shape[0]
        if self.normalize:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = MLPLayer(
            obs_dim,
            config["hidden_size"],
            config["layer_N"],
            config.get("use_orthogonal", True),
            config.get("use_ReLU", True),
        )

    def forward(self, x):
        if self.normalize:
            x = self.feature_norm(x)
        return self.mlp(x)
