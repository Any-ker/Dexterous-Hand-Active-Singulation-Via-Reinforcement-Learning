import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNNLayer(nn.Module):
    """Very small CNN block: Conv -> ReLU -> Linear -> ReLU."""

    def __init__(self, obs_shape, hidden_size, use_orthogonal=True, use_relu=True, kernel_size=3, stride=1):
        super().__init__()
        activation = nn.ReLU() if use_relu else nn.Tanh()
        conv = nn.Conv2d(obs_shape[0], hidden_size // 2, kernel_size=kernel_size, stride=stride)
        linear = nn.Linear((hidden_size // 2) * (obs_shape[1] - kernel_size + stride) * (obs_shape[2] - kernel_size + stride), hidden_size)

        if use_orthogonal:
            nn.init.orthogonal_(conv.weight)
            nn.init.zeros_(conv.bias)
            nn.init.orthogonal_(linear.weight)
            nn.init.zeros_(linear.bias)

        self.net = nn.Sequential(
            conv,
            activation,
            Flatten(),
            linear,
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
        )

    def forward(self, x):
        return self.net(x / 255.0)


class CNNBase(nn.Module):
    def __init__(self, args, obs_shape):
        super().__init__()
        self.cnn = CNNLayer(obs_shape, args.hidden_size, args.use_orthogonal, args.use_ReLU)

    def forward(self, x):
        return self.cnn(x)
