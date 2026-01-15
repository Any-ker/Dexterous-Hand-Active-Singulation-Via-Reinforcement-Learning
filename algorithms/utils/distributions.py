import torch
import torch.nn as nn


class _GaussianDistribution:
    """Simple helper so callers can ask for sample/mode/log_probs/entropy."""

    def __init__(self, mean, std):
        self.dist = torch.distributions.Normal(mean, std)

    def sample(self):
        return self.dist.rsample()

    def mode(self):
        return self.dist.mean

    def log_probs(self, actions):
        return self.dist.log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return self.dist.entropy().sum(-1, keepdim=True)

    @property
    def probs(self):
        # Not a true probability for continuous spaces, but keeps legacy calls safe.
        return self.dist.mean


class DiagGaussian(nn.Module):
    """Diagonal Gaussian head tailored for our continuous Dexterous Hand policy."""

    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, config=None):
        super().__init__()
        init_fn = nn.init.orthogonal_ if use_orthogonal else nn.init.xavier_uniform_
        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        init_fn(self.fc_mean.weight, gain=gain)
        nn.init.zeros_(self.fc_mean.bias)

        init_std = 0.5 if config is None else config.get("std_init", 0.5)
        self.log_std = nn.Parameter(torch.ones(num_outputs) * init_std)

    def forward(self, x, available_actions=None):
        mean = self.fc_mean(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return _GaussianDistribution(mean, std)







