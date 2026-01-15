from .distributions import DiagGaussian
import torch.nn as nn


class ACTLayer(nn.Module):
    """Extremely small action head: DexGrasp only needs continuous (Box) outputs."""

    def __init__(self, action_space, inputs_dim, use_orthogonal=True, gain=0.01, args=None):
        super().__init__()
        if action_space.__class__.__name__ != "Box":
            raise ValueError("Current policy only supports continuous Box action spaces.")

        action_dim = action_space.shape[0]
        self.dist_head = DiagGaussian(inputs_dim, action_dim, use_orthogonal, gain, args)

    def forward(self, x, available_actions=None, deterministic=False):
        dist = self.dist_head(x, available_actions)
        actions = dist.mode() if deterministic else dist.sample()
        log_probs = dist.log_probs(actions)
        return actions, log_probs

    def get_probs(self, x, available_actions=None):
        # For Gaussians, returning the mean is often good enough for debugging.
        return self.dist_head(x, available_actions).probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        dist = self.dist_head(x, available_actions)
        log_probs = dist.log_probs(action)
        entropy = dist.entropy()
        if active_masks is not None:
            entropy = (entropy * active_masks).sum() / active_masks.sum()
        else:
            entropy = entropy.mean()
        return log_probs, entropy
