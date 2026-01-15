import torch
from torch.utils.data import BatchSampler, SequentialSampler


class RolloutStorage:
    """Simplified rollout buffer used by PPO."""

    def __init__(self, num_envs, num_steps, obs_shape, states_shape, actions_shape, device="cpu", sampler="sequential"):
        self.device = device
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.sampler = sampler

        shape = (num_steps, num_envs)
        self.observations = torch.zeros(*shape, *obs_shape, device=device)
        self.states = torch.zeros(*shape, *states_shape, device=device)
        self.rewards = torch.zeros(*shape, 1, device=device)
        self.actions = torch.zeros(*shape, *actions_shape, device=device)
        self.dones = torch.zeros(*shape, 1, dtype=torch.bool, device=device)
        self.values = torch.zeros(*shape, 1, device=device)
        self.log_probs = torch.zeros(*shape, 1, device=device)
        self.returns = torch.zeros(*shape, 1, device=device)
        self.advantages = torch.zeros(*shape, 1, device=device)
        # Additional fields for PPO
        self.actions_log_prob = torch.zeros(*shape, 1, device=device)
        self.mu = torch.zeros(*shape, *actions_shape, device=device)
        self.sigma = torch.zeros(*shape, *actions_shape, device=device)

        self.step = 0

    def add(self, obs, states, actions, rewards, dones, values, log_probs):
        if self.step >= self.num_steps:
            raise RuntimeError("Rollout buffer overflow")
        self.observations[self.step].copy_(obs)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.log_probs[self.step].copy_(log_probs.view(-1, 1))
        self.step += 1

    def add_transitions(self, obs, states, actions, rewards, dones, values, actions_log_prob, mu, sigma):
        """Add transition data including mu and sigma for PPO."""
        if self.step >= self.num_steps:
            raise RuntimeError("Rollout buffer overflow")
        self.observations[self.step].copy_(obs)
        # Handle states shape mismatch: only copy if shapes match
        if states.shape == self.states[self.step].shape:
            self.states[self.step].copy_(states)
        elif states.numel() > 0 and self.states[self.step].numel() > 0:
            # Try to reshape if possible
            if states.shape[-1] == self.states[self.step].shape[-1]:
                self.states[self.step].copy_(states)
            # Otherwise skip copy (states may be empty or shape mismatch)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = torch.zeros_like(last_values)
        for t in reversed(range(self.num_steps)):
            next_value = last_values if t == self.num_steps - 1 else self.values[t + 1]
            mask = 1.0 - self.dones[t].float()
            delta = self.rewards[t] + gamma * next_value * mask - self.values[t]
            advantage = delta + gamma * lam * advantage * mask
            self.returns[t] = advantage + self.values[t]
        self.advantages = self.returns - self.values
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std().clamp(min=1e-8)
        self.advantages = (self.advantages - adv_mean) / adv_std

    def mini_batch_generator(self, num_mini_batches):
        batch_size = self.num_envs * self.num_steps
        mini_batch_size = batch_size // num_mini_batches
        indices = range(batch_size)
        sampler = SequentialSampler(indices)
        return BatchSampler(sampler, mini_batch_size, drop_last=True)

    def get_statistics(self):
        """Get mean trajectory length and mean reward."""
        mean_length = self.step
        mean_reward = self.rewards[:self.step].mean().item()
        return mean_length, mean_reward
