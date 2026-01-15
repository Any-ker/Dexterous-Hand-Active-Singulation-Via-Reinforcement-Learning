# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from gym import spaces
import numpy as np
import torch


class VecTask:
    """A tiny base wrapper that keeps track of task sizes and clipping limits."""

    def __init__(self, task, rl_device, clip_observations=5.0, clip_actions=1.0):
        self.task = task
        self.rl_device = rl_device
        self.clip_obs = clip_observations
        self.clip_actions = clip_actions

        self.num_environments = task.num_envs
        self.num_agents = 1
        self.num_observations = task.num_obs
        self.num_states = task.num_states
        self.num_actions = task.num_actions

        inf_obs = np.ones(self.num_observations) * np.inf
        self.obs_space = spaces.Box(-inf_obs, inf_obs)
        # Ensure state_space shape matches num_states
        if self.num_states > 0:
            inf_state = np.ones(self.num_states) * np.inf
            self.state_space = spaces.Box(-inf_state, inf_state)
        else:
            # If num_states is 0, create a dummy space with shape (0,)
            # Note: gym Box doesn't support empty shape, so we use shape (1,) but handle it specially
            inf_state = np.ones(1) * np.inf
            self.state_space = spaces.Box(-inf_state, inf_state)
        self.act_space = spaces.Box(
            low=np.ones(self.num_actions) * -1.0,
            high=np.ones(self.num_actions) * 1.0,
        )

        print("RL device:", rl_device)

    def step(self, actions):
        raise NotImplementedError("VecTask subclasses must implement step().")

    def reset(self):
        raise NotImplementedError("VecTask subclasses must implement reset().")

    def get_number_of_agents(self):
        return self.num_agents

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @property
    def num_envs(self):
        return self.num_environments

    @property
    def num_acts(self):
        return self.num_actions

    @property
    def num_obs(self):
        return self.num_observations


class VecTaskPython(VecTask):
    """Simple Python vec wrapper used by PPO."""

    def get_state(self):
        return torch.clamp(self.task.states_buf, -self.clip_obs, self.clip_obs).to(
            self.rl_device
        )

    def step(self, actions, id=0):
        clipped_actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        self.task.step(clipped_actions, id)

        obs = torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(
            self.rl_device
        )
        rewards = self.task.rew_buf.to(self.rl_device)
        resets = self.task.reset_buf.to(self.rl_device)
        extras = self.task.extras
        return obs, rewards, resets, extras

    def reset(self):
        zero_actions = torch.zeros(
            (self.task.num_envs, self.task.num_actions),
            dtype=torch.float32,
            device=self.rl_device,
        )
        self.task.step(zero_actions, id=-1)
        return torch.clamp(self.task.obs_buf, -self.clip_obs, self.clip_obs).to(
            self.rl_device
        )