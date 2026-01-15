# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import torch
from isaacgym import gymapi


class BaseTask:
    """Small helper that sets up Isaac Gym, buffers, and optional viewers."""

    def __init__(self, cfg, enable_camera_sensors=False, is_meta=False, task_num=0):
        self.id = -1
        self.gym = gymapi.acquire_gym()

        # ------------------------------------------------------------------ #
        # Device and viewer setup
        # ------------------------------------------------------------------ #
        self.device_id = cfg.get("device_id", 0)
        device_type = cfg.get("device_type", "cuda").lower()
        self.device = f"cuda:{self.device_id}" if device_type in ("cuda", "gpu") else "cpu"

        self.headless = cfg.get("headless", True)
        self.enable_camera_sensors = enable_camera_sensors
        self.graphics_device_id = cfg.get("graphics_device_id", self.device_id)
        if self.headless and not self.enable_camera_sensors:
            self.graphics_device_id = -1

        # ------------------------------------------------------------------ #
        # Environment sizes
        # ------------------------------------------------------------------ #
        self.num_envs = cfg["env"]["numEnvs"]
        if is_meta:
            self.num_envs *= task_num
        self.num_obs = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]
        self.control_freq_inv = cfg["env"].get("controlFrequencyInv", 1)

        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # ------------------------------------------------------------------ #
        # Buffers shared with child classes
        # ------------------------------------------------------------------ #
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device)
        self.states_buf = torch.zeros((self.num_envs, self.num_states), device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.object_id_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        # Isaac Gym simulation
        self.create_sim()
        self.gym.prepare_sim(self.sim)

        self.viewer = None
        self.enable_viewer_sync = True
        if not self.headless:
            self._create_viewer()

    # ------------------------------------------------------------------ #
    # Helper utilities
    # ------------------------------------------------------------------ #
    def _create_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "TOGGLE_V")

        cam_pos = gymapi.Vec3(10.0, 10.0, 3.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def set_sim_params_up_axis(self, sim_params, axis):
        """Set gravity for the desired up-axis. Returns the axis index used by Gym."""
        if axis == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
            return 2
        return 1

    def create_sim(self, compute_device, graphics_device, physics_engine, sim_params):
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            raise RuntimeError("Failed to create Isaac Gym simulation.")
        return sim

    # ------------------------------------------------------------------ #
    # Simulation control
    # ------------------------------------------------------------------ #
    def step(self, actions, id):
        """Apply actions, step physics, then let the task update buffers."""
        self.id = id
        self.pre_physics_step(actions)

        for _ in range(self.control_freq_inv):
            self.render()
            self.gym.simulate(self.sim)

        if self.device == "cpu":
            self.gym.fetch_results(self.sim, True)

        self.post_physics_step()

    def get_states(self):
        return self.states_buf

    def render(self, sync_frame_time=False):
        """Update the viewer (if visible) and keep camera sensors fresh."""
        if self.viewer:
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                if evt.action == "TOGGLE_V" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)

            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)
        elif self.enable_camera_sensors:
            if self.device != "cpu":
                self.gym.fetch_results(self.sim, True)
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)

    # ------------------------------------------------------------------ #
    # Abstract hooks for derived classes
    # ------------------------------------------------------------------ #
    def pre_physics_step(self, actions):
        raise NotImplementedError

    def post_physics_step(self):
        raise NotImplementedError







