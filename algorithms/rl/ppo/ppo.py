import os
import yaml
import numpy as np
import os.path as osp
import statistics
import time
import glob
import wandb

from gym.spaces import Space
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils.general_utils import *
from algorithms.rl.ppo import RolloutStorage


class PPO:

    def __init__(self,
                 vec_env,
                 actor_critic_class,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 init_noise_std=1.0,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=None,
                 model_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False,
                 is_vision=False
                 ):
        # Store configuration
        self.is_vision = is_vision
        self.config = vec_env.task.config

        # Validate environment spaces
        self._validate_spaces(vec_env)
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space

        # Device and mode settings
        self.device = device
        self.asymmetric = asymmetric
        self.is_testing = is_testing
        self.apply_reset = apply_reset

        # Learning rate and schedule management
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.current_lr = learning_rate

        # Initialize core components
        self.vec_env = vec_env
        self.actor_critic = actor_critic_class(
            self.observation_space.shape,
            self.state_space.shape,
            self.action_space.shape,
            init_noise_std,
            self.config['Models'],
            asymmetric=asymmetric
        )
        self.actor_critic.to(self.device)
        
        # Experience buffer
        self.storage = RolloutStorage(
            self.vec_env.num_envs,
            num_transitions_per_env,
            self.observation_space.shape,
            self.state_space.shape,
            self.action_space.shape,
            self.device,
            sampler
        )
        
        # Optimizer
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # PPO hyperparameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Logging setup
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10) if self.print_log else None
        self.vec_env.task.log_dir = log_dir
        
        # Training statistics
        self.total_timesteps = 0
        self.total_time_elapsed = 0
        self.save_traj = False
        self.current_learning_iteration = 0

    def _validate_spaces(self, vec_env):
        """Validate that environment spaces are correct types."""
        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")

    def test(self, path):
        """Load model weights for testing."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint)
        self.actor_critic.eval()

    def load(self, path):
        """Load model weights for resuming training."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint)
        self.current_learning_iteration = 0
        self.actor_critic.train()

    def save(self, path):
        """Save current model weights."""
        torch.save(self.actor_critic.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        """Main training/testing loop."""
        # Setup for testing mode
        if self.is_testing:
            self._prepare_testing_mode()
        
        # Initialize environment
        frame_counter = -1
        if self.is_testing:
            self.vec_env.task.random_time = False
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()
        self.vec_env.task.is_testing = self.is_testing

        # Save configuration
        os.makedirs(self.log_dir, exist_ok=True)
        save_yaml(osp.join(self.log_dir, 'train.yaml'), self.vec_env.task.config)

        # Branch to testing or training
        if self.is_testing:
            self._execute_testing_loop(current_obs, frame_counter)
        else:
            self._execute_training_loop(num_learning_iterations, log_interval, current_obs, current_states, frame_counter)

    def _prepare_testing_mode(self):
        """Prepare environment for testing mode."""
        pc_feat_path = osp.join(self.log_dir, 'pc_feat.npy')
        if osp.exists(pc_feat_path):
            self.vec_env.task.config['Modes']['zero_object_visual_feature'] = False
            loaded_features = torch.tensor(np.load(pc_feat_path), device=self.device)
            num_repeats = self.vec_env.task.visual_feat_buf.shape[0]
            self.vec_env.task.visual_feat_buf = loaded_features.repeat(num_repeats, 1)

    def _setup_test_directories(self):
        """Setup directories for saving test results."""
        if self.config['Save']:
            save_name = 'results_trajectory_train' if self.config['Save_Train'] else 'results_trajectory_test'
            self.vec_env.task.render_folder = osp.join(
                self.log_dir.replace('results_train', save_name), 'trajectory'
            )
            if self.vec_env.task.config['Save_Render']:
                self.vec_env.task.render_folder = osp.join(
                    self.log_dir.replace('results_train', 'results_trajectory_render'), 'trajectory'
                )
                self.vec_env.task.pointcloud_folder = osp.join(
                    self.log_dir.replace('results_train', 'results_trajectory_render'), 'pointcloud'
                )
                os.makedirs(self.vec_env.task.pointcloud_folder, exist_ok=True)
        elif self.vec_env.task.render_folder is None:
            test_dirs = glob.glob(osp.join(self.log_dir, 'test_*'))
            test_idx = len(test_dirs)
            self.vec_env.task.render_folder = osp.join(self.log_dir, f'test_{test_idx}')
            os.makedirs(self.vec_env.task.render_folder, exist_ok=True)
        save_list_strings(
            os.path.join(self.vec_env.task.render_folder, 'env_object_scale.txt'),
            self.vec_env.task.env_object_scale
        )

    def _execute_testing_loop(self, initial_obs, frame_counter):
        """Execute the testing loop."""
        self._setup_test_directories()
        
        success_rates = []
        average_steps = []
        num_test_iterations = self.vec_env.task.test_iteration
        
        for test_idx in range(num_test_iterations):
            iteration_start = time.time()
            self.vec_env.task.current_test_iteration = test_idx + 1
            
            # Initialize trajectory storage
            pc_traj = self._init_pointcloud_trajectory()
            obs_traj = self._init_observation_trajectory()
            
            # Run test episode
            print(f'Testing iteration {self.log_dir.split("/")[-1]}/{self.device}//{test_idx}/{num_test_iterations}')
            
            # Adjust episode length for initialization
            if self.vec_env.task.config['Init']:
                self.vec_env.task.max_episode_length = 3
            
            obs = initial_obs.clone()
            for step in range(self.vec_env.task.max_episode_length):
                self.vec_env.task.frame = step
                with torch.no_grad():
                    if self.apply_reset:
                        obs = self.vec_env.reset()
                    frame_counter = (frame_counter + 1) % self.vec_env.task.max_episode_length
                    
                    # Get action from policy
                    actions, values = self.actor_critic.act_inference(
                        obs, act_value=self.vec_env.task.config['Save']
                    )
                    
                    # Store trajectory data
                    if obs_traj is not None:
                        self._append_observation_data(obs_traj, obs, actions, values)
                    if pc_traj is not None:
                        self._append_pointcloud_data(pc_traj)
                    
                    # Step environment
                    next_obs, rewards, dones, infos = self.vec_env.step(actions, frame_counter)
                    obs.copy_(next_obs)
                
                # Record success at second-to-last step
                if step == self.vec_env.task.max_episode_length - 2:
                    success_rate = self.vec_env.task.successes.sum() / self.vec_env.num_envs
                    if obs_traj is not None:
                        obs_traj['successes'] = self.vec_env.task.successes.clone().unsqueeze(-1)
                        obs_traj['final_successes'] = self.vec_env.task.final_successes.clone().unsqueeze(-1)
            
            # Update statistics
            self.vec_env.task.current_iteration += 1
            success_rates.append(success_rate.item())
            average_steps.append(self.vec_env.task.current_avg_steps)
            
            # Save statistics
            np.savetxt(
                os.path.join(self.vec_env.task.render_folder, 'final_success_rate.txt'),
                np.asarray(success_rates)
            )
            np.savetxt(
                os.path.join(self.vec_env.task.render_folder, 'final_average_step.txt'),
                np.asarray(average_steps)
            )
            
            # Save trajectories
            if obs_traj is not None:
                self._save_observation_trajectory(obs_traj)
            if pc_traj is not None:
                self._save_pointcloud_trajectory(pc_traj, obs_traj)
            
            elapsed_minutes = (time.time() - iteration_start) / 60
            print(f'process time: {elapsed_minutes}')
        
        # Final summary
        avg_success = np.mean(success_rates)
        print(f"Final success_rate: {self.log_dir.split('/')[-1]}/{avg_success:.3f}/{num_test_iterations}")
        exit()

    def _init_pointcloud_trajectory(self):
        """Initialize point cloud trajectory storage."""
        if self.vec_env.task.config['Save'] and self.vec_env.task.config['Save_Render']:
            return {
                'canonical': [], 'rendered': [], 'centers': [], 'appears': [],
                'features': [], 'pcas': [], 'hand_body': [], 'hand_object': []
            }
        return None

    def _init_observation_trajectory(self):
        """Initialize observation-action trajectory storage."""
        if self.vec_env.task.config['Save']:
            return {
                'observations': [], 'actions': [], 'values': [], 'states': [],
                'features': [], 'successes': None, 'final_successes': None
            }
        return None

    def _append_observation_data(self, traj, obs, actions, values):
        """Append observation data to trajectory."""
        traj['observations'].append(obs.clone())
        traj['actions'].append(actions.clone())
        traj['values'].append(values.clone())
        traj['features'].append(self.vec_env.task.object_points_visual_features.clone())

    def _append_pointcloud_data(self, traj):
        """Append point cloud data to trajectory."""
        traj['canonical'].append(self.vec_env.task.object_points.clone().to(torch.float16))
        traj['rendered'].append(self.vec_env.task.rendered_object_points.clone().to(torch.float16))
        traj['centers'].append(self.vec_env.task.rendered_object_points_centers.clone())
        traj['appears'].append(self.vec_env.task.rendered_object_points_appears.clone())
        traj['features'].append(self.vec_env.task.rendered_points_visual_features.clone())
        traj['pcas'].append(self.vec_env.task.rendered_object_pcas.clone())
        traj['hand_body'].append(self.vec_env.task.hand_body_pos.clone())
        traj['hand_object'].append(self.vec_env.task.rendered_hand_object_dists.clone())

    def _save_observation_trajectory(self, traj):
        """Save observation trajectory to disk."""
        # Stack all trajectory data
        traj['observations'] = torch.stack(traj['observations'], dim=1).cpu().numpy()
        traj['actions'] = torch.stack(traj['actions'], dim=1).cpu().numpy()
        traj['values'] = torch.stack(traj['values'], dim=1).cpu().numpy()
        traj['features'] = torch.stack(traj['features'], dim=1).cpu().numpy()
        traj['successes'] = traj['successes'].cpu().numpy()
        traj['final_successes'] = traj['final_successes'].cpu().numpy()
        traj['valids'] = compute_trajectory_valids(traj['observations'][:, :, 191:194])
        
        # Save in groups
        group_size = 10
        num_groups = self.vec_env.task.num_envs // group_size
        existing_files = glob.glob(osp.join(self.vec_env.task.render_folder, 'trajectory_*.pkl'))
        start_idx = len(existing_files)
        
        for group_idx in range(num_groups):
            group_start = group_idx * group_size
            group_end = (group_idx + 1) * group_size
            sub_traj = {k: v[group_start:group_end] for k, v in traj.items()}
            file_path = osp.join(
                self.vec_env.task.render_folder,
                f'trajectory_{start_idx + group_idx:03d}.pkl'
            )
            save_pickle(file_path, sub_traj)

    def _save_pointcloud_trajectory(self, pc_traj, obs_traj):
        """Save point cloud trajectory to disk."""
        # Stack all point cloud data
        pc_traj['canonical'] = torch.stack(pc_traj['canonical'], dim=1).cpu().numpy()
        pc_traj['rendered'] = torch.stack(pc_traj['rendered'], dim=1).cpu().numpy()
        pc_traj['centers'] = torch.stack(pc_traj['centers'], dim=1).cpu().numpy()
        pc_traj['appears'] = torch.stack(pc_traj['appears'], dim=1).cpu().numpy()
        pc_traj['features'] = torch.stack(pc_traj['features'], dim=1).cpu().numpy()
        pc_traj['pcas'] = torch.stack(pc_traj['pcas'], dim=1).cpu().numpy()
        pc_traj['hand_body'] = torch.stack(pc_traj['hand_body'], dim=1).cpu().numpy()
        pc_traj['hand_object'] = torch.stack(pc_traj['hand_object'], dim=1).cpu().numpy()
        pc_traj['successes'] = obs_traj['successes']
        pc_traj['final_successes'] = obs_traj['final_successes']
        pc_traj['valids'] = obs_traj['valids']
        
        # Save in groups
        group_size = 10
        num_groups = self.vec_env.task.num_envs // group_size
        existing_files = glob.glob(osp.join(self.vec_env.task.pointcloud_folder, 'pointcloud_*.pkl'))
        start_idx = len(existing_files)
        
        for group_idx in range(num_groups):
            group_start = group_idx * group_size
            group_end = (group_idx + 1) * group_size
            sub_traj = {k: v[group_start:group_end] for k, v in pc_traj.items()}
            file_path = osp.join(
                self.vec_env.task.pointcloud_folder,
                f'pointcloud_{start_idx + group_idx:03d}.pkl'
            )
            save_pickle(file_path, sub_traj)

    def _execute_training_loop(self, num_iterations, log_interval, initial_obs, initial_states, frame_counter):
        """Execute the main training loop."""
        # Save environment object scales
        save_list_strings(
            os.path.join(self.log_dir, 'env_object_scale.txt'),
            self.vec_env.task.env_object_scale
        )
        
        # Initialize wandb if needed
        if self.vec_env.task.init_wandb:
            project_name = self.log_dir.split('/')[-1]
            wandb.init(project=project_name)
        
        # Initialize statistics buffers
        reward_buffer = deque(maxlen=100)
        length_buffer = deque(maxlen=100)
        episode_rewards = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
        episode_lengths = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
        completed_rewards = []
        completed_lengths = []
        log_messages = []
        
        # Training loop
        training_start = time.time()
        self.vec_env.task.frame = -1
        current_obs = initial_obs
        current_states = initial_states
        
        # Ensure num_iterations is an integer
        num_iterations = int(num_iterations)
        start_iteration = int(self.current_learning_iteration)
        
        for iteration in range(start_iteration, num_iterations):
            episode_infos = []
            iter_start = time.time()
            elapsed_minutes = (iter_start - training_start) / 60
            self.vec_env.task.current_iteration += 1
            
            # Logging
            if iteration % max(1, 100) == 0:
                estimated_total = (elapsed_minutes / (iteration + 1)) * num_iterations
                remaining_hours = (estimated_total - elapsed_minutes) / 60
                log_msg = (
                    f"Training iteration: {self.log_dir.split('/')[-1]}/{self.device}//"
                    f"{iteration}/{num_iterations}; Processed time:{int(elapsed_minutes)}/"
                    f"{int(estimated_total)} mins; Left time: {remaining_hours:.2f} hours"
                )
                log_messages.append(log_msg)
                log_messages.append(
                    f"Current success rate: {self.vec_env.task.current_success_rate}, "
                    f"Cumulative successes: {self.vec_env.task.cumulative_successes}"
                )
                log_messages.append(
                    f"Current average step: {self.vec_env.task.current_avg_steps}, "
                    f"Cumulative average step: {self.vec_env.task.avg_succ_steps}"
                )
                log_messages.append(f"Current reward: {self.vec_env.task.reward_value}")
                print(log_messages[-4:])
                save_list_strings(osp.join(self.log_dir, 'train.log'), log_messages)

            # Collect rollout
            for step in range(self.num_transitions_per_env):
                self.vec_env.task.frame += 1
                if self.apply_reset:
                    current_obs = self.vec_env.reset()
                    current_states = self.vec_env.get_state()
                frame_counter = (frame_counter + 1) % self.vec_env.task.max_episode_length
                
                # Get action from policy
                actions, log_probs, values, mu, sigma = self.actor_critic.act(current_obs, current_states)
                
                # Step environment
                next_obs, rewards, dones, infos = self.vec_env.step(actions, frame_counter)
                next_states = self.vec_env.get_state()
                
                # Store transition
                self.storage.add_transitions(
                    current_obs, current_states, actions, rewards, dones,
                    values, log_probs, mu, sigma
                )
                current_obs.copy_(next_obs)
                current_states.copy_(next_states)
                episode_infos.append(infos)
                
                # Track episode statistics
                if self.print_log:
                    episode_rewards += rewards
                    episode_lengths += 1
                    done_indices = (dones > 0).nonzero(as_tuple=False)
                    if done_indices.numel() > 0:
                        completed_rewards.extend(episode_rewards[done_indices][:, 0].cpu().numpy().tolist())
                        completed_lengths.extend(episode_lengths[done_indices][:, 0].cpu().numpy().tolist())
                        episode_rewards[done_indices] = 0
                        episode_lengths[done_indices] = 0
            
            # Update statistics buffers
            if self.print_log:
                reward_buffer.extend(completed_rewards)
                length_buffer.extend(completed_lengths)
            
            # Compute final values and statistics
            _, _, final_values, _, _ = self.actor_critic.act(current_obs, current_states)
            collection_time = time.time() - iter_start
            avg_traj_length, avg_reward = self.storage.get_statistics()
            
            # Learning update
            update_start = time.time()
            self.storage.compute_returns(final_values, self.gamma, self.lam)
            value_loss, policy_loss = self.update()
            self.storage.clear()
            learning_time = time.time() - update_start
            
            # Logging
            if self.print_log:
                local_vars = {
                    'it': iteration,
                    'num_learning_iterations': num_iterations,
                    'collection_time': collection_time,
                    'learn_time': learning_time,
                    'mean_reward': avg_reward,
                    'mean_trajectory_length': avg_traj_length,
                    'mean_value_loss': value_loss,
                    'mean_surrogate_loss': policy_loss,
                    'rewbuffer': reward_buffer,
                    'lenbuffer': length_buffer,
                    'ep_infos': episode_infos
                }
                self.log(local_vars, show=False)
            
            # Save checkpoint
            if iteration % log_interval == 0 and iteration != 0:
                checkpoint_path = os.path.join(self.log_dir, f'model_{iteration}.pt')
                self.save(checkpoint_path)
            
            episode_infos.clear()
            completed_rewards.clear()
            completed_lengths.clear()
        
        # Final save
        final_checkpoint = os.path.join(self.log_dir, f'model_{num_iterations}.pt')
        self.save(final_checkpoint)
        exit()

    def log(self, local_vars, width=80, pad=35, show=False):
        """Log training statistics to TensorBoard and console."""
        # Update global statistics
        timesteps_this_iter = self.num_transitions_per_env * self.vec_env.num_envs
        self.total_timesteps += timesteps_this_iter
        iter_time = local_vars['collection_time'] + local_vars['learn_time']
        self.total_time_elapsed += iter_time
        
        log_lines = []
        episode_metrics = []

        # Log episode information
        if local_vars['ep_infos']:
            info_keys = local_vars['ep_infos'][0].keys()
            for key in info_keys:
                # Handle both tensor and float values
                info_values = []
                for info in local_vars['ep_infos']:
                    val = info[key]
                    if isinstance(val, torch.Tensor):
                        info_values.append(val.to(self.device))
                    else:
                        # Convert float/int to tensor
                        info_values.append(torch.tensor(val, device=self.device, dtype=torch.float32))
                
                if len(info_values) > 0:
                    if isinstance(info_values[0], torch.Tensor) and info_values[0].dim() > 0:
                        combined = torch.cat(info_values, dim=0)
                        mean_val = combined.mean()
                    else:
                        # Stack scalar tensors
                        combined = torch.stack(info_values)
                        mean_val = combined.mean()
                    self.writer.add_scalar(f'Episode/{key}', mean_val, local_vars['it'])
                    episode_metrics.append(f"{f'Mean episode {key}:':>{pad}} {mean_val:.4f}")
                if key in ['successes', 'current_successes', 'consecutive_successes']:
                    if self.vec_env.task.init_wandb:
                        wandb.log({f'Mean episode {key}': mean_val})

        # Log policy statistics
        policy_std = self.actor_critic.log_std.exp().mean()
        self.writer.add_scalar('Loss/value_function', local_vars['mean_value_loss'], local_vars['it'])
        self.writer.add_scalar('Loss/surrogate', local_vars['mean_surrogate_loss'], local_vars['it'])
        self.writer.add_scalar('Policy/mean_noise_std', policy_std.item(), local_vars['it'])

        # Log reward statistics
        if local_vars['rewbuffer']:
            mean_reward = statistics.mean(local_vars['rewbuffer'])
            mean_length = statistics.mean(local_vars['lenbuffer'])
            self.writer.add_scalar('Train/mean_reward', mean_reward, local_vars['it'])
            self.writer.add_scalar('Train/mean_episode_length', mean_length, local_vars['it'])
            self.writer.add_scalar('Train/mean_reward/time', mean_reward, self.total_time_elapsed)
            self.writer.add_scalar('Train/mean_episode_length/time', mean_length, self.total_time_elapsed)
            if self.vec_env.task.init_wandb:
                wandb.log({
                    'Mean reward': mean_reward,
                    'Mean reward per step': local_vars['mean_reward']
                })

        self.writer.add_scalar('Train2/mean_reward/step', local_vars['mean_reward'], local_vars['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', local_vars['mean_trajectory_length'], local_vars['it'])

        # Log task-specific metrics
        task_obj = getattr(self, "vec_env", None)
        task_obj = getattr(task_obj, "task", None) if task_obj else None
        if task_obj is not None:
            if hasattr(task_obj, 'current_success_rate'):
                self.writer.add_scalar('Train/success_rate', task_obj.current_success_rate, local_vars['it'])
            if hasattr(task_obj, 'cumulative_successes'):
                self.writer.add_scalar('Train/cumulative_successes', task_obj.cumulative_successes, local_vars['it'])
            if hasattr(task_obj, 'singulation_success_rate'):
                self.writer.add_scalar('Singulation/success_rate', task_obj.singulation_success_rate, local_vars['it'])
            if hasattr(task_obj, 'reward_value'):
                self.writer.add_scalar('Train/episode_reward', task_obj.reward_value, local_vars['it'])

        # Compute FPS
        fps = int(timesteps_this_iter / max(iter_time, 1e-6))
        header_text = f" \033[1m Learning iteration {local_vars['it']}/{local_vars['num_learning_iterations']} \033[0m "

        # Build log message
        log_lines.append("#" * width)
        log_lines.append(header_text.center(width, " "))
        log_lines.append("")
        log_lines.append(
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {local_vars['collection_time']:.3f}s, learning {local_vars['learn_time']:.3f}s)"
        )
        log_lines.append(f"{'Value function loss:':>{pad}} {local_vars['mean_value_loss']:.4f}")
        log_lines.append(f"{'Surrogate loss:':>{pad}} {local_vars['mean_surrogate_loss']:.4f}")
        log_lines.append(f"{'Mean action noise std:':>{pad}} {policy_std.item():.2f}")
        log_lines.append(f"{'Mean reward/step:':>{pad}} {local_vars['mean_reward']:.2f}")
        log_lines.append(f"{'Mean episode length/episode:':>{pad}} {local_vars['mean_trajectory_length']:.2f}")
        log_lines.extend(episode_metrics)
        log_lines.append("-" * width)
        log_lines.append(f"{'Total timesteps:':>{pad}} {self.total_timesteps}")
        log_lines.append(f"{'Iteration time:':>{pad}} {iter_time:.2f}s")
        log_lines.append(f"{'Total time:':>{pad}} {self.total_time_elapsed:.2f}s")
        remaining_time = (self.total_time_elapsed / max(local_vars['it'] + 1, 1)) * (
            local_vars['num_learning_iterations'] - local_vars['it']
        )
        log_lines.append(f"{'ETA:':>{pad}} {remaining_time:.1f}s")

        if show:
            print("\n".join(log_lines))

    def update(self):
        """Perform PPO policy and value function updates."""
        total_value_loss = 0.0
        total_policy_loss = 0.0
        batch_indices = self.storage.mini_batch_generator(self.num_mini_batches)
        
        # Multiple epochs of updates
        for epoch_idx in range(self.num_learning_epochs):
            for batch_idx in batch_indices:
                # Extract batch data
                batch_data = self._extract_batch_data(batch_idx)
                
                # Evaluate current policy
                policy_outputs = self.actor_critic.evaluate(
                    batch_data['observations'],
                    batch_data['states'],
                    batch_data['actions']
                )
                new_log_probs, entropy, new_values, new_mu, new_sigma = policy_outputs
                
                # Adjust learning rate if using adaptive schedule
                if self.desired_kl is not None and self.schedule == 'adaptive':
                    self._adjust_learning_rate(new_mu, new_sigma, batch_data['old_mu'], batch_data['old_sigma'])
                
                # Compute losses
                policy_loss = self._compute_policy_loss(
                    new_log_probs,
                    batch_data['old_log_probs'],
                    batch_data['advantages']
                )
                value_loss = self._compute_value_loss(
                    new_values,
                    batch_data['returns'],
                    batch_data['old_values']
                )

                # Total loss
                total_loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss -
                    self.entropy_coef * entropy.mean()
                )
                
                # Optimizer step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate losses
                total_value_loss += value_loss.item()
                total_policy_loss += policy_loss.item()
        
        # Average losses
        num_updates = self.num_learning_epochs * self.num_mini_batches
        avg_value_loss = total_value_loss / num_updates
        avg_policy_loss = total_policy_loss / num_updates
        return avg_value_loss, avg_policy_loss

    def _extract_batch_data(self, indices):
        """Extract batch data from storage."""
        flat_obs = self.storage.observations.view(-1, *self.storage.observations.size()[2:])
        flat_actions = self.storage.actions.view(-1, self.storage.actions.size(-1))
        flat_values = self.storage.values.view(-1, 1)
        flat_returns = self.storage.returns.view(-1, 1)
        flat_advantages = self.storage.advantages.view(-1, 1)
        flat_mu = self.storage.mu.view(-1, self.storage.actions.size(-1))
        flat_sigma = self.storage.sigma.view(-1, self.storage.actions.size(-1))
        flat_log_probs = self.storage.actions_log_prob.view(-1, 1)
        
        batch_obs = flat_obs[indices]
        batch_states = None
        if self.asymmetric:
            flat_states = self.storage.states.view(-1, *self.storage.states.size()[2:])
            batch_states = flat_states[indices]
        
        return {
            'observations': batch_obs,
            'states': batch_states,
            'actions': flat_actions[indices],
            'old_values': flat_values[indices],
            'returns': flat_returns[indices],
            'advantages': flat_advantages[indices],
            'old_mu': flat_mu[indices],
            'old_sigma': flat_sigma[indices],
            'old_log_probs': flat_log_probs[indices]
        }

    def _adjust_learning_rate(self, new_mu, new_sigma, old_mu, old_sigma):
        """Adjust learning rate based on KL divergence."""
        kl_components = (
            new_sigma - old_sigma +
            (torch.square(old_sigma.exp()) + torch.square(old_mu - new_mu)) /
            (2.0 * torch.square(new_sigma.exp())) - 0.5
        )
        kl_mean = torch.sum(kl_components, axis=-1).mean()
        
        if kl_mean > self.desired_kl * 2.0:
            self.current_lr = max(1e-5, self.current_lr / 1.5)
        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.current_lr = min(1e-2, self.current_lr * 1.5)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.current_lr

    def _compute_policy_loss(self, new_log_probs, old_log_probs, advantages):
        """Compute clipped surrogate policy loss."""
        advantage_squeezed = torch.squeeze(advantages)
        old_log_probs_squeezed = torch.squeeze(old_log_probs)
        
        # Importance sampling ratio
        ratio = torch.exp(new_log_probs - old_log_probs_squeezed)
        
        # Unclipped and clipped surrogate losses
        unclipped_loss = -advantage_squeezed * ratio
        clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        clipped_loss = -advantage_squeezed * clipped_ratio
        
        # Take maximum (more conservative)
        return torch.max(unclipped_loss, clipped_loss).mean()

    def _compute_value_loss(self, new_values, returns, old_values):
        """Compute value function loss."""
        if self.use_clipped_value_loss:
            # Clipped value loss
            value_clipped = old_values + (new_values - old_values).clamp(
                -self.clip_param, self.clip_param
            )
            unclipped_loss = (new_values - returns).pow(2)
            clipped_loss = (value_clipped - returns).pow(2)
            return torch.max(unclipped_loss, clipped_loss).mean()
        else:
            # Standard MSE loss
            return (returns - new_values).pow(2).mean()
