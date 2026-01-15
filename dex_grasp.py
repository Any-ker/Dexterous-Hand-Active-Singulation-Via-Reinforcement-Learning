
import numpy as np
import os.path as osp
import os, glob, tqdm
import random, torch, trimesh
import json, math, time, atexit, sys
from datetime import datetime
from collections import defaultdict
from typing import Dict

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import quat_apply, tensor_clamp, to_torch, torch_rand_float, quat_from_euler_xyz, get_euler_xyz, quat_mul, quat_conjugate, scale, unscale

from utils.general_utils import *
from utils.torch_jit_utils import *


from sklearn.decomposition import PCA
from tasks.hand_base.base_task import BaseTask

from dexgrasp.interfaces import RunMetadata, VideoLoggingConfig, VisualizationConfig
from dexgrasp.plugins.manager import PluginManager
from dexgrasp.plugins.visualization import VideoCapturePlugin, VisualizationPlugin

sys.path.append(osp.join(BASE_DIR, 'dexgrasp/autoencoding'))

class DexGrasp(BaseTask):
    """Isaac Gym task that simulates dexterous grasping with clutter."""
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg
        self.sim_params = sim_params
        self.agent_index = agent_index
        self.physics_engine = physics_engine
        self.is_multi_agent = is_multi_agent

        self.init_dof_state = [0 for _ in range(22)]

        self.reward_value = 0
        
        self.mean_singulation_distance = None
        self.min_singulation_distance = None
        self.invalid_env_mask = None
        self.invalid_env_num = 0
        self.remove_invalid_arrangement = False
        self.init_wandb = False  # 默认不使用wandb

        # Counters for logging resets/steps.
        self.step_count = 0
        self.reset_count = 0
        # Pre-defined grid used to scatter surrounding objects in a repeatable way.
        self.sudoku_grid = [
                {'x':0, 'y':0},        # Middle center
                {'x':-1, 'y':1},   # Top left
                {'x':0, 'y':1},      # Top center  
                {'x':1, 'y':1},    # Top right
                {'x':1, 'y':0},      # Middle right  
                {'x':1, 'y':-1},   # Bottom right
                {'x':0, 'y':-1},     # Bottom center
                {'x':-1, 'y':-1},  # Bottom left
                {'x':-1, 'y':0},     # Middle left
            ]

        self.algo = cfg['algo']
        self.config = cfg['config']
        debug_cfg = self.config.get('Debug', {})
        self.debug_log_interval = debug_cfg.get('reward_log_interval', 0)
        self.randomize = self.cfg["task"]["randomize"]
        self.up_axis = 'z'
        self.num_envs = self.cfg["env"]["numEnvs"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.is_testing, self.test_epoch, self.test_iteration, self.current_test_iteration = cfg['test'], cfg['test_epoch'], self.cfg["test_iteration"], 0
        self.current_iteration = 0
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]
        self.dex_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)

        self.separation_dist = self.cfg["env"].get("separation_dist", 0.1)
        self.surrounding_obj_num = self.cfg["env"].get("surrounding_obj_num", 8)
        self.random_grid_sequences = self.cfg["env"].get("random_grid_sequences", True)
        self.random_surrounding_positions = self.cfg["env"].get("random_surrounding_positions", True)
        self.random_surrounding_orientations = self.cfg["env"].get("random_surrounding_orientations", True)

        self.use_hand_rotation = self.cfg["env"].get("use_hand_rotation", False)
        self.hand_rotation_coef = self.cfg["env"].get("hand_rotation_coef", 0.01)

        self.use_hand_link_pose = self.cfg["env"].get("use_hand_link_pose", True)
        self.include_dummy_dofs = self.cfg["env"].get("include_dummy_dofs", True)

        self.shuffle_object_arrangements = self.cfg["env"].get("shuffle_object_arrangements", True)
        self.object_arrangements = None

        self.remove_invalid_arrangement = self.cfg["env"].get("remove_invalid_arrangement", False)
        self.randomize_object_mass = self.cfg["env"].get("randomize_object_mass", False)
        self.randomize_object_mass_range = self.cfg["env"].get("randomize_object_mass_range", 0.05)

        self.cumulative_successes = 0
        self.current_success_rate = 0

        self.total_valid_successes = []
        self.total_valid_envs = []
        
        # 累计分离成功率统计（基于singulation_success）
        self.cumulative_singulation_success_rate = 0.0
        self.total_valid_singulation_successes = []
        self.total_valid_singulation_envs = []

        self.object_position_assignments = None

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        
        self.image_size = 256
        self.table_dims = gymapi.Vec3(1, 1, self.cfg["env"].get("table_dim_z", 0.6)) # 0.01 for training, 0.6 for recording trajecotry and pointcloud
        self.table_center = np.array([0.0, 0.0, self.table_dims.z])
        
        self.fingertips = ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"]
        self.num_fingertips = len(self.fingertips) 
        self.use_vel_obs = False
        self.fingertip_obs = True
        num_states = 0
        self.cfg["env"]["numStates"] = num_states
        self.num_agents = 1
        self.cfg["env"]["numActions"] = 22
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["graphics_device_id"] = device_id
        self.cfg["headless"] = headless

        self.render_folder = None
        self.render_env_list = None
        self.render_env_list = self.render_env_list

        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.run_name = self.cfg.get("run_name", f"{self.algo}_{timestamp_str}")
        self.task_name = self.cfg.get("task", {}).get("name", "dexgrasp")
        self.reward_mode = self.cfg["env"].get("reward_mode", "new")
        self.curriculum_enabled = self.cfg["env"].get("use_curriculum", False)
        self.training_strategy = self.cfg["env"].get("training_strategy", "standard")

        default_viz_dir = osp.join(BASE_DIR, "outputs", "visualizations")
        # Optional plug-in driven visualizations/video logging.
        self.visualization_cfg = VisualizationConfig.from_dict(self.cfg["env"].get("visualization", {}), default_viz_dir)
        self.visualization_output_dir = self.visualization_cfg.output_dir
        os.makedirs(self.visualization_output_dir, exist_ok=True)

        video_cfg_dict = self.cfg["env"].get("video_logging", {})
        self.video_cfg = VideoLoggingConfig.from_dict(video_cfg_dict)
        if not self.video_cfg.output_dir:
            self.video_cfg.output_dir = self.visualization_output_dir

        self.target_camera_configs = {}
        if self.video_cfg.camera_setups:
            for setup in self.video_cfg.camera_setups:
                env_id = setup.get("env_id")
                if env_id is None:
                    continue
                self.target_camera_configs[env_id] = setup
        elif self.video_cfg.camera_env_ids:
            for env_id in self.video_cfg.camera_env_ids:
                self.target_camera_configs[env_id] = {
                    "env_id": env_id,
                    "position": self.video_cfg.default_pos,
                    "target": self.video_cfg.default_target,
                }

        self.camera_handles_map = {}
        self.completed_video_manifest = []
        self.global_step = 0

        # Plug-ins enable report-friendly plots and videos without cluttering core logic.
        self.plugin_manager = PluginManager()
        self._plugins_registered = False
        self._plugins_finalized = False
        self.run_metadata = RunMetadata(
            run_name=self.run_name,
            task_name=self.task_name,
            reward_mode=self.reward_mode,
            curriculum_enabled=self.curriculum_enabled,
            obstacle_count=self.surrounding_obj_num,
            training_strategy=self.training_strategy,
        )

        

        self.frame = -1

        
        super().__init__(cfg=self.cfg, enable_camera_sensors=False)
        self._activate_plugins()
        atexit.register(self._finalize_plugins)

        self.invalid_env_num = 0
        self.invalid_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.look_at_env = None
        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.8, 0, 1.5)
            cam_target = gymapi.Vec3(0, 0, 0.6)
            self.look_at_env = self.envs[len(self.envs) // 2]
            self.gym.viewer_camera_look_at(self.viewer, self.look_at_env, cam_pos, cam_target)


        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dex_hand_dofs + self.num_object_dofs)
        self.dof_force_tensor = self.dof_force_tensor[:, :self.num_dex_hand_dofs]
        
        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "hand")
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.z_theta = torch.zeros(self.num_envs, device=self.device)
        self.dex_hand_default_dof_pos = torch.tensor(self.init_dof_state, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        
        self.dex_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dex_hand_dofs]
        self.dex_hand_dof_pos = self.dex_hand_dof_state[..., 0]
        self.dex_hand_dof_vel = self.dex_hand_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.cur_targets = torch.tensor(self.init_dof_state, dtype=torch.float, device=self.device).repeat(self.num_envs, 1)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,-1)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.final_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.succ_steps = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.singulation_success = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.singulation_success_rate = 0.0
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.singulation_thresh = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0
        self.avg_succ_steps = 0
        self.current_avg_steps = 0
        self.right_hand_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.right_hand_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.object_points_visual_features = torch.zeros((self.num_envs, 128), device=self.device)
        self.prev_obstacle_distances = None
        self.prev_obstacle_projected = None



    def create_sim(self):
        """Build the simulation, ground plane, and all environments."""
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        
        self.mean_singulation_distance = torch.zeros(self.num_envs, device=self.device)
        self.min_singulation_distance = torch.zeros(self.num_envs, device=self.device)
        self.invalid_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _create_ground_plane(self):
        """Add a simple ground plane for stability."""
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        """Instantiate actors, textures, and optional cameras for each env."""
        # Initialize configuration and paths
        self._initialize_env_config()
        
        # Load all required assets
        assets = self._load_all_assets()
        dex_hand_asset, dex_hand_start_pose, dex_hand_dof_props = assets['hand']
        goal_asset, table_asset = assets['goal'], assets['table']
        object_start_pose, goal_start_pose, table_pose = assets['poses']
        table_texture_handle = assets['texture']
        
        # Prepare aggregation parameters
        aggregation_params = self._compute_aggregation_params()
        
        # Initialize all collection lists
        collections = self._initialize_collections()
        
        # Initialize buffers that need to be accessed during environment creation
        self.object_scale_buf = collections['object_scale_buf']
        self.object_clutter_indices = collections['object_clutter_indices']
        if not hasattr(self, 'object_id_buf'):
            # object_id_buf should be initialized in base class, but ensure it exists
            self.object_id_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        
        # Setup environment boundaries and camera configuration
        env_bounds = (gymapi.Vec3(-spacing, -spacing, 0.0), gymapi.Vec3(spacing, spacing, spacing))
        camera_setup = self._prepare_camera_config()
        
        print(f'Create num_envs {self.num_envs}')
        
        # Create each environment
        for env_id in tqdm.tqdm(range(self.num_envs), desc='Creating environments'):
            env_data = self._build_single_environment(
                env_id, env_bounds, num_per_row, aggregation_params,
                dex_hand_asset, dex_hand_start_pose, dex_hand_dof_props,
                goal_asset, table_asset, object_start_pose, goal_start_pose, table_pose,
                table_texture_handle, camera_setup
            )
            self._append_env_data(collections, env_data)
        
        # Convert all lists to tensors
        self._finalize_tensors(collections)
    
    def _initialize_env_config(self):
        """Extract and initialize configuration parameters."""
        self.object_scale_dict = self.cfg['env']['object_code_dict']
        self.object_code_list = list(self.object_scale_dict.keys())
        
        asset_root = self.cfg["env"]['asset']['assetRoot']
        self.assets_path = asset_root if osp.exists(asset_root) else '../' + asset_root
        
        self.goal_cond = self.cfg["env"]["goal_cond"]
        self.random_prior = self.cfg['env']['random_prior']
        self.random_time = self.cfg["env"]["random_time"]
        if 'random_time' in self.config.get('Modes', {}):
            self.random_time = bool(self.config['Modes']['random_time'])
        
        # Initialize target state tensors
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)
        
        # Scale conversion mappings
        self.scale2str = {0.06: '006', 0.08: '008', 0.10: '010', 0.12: '012', 0.15: '015'}
        self.str2scale = {v: k for k, v in self.scale2str.items()}
    
    def _load_all_assets(self):
        """Load all assets needed for environment creation."""
        hand_assets = self._load_dex_hand_assets(self.assets_path)
        object_dict, goal_asset, table_asset, obj_pose, goal_pose, tbl_pose = \
            self._load_object_table_goal_assets(self.assets_path, self.scale2str)
        self.object_asset_dict = object_dict
        
        texture_path = osp.join(self.assets_path, "textures/texture_stone_stone_texture_0.jpg")
        texture_handle = self.gym.create_texture_from_file(self.sim, texture_path)
        
        return {
            'hand': hand_assets,
            'goal': goal_asset,
            'table': table_asset,
            'poses': (obj_pose, goal_pose, tbl_pose),
            'texture': texture_handle
        }
    
    def _compute_aggregation_params(self):
        """Calculate aggregation parameters for physics optimization."""
        extra_bodies = 8
        max_bodies = (self.num_dex_hand_bodies + 2 * self.num_object_bodies + 
                     1 + extra_bodies)
        max_shapes = (self.num_dex_hand_shapes + 20 * self.num_object_shapes + 
                     1 + extra_bodies)
        return {'max_bodies': max_bodies, 'max_shapes': max_shapes}
    
    def _initialize_collections(self):
        """Initialize all lists that will collect environment data."""
        return {
            'envs': [],
            'env_object_scale': [],
            'hand_indices': [],
            'object_indices': [],
            'goal_object_indices': [],
            'table_indices': [],
            'hand_start_states': [],
            'object_init_state': [],
            'goal_init_state': [],
            'object_init_mesh': {
                'mesh': [], 'mesh_vertices': [], 'mesh_faces': [],
                'points': [], 'points_centered': [], 'pca_axes': []
            },
            'object_scale_buf': {},
            'object_clutter_indices': [[] for _ in range(self.surrounding_obj_num + 1)],
            'hand_point_handles': [],
            'object_point_handles': [],
            'hand_point_indices': [],
            'object_point_indices': [],
            'env_object_scale_id': []
        }
    
    def _prepare_camera_config(self):
        """Prepare camera configuration if video logging is enabled."""
        self.camera_handles_map = {}
        if not self.video_cfg.enable:
            return None
        
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.video_cfg.width
        camera_props.height = self.video_cfg.height
        camera_props.enable_tensors = True
        return camera_props
    
    def _build_single_environment(self, env_id, env_bounds, num_per_row, agg_params,
                                  hand_asset, hand_pose, hand_dof_props,
                                  goal_asset, table_asset, obj_pose, goal_pose, tbl_pose,
                                  texture_handle, camera_props):
        """Construct a single environment with all actors and properties."""
        lower, upper = env_bounds
        env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
        
        if self.aggregate_mode >= 1:
            self.gym.begin_aggregate(env_ptr, agg_params['max_bodies'], 
                                    agg_params['max_shapes'], True)
        
        # Create and configure hand
        hand_data = self._setup_hand_actor(env_ptr, env_id, hand_asset, hand_pose, hand_dof_props)
        
        # Determine object configuration for this environment
        obj_config = self._select_object_configuration(env_id)
        
        # Create object clutter
        clutter_data = self._setup_object_clutter(env_ptr, env_id, obj_config, obj_pose)
        
        # Create goal and table actors
        goal_data = self._setup_goal_actor(env_ptr, env_id, goal_asset, goal_pose)
        table_data = self._setup_table_actor(env_ptr, env_id, table_asset, tbl_pose, texture_handle)
        
        # Configure physical properties
        self._apply_physical_properties(env_ptr, table_data['handle'], clutter_data['focus_handle'])
        
        # Setup visual appearance
        self._apply_visual_styling(env_ptr, clutter_data['focus_handle'], 
                                  table_data['handle'], goal_data['handle'])
        
        # Optional camera setup
        camera_handle = None
        if camera_props and env_id in self.target_camera_configs:
            camera_handle = self._setup_camera(env_ptr, env_id, camera_props)
        
        if self.aggregate_mode > 0:
            self.gym.end_aggregate(env_ptr)
        
        return {
            'env_ptr': env_ptr,
            'hand': hand_data,
            'clutter': clutter_data,
            'goal': goal_data,
            'table': table_data,
            'camera': camera_handle,
            'object_config': obj_config
        }
    
    def _setup_hand_actor(self, env_ptr, env_id, hand_asset, hand_pose, dof_props):
        """Create and configure the dexterous hand actor."""
        actor_id = env_id - 5 * self.num_envs
        hand_actor = self.gym.create_actor(env_ptr, hand_asset, hand_pose, "hand", 
                                          actor_id, -1, 0)
        
        # Configure DOF properties
        dof_props["driveMode"][:6] = gymapi.DOF_MODE_POS
        dof_props["driveMode"][6:] = gymapi.DOF_MODE_POS
        dof_props["stiffness"] = [50, 50, 150] + [100] * 3 + [100] * 16
        dof_props["armature"] = [0.001] * 6 + [0.0001] * 16
        dof_props["damping"] = [20] * 6 + [5] * 16
        self.gym.set_actor_dof_properties(env_ptr, hand_actor, dof_props)
        
        hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
        self.gym.enable_actor_dof_force_sensors(env_ptr, hand_actor)
        
        hand_state = [
            hand_pose.p.x, hand_pose.p.y, hand_pose.p.z,
            hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w,
            0, 0, 0, 0, 0, 0
        ]
        
        return {'actor': hand_actor, 'index': hand_idx, 'state': hand_state}
    
    def _select_object_configuration(self, env_id):
        """Select object code and scale for the current environment."""
        code_idx = env_id % len(self.object_code_list)
        object_code = self.object_code_list[code_idx]
        available_scales = self.object_scale_dict[object_code]
        selected_scale = available_scales[env_id % len(available_scales)]
        scale_str = self.scale2str[selected_scale]
        
        self.object_id_buf[env_id] = env_id
        self.object_scale_buf[env_id] = selected_scale
        
        # Prepare clutter object codes
        clutter_codes = list(self.object_asset_dict.keys())
        clutter_scales = [list(self.object_asset_dict[code].keys())[0] 
                         for code in clutter_codes]
        focus_idx = 0
        
        if self.use_object_asset_dict:
            asset_info = self.object_asset_dict[clutter_codes[focus_idx]][clutter_scales[focus_idx]]
        else:
            asset_info = self._load_object_asset_info(self.assets_path, object_code, scale_str)
        
        return {
            'code_idx': code_idx,
            'code': object_code,
            'scale': selected_scale,
            'scale_str': scale_str,
            'clutter_codes': clutter_codes,
            'clutter_scales': clutter_scales,
            'focus_idx': focus_idx,
            'asset_info': asset_info
        }
    
    def _setup_object_clutter(self, env_ptr, env_id, obj_config, obj_pose):
        """Create object clutter for the environment."""
        self.object_clutter_handles = []
        self.env_id = env_id
        self.obj_focus_id = obj_config['focus_idx']
        self._create_objects(env_ptr, env_id, obj_config['clutter_codes'], 
                           obj_config['clutter_scales'], type=2)
        
        focus_handle = self.object_clutter_handles[obj_config['focus_idx']]
        focus_idx = self.gym.get_actor_index(env_ptr, focus_handle, gymapi.DOMAIN_SIM)
        
        obj_state = [
            obj_pose.p.x, obj_pose.p.y, obj_pose.p.z,
            obj_pose.r.x, obj_pose.r.y, obj_pose.r.z, obj_pose.r.w,
            0, 0, 0, 0, 0, 0
        ]
        
        mesh_data = {
            'mesh': obj_config['asset_info'][2],
            'vertices': obj_config['asset_info'][2].vertices,
            'faces': obj_config['asset_info'][2].faces,
            'points': obj_config['asset_info'][3],
            'points_centered': obj_config['asset_info'][4],
            'pca_axes': obj_config['asset_info'][5]
        }
        
        return {
            'focus_handle': focus_handle,
            'focus_idx': focus_idx,
            'state': obj_state,
            'mesh': mesh_data,
            'scale': obj_config['asset_info'][1]
        }
    
    def _setup_goal_actor(self, env_ptr, env_id, goal_asset, goal_pose):
        """Create the goal marker actor."""
        goal_actor = self.gym.create_actor(env_ptr, goal_asset, goal_pose, "goal_object",
                                          env_id + self.num_envs, 0, 0)
        goal_idx = self.gym.get_actor_index(env_ptr, goal_actor, gymapi.DOMAIN_SIM)
        self.gym.set_actor_scale(env_ptr, goal_actor, 0.0001)
        
        goal_state = [
            goal_pose.p.x, goal_pose.p.y, goal_pose.p.z,
            goal_pose.r.x, goal_pose.r.y, goal_pose.r.z, goal_pose.r.w,
            0, 0, 0, 0, 0, 0
        ]
        
        return {'handle': goal_actor, 'index': goal_idx, 'state': goal_state}
    
    def _setup_table_actor(self, env_ptr, env_id, table_asset, table_pose, texture):
        """Create the table actor with texture."""
        actor_id = env_id - 5 * self.num_envs
        table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table",
                                           actor_id, -1, 0)
        self.gym.set_rigid_body_texture(env_ptr, table_actor, 0, 
                                       gymapi.MESH_VISUAL, texture)
        table_idx = self.gym.get_actor_index(env_ptr, table_actor, gymapi.DOMAIN_SIM)
        
        return {'handle': table_actor, 'index': table_idx}
    
    def _apply_physical_properties(self, env_ptr, table_handle, object_handle):
        """Apply friction and other physical properties to actors."""
        table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
        obj_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
        table_props[0].friction = 1.0
        obj_props[0].friction = 1.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_props)
        self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, obj_props)
    
    def _apply_visual_styling(self, env_ptr, obj_handle, table_handle, goal_handle):
        """Apply colors to actors for visual distinction."""
        obj_color = gymapi.Vec3(90/255, 90/255, 173/255)
        table_color = gymapi.Vec3(150/255, 150/255, 150/255)
        goal_color = gymapi.Vec3(173/255, 90/255, 90/255)
        
        self.gym.set_rigid_body_color(env_ptr, obj_handle, 0, 
                                     gymapi.MESH_VISUAL, obj_color)
        self.gym.set_rigid_body_color(env_ptr, table_handle, 0, 
                                     gymapi.MESH_VISUAL, table_color)
        self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, 
                                     gymapi.MESH_VISUAL, goal_color)
    
    def _setup_camera(self, env_ptr, env_id, camera_props):
        """Setup camera sensor if configured for this environment."""
        camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)
        cam_cfg = self.target_camera_configs[env_id]
        pos = cam_cfg.get("position", self.video_cfg.default_pos)
        target = cam_cfg.get("target", self.video_cfg.default_target)
        self.gym.set_camera_location(camera_handle, env_ptr, 
                                    gymapi.Vec3(*pos), gymapi.Vec3(*target))
        self.camera_handles_map[env_id] = camera_handle
        return camera_handle
    
    def _append_env_data(self, collections, env_data):
        """Append all data from a single environment to collection lists."""
        collections['envs'].append(env_data['env_ptr'])
        collections['hand_indices'].append(env_data['hand']['index'])
        collections['hand_start_states'].append(env_data['hand']['state'])
        collections['object_indices'].append(env_data['clutter']['focus_idx'])
        collections['object_init_state'].append(env_data['clutter']['state'])
        collections['goal_object_indices'].append(env_data['goal']['index'])
        collections['goal_init_state'].append(env_data['goal']['state'])
        collections['table_indices'].append(env_data['table']['index'])
        collections['env_object_scale'].append(env_data['clutter']['scale'])
        collections['env_object_scale_id'].append(env_data['object_config']['code_idx'])
        
        # Store mesh data
        mesh = env_data['clutter']['mesh']
        collections['object_init_mesh']['mesh'].append(mesh['mesh'])
        collections['object_init_mesh']['mesh_vertices'].append(mesh['vertices'])
        collections['object_init_mesh']['mesh_faces'].append(mesh['faces'])
        collections['object_init_mesh']['points'].append(mesh['points'])
        collections['object_init_mesh']['points_centered'].append(mesh['points_centered'])
        collections['object_init_mesh']['pca_axes'].append(mesh['pca_axes'])
        
        # Store clutter indices
        for i, handle in enumerate(self.object_clutter_handles):
            idx = self.gym.get_actor_index(env_data['env_ptr'], handle, gymapi.DOMAIN_SIM)
            collections['object_clutter_indices'][i].append(idx)
        
        # Initialize point handles
        collections['hand_point_handles'].append([])
        collections['object_point_handles'].append([])
    
    def _finalize_tensors(self, collections):
        """Convert all collected lists to PyTorch tensors."""
        self.envs = collections['envs']
        self.hand_indices = to_torch(collections['hand_indices'], dtype=torch.long, device=self.device)
        self.object_indices = to_torch(collections['object_indices'], dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(collections['goal_object_indices'], dtype=torch.long, device=self.device)
        self.table_indices = to_torch(collections['table_indices'], dtype=torch.long, device=self.device)
        self.object_clutter_indices = to_torch(collections['object_clutter_indices'], dtype=torch.long, device=self.device)
        
        self.hand_start_states = to_torch(collections['hand_start_states'], device=self.device).view(self.num_envs, 13)
        self.object_init_state = to_torch(collections['object_init_state'], device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(collections['goal_init_state'], device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.goal_init_state.clone()
        self.goal_init_state = self.goal_states.clone()
        
        self.env_object_scale = collections['env_object_scale']
        self.object_scale_buf = collections['object_scale_buf']
        self.env_object_scale_id = collections['env_object_scale_id']
        
        # Convert mesh data
        self.object_init_mesh = collections['object_init_mesh']
        self.object_init_mesh['points'] = to_torch(
            np.stack(self.object_init_mesh['points'], axis=0), 
            device=self.device, dtype=torch.float
        )
        self.object_init_mesh['points_centered'] = to_torch(
            np.stack(self.object_init_mesh['points_centered'], axis=0), 
            device=self.device, dtype=torch.float
        )
        self.object_init_mesh['pca_axes'] = to_torch(
            np.stack(self.object_init_mesh['pca_axes'], axis=0), 
            device=self.device, dtype=torch.float
        )
        
        self.hand_point_handles = collections['hand_point_handles']
        self.object_point_handles = collections['object_point_handles']
        self.hand_point_indices = to_torch(collections['hand_point_indices'], dtype=torch.long, device=self.device)
        self.object_point_indices = to_torch(collections['object_point_indices'], dtype=torch.long, device=self.device)
        
        # Check if single object type is used
        first_scale_id = self.env_object_scale_id[0]
        self.load_single_object = all(sid == first_scale_id for sid in self.env_object_scale_id)
        
        if self.load_single_object:
            self.object_init_mesh['mesh_vertices'] = to_torch(
                np.stack(self.object_init_mesh['mesh_vertices'], axis=0), 
                device=self.device, dtype=torch.float
            )
            self.object_init_mesh['mesh_faces'] = to_torch(
                np.stack(self.object_init_mesh['mesh_faces'], axis=0), 
                device=self.device, dtype=torch.long
            )
        
        self.env_origin = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

    def _create_objects(self, env_ptr, env_id, object_codes_in_clutter, scale_strs_in_clutter,type=1):
        """Spawn the target object plus surrounding clutter for one env."""


        object_asset_options = gymapi.AssetOptions()
        object_asset_options.flip_visual_attachments = False
        object_asset_options.fix_base_link = False
        object_asset_options.collapse_fixed_joints = True
        object_asset_options.disable_gravity = False

        object_asset_info = self.object_asset_dict[object_codes_in_clutter[0]][scale_strs_in_clutter[0]]
        
        obj_mesh_scale = list(self.cfg['env']['object_code_dict'].values())[0][0]
        if obj_mesh_scale == 0.08: side_len = 0.09
        elif obj_mesh_scale == 0.06: side_len = 0.0675
        elif obj_mesh_scale == 0.10: side_len = 0.1125
        else: print(f"obj_mesh_scale {obj_mesh_scale} not supported"); exit()
        self.side_len = side_len
        
        object_1x1_asset = object_asset_info[0]

        object_1x1_asset_else = self.gym.create_box(self.sim, side_len, side_len, side_len, object_asset_options)

        object_1x2_asset = self.gym.create_box(self.sim, side_len, side_len*2, side_len, object_asset_options)

        object_1x3_asset = self.gym.create_box(self.sim, side_len, side_len*3, side_len, object_asset_options)
        
        for i in range(self.surrounding_obj_num+1):
            object_start_pose_i = gymapi.Transform()
            
            if i == 0:
                object_handle_i = self.gym.create_actor(env_ptr, object_1x1_asset, object_start_pose_i, "object_i", env_id - 5*self.num_envs, 0, 0)
                
                if self.randomize_object_mass:
                    object_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle_i)
                    object_props[0].mass += random.uniform(-self.randomize_object_mass_range, self.randomize_object_mass_range)
                    self.gym.set_actor_rigid_body_properties(env_ptr, object_handle_i, object_props, recomputeInertia=True)
            else:
                rand_val = random.random()
                if rand_val < 0.75:
                    object_handle_i = self.gym.create_actor(env_ptr, object_1x1_asset, object_start_pose_i, "object_i", env_id - 5*self.num_envs, 0, 0)
                elif rand_val < 0.95:
                    object_handle_i = self.gym.create_actor(env_ptr, object_1x2_asset, object_start_pose_i, "object_i", env_id - 5*self.num_envs, 0, 0)
                else:
                    object_handle_i = self.gym.create_actor(env_ptr, object_1x3_asset, object_start_pose_i, "object_i", env_id - 5*self.num_envs, 0, 0)

                if self.randomize_object_mass:
                    object_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle_i)
                    object_props[0].mass += random.uniform(-self.randomize_object_mass_range, self.randomize_object_mass_range)
                    self.gym.set_actor_rigid_body_properties(env_ptr, object_handle_i, object_props, recomputeInertia=True)
            
            self.object_clutter_handles.append(object_handle_i)

            object_idx_i = self.gym.get_actor_index(env_ptr, object_handle_i, gymapi.DOMAIN_SIM)
            self.object_clutter_indices[i].append(object_idx_i)
        
    def _load_dex_hand_assets(self, assets_path):        
        """Load the ShadowHand asset and cache joint metadata."""
        dex_hand_asset_file = "urdf/dummy_leap_vertical_moving.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100 
        asset_options.linear_damping = 100 
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        
        dex_hand_asset = self.gym.load_asset(self.sim, assets_path, dex_hand_asset_file, asset_options)
        self.num_dex_hand_bodies = self.gym.get_asset_rigid_body_count(dex_hand_asset)  # 24
        self.num_dex_hand_shapes = self.gym.get_asset_rigid_shape_count(dex_hand_asset)  # 20
        self.num_dex_hand_dofs = self.gym.get_asset_dof_count(dex_hand_asset)  # 22
        
        self.valid_dex_hand_bodies = [i for i in range(8,self.num_dex_hand_bodies)]


        self.actuated_dof_indices = [i for i in range(self.num_dex_hand_dofs)]
        dex_hand_dof_props = self.gym.get_asset_dof_properties(dex_hand_asset)
        self.dex_hand_dof_lower_limits = []
        self.dex_hand_dof_upper_limits = []
        self.dex_hand_dof_default_pos = []
        self.dex_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_dex_hand_dofs):
            self.dex_hand_dof_lower_limits.append(dex_hand_dof_props['lower'][i])
            self.dex_hand_dof_upper_limits.append(dex_hand_dof_props['upper'][i])
            self.dex_hand_dof_default_pos.append(0.0)
            self.dex_hand_dof_default_vel.append(0.0)
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.dex_hand_dof_lower_limits = to_torch(self.dex_hand_dof_lower_limits, device=self.device)
        self.dex_hand_dof_upper_limits = to_torch(self.dex_hand_dof_upper_limits, device=self.device)
        self.dex_hand_dof_default_pos = to_torch(self.dex_hand_dof_default_pos, device=self.device)
        self.dex_hand_dof_default_vel = to_torch(self.dex_hand_dof_default_vel, device=self.device)

        if 'static_init' in self.config['Modes'] and self.config['Modes']['static_init']:
            hand_shift_x = 0 #-0.35
            hand_shift_y = 0.0
            hand_shift_z = 0.3
            dex_hand_start_pose = gymapi.Transform()
            dex_hand_start_pose.p = gymapi.Vec3(hand_shift_x, hand_shift_y, self.table_dims.z + hand_shift_z)  # gymapi.Vec3(0.1, 0.1, 0.65)
            dex_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)
        else:
            dex_hand_start_pose = gymapi.Transform()
            dex_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, self.table_dims.z + 0.3)  # gymapi.Vec3(0.1, 0.1, 0.65)
            dex_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)

        body_names = {'wrist': 'base', 'palm': 'palm_lower', 'thumb': 'thumb_fingertip',
                      'index': 'fingertip', 'middle': 'fingertip_2', 'ring': 'fingertip_3'}
        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(dex_hand_asset, body_name)
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(dex_hand_asset, name) for name in self.fingertips]

        
        return dex_hand_asset, dex_hand_start_pose, dex_hand_dof_props

    def _load_object_asset_info(self, assets_path, object_code, scale_str):
        """Load a single object mesh plus sampled points and PCA axes."""
        mesh_path = osp.join(assets_path, 'meshdatav3_scaled')
        scaled_object_asset_file = object_code + f"/coacd/coacd_{scale_str}.urdf"
        scaled_object_asset = self.gym.load_asset(self.sim, mesh_path, scaled_object_asset_file, self.object_asset_options)
        scaled_object_mesh_file = os.path.join(mesh_path, object_code + f"/coacd/decomposed_{scale_str}.obj")
        scaled_object_mesh = trimesh.load(scaled_object_mesh_file)
        scaled_object_points, _ = trimesh.sample.sample_surface(scaled_object_mesh, 1024)
        pca = PCA(n_components=3)
        pca.fit(scaled_object_points)
        pca_axes = pca.components_
        scaled_object_pc_file = osp.join(assets_path, 'meshdatav3_pc_fps', object_code + f"/coacd/pc_fps1024_{scale_str}.npy")
        with open(scaled_object_pc_file, 'rb') as f: scaled_object_pc_fps = np.asarray(np.load(f))[:, :3]

        scaled_object_mesh = simplify_trimesh(scaled_object_mesh, ratio=0.1, min_faces=500)

        return [scaled_object_asset, '{}/{}'.format(object_code, scale_str), scaled_object_mesh, scaled_object_pc_fps, scaled_object_pc_fps-scaled_object_pc_fps.mean(0), pca_axes]

    def _load_object_table_goal_assets(self, assets_path, scale2str):
        """Load every object mesh along with goal and table assets."""
        object_asset_dict = {}
        self.use_object_asset_dict = True
        
        self.object_asset_options = gymapi.AssetOptions()
        self.object_asset_options.density = 500
        self.object_asset_options.fix_base_link = False
        self.object_asset_options.use_mesh_materials = True
        self.object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        self.object_asset_options.override_com = True
        self.object_asset_options.override_inertia = True
        self.object_asset_options.vhacd_enabled = True
        self.object_asset_options.vhacd_params = gymapi.VhacdParams()
        self.object_asset_options.vhacd_params.resolution = 300000
        self.object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        
    
        
        object_asset, self.num_object_bodies, self.num_object_shapes = None, -1, -1
        loop = tqdm.tqdm(range(len(self.object_code_list)))
        for object_id in loop:
            object_code = self.object_code_list[object_id]
            loop.set_description('Loading object_mesh {}'.format(object_id))
            if not self.use_object_asset_dict and object_id != 0: continue
            object_asset_dict[object_code] = {}
            for scale in self.object_scale_dict[object_code]:
                scale_str = scale2str[scale]
                object_asset_info = self._load_object_asset_info(assets_path, object_code, scale_str)
                object_asset_dict[object_code][scale_str] = object_asset_info
                self.num_object_bodies = max(self.num_object_bodies, self.gym.get_asset_rigid_body_count(object_asset_info[0]))
                self.num_object_shapes = max(self.num_object_shapes, self.gym.get_asset_rigid_shape_count(object_asset_info[0]))
                if object_asset is None: object_asset = object_asset_info[0]

        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
        self.object_dof_props = self.gym.get_asset_dof_properties(object_asset)
        self.object_dof_lower_limits = to_torch([self.object_dof_props['lower'][i] for i in range(self.num_object_dofs)], device=self.device)
        self.object_dof_upper_limits = to_torch([self.object_dof_props['upper'][i] for i in range(self.num_object_dofs)], device=self.device)

        goal_asset_options = gymapi.AssetOptions()
        goal_asset_options.density = 500
        goal_asset_options.fix_base_link = False
        goal_asset_options.disable_gravity = True
        goal_asset_options.use_mesh_materials = True
        goal_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        goal_asset_options.override_com = True
        goal_asset_options.override_inertia = True
        goal_asset_options.vhacd_enabled = True
        goal_asset_options.vhacd_params = gymapi.VhacdParams()
        goal_asset_options.vhacd_params.resolution = 300000
        goal_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        goal_asset = self.gym.create_sphere(self.sim, 0.01, goal_asset_options)

        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, self.table_dims.x, self.table_dims.y, self.table_dims.z, table_asset_options)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, self.table_dims.z + 0.05)  # gymapi.Vec3(0.0, 0.0, 0.72)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        self.goal_displacement = gymapi.Vec3(-0, 0.0, 0.2) # gymapi.Vec3(-0., 0.0, 0.2)
        self.goal_displacement_tensor = to_torch([self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        goal_start_pose.p.z -= 0.0
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * self.table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        self.table_mesh = trimesh.creation.box(extents=(self.table_dims.x, self.table_dims.y, self.table_dims.z))
        self.table_mesh.vertices += np.array([0.0, 0.0, 0.5 * self.table_dims.z])
        self.table_vertices = torch.tensor(self.table_mesh.vertices, dtype=torch.float).repeat(self.num_envs, 1, 1).to(self.device)
        self.table_faces = torch.tensor(self.table_mesh.faces, dtype=torch.long).repeat(self.num_envs, 1, 1).to(self.device)
        self.table_colors = torch.tensor([0, 0, 0]).repeat(self.table_vertices.shape[0], self.table_vertices.shape[1], 1).to(self.device) / 255.

        return object_asset_dict, goal_asset, table_asset, object_start_pose, goal_start_pose, table_pose



    def pre_physics_step(self, actions):
        """Handle resets and push the latest control actions to the sim."""

        if self.config['Init']: actions *= 0
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            # All envs slated for reset get zeroed actions and fresh randomized states.
            if 'reset_actions' in self.config['Modes'] and self.config['Modes']['reset_actions']: actions[env_ids] *= 0.
            if 'static_init' in self.config['Modes'] and self.config['Modes']['static_init']: actions[env_ids] *= 0.
            self.reset(env_ids, goal_env_ids)

        self.get_pose_quat()

        self.actions = actions.clone().to(self.device)

        self.cur_targets[:, self.actuated_dof_indices[6:]] = scale(self.actions[:, 6:],self.dex_hand_dof_lower_limits[self.actuated_dof_indices[6:]],self.dex_hand_dof_upper_limits[self.actuated_dof_indices[6:]])
        self.cur_targets[:, self.actuated_dof_indices[6:]] = self.act_moving_average * self.cur_targets[:,self.actuated_dof_indices[6:]] + (1.0 - self.act_moving_average) * self.prev_targets[:,self.actuated_dof_indices[6:]]
        
        targets = self.prev_targets[:, self.actuated_dof_indices[:3]] + self.dex_hand_dof_speed_scale * self.dt * self.actions[:, :3] 
        self.cur_targets[:, self.actuated_dof_indices[:3]] = tensor_clamp(targets, self.dex_hand_dof_lower_limits[self.actuated_dof_indices[:3]],self.dex_hand_dof_upper_limits[self.actuated_dof_indices[:3]])

        if self.use_hand_rotation:
            targets = self.prev_targets[:, self.actuated_dof_indices[3:6]] + self.dex_hand_dof_speed_scale * self.dt * self.actions[:, 3:6] * self.hand_rotation_coef   
            self.cur_targets[:, self.actuated_dof_indices[3:6]] = tensor_clamp(targets, self.dex_hand_dof_lower_limits[self.actuated_dof_indices[3:6]],self.dex_hand_dof_upper_limits[self.actuated_dof_indices[3:6]])
                  
        self.prev_targets = self.cur_targets  
        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

    def post_physics_step(self):
        """Refresh rewards/observations and dispatch plugin hooks."""
        self.get_unpose_quat()
        self.progress_buf += 1
        self.randomize_buf += 1
        if self.config['Modes']['train_default']: self.compute_observations_default()
        else: self.compute_observations()
        self.compute_reward(self.actions, self.id)
        self.log_reward_debug()
        if self.step_count % 10 == 0:
            print(f"Step {self.step_count}: "
                f"right_hand_joint_dist (mean) = {self.right_hand_joint_pc_dist.mean().item():.4f}, "
                f"right_hand_dist (mean) = {self.right_hand_pc_dist.mean().item():.4f}, "
                f"right_hand_joint_dist (min) = {self.right_hand_joint_pc_dist.min().item():.4f}, "
                f"right_hand_joint_dist (max) = {self.right_hand_joint_pc_dist.max().item():.4f}")

        self.step_count += 1
        self.global_step += 1
        if self.step_count == self.max_episode_length - 1:
            self.reward_value = self.rew_buf.mean()

        self.plugin_manager.on_step(self)


        if self.viewer:
            if self.debug_viz:
                self.gym.clear_lines(self.viewer)
                self.gym.refresh_rigid_body_state_tensor(self.sim)
                for i in range(self.num_envs):
                    self.add_debug_lines(self.envs[i], self.right_hand_pos[i], self.right_hand_rot[i])
                    self.add_debug_lines(self.envs[i], torch.tensor([0,0,0.601], device=self.device, dtype=torch.float32), torch.tensor([0,0,0,1], device=self.device, dtype=torch.float32))
                    self.add_debug_lines(self.envs[i], self.hand_body_pos[0], self.right_hand_rot[0])
                    
                 

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.3)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.3)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.3)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])
    
    def _activate_plugins(self):
        if self._plugins_registered:
            return
        if self.visualization_cfg.enable:
            self.plugin_manager.register(VisualizationPlugin(self.visualization_cfg, self.run_metadata), self)
        if self.video_cfg.enable:
            self.plugin_manager.register(VideoCapturePlugin(self.video_cfg, self.run_metadata), self)
        self._plugins_registered = True

    def _finalize_plugins(self):
        if self._plugins_finalized:
            return
        self._plugins_finalized = True
        self.plugin_manager.finalize(self)

    def compute_reward(self, actions, id=-1):
        """Assemble the full reward tensor and auxiliary success metrics."""
        
        object_init_pos = self.object_init_state[:, :3]
        dist_target_to_centroid = torch.zeros_like(self.object_pos[:, 0])
        
        

        if self.surrounding_obj_num > 0 and self.prev_obstacle_distances is not None:
            # Interactive reward balances obstacle clearing (singulation) and target stability.
            weights = self.config.get('Weights', {})
            rew_scale_clearing = weights.get('interactive_clearing_scale', 10.0)
            rew_scale_stability = weights.get('interactive_stability_scale', 5.0)
            roi_radius = weights.get('interactive_roi_radius', 0.20)
            max_push_reward_clip = weights.get('interactive_max_push_clip', 0.05)
            
            target_object_pos = self.object_pos
            obstacle_positions = []
            
            for i in range(self.surrounding_obj_num + 1):
                if i != self.obj_focus_id:
                    obstacle_indices = self.object_clutter_indices[i][:self.num_envs]
                    obstacle_pos = self.root_state_tensor[obstacle_indices, 0:3]
                    obstacle_positions.append(obstacle_pos)

            if len(obstacle_positions) > 0:
                real_obstacles_tensor = torch.stack(obstacle_positions, dim=1)
                
                clutter_centroid = torch.mean(real_obstacles_tensor, dim=1) # (Num_Envs, 3)
                
                dist_target_to_centroid = torch.norm(self.object_pos - clutter_centroid, p=2, dim=-1)
                
            else:
                dist_target_to_centroid = torch.zeros_like(self.object_pos[:, 0])

            self.current_dist_to_centroid = dist_target_to_centroid
            
            while len(obstacle_positions) < 8:
                obstacle_positions.append(torch.zeros_like(target_object_pos))
            
            obstacle_pos_tensor = torch.stack(obstacle_positions, dim=1)
            
            max_finger_dist = weights.get('max_finger_dist', 0.20)
            max_hand_dist = weights.get('max_hand_dist', 0.06)
            target_separation_distance = weights.get('target_separation_distance', 0.02)
            is_joint_dist_close = (self.right_hand_joint_pc_dist <= max_finger_dist).float()
            is_hand_dist_close = (self.right_hand_pc_dist <= max_hand_dist).float()
            is_singulation_complete = (self.min_singulation_distance >= target_separation_distance).float()
            

        elif self.surrounding_obj_num > 0:
            if self.prev_obstacle_distances is None:
                if hasattr(self, 'object_distances') and self.object_distances is not None:
                    self.prev_obstacle_distances = self.object_distances.transpose(0, 1).clone()
        
        (self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:],
         self.current_successes[:], self.consecutive_successes[:], self.final_successes[:], self.succ_steps[:],
         self.singulation_success) = compute_hand_reward(
            self.config['Modes'], self.config['Weights'],
            self.object_init_z, self.delta_target_hand_pos, self.delta_target_hand_rot,
            self.id, self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, 
            self.successes, self.current_successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, 
            self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, 
            self.fall_penalty, self.max_consecutive_successes, self.av_factor, self.goal_cond,
            self.object_points, self.right_hand_pc_dist, self.right_hand_finger_pc_dist, self.right_hand_joint_pc_dist, self.right_hand_body_pc_dist,
            self.mean_singulation_distance, self.min_singulation_distance, self.table_dims.z, self.invalid_env_mask, self.invalid_env_num, self.remove_invalid_arrangement, self.succ_steps[:],
            self.surrounding_obj_num, dist_target_to_centroid
        )

        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes
        self.extras['final_successes'] = self.final_successes
        self.singulation_success_rate = float(self.singulation_success.mean().item())
        self.extras['singulation_success'] = self.singulation_success
        self.extras['singulation_success_rate'] = self.singulation_success_rate

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes / self.total_resets))

    def log_reward_debug(self):
        """定期打印关键奖励分量和 hold_flag 统计，帮助观察训练过程。"""
        if self.debug_log_interval <= 0:
            return
        if (self.step_count % self.debug_log_interval) != 0:
            return

        
        weights = self.config.get('Weights', {})
        if self.surrounding_obj_num > 0:
            if hasattr(self, 'obstacle_pos'): 
                clutter_centroid = torch.mean(self.obstacle_pos, dim=1) # (N, 3)
                dist_target_to_centroid = torch.norm(self.object_pos - clutter_centroid, p=2, dim=-1)
            else:
                dist_target_to_centroid = torch.zeros_like(self.object_pos[:, 0])
            if hasattr(self, 'current_dist_to_centroid'):
                dist_target_to_centroid = self.current_dist_to_centroid
            else:
                dist_target_to_centroid = torch.zeros_like(self.object_pos[:, 0])
        else:
            dist_target_to_centroid = torch.zeros_like(self.object_pos[:, 0])
        max_finger_dist_log = weights.get('max_finger_dist', 0.10) 
        max_hand_dist_log = weights.get('max_hand_dist', 0.06)
        target_separation_dist_log = weights.get('target_separation_distance', 0.02)

        is_joint_close = (self.right_hand_joint_pc_dist <= max_finger_dist_log).int()
        is_hand_close = (self.right_hand_pc_dist <= max_hand_dist_log).int()
        hold_value = 2
        if self.surrounding_obj_num > 0:
            is_sing_complete = (dist_target_to_centroid >= target_separation_dist_log).int()
            
            hold_flag = is_joint_close + is_sing_complete
        else:
            hold_flag = is_joint_close + is_hand_close

        hold_ratio = torch.mean((hold_flag == hold_value).float()).item()
        goal_dist = torch.norm(self.goal_pos - self.object_pos, p=2, dim=-1)
    
        fingertip_pos = torch.stack(
            [self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_th_pos],
            dim=1
        )
        fingertip_center_pos = torch.mean(fingertip_pos, dim=1)
        encapsulation_dist = torch.norm(self.object_pos - fingertip_center_pos, p=2, dim=-1)

        init_reward_val = (
            weights.get('right_hand_dist', 0.0) * self.right_hand_pc_dist +
            weights.get('right_hand_joint_dist', 0.0) * self.right_hand_joint_pc_dist
        )

        goal_rew_val = torch.where(
            hold_flag == hold_value, 0.9 - 2.0 * goal_dist, torch.zeros_like(goal_dist)
        )

        lowest = torch.min(self.object_points[:, :, -1], dim=1)[0]
        table_z = self.table_dims.z
        hand_up_val = torch.zeros_like(goal_dist)
        hand_up_val = torch.where(
            lowest >= table_z - 0.01,
            torch.where(hold_flag == hold_value, 0.1 - 0.3 * self.actions[:, 2], hand_up_val),
            hand_up_val
        )
        hand_up_val = torch.where(
            lowest >= table_z - 0.01 + 0.2,
            torch.where(
                hold_flag == hold_value,
                0.2 + weights.get('hand_up_goal_dist', 0.0) * (0.2 - goal_dist),
                hand_up_val
            ),
            hand_up_val
        )

        max_goal_dist = weights.get('max_goal_dist', 0.05)
        bonus_val = torch.where(
            hold_flag == hold_value,
            torch.where(goal_dist <= max_goal_dist, 1.0 / (1 + 10 * goal_dist), torch.zeros_like(goal_dist)),
            torch.zeros_like(goal_dist)
        )

        singulation_scale = weights.get('singulation_reward_scale', 50.0)
        singulation_val = singulation_scale * torch.clamp(self.min_singulation_distance, 0.0, 0.03)

        encapsulation_val = weights.get('encapsulation', 0.0) * encapsulation_dist

        finger_vecs = fingertip_pos - self.right_hand_pos.unsqueeze(1)
        vec_lengths = torch.norm(finger_vecs, p=2, dim=-1, keepdim=True)
        finger_dirs = finger_vecs / (vec_lengths + 1e-6)
        avg_finger_z_proj = torch.mean(finger_dirs[:, :, 2], dim=1)
        r_finger_vertical = -1.0 * avg_finger_z_proj
        r_top_down = r_finger_vertical * 0.5
        top_down_scale = weights.get('top_down_reward_scale', 1.0)
        top_down_val = top_down_scale * r_top_down

        grasp_reward_val = (
            weights.get('right_hand_joint_dist', 0.0) * self.right_hand_joint_pc_dist +
            weights.get('right_hand_finger_dist', 0.0) * self.right_hand_finger_pc_dist +
            weights.get('right_hand_dist', 0.0) * self.right_hand_pc_dist +
            weights.get('goal_dist', 0.0) * goal_dist +
            weights.get('goal_rew', 0.0) * goal_rew_val +
            weights.get('hand_up', 0.0) * hand_up_val +
            weights.get('bonus', 0.0) * bonus_val
        )
        total_reward_val = torch.where(hold_flag != hold_value, init_reward_val, grasp_reward_val)
        total_reward_val = total_reward_val + singulation_val + encapsulation_val

        print(f"--- Step {self.step_count} Reward Debug ---")
        print(f"  Hold ratio (flag=={hold_value}): {hold_ratio:.4f}")
        print(f"  Hand joint dist mean: {self.right_hand_joint_pc_dist.mean().item():.4f} "
              f"(min: {self.right_hand_joint_pc_dist.min().item():.4f}, "
              f"max: {self.right_hand_joint_pc_dist.max().item():.4f})")
        print(f"  Hand palm dist mean: {self.right_hand_pc_dist.mean().item():.4f}")
        print(f"  Goal dist mean: {goal_dist.mean().item():.4f}")
        print(f"  Encapsulation dist mean: {encapsulation_dist.mean().item():.4f}")
        print(f"  dist_target_to_centroid mean: {dist_target_to_centroid.mean().item():.4f}")
        print(f"  Init reward mean: {init_reward_val.mean().item():.4f}")
        print(f"  Goal reward mean: {goal_rew_val.mean().item():.4f}")
        print(f"  Hand-up reward mean: {hand_up_val.mean().item():.4f}")
        print(f"  Bonus mean: {bonus_val.mean().item():.4f}")
        print(f"  Singulation mean: {singulation_val.mean().item():.4f}")
        print(f"  Encapsulation mean: {encapsulation_val.mean().item():.4f}")
        print(f"  Top-down reward mean: {top_down_val.mean().item():.4f} (r_top_down: {r_top_down.mean().item():.4f})")
        print(f"  Total reward mean: {total_reward_val.mean().item():.4f}")


    def compute_observations(self):
        """Gather simulator state into the observation buffer."""
        self._refresh_simulator_tensors()
        self._extract_object_state()
        self._compute_object_point_clouds()
        self._extract_hand_state()
        self._compute_finger_positions()
        self._extract_goal_state()
        self._compute_coordinate_transforms()
        self._compute_point_cloud_distances()
        self.compute_full_state()
    
    def _refresh_simulator_tensors(self):
        """Refresh all simulator state tensors."""
        refresh_ops = [
            lambda: self.gym.refresh_dof_state_tensor(self.sim),
            lambda: self.gym.refresh_actor_root_state_tensor(self.sim),
            lambda: self.gym.refresh_rigid_body_state_tensor(self.sim),
            lambda: self.gym.refresh_jacobian_tensors(self.sim),
            lambda: self.gym.refresh_dof_force_tensor(self.sim),
        ]
        for op in refresh_ops:
            op()
    
    def _extract_object_state(self):
        """Extract object pose, position, rotation, and velocities."""
        # Use same indexing as original code
        # Ensure object_rot has shape [num_envs, 4] by taking only first num_envs indices
        obj_indices = self.object_indices[:self.num_envs]
        self.object_pose = self.root_state_tensor[obj_indices, 0:7]
        self.object_pos = self.root_state_tensor[obj_indices, 0:3]
        self.object_rot = self.root_state_tensor[obj_indices, 3:7]
        self.object_linvel = self.root_state_tensor[obj_indices, 7:10]
        self.object_angvel = self.root_state_tensor[obj_indices, 10:13]
        
        # Compute derived positions
        forward_vec_base = to_torch([1, 0, 0], device=self.device) * 0.04
        forward_vec = forward_vec_base.repeat(self.num_envs, 1)
        self.object_handle_pos = self.object_pos
        # Manual quaternion rotation to avoid shape issues
        # q * v * q^-1 where q is [N, 4] and v is [N, 3]
        quat_xyz = self.object_rot[:, :3]  # [N, 3]
        quat_w = self.object_rot[:, 3:4]  # [N, 1]
        # v' = v + 2 * cross(q_xyz, cross(q_xyz, v) + q_w * v)
        cross1 = torch.cross(quat_xyz, forward_vec, dim=-1)  # [N, 3]
        cross2 = quat_w * forward_vec + cross1  # [N, 3]
        rotated = forward_vec + 2.0 * torch.cross(quat_xyz, cross2, dim=-1)  # [N, 3]
        self.object_back_pos = self.object_pos + rotated
    
    def _compute_object_point_clouds(self):
        """Compute object point clouds in world and centered coordinates."""
        # Ensure shapes are correct: quat [N, 4], points [N, M, 3]
        # object_rot should be [N, 4], but might be flattened, reshape it
        quat = self.object_rot.reshape(-1, 4)
        points = self.object_init_mesh['points'].reshape(self.num_envs, -1, 3)
        rotated_points = batch_quat_apply(quat, points)
        self.object_points = rotated_points + self.object_pos.unsqueeze(1)
        points_centered = self.object_init_mesh['points_centered'].reshape(self.num_envs, -1, 3)
        self.object_points_centered = batch_quat_apply(quat, points_centered)
        
        # Compute visual features if saving is enabled
        if self.config.get('Save', False):
            with torch.no_grad():
                pc_input = self.object_points_centered.permute(0, 2, 1)
                features, _ = self.object_visual_encoder(pc_input)
                normalized = (features.squeeze(-1) - self.object_visual_scaler_mean) / self.object_visual_scaler_scale
                self.object_points_visual_features = normalized.float()
    
    def _extract_hand_state(self):
        """Extract hand position and rotation with offset adjustments."""
        palm_idx = self.hand_body_idx_dict['palm']
        base_pos = self.rigid_body_states[:, palm_idx, 0:3]
        base_rot = self.rigid_body_states[:, palm_idx, 3:7]
        
        # Apply multiple offset adjustments
        offset_vectors = [
            (to_torch([0, 0, 1], device=self.device), -0.04),
            (to_torch([0, 1, 0], device=self.device), -0.05),
            (to_torch([1, 0, 0], device=self.device), -0.01),
        ]
        
        adjusted_pos = base_pos.clone()
        for vec, scale in offset_vectors:
            offset = quat_apply(base_rot, vec.repeat(self.num_envs, 1) * scale)
            adjusted_pos = adjusted_pos + offset
        
        self.right_hand_pos = adjusted_pos
        self.right_hand_rot = base_rot
        
        # Extract joint positions and rotations
        self.hand_joint_pos = self.rigid_body_states[:, self.valid_dex_hand_bodies, 0:3]
        self.hand_joint_rot = self.rigid_body_states[:, self.valid_dex_hand_bodies, 3:7]
        self.hand_body_pos = self.hand_joint_pos
    
    def _compute_finger_positions(self):
        """Compute positions for all fingertips with appropriate offsets."""
        finger_configs = [
            ('index', 'ff'),
            ('middle', 'mf'),
            ('ring', 'rf'),
            ('thumb', 'th'),
        ]
        
        # Determine finger shift amount
        shift_amount = 0.01 if (self.config.get('Modes', {}).get('half_finger_shift', False)) else 0.02
        self.finger_shift = shift_amount
        shift_vector = to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * shift_amount
        
        for body_name, prefix in finger_configs:
            body_idx = self.hand_body_idx_dict[body_name]
            base_pos = self.rigid_body_states[:, body_idx, 0:3]
            base_rot = self.rigid_body_states[:, body_idx, 3:7]
            offset = quat_apply(base_rot, shift_vector)
            adjusted_pos = base_pos + offset
            
            setattr(self, f'right_hand_{prefix}_pos', adjusted_pos)
            setattr(self, f'right_hand_{prefix}_rot', base_rot)
        
        # Extract fingertip states
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]

    def compute_reward_breakdown(self, actions):
        """
        计算奖励分解，用于分析各个奖励分量的贡献。
        
        Args:
            actions: 当前动作
            
        Returns:
            dict: 奖励分量字典
        """
        breakdown = {}
        
        # 基础距离奖励
        goal_dist = torch.norm(self.goal_pos - self.object_pos, p=2, dim=-1)
        breakdown['goal_distance'] = goal_dist.mean().item()
        
        # 手部距离奖励
        breakdown['hand_palm_dist'] = self.right_hand_pc_dist.mean().item()
        breakdown['hand_joint_dist'] = self.right_hand_joint_pc_dist.mean().item()
        breakdown['hand_finger_dist'] = self.right_hand_finger_pc_dist.mean().item()
        
        # 封装距离
        fingertip_pos = torch.stack([
            self.right_hand_ff_pos, self.right_hand_mf_pos,
            self.right_hand_rf_pos, self.right_hand_th_pos
        ], dim=1)
        fingertip_center = torch.mean(fingertip_pos, dim=1)
        encapsulation_dist = torch.norm(self.object_pos - fingertip_center, p=2, dim=-1)
        breakdown['encapsulation_dist'] = encapsulation_dist.mean().item()
        
        # 分离距离（如果有障碍物）
        if self.surrounding_obj_num > 0 and hasattr(self, 'min_singulation_distance'):
            breakdown['min_singulation_dist'] = self.min_singulation_distance.mean().item()
            breakdown['mean_singulation_dist'] = self.mean_singulation_distance.mean().item()
        
        # 动作惩罚
        action_penalty = torch.sum(actions ** 2, dim=-1)
        breakdown['action_penalty'] = action_penalty.mean().item()
        
        return breakdown   

    def _extract_goal_state(self):
        """Extract goal pose, position, and rotation."""
        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
    
    def _compute_coordinate_transforms(self):
        """Define and apply coordinate transformation functions."""
        obj_rot_conj = quat_conjugate(self.object_rot)
        
        def transform_vec_to_obj(vec):
            return quat_apply(obj_rot_conj, vec - self.object_pos)
        
        def transform_quat_to_obj(quat):
            return quat_mul(obj_rot_conj, quat)
        
        # Compute DOF position and relative transforms
        self.dof_pos = self.dex_hand_dof_pos
        self.delta_target_hand_pos = transform_vec_to_obj(self.right_hand_pos) - self.target_hand_pos
        self.rel_hand_rot = transform_quat_to_obj(self.right_hand_rot)
        self.delta_target_hand_rot = quat_mul(self.rel_hand_rot, quat_conjugate(self.target_hand_rot))
    
    def _compute_point_cloud_distances(self):
        """Compute distances between hand components and object point cloud."""
        pc = self.object_points
        hand_pos_expanded = self.right_hand_pos.unsqueeze(1)
        
        # Use memory-efficient batch processing for all distance computations
        batch_size = 64  # Process 64 environments at a time
        
        # Hand palm to point cloud distance
        palm_dist = torch.zeros(self.num_envs, device=self.device, dtype=self.right_hand_pos.dtype)
        for i in range(0, self.num_envs, batch_size):
            end_idx = min(i + batch_size, self.num_envs)
            batch_hand_pos = hand_pos_expanded[i:end_idx]
            batch_pc = pc[i:end_idx]
            batch_dist = batch_sided_distance(batch_hand_pos, batch_pc).squeeze(-1)
            palm_dist[i:end_idx] = batch_dist
        self.right_hand_pc_dist = torch.clamp(palm_dist, max=0.5)
        
        # Finger positions to point cloud distance
        finger_positions = torch.stack([
            self.right_hand_ff_pos, self.right_hand_mf_pos,
            self.right_hand_rf_pos, self.right_hand_th_pos
        ], dim=1)
        self.right_hand_finger_pos = finger_positions
        finger_dist = torch.zeros(self.num_envs, device=self.device, dtype=finger_positions.dtype)
        for i in range(0, self.num_envs, batch_size):
            end_idx = min(i + batch_size, self.num_envs)
            batch_finger_pos = finger_positions[i:end_idx]
            batch_pc = pc[i:end_idx]
            batch_dist = torch.sum(batch_sided_distance(batch_finger_pos, batch_pc), dim=-1)
            finger_dist[i:end_idx] = batch_dist
        self.right_hand_finger_pc_dist = torch.clamp(finger_dist, max=3.0)
        
        # Joint positions to point cloud distance - use memory-efficient batch processing
        # Split into smaller batches to avoid OOM
        batch_size = 64  # Process 64 environments at a time
        num_joints = self.hand_joint_pos.shape[1]
        joint_batch_dist = torch.zeros((self.num_envs, num_joints), device=self.device, dtype=self.hand_joint_pos.dtype)
        
        for i in range(0, self.num_envs, batch_size):
            end_idx = min(i + batch_size, self.num_envs)
            batch_joint_pos = self.hand_joint_pos[i:end_idx]
            batch_pc = pc[i:end_idx]
            batch_dist = batch_sided_distance(batch_joint_pos, batch_pc)
            joint_batch_dist[i:end_idx] = batch_dist
        
        joint_dist = torch.sum(joint_batch_dist, dim=-1) * 5.0 / num_joints
        self.right_hand_joint_pc_batch_dist = joint_batch_dist
        self.right_hand_joint_pc_dist = torch.clamp(joint_dist, max=3.0)
        
        # Body positions to point cloud distance - use memory-efficient batch processing
        num_bodies = self.hand_body_pos.shape[1]
        body_batch_dist = torch.zeros((self.num_envs, num_bodies), device=self.device, dtype=self.hand_body_pos.dtype)
        
        for i in range(0, self.num_envs, batch_size):
            end_idx = min(i + batch_size, self.num_envs)
            batch_body_pos = self.hand_body_pos[i:end_idx]
            batch_pc = pc[i:end_idx]
            batch_dist = batch_sided_distance(batch_body_pos, batch_pc)
            body_batch_dist[i:end_idx] = batch_dist
        
        body_dist = torch.sum(body_batch_dist, dim=-1) * 5.0 / num_bodies
        self.right_hand_body_pc_batch_dist = body_batch_dist
        self.right_hand_body_pc_dist = torch.clamp(body_dist, max=3.0)

    def get_unpose_quat(self):
        """Prepare quaternions that rotate data into object space."""
        self.unpose_z_theta_quat = quat_from_euler_xyz(torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta), -self.z_theta)
        return

    def unpose_point(self, point):
        """Rotate a world-space point into the object frame."""
        return self.unpose_vec(point)
        return point

    def unpose_vec(self, vec):
        """Rotate a world-space vector into the object frame."""
        return quat_apply(self.unpose_z_theta_quat, vec)
        return vec

    def unpose_quat(self, quat):
        """Apply the unpose quaternion to another quaternion."""
        return quat_mul(self.unpose_z_theta_quat, quat)
        return quat

    def find_nearest_points(self, query_points, reference_points):
        """找到查询点到参考点的最近点"""
        distances = self.compute_pairwise_distances(query_points, reference_points)
        min_distances, nearest_indices = torch.min(distances, dim=-1)
        return min_distances, nearest_indices
    
    def compute_relative_position(self, pos1, pos2):
        """计算相对位置"""
        return pos2 - pos1
        
    def unpose_state(self, state):
        """Transform a rigid-body state into the object-aligned frame."""
        state = state.clone()
        state[:, 0:3] = self.unpose_point(state[:, 0:3])
        state[:, 3:7] = self.unpose_quat(state[:, 3:7])
        state[:, 7:10] = self.unpose_vec(state[:, 7:10])
        state[:, 10:13] = self.unpose_vec(state[:, 10:13])
        return state
    
    def get_pose_quat(self):
        """Prepare quaternions that rotate data back to world space."""
        self.pose_z_theta_quat = quat_from_euler_xyz(torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta), self.z_theta)
        return

    def pose_vec(self, vec):
        """Rotate a vector from object space into the world frame."""
        return quat_apply(self.pose_z_theta_quat, vec)

    def pose_point(self, point):
        """Rotate a point from object space into the world frame."""
        return self.pose_vec(point)

    def pose_quat(self, quat):
        """Apply the pose quaternion to another quaternion."""
        return quat_mul(self.pose_z_theta_quat, quat)


    def compute_full_state(self):
        """Assemble auxiliary tensors such as distances and forces."""

        self.get_unpose_quat()
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ## 52
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ## 24

        obs_dict = dict()
        hand_dof_pos = unscale(self.dex_hand_dof_pos, self.dex_hand_dof_lower_limits, self.dex_hand_dof_upper_limits)
        hand_dof_vel = self.vel_obs_scale * self.dex_hand_dof_vel
        hand_dof_force = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]
        if self.include_dummy_dofs:
            obs_dict['hand_dofs'] = torch.cat([hand_dof_pos, hand_dof_vel, hand_dof_force], dim=-1)
        else:
            obs_dict['hand_dofs'] = torch.cat([hand_dof_pos[:,6:], hand_dof_vel[:,6:], hand_dof_force[:,6:]], dim=-1) # remove the first 6 dummy dofs
        
        aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        for i in range(int(self.num_fingertips)): aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])

        hand_pos = self.unpose_point(self.right_hand_pos)
        hand_pos[:, 2] -= self.table_dims.z
        hand_euler_xyz = get_euler_xyz(self.unpose_quat(self.hand_orientations[self.hand_indices, :]))

        obs_dict['hand_states'] = torch.cat([hand_pos, hand_euler_xyz[0].unsqueeze(-1), hand_euler_xyz[1].unsqueeze(-1), hand_euler_xyz[2].unsqueeze(-1)], dim=-1)

        self.actions[:, 0:3] = self.unpose_vec(self.actions[:, 0:3])
        self.actions[:, 3:6] = self.unpose_vec(self.actions[:, 3:6])
        obs_dict['actions'] = self.actions

        object_pos = self.unpose_point(self.object_pose[:, 0:3])  # 3
        object_pos[:, 2] -= self.table_dims.z
        object_rot = self.unpose_quat(self.object_pose[:, 3:7])  # 4
        object_linvel = self.unpose_vec(self.object_linvel)  # 3
        object_angvel = self.vel_obs_scale * self.unpose_vec(self.object_angvel)  # 3
        object_hand_dist = self.unpose_vec(self.goal_pos - self.object_pos)  # 3
        obs_dict['objects'] = torch.cat([object_pos, object_rot, object_linvel, object_angvel, object_hand_dist], dim=-1)

        obs_dict['object_visual'] = self.object_points_visual_features * 0
        if self.algo == 'ppo' and 'zero_object_visual_feature' in self.config['Modes'] and self.config['Modes']['zero_object_visual_feature']:
            obs_dict['object_visual'] = torch.zeros_like(obs_dict['object_visual'], device=self.device)

        obs_dict['times'] = torch.cat([self.progress_buf.unsqueeze(-1), compute_time_encoding(self.progress_buf, 28)], dim=-1)
            
        if 'encode_hand_object_dist' in self.config['Modes'] and self.config['Modes']['encode_hand_object_dist']:

            obs_dict['hand_objects'] = self.right_hand_joint_pc_batch_dist

        

        # Track pairwise spacings so reward can encourage obstacle separation.
        object_distances = []
        projected_distances = []
        singulation_distances = []
        # Ensure indices are properly shaped
        target_indices = self.object_clutter_indices[self.obj_focus_id][:self.num_envs]
        target_object_pos = self.root_state_tensor[target_indices, 0:3]
        proj_target_object_pos = self.root_state_tensor[target_indices, 0:2]

        for i in range(self.surrounding_obj_num+1):
            if i != self.obj_focus_id:
                other_indices = self.object_clutter_indices[i][:self.num_envs]
                other_object_pos = self.root_state_tensor[other_indices, 0:3]
                dist = torch.norm(target_object_pos - other_object_pos, p=2, dim=-1)
                object_distances.append(dist)

                proj_other_object_pos = self.root_state_tensor[other_indices, 0:2]
                proj_dist = torch.norm(proj_target_object_pos - proj_other_object_pos, p=2, dim=-1)
                projected_distances.append(proj_dist)

                grid_positions = self.object_position_assignments[:, i]  # Shape: (num_envs,)

                init_x = self.sudoku_grid_x[grid_positions]
                init_y = self.sudoku_grid_y[grid_positions]
                init_object_pos = torch.stack([init_x, init_y], dim=1)

                
                sing_dist = torch.norm(init_object_pos - proj_other_object_pos, p=2, dim=-1)
                singulation_distances.append(sing_dist)

        while len(object_distances) < 8:
            object_distances.append(torch.zeros_like(target_object_pos[:, 0]))
            projected_distances.append(torch.zeros_like(proj_target_object_pos[:, 0]))
            singulation_distances.append(torch.zeros_like(proj_target_object_pos[:, 0]))

        self.object_distances = torch.stack(object_distances, dim=0)
        non_zero_mask = self.object_distances > 0
        self.mean_object_distance = torch.where(
            torch.any(non_zero_mask, dim=0),
            torch.sum(self.object_distances, dim=0) / torch.count_nonzero(non_zero_mask, dim=0).float(),
            torch.zeros_like(self.object_distances[0])
        )
        self.min_object_distance = torch.where(
            torch.any(non_zero_mask, dim=0),
            torch.min(torch.where(non_zero_mask, self.object_distances, torch.inf), dim=0)[0],
            torch.zeros_like(self.object_distances[0])
        )

        self.projected_distances = torch.stack(projected_distances, dim=0)
        non_zero_mask = self.projected_distances > 0
        self.mean_projected_distance = torch.where(
            torch.any(non_zero_mask, dim=0),
            torch.sum(self.projected_distances, dim=0) / torch.count_nonzero(non_zero_mask, dim=0).float(),
            torch.zeros_like(self.projected_distances[0])
        )
        self.min_projected_distance = torch.where(
            torch.any(non_zero_mask, dim=0),
            torch.min(torch.where(non_zero_mask, self.projected_distances, torch.inf), dim=0)[0],
            torch.zeros_like(self.projected_distances[0])
        )

        self.singulation_distances = torch.stack(singulation_distances, dim=0)
        non_zero_mask = self.singulation_distances > 0
        self.mean_singulation_distance = torch.where(
            torch.any(non_zero_mask, dim=0),
            torch.sum(self.singulation_distances, dim=0) / torch.count_nonzero(non_zero_mask, dim=0).float(),
            torch.zeros_like(self.singulation_distances[0])
        )
        self.min_singulation_distance = torch.where(
            torch.any(non_zero_mask, dim=0),
            torch.min(torch.where(non_zero_mask, self.singulation_distances, torch.inf), dim=0)[0],
            torch.zeros_like(self.singulation_distances[0])
        )

        self.invalid_env_mask = torch.any(self.singulation_distances > 0.3, dim=0)
        self.invalid_env_num = torch.sum(self.invalid_env_mask).item()
        obs_dict['singulation_distances'] = self.object_distances.transpose(0,1)


        self.obs_names = ['hand_dofs', 'hand_states', 'actions', 'objects', 'object_visual', 'times', 'hand_objects', 'singulation_distances']
        
        self.obs_buf = torch.cat([obs_dict[name] for name in self.obs_names if name in obs_dict], dim=-1)
        
        
        start_temp, self.obs_infos = 0, {'names': [name for name in self.obs_names if name in obs_dict], 'intervals': {}}
        for name in self.obs_names:
            if name not in obs_dict: continue
            self.obs_infos['intervals'][name] = [start_temp, start_temp + obs_dict[name].shape[-1]]
            start_temp += obs_dict[name].shape[-1]

        return

    def reset_target_pose(self, env_ids, apply_reset=False):
        """Reset the goal marker position and orientation."""

        goal_position = torch.tensor([0.0, 0.0, self.table_dims.z + 0.305], device=self.device)
        self.goal_states[env_ids, 0:3] = goal_position


        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3]   #+ self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset(self, env_ids, goal_env_ids):
        """Randomize the environment state and rebuild cached tensors."""
        self.step_count = 0

        if self.reset_count > 0: # skip the first reset when init
            # Print consolidated success stats so users can monitor curriculum progress.
            if self.remove_invalid_arrangement:
                if self.num_envs-self.invalid_env_num != 0:
                    print("successes", torch.sum(self.successes[:]).item(),"out of", self.num_envs-self.invalid_env_num, "rate", torch.sum(self.successes[:]).item()/(self.num_envs-self.invalid_env_num))
                else:
                    print("successes", torch.sum(self.successes[:]).item(),"out of", self.num_envs-self.invalid_env_num, "rate", 0)
                self.total_valid_successes.append(torch.sum(self.successes[:]).item())
                self.total_valid_envs.append(self.num_envs-self.invalid_env_num)
            else:
                print("successes", torch.sum(self.successes[:]).item(),"out of", self.num_envs, "rate", torch.sum(self.successes[:]).item()/self.num_envs)
                self.total_valid_successes.append(torch.sum(self.successes[:]).item())
                self.total_valid_envs.append(self.num_envs)
            self.cumulative_successes = sum(self.total_valid_successes)/sum(self.total_valid_envs)
            self.current_success_rate = torch.sum(self.successes[:]).item()/self.num_envs
            print('=== Cumulative Success {} out of {}, rate {} ==='.format(sum(self.total_valid_successes), sum(self.total_valid_envs), self.cumulative_successes))
            
            # 更新累计分离成功率统计
            if self.remove_invalid_arrangement:
                singulation_success_count = torch.sum(self.singulation_success[:]).item()
                valid_env_count = self.num_envs - self.invalid_env_num
                self.total_valid_singulation_successes.append(singulation_success_count)
                self.total_valid_singulation_envs.append(valid_env_count)
            else:
                singulation_success_count = torch.sum(self.singulation_success[:]).item()
                self.total_valid_singulation_successes.append(singulation_success_count)
                self.total_valid_singulation_envs.append(self.num_envs)
            
            # 计算累计分离成功率
            total_singulation_successes = sum(self.total_valid_singulation_successes)
            total_singulation_envs = sum(self.total_valid_singulation_envs)
            self.cumulative_singulation_success_rate = total_singulation_successes / total_singulation_envs if total_singulation_envs > 0 else 0.0
            print('=== Cumulative Singulation Success {} out of {}, rate {:.4f} ==='.format(
                total_singulation_successes, total_singulation_envs, self.cumulative_singulation_success_rate))
            push_threshold = 0.015
            push_success = (self.min_singulation_distance > push_threshold).float()
            if self.remove_invalid_arrangement:
                valid_mask = ~self.invalid_env_mask
                valid_push = push_success[valid_mask]
                valid_total = valid_mask.sum().item()
                push_count = valid_push.sum().item() if valid_push.numel() > 0 else 0.0
                push_rate = (push_count / valid_total) if valid_total > 0 else 0.0
            else:
                push_count = push_success.sum().item()
                push_rate = push_count / self.num_envs
                valid_total = self.num_envs
            print("Singulation Success {} out of {} rate {:.3f} (threshold {:.3f})".format(push_count, valid_total, push_rate, push_threshold))
            self.plugin_manager.notify(
                "curriculum_checkpoint",
                self,
                step=self.reset_count,
                success_rate=float(self.current_success_rate),
            )
            
            if self.surrounding_obj_num > 0:
                valid_mean_distances = self.mean_singulation_distance[~self.invalid_env_mask] if self.remove_invalid_arrangement else self.mean_singulation_distance
                valid_min_distances = self.min_singulation_distance[~self.invalid_env_mask] if self.remove_invalid_arrangement else self.min_singulation_distance
                
                if len(valid_mean_distances) > 0:
                    avg_mean_distance = torch.mean(valid_mean_distances).item()
                    avg_min_distance = torch.mean(valid_min_distances).item()
                    max_mean_distance = torch.max(valid_mean_distances).item()
                    min_mean_distance = torch.min(valid_mean_distances).item()
                    
                    print(f"=== Singulation Distance Stats ===")
                    print(f"Mean Distance - Avg: {avg_mean_distance:.4f}, Max: {max_mean_distance:.4f}, Min: {min_mean_distance:.4f}")
                    print(f"Min Distance - Avg: {avg_min_distance:.4f}")
                    print(f"Success Threshold (avg): {self.singulation_thresh.mean().item():.4f}")
                else:
                    print("=== Singulation Distance Stats ===")
                    print("No valid environments for distance calculation")
            else:
                goal_distances = torch.norm(self.goal_pos - self.object_pos, p=2, dim=-1)
                avg_goal_distance = torch.mean(goal_distances).item()
                min_goal_distance = torch.min(goal_distances).item()
                max_goal_distance = torch.max(goal_distances).item()
                
                print(f"=== Goal Distance Stats ===")
                print(f"Goal Distance - Avg: {avg_goal_distance:.4f}, Max: {max_goal_distance:.4f}, Min: {min_goal_distance:.4f}")
                print(f"Success Threshold: {self.config['Weights']['max_goal_dist']:.4f}")

        non_zero_steps = self.succ_steps[self.succ_steps > 0]
        if len(non_zero_steps) > 0:
            self.current_avg_steps = round(non_zero_steps.float().mean().item())
            if not hasattr(self, 'total_succ_steps'):
                self.total_succ_steps = []
            self.total_succ_steps.append(self.current_avg_steps)
            self.avg_succ_steps = round(sum(self.total_succ_steps) / len(self.total_succ_steps))
        else:
            self.current_avg_steps = 0
            self.avg_succ_steps = 0 if not hasattr(self, 'total_succ_steps') else round(sum(self.total_succ_steps) / len(self.total_succ_steps))
        print(f"Current average steps: {self.current_avg_steps}, Cumulative average steps: {self.avg_succ_steps}")

            


        
        rand_object_index =  0 
        self.obj_focus_id = rand_object_index
        
        
        self.object_indices = self.object_clutter_indices[rand_object_index]

        for env_id in env_ids:
            self.gym.set_rigid_body_color(self.envs[env_id], self.object_clutter_handles[rand_object_index], 0, gymapi.MESH_VISUAL, gymapi.Vec3(*[90/255, 90/255, 173/255]))
            for i in range(len(self.object_clutter_handles)):
                if i != rand_object_index:
                    self.gym.set_rigid_body_color(self.envs[env_id], self.object_clutter_handles[i], 0, gymapi.MESH_VISUAL, gymapi.Vec3(*[150/255, 150/255, 150/255]))


        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_dex_hand_dofs * 2 + 5), device=self.device)

        self.reset_target_pose(env_ids)

        delta_max = self.dex_hand_dof_upper_limits - self.dex_hand_dof_default_pos
        delta_min = self.dex_hand_dof_lower_limits - self.dex_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_dex_hand_dofs]
        pos = self.dex_hand_default_dof_pos  # + self.reset_dof_pos_noise * rand_delta
        self.dex_hand_dof_pos[env_ids, :] = pos
        self.dex_hand_dof_vel[env_ids, :] = self.dex_hand_dof_default_vel + self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_dex_hand_dofs:5 + self.num_dex_hand_dofs * 2]

        self.prev_targets[env_ids, :self.num_dex_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_dex_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))


        self.reset_dof_state = self.dof_state.clone().view(self.num_envs, self.num_dex_hand_dofs,-1)
        self.reset_dof_state[:,:,0] = torch.tensor(self.init_dof_state, device=self.device).repeat(self.num_envs, 1)

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.reset_dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        all_indices = torch.unique(torch.cat([all_hand_indices, self.object_indices[env_ids], self.table_indices[env_ids], ]).to(torch.int32))  ##

        theta = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device)[:, 0]
        if not self.random_prior: theta *= 0
        new_object_rot = quat_from_euler_xyz(self.object_init_euler_xy[env_ids,0], self.object_init_euler_xy[env_ids,1], theta)
        prior_rot_z = get_euler_xyz(quat_mul(new_object_rot, self.target_hand_rot[env_ids]))[2]

        self.z_theta[env_ids] = prior_rot_z
        prior_rot_quat = quat_from_euler_xyz(torch.tensor(0.0, device=self.device).repeat(len(env_ids), 1)[:, 0], torch.zeros_like(theta), prior_rot_z)


        if self.num_envs == len(env_ids):
            self.hand_prior_rot_quat = quat_from_euler_xyz(torch.tensor(1.57, device=self.device).repeat(self.num_envs, 1)[:, 0], torch.zeros_like(theta), prior_rot_z)
      
        if 'static_init' in self.config['Modes'] and self.config['Modes']['static_init']:

            target_pos_rot = torch.zeros((len(env_ids), 7), device=self.device)
            if 'central_object' in self.config['Modes'] and self.config['Modes']['central_object']: target_pos_rot[:, :2] *= 0.  # central object
            target_pos_rot[:, :3] = torch.tensor([self.separation_dist*self.sudoku_grid[0]['x'], 
                                                self.separation_dist*self.sudoku_grid[0]['y'],
                                                self.table_dims.z + 0.05],
                                                device=target_pos_rot.device,
                                                dtype=self.root_state_tensor.dtype).repeat(target_pos_rot.shape[0], 1)
            
            self.root_state_tensor[self.object_clutter_indices[0][env_ids], :3] = target_pos_rot[:, :3].clone()
            self.root_state_tensor[self.object_clutter_indices[0][env_ids], 3:7] = quat_from_euler_xyz(
                torch.zeros(target_pos_rot.shape[0], device=target_pos_rot.device, dtype=self.root_state_tensor.dtype),
                torch.zeros(target_pos_rot.shape[0], device=target_pos_rot.device, dtype=self.root_state_tensor.dtype),
                torch.zeros(target_pos_rot.shape[0], device=target_pos_rot.device, dtype=self.root_state_tensor.dtype))
            self.root_state_tensor[self.object_clutter_indices[0][env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

            self.init_x = torch.zeros((len(env_ids), self.surrounding_obj_num + 1), device=self.device)
            self.init_y = torch.zeros((len(env_ids), self.surrounding_obj_num + 1), device=self.device)
            if self.surrounding_obj_num > 0:
                available_positions = [list(range(1, len(self.sudoku_grid))) for _ in range(len(env_ids))]
                self.object_position_assignments = torch.zeros((self.num_envs, self.surrounding_obj_num + 1), 
                                                            device=self.device, dtype=torch.long)
                
                self.object_position_assignments[:, 0] = 0

                for i in range(1, self.surrounding_obj_num + 1):
                    selected_positions = torch.zeros((len(env_ids), 3), device=target_pos_rot.device, dtype=self.root_state_tensor.dtype)
                    
                    for env_idx in range(len(env_ids)):
                        if len(available_positions[env_idx]) > 0:
                            if self.random_grid_sequences:
                                pos_idx = available_positions[env_idx].pop(
                                    torch.randint(0, len(available_positions[env_idx]), (1,)).item()
                                )
                            else:
                                pos_idx = available_positions[env_idx].pop(0)
                            
                            self.object_position_assignments[env_ids[env_idx], i] = pos_idx

                            selected_positions[env_idx] = torch.tensor([
                                self.separation_dist*self.sudoku_grid[pos_idx]['x'],
                                self.separation_dist*self.sudoku_grid[pos_idx]['y'],
                                self.table_dims.z + 0.05
                            ], device=target_pos_rot.device, dtype=self.root_state_tensor.dtype)
                    
                    

                    self.root_state_tensor[self.object_clutter_indices[i][env_ids], :3] = selected_positions
                    if hasattr(self, 'random_surrounding_positions') and self.random_surrounding_positions:
                        safe_gap = (self.separation_dist - getattr(self, 'side_len', 0.06)) * 0.5
                        max_noise = torch.clamp(torch.tensor(0.005, device=selected_positions.device, dtype=selected_positions.dtype),
                                                min=0.0,
                                                max=torch.tensor(max(0.0, float(safe_gap)), device=selected_positions.device, dtype=selected_positions.dtype))
                        noise_xy = (torch.rand_like(selected_positions[:, :2]) * 2.0 - 1.0) * max_noise
                        position_noise = torch.cat([noise_xy, torch.zeros((selected_positions.shape[0], 1), device=selected_positions.device, dtype=selected_positions.dtype)], dim=-1)
                        self.root_state_tensor[self.object_clutter_indices[i][env_ids], :3] += position_noise
                    
                    if hasattr(self, 'random_surrounding_orientations') and self.random_surrounding_orientations:
                        rand_z_rot = torch.rand(target_pos_rot.shape[0], device=target_pos_rot.device) * 2 * torch.pi - torch.pi
                        self.root_state_tensor[self.object_clutter_indices[i][env_ids], 3:7] = quat_from_euler_xyz(
                            torch.zeros_like(rand_z_rot),
                            torch.zeros_like(rand_z_rot),
                            rand_z_rot)
                    self.root_state_tensor[self.object_clutter_indices[i][env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])
                    
            
            self.sudoku_grid_x = torch.tensor([pos['x'] for pos in self.sudoku_grid], 
                                            device=self.device) * self.separation_dist
            self.sudoku_grid_y = torch.tensor([pos['y'] for pos in self.sudoku_grid], 
                                            device=self.device) * self.separation_dist
        else:
            self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
            self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot  # reset object rotation
            self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])
        self.prev_obstacle_distances = None
        self.prev_obstacle_projected = None

        indices_to_cat = [all_hand_indices]
        
        for i in range(self.surrounding_obj_num + 1):
            indices_to_cat.append(self.object_clutter_indices[i][env_ids])
            
        indices_to_cat.extend([
            self.goal_object_indices[env_ids],
            self.table_indices[env_ids]
        ])
        
        all_indices = torch.unique(torch.cat(indices_to_cat).to(torch.int32))

        if self.surrounding_obj_num > 0:
            proj_target_object_pos = self.root_state_tensor[self.object_clutter_indices[self.obj_focus_id], 0:2]

            proj_dists = []
            for i in range(self.surrounding_obj_num+1):
                if i != self.obj_focus_id:
                    proj_other_object_pos = self.root_state_tensor[self.object_clutter_indices[i], 0:2]
                    dist = torch.norm(proj_target_object_pos - proj_other_object_pos, p=2, dim=-1)
                    proj_dists.append(dist)

            while len(proj_dists) < max(0, self.surrounding_obj_num):
                proj_dists.append(torch.zeros_like(proj_target_object_pos[:, 0]))

            if len(proj_dists) > 0:
                proj_dists = torch.stack(proj_dists, dim=0)
                non_zero_mask = proj_dists > 0
                mean_proj_dist = torch.where(
                    torch.any(non_zero_mask, dim=0),
                    torch.sum(proj_dists, dim=0) / torch.count_nonzero(non_zero_mask, dim=0).float(),
                    torch.zeros_like(proj_dists[0])
                )
                min_proj_dist = torch.where(
                    torch.any(non_zero_mask, dim=0),
                    torch.min(torch.where(non_zero_mask, proj_dists, torch.inf), dim=0)[0],
                    torch.zeros_like(proj_dists[0])
                )
                self.singulation_thresh[env_ids] = 1.3 * mean_proj_dist[env_ids]
                self.mean_singulation_distance[env_ids] = mean_proj_dist[env_ids]
                self.min_singulation_distance[env_ids] = min_proj_dist[env_ids]

        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))

        if self.random_time:
            self.random_time = False
            self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        else:
            self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

        self.reset_count += 1

    def set_goal_displacement(self, displacement):
        """Set new goal displacement - can be called before reset or during episode"""
        if isinstance(displacement, np.ndarray):
            displacement = torch.from_numpy(displacement).to(self.device).float()  # Add .float()
        elif isinstance(displacement, list):
            displacement = torch.tensor(displacement, device=self.device, dtype=torch.float32)  # Specify dtype
        
        self.goal_displacement = gymapi.Vec3(displacement[0], displacement[1], displacement[2])
        self.goal_displacement_tensor = displacement



    
    
@torch.jit.script
def compute_hand_reward(
    #"""TorchScript reward function used by PPO."""
    modes: Dict[str, bool], weights: Dict[str, float],
    object_init_z, delta_target_hand_pos, delta_target_hand_rot,
    id: int, object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf,
    successes, current_successes, consecutive_successes,
    max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,
    right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_th_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool,
    object_points, right_hand_pc_dist, right_hand_finger_pc_dist, right_hand_joint_pc_dist, right_hand_body_pc_dist,
    mean_singulation_distance, min_singulation_distance, table_z: float, invalid_env_mask, invalid_env_num: int, remove_invalid_arrangement: bool,
    succ_steps: torch.Tensor,
    surrounding_obj_num: int,
    dist_target_to_centroid
):

    goal_dist_pre = torch.norm(target_pos - object_pos, p=2, dim=-1)
    fingertip_pos_pre = torch.stack([right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_th_pos], dim=1)
    fingertip_center_pos_pre = torch.mean(fingertip_pos_pre, dim=1)
    right_hand_joint_dist_pre = right_hand_joint_pc_dist
    
    exp_decay_coef = 2.5
    r_exp_distance = torch.exp(-exp_decay_coef * goal_dist_pre)
    

    linear_threshold_1 = 0.05
    linear_threshold_2 = 0.15
    linear_slope_1 = -20.0
    linear_slope_2 = -5.0
    r_piecewise_linear = torch.where(
        goal_dist_pre < linear_threshold_1,
        linear_slope_1 * goal_dist_pre + 1.0,
        torch.where(goal_dist_pre < linear_threshold_2, linear_slope_2 * goal_dist_pre + 0.5, -0.5)
    )
    
    gaussian_sigma = 0.08
    gaussian_peak = 1.0
    r_gaussian = gaussian_peak * torch.exp(-0.5 * (goal_dist_pre / gaussian_sigma) ** 2)
    
    inverse_scale = 0.1
    inverse_offset = 0.01
    r_inverse = inverse_scale / (goal_dist_pre + inverse_offset)
    
    inverse_square_scale = 0.05
    r_inverse_square = inverse_square_scale / (goal_dist_pre ** 2 + 0.001)
    
    log_scale = -2.0
    log_offset = 0.01
    r_log = log_scale * torch.log(goal_dist_pre + log_offset)
    
    tanh_scale = 1.0
    tanh_steepness = 10.0
    r_tanh = tanh_scale * torch.tanh(-tanh_steepness * goal_dist_pre + 1.0)
    

    object_vel = torch.zeros_like(goal_dist_pre) 
    vel_threshold = 0.1
    vel_reward_coef = 5.0
    r_velocity = torch.where(object_vel < vel_threshold, vel_reward_coef * (1.0 - object_vel / vel_threshold), 0.0)
    
    angle_threshold = 0.1  
    angle_reward_coef = 3.0
    angle_diff = torch.norm(object_rot - target_rot, p=2, dim=-1)
    r_angle_alignment = torch.where(angle_diff < angle_threshold, angle_reward_coef * (1.0 - angle_diff / angle_threshold), 0.0)
    

    min_contact_points = 3
    contact_reward_coef = 2.0
    contact_mask = (right_hand_joint_pc_dist < 0.02).int()
    num_contacts = torch.sum(contact_mask, dim=-1)
    r_contact_count = torch.where(num_contacts >= min_contact_points, contact_reward_coef * (num_contacts.float() / 10.0), 0.0)

    symmetry_threshold = 0.05
    symmetry_coef = 1.5
    finger_pair_dist_1 = torch.norm(right_hand_ff_pos - right_hand_rf_pos, p=2, dim=-1)
    finger_pair_dist_2 = torch.norm(right_hand_mf_pos - right_hand_th_pos, p=2, dim=-1)
    symmetry_diff = torch.abs(finger_pair_dist_1 - finger_pair_dist_2)
    r_symmetry = torch.where(symmetry_diff < symmetry_threshold, symmetry_coef * (1.0 - symmetry_diff / symmetry_threshold), 0.0)
    

    stability_threshold = 0.02
    stability_coef = 2.0
    position_stability = torch.ones_like(goal_dist_pre) * 0.01
    r_stability = torch.where(position_stability < stability_threshold, stability_coef * (1.0 - position_stability / stability_threshold), 0.0)
    
    energy_threshold = 0.5
    energy_coef = 1.0
    energy_consumption = torch.sum(torch.abs(actions), dim=-1)
    r_energy_efficiency = torch.where(energy_consumption < energy_threshold, energy_coef * (1.0 - energy_consumption / energy_threshold), -energy_coef * (energy_consumption - energy_threshold))
    

    time_discount = 0.99
    time_reward = time_discount ** progress_buf.float()
    
    
    exploration_radius = 0.15
    exploration_coef = 0.5
    exploration_dist = torch.norm(right_hand_pos - object_pos, p=2, dim=-1)
    r_exploration = torch.where(exploration_dist > exploration_radius, exploration_coef * (exploration_dist - exploration_radius), 0.0)
    
    
    force_threshold = 10.0
    force_coef = 0.3
    
    grasp_force = torch.ones_like(goal_dist_pre) * 5.0  
    r_grasp_force = torch.where(grasp_force > force_threshold, force_coef * (grasp_force - force_threshold) / force_threshold, 0.0)
    
   
    coordination_threshold = 0.03
    coordination_coef = 1.2
    finger_distances = torch.stack([
        torch.norm(right_hand_ff_pos - object_pos, p=2, dim=-1),
        torch.norm(right_hand_mf_pos - object_pos, p=2, dim=-1),
        torch.norm(right_hand_rf_pos - object_pos, p=2, dim=-1),
        torch.norm(right_hand_th_pos - object_pos, p=2, dim=-1)
    ], dim=1)
    finger_std = torch.std(finger_distances, dim=1)
    r_coordination = torch.where(finger_std < coordination_threshold, coordination_coef * (1.0 - finger_std / coordination_threshold), 0.0)
    
    
    height_target = table_z + 0.3
    height_coef = 3.0
    height_diff = torch.abs(object_pos[:, 2] - height_target)
    r_height = height_coef * torch.exp(-5.0 * height_diff)
    
    
    direction_threshold = 0.2
    direction_coef = 1.0
    hand_to_object = object_pos - right_hand_pos
    hand_to_target = target_pos - right_hand_pos
    direction_similarity = torch.sum(hand_to_object * hand_to_target, dim=-1) / (torch.norm(hand_to_object, p=2, dim=-1) * torch.norm(hand_to_target, p=2, dim=-1) + 1e-6)
    r_direction = torch.where(direction_similarity > direction_threshold, direction_coef * direction_similarity, 0.0)
    
   
    smoothness_threshold = 0.1
    smoothness_coef = 0.8
   
    action_smoothness = torch.sum(torch.abs(actions[:, 1:] - actions[:, :-1]), dim=-1)
    r_smoothness = torch.where(action_smoothness < smoothness_threshold, smoothness_coef * (1.0 - action_smoothness / smoothness_threshold), 0.0)
    
    shape_coef_param = 0.1
    shape_reward_coef = 1.5
    finger_span = torch.norm(right_hand_ff_pos - right_hand_th_pos, p=2, dim=-1)
    optimal_span = 0.12
    shape_error = torch.abs(finger_span - optimal_span)
    r_shape = shape_reward_coef * torch.exp(-shape_coef_param * shape_error)
    
    stage_1_threshold = 0.1
    stage_2_threshold = 0.05
    stage_1_reward = 0.5
    stage_2_reward = 1.0
    r_progressive = torch.where(
        goal_dist_pre < stage_2_threshold, stage_2_reward,
        torch.where(goal_dist_pre < stage_1_threshold, stage_1_reward, 0.0)
    )
    
    relative_weight = 0.5
    relative_pos = object_pos - right_hand_pos
    r_relative = -relative_weight * torch.norm(relative_pos, p=2, dim=-1)
    
    velocity_match_coef = 2.0
    # 需要速度数据，这里仅示例
    hand_velocity = torch.zeros_like(right_hand_pos)
    object_velocity = torch.zeros_like(object_pos)
    velocity_match = -torch.norm(hand_velocity - object_velocity, p=2, dim=-1)
    r_velocity_match = velocity_match_coef * velocity_match
    
    
    quality_coef = 1.0
    grasp_quality = (1.0 / (right_hand_joint_pc_dist + 0.01)).mean(dim=-1)
    r_quality = quality_coef * grasp_quality
    
    w1, w2, w3, w4 = 0.3, 0.3, 0.2, 0.2
    r_multi_objective = w1 * r_exp_distance + w2 * r_gaussian + w3 * r_contact_count + w4 * r_symmetry
    
    adaptive_coef = 0.1
    adaptive_threshold = torch.mean(goal_dist_pre) + adaptive_coef * torch.std(goal_dist_pre)
    r_adaptive = torch.where(goal_dist_pre < adaptive_threshold, 1.0, -1.0)
    
    layer1_weight, layer2_weight, layer3_weight = 0.5, 0.3, 0.2
    layer1_reward = r_exp_distance
    layer2_reward = r_contact_count
    layer3_reward = r_height
    r_layered = layer1_weight * layer1_reward + layer2_weight * layer2_reward + layer3_weight * layer3_reward
    
    sparse_threshold = 0.02
    sparse_reward_value = 10.0
    r_sparse = torch.where(goal_dist_pre < sparse_threshold, sparse_reward_value, 0.0)
    
    dense_coef = 5.0
    r_dense = -dense_coef * goal_dist_pre
    
    cosine_coef = 2.0
    hand_to_obj = object_pos - right_hand_pos
    hand_to_goal = target_pos - right_hand_pos
    cosine_sim = torch.sum(hand_to_obj * hand_to_goal, dim=-1) / (torch.norm(hand_to_obj, p=2, dim=-1) * torch.norm(hand_to_goal, p=2, dim=-1) + 1e-6)
    r_cosine = cosine_coef * cosine_sim
    
    depth_threshold = 0.03
    depth_coef = 2.5
    # right_hand_joint_pc_dist 是一维张量（每个环境的距离）
    grasp_depth = right_hand_joint_dist_pre
    r_depth = torch.where(grasp_depth < depth_threshold, depth_coef * (1.0 - grasp_depth / depth_threshold), 0.0)
    
    coverage_angle = 0.5
    coverage_coef = 1.8
    finger_vectors = fingertip_pos_pre - right_hand_pos.unsqueeze(1)
    finger_angles = torch.acos(torch.clamp(torch.sum(finger_vectors[:, 0] * finger_vectors[:, 1], dim=-1) / (torch.norm(finger_vectors[:, 0], p=2, dim=-1) * torch.norm(finger_vectors[:, 1], p=2, dim=-1) + 1e-6), -1.0, 1.0))
    r_coverage = torch.where(finger_angles > coverage_angle, coverage_coef * (finger_angles / torch.pi), 0.0)
    
    balance_threshold = 0.02
    balance_coef = 1.5
    object_center = object_pos
    support_center = fingertip_center_pos_pre
    balance_error = torch.norm(object_center - support_center, p=2, dim=-1)
    r_balance = torch.where(balance_error < balance_threshold, balance_coef * (1.0 - balance_error / balance_threshold), 0.0)
    
    action_magnitude_threshold = 0.5
    action_magnitude_coef = 0.5
    action_magnitude = torch.norm(actions, p=2, dim=-1)
    r_action_magnitude = torch.where(action_magnitude < action_magnitude_threshold, action_magnitude_coef * (1.0 - action_magnitude / action_magnitude_threshold), -action_magnitude_coef * (action_magnitude - action_magnitude_threshold))
    
    multi_finger_coef = 1.5
    finger_distances_to_obj = torch.stack([
        torch.norm(right_hand_ff_pos - object_pos, p=2, dim=-1),
        torch.norm(right_hand_mf_pos - object_pos, p=2, dim=-1),
        torch.norm(right_hand_rf_pos - object_pos, p=2, dim=-1),
        torch.norm(right_hand_th_pos - object_pos, p=2, dim=-1)
    ], dim=1)
    finger_mean_dist = torch.mean(finger_distances_to_obj, dim=1)
    finger_std_dist = torch.std(finger_distances_to_obj, dim=1)
    r_multi_finger = multi_finger_coef * torch.exp(-finger_mean_dist) * torch.exp(-5.0 * finger_std_dist)
    
    stage1_dist, stage2_dist, stage3_dist = 0.15, 0.08, 0.03
    stage1_rew, stage2_rew, stage3_rew = 0.2, 0.5, 1.0
    r_progressive_grasp = torch.where(
        right_hand_joint_dist_pre < stage3_dist, stage3_rew,
        torch.where(right_hand_joint_dist_pre < stage2_dist, stage2_rew,
        torch.where(right_hand_joint_dist_pre < stage1_dist, stage1_rew, 0.0))
    )
    
    shape_match_coef = 1.3
    optimal_finger_spread = 0.1
    actual_finger_spread = torch.norm(right_hand_ff_pos - right_hand_rf_pos, p=2, dim=-1)
    shape_match_error = torch.abs(actual_finger_spread - optimal_finger_spread)
    r_shape_match = shape_match_coef * torch.exp(-10.0 * shape_match_error)
    
    efficiency_coef = 0.8
    grasp_efficiency = (1.0 / (right_hand_joint_dist_pre + 0.01)) / (torch.sum(torch.abs(actions), dim=-1) + 0.1)
    r_efficiency = efficiency_coef * grasp_efficiency
    
    target_zone_radius = 0.05
    target_zone_coef = 2.0
    in_target_zone = (goal_dist_pre < target_zone_radius).float()
    r_target_zone = target_zone_coef * in_target_zone
    
    grasp_strength_threshold = 0.02
    grasp_strength_coef = 2.5
    grasp_strength = 1.0 / (right_hand_joint_dist_pre + 0.01)
    r_grasp_strength = torch.where(grasp_strength > 1.0 / grasp_strength_threshold, grasp_strength_coef * (grasp_strength - 1.0 / grasp_strength_threshold) / (1.0 / grasp_strength_threshold), 0.0)
    

    quality_w1, quality_w2, quality_w3 = 0.4, 0.3, 0.3
    quality_metric_1 = r_exp_distance
    quality_metric_2 = r_contact_count
    quality_metric_3 = r_symmetry
    r_comprehensive_quality = quality_w1 * quality_metric_1 + quality_w2 * quality_metric_2 + quality_w3 * quality_metric_3
    
    
    action_penalty = torch.sum(actions ** 2, dim=-1)
    heighest = torch.max(object_points[:, :, -1], dim=1)[0]
    lowest = torch.min(object_points[:, :, -1], dim=1)[0]

    target_z = heighest + 0.05
    target_xy = object_pos[:, :2]
    target_init_pos = torch.cat([target_xy, target_z.unsqueeze(-1)], dim=-1)
    right_hand_axis_dist = torch.norm(target_xy - right_hand_pos[:, :2], p=2, dim=-1)
    right_hand_init_dist = torch.norm(target_init_pos - right_hand_pos, p=2, dim=-1)



    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)
    

    right_hand_dist = right_hand_pc_dist
    right_hand_body_dist = right_hand_body_pc_dist
    right_hand_joint_dist = right_hand_joint_pc_dist
    right_hand_finger_dist = right_hand_finger_pc_dist
    fingertip_pos = torch.stack([right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_th_pos], dim=1)
    fingertip_center_pos = torch.mean(fingertip_pos, dim=1)
    # encapsulation_dist = torch.norm(object_pos - fingertip_center_pos, p=2, dim=-1)
    # 假设张量的最后一个维度格式为 (x, y, z)
    encapsulation_dist = torch.norm(object_pos - fingertip_center_pos, p=2, dim=-1)
    encapsulation_reward = -10 * encapsulation_dist


    max_finger_dist, max_hand_dist, max_goal_dist, target_separation_distance = weights['max_finger_dist'], weights['max_hand_dist'], weights['max_goal_dist'], weights['target_separation_distance']
    max_finger_dist = 0.20 # NOTE default is 0.3, smaller value helps grasp tighter

    if 'right_hand_body_dist' not in weights: weights['right_hand_body_dist'] = 0.


    singulation_reward = torch.clamp(min_singulation_distance, 0.0, 0.02)  # 0.3 used to be very large
    singulation_reward = 100 * singulation_reward


    singulation_success = torch.zeros_like(successes)
    is_joint_dist_close = (right_hand_joint_dist <= max_finger_dist).int()
    is_hand_dist_close = (right_hand_dist <= max_hand_dist).int()
    hold_value = 2
    
    if surrounding_obj_num > 0:
        
        is_singulation_complete = (dist_target_to_centroid >= target_separation_distance).int()

        hold_flag = is_hand_dist_close + is_singulation_complete
        singulation_success = is_singulation_complete.float()
        

    else:
        hold_flag = is_joint_dist_close + is_hand_dist_close


    object_points_sorted, _ = torch.sort(object_points, dim=-1)
    object_points_sorted = object_points_sorted[:, :object_points_sorted.shape[1]//4, :]
    random_indices = torch.randint(0, object_points_sorted.shape[1], (object_points_sorted.shape[0], 1))
    exploration_target_pos = object_points_sorted[torch.arange(object_points_sorted.shape[0]).unsqueeze(1), random_indices].squeeze(1)
    right_hand_exploration_dist = torch.norm(exploration_target_pos - right_hand_pos, p=2, dim=-1)
    r_move_away_from_clutter = 10 * dist_target_to_centroid

    goal_rew = torch.zeros_like(goal_dist)
    goal_rew = torch.where(hold_flag == hold_value, 1.0 * (0.9 - 2.0 * goal_dist), goal_rew)
    hand_up = torch.zeros_like(goal_dist)

    hand_up = torch.where(lowest >= table_z-0.01 + 0, torch.where(hold_flag == hold_value, 0.1 - 0.3 * actions[:, 2], hand_up), hand_up) # NOTE, hand urdf is flipped, actions[:, 2] is negative for +z

    hand_up = torch.where(lowest >= table_z-0.01 + 0.2, torch.where(hold_flag == hold_value, 0.2 - goal_hand_dist * 0 + weights['hand_up_goal_dist'] * (0.2 - goal_dist), hand_up), hand_up)
    bonus = torch.zeros_like(goal_dist)
    bonus = torch.where(hold_flag == hold_value, torch.where(goal_dist <= max_goal_dist, 1.0 / (1 + 10 * goal_dist), bonus), bonus)
    

    init_reward = weights['right_hand_dist'] * right_hand_dist 
    init_reward += weights['right_hand_joint_dist'] * right_hand_joint_dist # r_j
    
    
    grasp_reward =  weights['right_hand_joint_dist'] * right_hand_joint_dist # r_j
    grasp_reward += weights['right_hand_finger_dist'] * right_hand_finger_dist + weights['right_hand_dist'] * right_hand_dist # r_p r_f
    grasp_reward += weights['goal_dist'] * goal_dist + weights['goal_rew'] * goal_rew + weights['hand_up'] * hand_up + weights['bonus'] * bonus
    grasp_reward += weights['encapsulation'] * encapsulation_reward 

    reward = torch.where(hold_flag != hold_value, init_reward, grasp_reward)
    reward += r_move_away_from_clutter
    reward += singulation_reward

    

    resets = reset_buf
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets) # NOTE make equal length episode
    goal_resets = resets


    new_successes = torch.where(goal_dist <= max_goal_dist, torch.ones_like(successes), torch.zeros_like(successes))
    succ_steps = torch.where((new_successes == 1) & (successes == 0), progress_buf, succ_steps)
    successes = torch.where(new_successes == 1, torch.ones_like(successes), successes)

    if remove_invalid_arrangement:
        successes = torch.where(invalid_env_mask, torch.zeros_like(successes), successes)

    final_successes = torch.where(goal_dist <= max_goal_dist, torch.ones_like(successes), torch.zeros_like(successes))
    current_successes = torch.where(resets == 1, successes, current_successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes, final_successes, succ_steps, singulation_success
