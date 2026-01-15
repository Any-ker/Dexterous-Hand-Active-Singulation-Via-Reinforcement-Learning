# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import sys
import yaml

from isaacgym import gymapi
from isaacgym import gymutil

import torch
import random
import numpy as np
from utils.general_utils import *


def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)


def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def retrieve_cfg(args):
    """Return default logdir/cfg paths for a task."""
    if args.task == "DexGrasp":
        default_log = os.path.join("logs", "dex_grasp", args.algo, args.algo)
        train_cfg = f"cfg/{args.algo}/config.yaml"
        env_cfg = "cfg/dex_grasp.yaml"
        return default_log, train_cfg, env_cfg
    # Fallback (never used in our current workflow, but keeps the API simple)
    return "logs", f"cfg/{args.algo}/config.yaml", f"cfg/{args.task}.yaml"




def load_cfg(args):
    with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)

    with open(os.path.join(os.getcwd(), args.cfg_env), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # Override number of environments if passed on the command line
    if args.num_envs > 0:
        cfg["env"]["numEnvs"] = args.num_envs

    if args.episode_length > 0:
        cfg["env"]["episodeLength"] = args.episode_length

    cfg["name"] = args.task
    cfg["wandb"] = args.wandb
    cfg["headless"] = args.headless
    # test_iteration
    cfg['test'] = args.test
    cfg['config'] = args.config
    cfg["test_epoch"] = args.test_epoch
    cfg["test_iteration"] = args.test_iteration

    # Set physics domain randomization
    if "task" in cfg:
        if "randomize" not in cfg["task"]:
            cfg["task"]["randomize"] = args.randomize
        else:
            cfg["task"]["randomize"] = args.randomize or cfg["task"]["randomize"]
    else:
        cfg["task"] = {"randomize": False}

    logdir = os.path.realpath(args.logdir)

    if args.torch_deterministic:
        cfg_train["torch_deterministic"] = True

    if args.seed is not None:
        cfg_train["seed"] = args.seed

    cfg['logdir'] = logdir
    cfg['algo'] = args.algo

    # modify PPO params
    if args.algo == 'ppo':
        # double_update_step
        if args.config['Modes']['double_update_step'] or args.config['Modes']['double_update_half_iteration_step']:
            cfg_train['learn']['nsteps'] = 16
    # modify DaggerValue params
    elif args.algo == 'dagger_value':
        # double_update_step
        if args.config['Distills']['double_update_step'] or args.config['Distills']['double_update_half_iteration_step']:
            cfg_train['learn']['nsteps'] = 16
    
    # num_observation
    if 'num_observation' in args.config['Weights']: cfg["env"]["numObservations"] = args.config['Weights']['num_observation']
    return cfg, cfg_train, logdir


def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args(benchmark=False):
    custom_parameters = [
        {"name": "--task", "type": str, "default": "DexGrasp", "help": "Task name"},
        {"name": "--algo", "type": str, "default": "ppo", "help": "Algorithm name"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "Device for RL policy"},
        {"name": "--logdir", "type": str, "default": "", "help": "Output log directory"},
        {"name": "--num_envs", "type": int, "default": 0, "help": "Override number of environments"},
        {"name": "--episode_length", "type": int, "default": 0, "help": "Override episode length"},
        {"name": "--seed", "type": int, "default": -1, "help": "Random seed"},
        {"name": "--max_iterations", "type": float, "default": 0, "help": "Maximum training iterations"},
        {"name": "--config", "type": str, "default": "dedicated_policy.yaml", "help": "Training config file"},
        {"name": "--test", "action": "store_true", "default": False, "help": "Test instead of train"},
        {"name": "--test_iteration", "type": int, "default": 10, "help": "Number of test runs"},
        {"name": "--test_epoch", "type": int, "default": 0, "help": "Epoch index for testing"},
        {"name": "--model_dir", "type": str, "default": "", "help": "Model checkpoint path"},
        {"name": "--surrounding_obj_num", "type": int, "default": None, "help": "Override surrounding object count"},
        {"name": "--table_dim_z", "type": float, "default": None, "help": "Override table height"},
        {"name": "--separation_dist", "type": float, "default": None, "help": "Override separation distance"},
        {"name": "--randomize", "action": "store_true", "default": False, "help": "Enable physics randomization"},
        {"name": "--torch_deterministic", "action": "store_true", "default": False, "help": "Use deterministic torch settings"},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Disable viewer"},
        {"name": "--wandb", "action": "store_true", "default": False, "help": "Enable wandb logging"},
    ]

    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)
    args.device_id = int(args.rl_device.split(':')[-1])
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'
    if not hasattr(args, "task_type"):
        args.task_type = "Python"
    args.train = not args.test

    # Resolve default paths based on selected task/algo.
    default_logdir, cfg_train_path, cfg_env_path = retrieve_cfg(args)
    if not args.logdir:
        args.logdir = default_logdir
    args.cfg_train = cfg_train_path
    args.cfg_env = cfg_env_path

    config_path = os.path.realpath(os.path.join('cfg/train', args.config))
    args.config = load_yaml(config_path)
    args.config['Save'] = False
    args.config['Save_Train'] = False
    args.config['Init'] = False

    logs_dir = LOG_DIR if os.path.exists(LOG_DIR) else 'logs'
    args.config['Save_Base'] = os.path.realpath(f"{logs_dir}/{args.config['Infos']['save_name']}")

    train_flag = True
    if args.config['Modes']['double_iteration_step']:
        args.max_iterations *= 2
    if args.config['Modes']['double_update_half_iteration_step']:
        args.max_iterations *= 0.5

    model_dir = os.path.realpath(f"{args.logdir}_seed0/model_{int(args.max_iterations)}.pt")
    if not args.test and os.path.exists(model_dir):
        train_flag = False
        print('======== Find Existing Trained Model! ========')
    if args.test and args.model_dir == "":
        args.model_dir = model_dir

    return args, train_flag
