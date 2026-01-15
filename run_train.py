# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from utils.config import (
    set_np_formatting,
    set_seed,
    get_args,
    parse_sim_params,
    load_cfg,
)
from utils.parse_task import parse_task
from utils.process_sarl import process_ppo
from utils.process_marl import get_AgentIndex


def train():
    """Train or test the PPO policy."""
    if args.algo != "ppo":
        raise ValueError("This script now only supports the PPO algorithm.")

    agent_index = get_AgentIndex(cfg)
    task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

    ppo_runner = process_ppo(args, env, cfg_train, logdir)

    total_iters = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        total_iters = args.max_iterations

    ppo_runner.run(
        num_learning_iterations=total_iters,
        log_interval=cfg_train["learn"]["save_interval"],
    )


if __name__ == '__main__':
    # set set_printoptions
    set_np_formatting()
    
    # init default args: task (shadow_hand_grasp), alog(ppo), num_envs, cfg_env, cfg_train
    args, train_flag = get_args()
    
    # start train or test process
    if train_flag:
        # load configs for cfg_env(shadow_hand_grasp.yaml), cfg_train(ppo/config.yaml)
        cfg, cfg_train, logdir = load_cfg(args)

        # overide param in yaml for easier training
        if args.separation_dist is not None: cfg["env"]["separation_dist"] = args.separation_dist
        if args.surrounding_obj_num is not None: cfg["env"]["surrounding_obj_num"] = args.surrounding_obj_num
        if args.table_dim_z is not None: cfg["env"]["table_dim_z"] = args.table_dim_z 
        # gymutil.parse_arguments with args and cfg
        sim_params = parse_sim_params(args, cfg, cfg_train)
        # set system random seed: 0
        set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
        # run train() with specific task, algo, train/test mode
        train()