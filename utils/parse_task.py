# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from dex_grasp import DexGrasp
from tasks.hand_base.vec_task import VecTaskPython


def parse_task(args, cfg, cfg_train, sim_params, agent_index):
    """Create the DexGrasp task and wrap it with the simple Python VecTask."""
    device_id = args.device_id
    rl_device = args.rl_device

    # Make sure seeds are set everywhere so runs are reproducible.
    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]

    # The current project only uses the Python task path.
    if args.task_type != "Python":
        raise ValueError("Only the Python DexGrasp task is supported in this repo.")

    env_name = cfg_task.get("env_name", "")
    print("Python")
    print(env_name)
    if env_name != "dex_grasp":
        raise ValueError(f"Unknown environment name: {env_name}")

    task = DexGrasp(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=device_id,
        headless=args.headless,
        is_multi_agent=False,
    )

    # VecTaskPython is the simple wrapper that exposes reset/step for PPO.
    env = VecTaskPython(task, rl_device)
    return task, env
