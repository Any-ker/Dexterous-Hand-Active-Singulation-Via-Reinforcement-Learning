from algorithms.rl.ppo import PPO, ActorCritic


def process_ppo(args, env, cfg_train, logdir):
    """Build the PPO trainer/tester used by run_train.py."""
    learn_cfg = cfg_train["learn"]
    is_testing = bool(args.test)
    checkpoint_path = args.model_dir.strip()

    logdir = f"{logdir}_seed{env.task.cfg['seed']}_obj{env.task.surrounding_obj_num}"

    ppo = PPO(
        vec_env=env,
        actor_critic_class=ActorCritic,
        num_transitions_per_env=learn_cfg["nsteps"],
        num_learning_epochs=learn_cfg["noptepochs"],
        num_mini_batches=learn_cfg["nminibatches"],
        clip_param=learn_cfg["cliprange"],
        gamma=learn_cfg["gamma"],
        lam=learn_cfg["lam"],
        init_noise_std=learn_cfg.get("init_noise_std", 0.3),
        value_loss_coef=learn_cfg.get("value_loss_coef", 2.0),
        entropy_coef=learn_cfg["ent_coef"],
        learning_rate=learn_cfg["optim_stepsize"],
        max_grad_norm=learn_cfg.get("max_grad_norm", 2.0),
        use_clipped_value_loss=learn_cfg.get("use_clipped_value_loss", False),
        schedule=learn_cfg.get("schedule", "fixed"),
        desired_kl=learn_cfg.get("desired_kl", None),
        model_cfg=cfg_train["policy"],
        device=env.rl_device,
        sampler=learn_cfg.get("sampler", "sequential"),
        log_dir=logdir,
        is_testing=is_testing,
        print_log=learn_cfg["print_log"],
        apply_reset=False,
        asymmetric=(env.num_states > 0),
    )

    if checkpoint_path:
        if is_testing:
            print(f"Loading ppo model for testing from {checkpoint_path}")
            ppo.test(checkpoint_path)
        else:
            print(f"Loading ppo model for training from {checkpoint_path}")
            ppo.load(checkpoint_path)

    return ppo