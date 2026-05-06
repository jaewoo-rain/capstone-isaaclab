from source.omy.scripts.train_runner import run_ppo_train


if __name__ == "__main__":
    run_ppo_train(
        env_cls_path="source.omy.tasks.lift.omy_lift_env.OmyLiftEnv",
        cfg_cls_path="source.omy.tasks.common.omy_env_cfg.OmyLiftEnvCfg",
        description="Train OMY lift PPO",
        default_save_path="checkpoints/omy_lift_ppo",
        default_resume_path="checkpoints/omy_lift_ppo.zip",
        default_log_dir="./logs/omy_lift/",
    )

