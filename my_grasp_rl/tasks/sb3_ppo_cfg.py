SB3_PPO_CFG = {
    "policy": "MlpPolicy",
    "n_steps": 64,
    "batch_size": 16384,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 1.0,
    "verbose": 1,
    "tensorboard_log": "./logs/sb3/grasp_franka",
}