from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train OMY place PPO")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from source.omy.omy_env_cfg import OmyLiftEnvCfg
from source.omy.omy_place_env import OmyPlaceEnv


def main():
    cfg = OmyLiftEnvCfg()
    cfg.scene.num_envs = 16
    cfg.n_steps = 256
    cfg.batch_size = 1024

    base_env = OmyPlaceEnv(cfg, render_mode=None)
    env = Sb3VecEnvWrapper(base_env)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        n_epochs=cfg.n_epochs,
        verbose=1,
        tensorboard_log="./logs/omy_place/",
        device="cpu",
    )

    model.learn(total_timesteps=50_000)

    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/omy_place_ppo")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()