from __future__ import annotations

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train OMY grasp PPO")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper   # 추가

from source.omy.omy_env_cfg import OmyLiftEnvCfg
from source.omy.omy_grasp_env import OmyGraspEnv


def main():
    cfg = OmyLiftEnvCfg()
    cfg.scene.num_envs = 16

    base_env = OmyGraspEnv(cfg, render_mode=None)

    # Isaac Lab env -> SB3 VecEnv로 감싸기
    env = Sb3VecEnvWrapper(base_env)

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,   # 16 env * 512 step = 8192 이므로 약수로 맞추기
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        n_epochs=5,
        verbose=1,
        tensorboard_log="./logs/omy_grasp/",
        device="cpu",   # MlpPolicy는 SB3 공식적으로 CPU 권장
    )

    model.learn(total_timesteps=1_000_000)

    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/omy_grasp_ppo")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()