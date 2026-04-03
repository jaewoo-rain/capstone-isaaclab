"""학습된 PPO 정책 재생."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained PPO policy")
parser.add_argument("--task", type=str, default="My-Grasp-Franka-Direct-v0")
parser.add_argument("--checkpoint", type=str, default="checkpoints/grasp_franka_ppo.zip")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from stable_baselines3 import PPO

import source.my_grasp_rl.tasks.grasp  # noqa: F401
from source.my_grasp_rl.tasks.grasp.grasp_franka_env_cfg import GraspFrankaEnvCfg
from source.my_grasp_rl.tasks.grasp.grasp_franka_env import GraspFrankaEnv
try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except Exception as e:
    raise RuntimeError(
        "isaaclab_rl의 SB3 wrapper를 불러오지 못했습니다. Isaac Lab 버전에 맞는 wrapper 경로를 확인하세요."
    ) from e

import time
def main():

    cfg = GraspFrankaEnvCfg()

    env = GraspFrankaEnv(cfg=cfg, render_mode=None)
    env = Sb3VecEnvWrapper(env)

    model = PPO.load(args_cli.checkpoint)

    obs = env.reset()
    while simulation_app.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

if __name__ == "__main__":
    main()
    simulation_app.close()