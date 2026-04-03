"""학습된 PPO Good Robot 정책 재생."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained PPO GoodRobot policy")
parser.add_argument("--checkpoint", type=str, default="checkpoints/good_robot_franka_ppo.zip")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from my_grasp_rl.tasks.good_robot.good_robot_franka_env_cfg import GoodRobotFrankaEnvCfg
from my_grasp_rl.tasks.good_robot.good_robot_franka_env import GoodRobotFrankaEnv

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except Exception as e:
    raise RuntimeError(
        "isaaclab_rl의 SB3 wrapper를 불러오지 못했습니다. Isaac Lab 버전에 맞는 wrapper 경로를 확인하세요."
    ) from e


def main():
    vecnorm_path = "checkpoints/good_robot_franka_vecnormalize.pkl"

    cfg = GoodRobotFrankaEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    env = GoodRobotFrankaEnv(cfg=cfg, render_mode="human")
    env = Sb3VecEnvWrapper(env)

    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

    model = PPO.load(args_cli.checkpoint, env=env)

    obs = env.reset()
    while simulation_app.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()