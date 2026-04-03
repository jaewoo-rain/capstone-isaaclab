"""SB3 PPO로 GoodRobotFrankaEnv 학습."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train PPO for GoodRobotFrankaEnv")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timesteps", type=int, default=1_000_000)
parser.add_argument("--resume", action="store_true", help="기존 good_robot 체크포인트 이어서 학습")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from my_grasp_rl.tasks.good_robot.good_robot_franka_env_cfg import GoodRobotFrankaEnvCfg
from my_grasp_rl.tasks.good_robot.good_robot_franka_env import GoodRobotFrankaEnv
from my_grasp_rl.tasks.sb3_ppo_cfg import SB3_PPO_CFG

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except Exception as e:
    raise RuntimeError(
        "isaaclab_rl의 SB3 wrapper를 불러오지 못했습니다. Isaac Lab 버전에 맞는 wrapper 경로를 확인하세요."
    ) from e


def main():
    checkpoint_dir = "checkpoints"
    checkpoint_path = os.path.join(checkpoint_dir, "good_robot_franka_ppo")
    vecnorm_path = os.path.join(checkpoint_dir, "good_robot_franka_vecnormalize.pkl")

    os.makedirs(checkpoint_dir, exist_ok=True)

    cfg = GoodRobotFrankaEnvCfg()
    if args_cli.num_envs is not None:
        cfg.scene.num_envs = args_cli.num_envs

    env = GoodRobotFrankaEnv(cfg=cfg, render_mode=None)
    env = Sb3VecEnvWrapper(env)

    if args_cli.resume and os.path.exists(checkpoint_path + ".zip") and os.path.exists(vecnorm_path):
        print("🔄 기존 good_robot 모델과 VecNormalize를 불러와 이어서 학습합니다.")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        env.norm_reward = True

        model = PPO.load(checkpoint_path, env=env)
        model.learn(
            total_timesteps=args_cli.timesteps,
            reset_num_timesteps=False,
        )
    else:
        print("🆕 새 good_robot 모델로 학습을 시작합니다.")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        model = PPO(
            env=env,
            seed=args_cli.seed,
            **SB3_PPO_CFG,
        )

        model.learn(
            total_timesteps=args_cli.timesteps,
            reset_num_timesteps=True,
        )

    model.save(checkpoint_path)
    env.save(vecnorm_path)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()