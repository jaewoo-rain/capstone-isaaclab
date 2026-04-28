"""학습된 example7 SAC 정책 재생 — VecNormalize 포함."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained example7 SAC Lift")
parser.add_argument("--checkpoint", type=str, default="checkpoints/example7.zip")
parser.add_argument("--vecnorm", type=str, default="checkpoints/example7_vecnorm.pkl")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper

from source.example7.tasks.lift.lift_env import LiftEnv
from source.example7.tasks.lift.lift_env_cfg import LiftEnvCfg


def main():
    cfg = LiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    env = LiftEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(env)

    if os.path.exists(args_cli.vecnorm):
        env = VecNormalize.load(args_cli.vecnorm, env)
        env.training = False
        env.norm_reward = False
        print(f"✅ VecNormalize 로드: {args_cli.vecnorm}")
    else:
        print(f"⚠️ VecNormalize 파일 없음 ({args_cli.vecnorm}) — 정규화 없이 실행")

    model = SAC.load(args_cli.checkpoint, env=env, device="cuda")
    print(f"✅ 모델 로드: {args_cli.checkpoint}")

    obs = env.reset()
    episode_count = 0
    episode_rewards = []
    current_reward = 0.0

    while simulation_app.is_running():
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)
        current_reward += float(rewards[0])

        if dones[0]:
            episode_count += 1
            episode_rewards.append(current_reward)
            print(
                f"Episode {episode_count:4d} | "
                f"reward={current_reward:8.2f} | "
                f"avg_last10={sum(episode_rewards[-10:])/min(10, len(episode_rewards)):.2f}"
            )
            current_reward = 0.0

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
