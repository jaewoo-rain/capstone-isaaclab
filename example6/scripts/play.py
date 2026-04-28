"""학습된 Place SAC 정책 재생 — VecNormalize 옵션 포함"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained SAC example6 Place")
parser.add_argument("--checkpoint", type=str, default="checkpoints/example6.zip")
parser.add_argument(
    "--vecnorm", type=str, default="checkpoints/example6_vecnorm.pkl",
    help="VecNormalize pkl 경로 (--vecnorm_on일 때만 사용)",
)
parser.add_argument("--vecnorm_on", action="store_true", help="VecNormalize 활성화")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, VecEnvWrapper

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper

from source.example6.tasks.place.place_env import PlaceEnv
from source.example6.tasks.place.place_env_cfg import PlaceEnvCfg


# train.py의 GoalEnvVecWrapper와 동일 (중복 정의 — 모듈 의존 최소화)
class GoalEnvVecWrapper(VecEnvWrapper):
    def __init__(self, venv, core_dim: int, goal_dim: int, compute_reward_fn):
        super().__init__(venv)
        self._core_dim = core_dim
        self._goal_dim = goal_dim
        self._compute_reward_fn = compute_reward_fn

        low = np.full(core_dim, -np.inf, dtype=np.float32)
        high = np.full(core_dim, np.inf, dtype=np.float32)
        goal_low = np.full(goal_dim, -np.inf, dtype=np.float32)
        goal_high = np.full(goal_dim, np.inf, dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=low, high=high, dtype=np.float32),
            "achieved_goal": gym.spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
            "desired_goal": gym.spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
        })

    def _split(self, obs: np.ndarray) -> dict:
        core = obs[..., : self._core_dim]
        achieved = obs[..., self._core_dim : self._core_dim + self._goal_dim]
        desired = obs[..., self._core_dim + self._goal_dim :]
        return {
            "observation": core.astype(np.float32),
            "achieved_goal": achieved.astype(np.float32),
            "desired_goal": desired.astype(np.float32),
        }

    def reset(self):
        obs = self.venv.reset()
        return self._split(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._split(obs), rewards, dones, infos

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._compute_reward_fn(achieved_goal, desired_goal, info)


def main():
    cfg = PlaceEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    raw_env = PlaceEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)
    env = GoalEnvVecWrapper(
        env,
        core_dim=PlaceEnv.OBS_CORE_DIM,
        goal_dim=PlaceEnv.GOAL_DIM,
        compute_reward_fn=raw_env.compute_reward,
    )

    if args_cli.vecnorm_on:
        if os.path.exists(args_cli.vecnorm):
            env = VecNormalize.load(args_cli.vecnorm, env)
            env.training = False
            env.norm_reward = False
            print(f"✅ VecNormalize 로드: {args_cli.vecnorm}")
        else:
            print(f"⚠️ VecNormalize 파일 없음 ({args_cli.vecnorm}). 정규화 없이 실행.")

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
