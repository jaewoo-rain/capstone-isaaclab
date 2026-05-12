"""motion1 — Insert RL v2 시각화 (SAC + HER).

학습된 SAC 정책으로 단독 InsertEnvV2 시각화. cell 4 walls 표시 없이
정렬 동작만 확인.

실행:
    ./isaaclab.sh -p source/motion1/scripts/play_insert_v2.py
    ./isaaclab.sh -p source/motion1/scripts/play_insert_v2.py \\
        --checkpoint checkpoints/motion1_insert_v2_best.zip
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="motion1 Insert v2 SAC+HER 시각화")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--checkpoint", type=str, default="checkpoints/motion1_insert_v2.zip")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnvWrapper

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper

from source.motion1.tasks.insert_v2.insert_env_v2 import InsertEnvV2
from source.motion1.tasks.insert_v2.insert_env_v2_cfg import InsertEnvV2Cfg


# train_insert_v2.py 의 GoalEnvVecWrapper 와 동일 (중복 정의)
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
    cfg = InsertEnvV2Cfg()
    cfg.scene.num_envs = args_cli.num_envs

    if args_cli.seed is not None:
        torch.manual_seed(args_cli.seed)

    raw_env = InsertEnvV2(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)
    env = GoalEnvVecWrapper(
        env,
        core_dim=InsertEnvV2.OBS_CORE_DIM,
        goal_dim=InsertEnvV2.GOAL_DIM,
        compute_reward_fn=raw_env.compute_reward,
    )

    if not os.path.exists(args_cli.checkpoint):
        print(f"❌ checkpoint 없음: {args_cli.checkpoint}")
        return

    print(f"🔄 SAC 정책 로드: {args_cli.checkpoint}")
    model = SAC.load(args_cli.checkpoint, env=env, device="cuda")

    obs = env.reset()
    ep_count = 0
    ep_reward = 0.0
    ep_len = 0
    success_count = 0

    while ep_count < args_cli.episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)
        ep_reward += float(np.asarray(reward).mean())
        ep_len += 1

        if dones.any():
            ep_count += 1
            # raw_env 의 _aligned_count 마지막 값 확인 (성공 판정)
            log = raw_env.reward_log
            aligned_rate = log.get("rate_aligned", 0.0)
            if aligned_rate > 0.5:
                success_count += 1
            print(
                f"[episode {ep_count}/{args_cli.episodes}] "
                f"len={ep_len} reward={ep_reward:.1f} | "
                f"xy_dist={log.get('dist_xy_mean', 0):.4f}m yaw_err={log.get('dist_yaw_abs_mean', 0):.3f}rad | "
                f"aligned_rate={aligned_rate:.3f}"
            )
            ep_reward = 0.0
            ep_len = 0

    print(f"\n✅ {args_cli.episodes} episode 완료. success rate: {success_count}/{args_cli.episodes}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    os._exit(0)
