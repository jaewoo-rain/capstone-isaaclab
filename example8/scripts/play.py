"""계층 정책 재생 스크립트

Lift policy (PPO + VecNormalize) → Place policy (SAC) 순차 실행.
물체 높이 > lift_to_place_threshold가 되면 policy를 전환.

사용 예:
    ./isaaclab.sh -p source/example8/scripts/play.py \\
        --lift_ckpt checkpoints/omy_lift.zip \\
        --lift_vecnorm checkpoints/omy_lift_vecnorm.pkl \\
        --place_ckpt checkpoints/example6_best.zip
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hierarchical Lift + Place play")
parser.add_argument("--lift_ckpt", type=str, default="checkpoints/omy_lift.zip")
parser.add_argument("--lift_vecnorm", type=str, default="checkpoints/omy_lift_vecnorm.pkl")
parser.add_argument("--place_ckpt", type=str, default="checkpoints/example6_best.zip")
parser.add_argument(
    "--place_vecnorm", type=str, default=None,
    help="Place 모델용 VecNormalize (사용 안 했으면 비워두기)",
)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument(
    "--switch_threshold", type=float, default=0.20,
    help="물체 높이가 이 값(m)을 넘으면 lift→place 전환",
)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import pickle

import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, VecEnvWrapper

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper

from source.example8.tasks.hierarchy.hier_env import HierEnv
from source.example8.tasks.hierarchy.hier_env_cfg import HierEnvCfg


# example6의 GoalEnvVecWrapper와 동일 (place policy 호환)
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


# Lift VecNormalize 통계만 로드해서 obs 정규화에 쓰는 헬퍼
class LiftObsNormalizer:
    """저장된 VecNormalize의 obs_rms 통계로 lift obs를 정규화."""

    def __init__(self, vecnorm_path: str):
        if not os.path.exists(vecnorm_path):
            raise FileNotFoundError(f"VecNormalize 파일 없음: {vecnorm_path}")
        with open(vecnorm_path, "rb") as f:
            vn = pickle.load(f)
        self.mean = vn.obs_rms.mean.astype(np.float32)
        self.var = vn.obs_rms.var.astype(np.float32)
        self.epsilon = float(getattr(vn, "epsilon", 1e-8))
        self.clip_obs = float(getattr(vn, "clip_obs", 10.0))
        print(f"✅ Lift VecNormalize 통계 로드: {vecnorm_path}")
        print(f"   obs dim = {self.mean.shape[0]}")

    def normalize(self, obs_np: np.ndarray) -> np.ndarray:
        normalized = (obs_np - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normalized, -self.clip_obs, self.clip_obs).astype(np.float32)


def main():
    cfg = HierEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    # ---- env 생성 ----
    raw_env = HierEnv(cfg=cfg)
    sb3_env = Sb3VecEnvWrapper(raw_env)

    # place용 wrapper 체인 (Dict obs)
    place_env = GoalEnvVecWrapper(
        sb3_env,
        core_dim=HierEnv.OBS_CORE_DIM,
        goal_dim=HierEnv.GOAL_DIM,
        compute_reward_fn=raw_env.compute_reward,
    )

    if args_cli.place_vecnorm and os.path.exists(args_cli.place_vecnorm):
        place_env = VecNormalize.load(args_cli.place_vecnorm, place_env)
        place_env.training = False
        place_env.norm_reward = False
        print(f"✅ Place VecNormalize 로드: {args_cli.place_vecnorm}")

    # ---- 모델 로드 ----
    lift_model = PPO.load(args_cli.lift_ckpt, device="cuda")
    print(f"✅ Lift 모델 로드: {args_cli.lift_ckpt}")

    # SAC + HER로 학습된 모델은 env 필요 (HerReplayBuffer 복원용)
    place_model = SAC.load(args_cli.place_ckpt, env=place_env, device="cuda")
    print(f"✅ Place 모델 로드: {args_cli.place_ckpt}")

    lift_normalizer = LiftObsNormalizer(args_cli.lift_vecnorm)

    # ---- 재생 루프 ----
    obs_dict = place_env.reset()  # place 형식 dict 반환
    phase = "lift"
    episode_count = 0
    lift_steps = 0
    place_steps = 0
    print(f"\n=== Episode {episode_count + 1} 시작 (LIFT) ===")

    while simulation_app.is_running():
        if phase == "lift":
            # 1) 현재 물체 높이 확인
            obj_height = float(raw_env.get_object_height()[0])

            # 전환 조건: 물체 높이 > 임계값
            if obj_height > args_cli.switch_threshold:
                phase = "place"
                print(f"  → 물체 높이 {obj_height:.3f}m 도달, PLACE 단계로 전환 (lift_steps={lift_steps})")
                continue

            # 2) lift 형식 obs 가져와서 정규화
            lift_obs = raw_env.get_lift_observation().detach().cpu().numpy()  # (num_envs, 34)
            lift_obs_norm = lift_normalizer.normalize(lift_obs)

            # 3) lift policy로 action 예측
            actions, _ = lift_model.predict(lift_obs_norm, deterministic=True)

            lift_steps += 1
        else:  # place
            # place_env가 자동으로 dict 형식 obs를 만들어줌
            actions, _ = place_model.predict(obs_dict, deterministic=True)
            place_steps += 1

        # 환경 step (action은 두 policy 모두 7차원 [-1,1])
        obs_dict, rewards, dones, infos = place_env.step(actions)

        if dones[0]:
            episode_count += 1
            print(f"=== Episode {episode_count} 종료: lift_steps={lift_steps}, place_steps={place_steps} ===\n")
            phase = "lift"
            lift_steps = 0
            place_steps = 0
            print(f"=== Episode {episode_count + 1} 시작 (LIFT) ===")

    place_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
