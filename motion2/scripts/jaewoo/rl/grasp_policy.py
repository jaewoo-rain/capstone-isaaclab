"""rl/grasp_policy.py — motion1 grasp PPO 정책 로더 (IsaacLab 무의존).

checkpoints/motion1_grasp.zip (SB3 PPO) + motion1_grasp_vecnorm.pkl (VecNormalize)
을 Isaac Sim/IsaacLab 없이 순수 stable-baselines3 + numpy 로 로드해 추론만 한다.

obs(6) → action(3). VecNormalize 통계로 obs 를 정규화한 뒤 deterministic predict.

의존성: stable-baselines3, numpy (torch 는 sb3 가 끌어옴).
"""
from __future__ import annotations

import pickle

import numpy as np

from config import CKPT_PATH, VECNORM_PATH


class GraspPolicy:
    """grasp 정책 추론기. obs(6,) → action(3,) ∈ [-1, 1]."""

    def __init__(self, ckpt_path=CKPT_PATH, vecnorm_path=VECNORM_PATH, device: str = "cpu"):
        from stable_baselines3 import PPO

        self.model = PPO.load(str(ckpt_path), device=device)

        # VecNormalize 는 env 객체 없이 pickle 로 직접 로드 → obs_rms 통계만 사용
        with open(vecnorm_path, "rb") as f:
            self._vecnorm = pickle.load(f)
        self._vecnorm.training = False      # 통계 업데이트 안 함
        self._vecnorm.norm_reward = False   # reward 정규화 불필요

        # 정규화 파라미터 직접 추출 (env-free 수동 정규화에 사용)
        self._mean = np.asarray(self._vecnorm.obs_rms.mean, dtype=np.float64)
        self._var = np.asarray(self._vecnorm.obs_rms.var, dtype=np.float64)
        self._epsilon = float(getattr(self._vecnorm, "epsilon", 1e-8))
        self._clip_obs = float(getattr(self._vecnorm, "clip_obs", 10.0))

        obs_dim = self.model.observation_space.shape[0]
        act_dim = self.model.action_space.shape[0]
        if obs_dim != 6 or act_dim != 3:
            raise RuntimeError(
                f"예상과 다른 정책 shape: obs={obs_dim}(기대 6), action={act_dim}(기대 3). "
                f"잘못된 체크포인트일 수 있음: {ckpt_path}")

    # ------------------------------------------------------------
    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """raw obs(6,) → 정규화 obs(6,). sim VecNormalize 와 동일한 식."""
        obs = np.asarray(obs, dtype=np.float64)
        norm = (obs - self._mean) / np.sqrt(self._var + self._epsilon)
        return np.clip(norm, -self._clip_obs, self._clip_obs)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """raw obs(6,) → action(3,) ∈ [-1, 1] (deterministic)."""
        norm_obs = self.normalize_obs(obs).astype(np.float32)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)


if __name__ == "__main__":
    # 단독 실행: 정책 로드 + dummy obs 추론 sanity check
    pol = GraspPolicy()
    print("[grasp_policy] loaded OK")
    print("  obs_rms.mean :", np.round(pol._mean, 5).tolist())
    print("  obs_rms.var  :", np.round(pol._var, 5).tolist())
    # 박스가 EE 기준 +x 10cm, +y 5cm, yaw 0.3rad 떨어진 상황 예시
    dummy = np.array([0.10, 0.05, 0.30, 0.0, 0.0, 0.0], dtype=np.float64)
    print("  dummy obs    :", dummy.tolist())
    print("  action       :", np.round(pol.predict(dummy), 4).tolist())
