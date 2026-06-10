"""rl/insert/insert_policy.py — motion3 insert PPO 정책 로더 (IsaacLab 무의존).

checkpoints/motion3_insert_v25.zip (SB3 PPO) + _vecnorm.pkl (VecNormalize) 을
순수 stable-baselines3 + numpy 로 로드해 추론만 한다.  obs(7) → action(3).

⚠️ yaw_only: 정책은 obs 7개를 받지만, 출력 action 중 yaw(action[2]) 만 사용한다
   (xy 는 IK 가 셀에 고정). 정책 추론 자체는 grasp 와 동일.

의존성: stable-baselines3, numpy.
"""
from __future__ import annotations

import pickle

import numpy as np

from config import CKPT_PATH, VECNORM_PATH


class InsertPolicy:
    """insert 정책 추론기. obs(7,) → action(3,) ∈ [-1, 1]."""

    def __init__(self, ckpt_path=CKPT_PATH, vecnorm_path=VECNORM_PATH, device: str = "cpu"):
        from stable_baselines3 import PPO

        self.model = PPO.load(str(ckpt_path), device=device)

        with open(vecnorm_path, "rb") as f:
            self._vecnorm = pickle.load(f)
        self._vecnorm.training = False
        self._vecnorm.norm_reward = False

        self._mean = np.asarray(self._vecnorm.obs_rms.mean, dtype=np.float64)
        self._var = np.asarray(self._vecnorm.obs_rms.var, dtype=np.float64)
        self._epsilon = float(getattr(self._vecnorm, "epsilon", 1e-8))
        self._clip_obs = float(getattr(self._vecnorm, "clip_obs", 10.0))

        obs_dim = self.model.observation_space.shape[0]
        act_dim = self.model.action_space.shape[0]
        if obs_dim != 7 or act_dim != 3:
            raise RuntimeError(
                f"예상과 다른 정책 shape: obs={obs_dim}(기대 7), action={act_dim}(기대 3). "
                f"잘못된 체크포인트일 수 있음: {ckpt_path}")

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float64)
        norm = (obs - self._mean) / np.sqrt(self._var + self._epsilon)
        return np.clip(norm, -self._clip_obs, self._clip_obs)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """raw obs(7,) → action(3,) ∈ [-1, 1] (deterministic)."""
        norm_obs = self.normalize_obs(obs).astype(np.float32)
        action, _ = self.model.predict(norm_obs, deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)


if __name__ == "__main__":
    pol = InsertPolicy()
    print("[insert_policy] loaded OK")
    print("  obs_rms.mean :", np.round(pol._mean, 5).tolist())
    print("  obs_rms.var  :", np.round(pol._var, 5).tolist())
    # 셀 위 호버, xy 정렬됨(0), yaw 0.3rad 오차 예시
    dummy = np.array([0.0, 0.0, 0.30, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    print("  dummy obs    :", dummy.tolist())
    print("  action       :", np.round(pol.predict(dummy), 4).tolist(), "(yaw=action[2] 만 사용)")
