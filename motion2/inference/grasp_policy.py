"""motion2 — Grasp RL 정책 inference.

stable_baselines3 PPO + VecNormalize 로드. obs (6-d) → action (3-d).

Obs (6) — 학습 시 정의 (sim/real 동일 식):
    [box_x - ee_x, box_y - ee_y, box_yaw - ee_target_yaw,
     ee_vel_x, ee_vel_y, ee_target_yaw_rate]
    모두 world-frame, env-rel 또는 robot-base 기준 — 학습 시 환경과 일치하게.

Action (3) — [Δee_x, Δee_y, Δee_yaw] (단위 normalize, [-1, 1]):
    action[:2] * action_scale_xy   → meters (e.g. 0.01 = 10mm)
    action[2]  * action_scale_yaw  → radians (e.g. 0.05 ≈ 2.86°)
"""
from __future__ import annotations

import pickle
import numpy as np


class GraspPolicy:
    """SB3 PPO + VecNormalize 통계 wrapper."""

    def __init__(self, ckpt_path: str, vecnorm_path: str, device: str = "cpu"):
        """Args:
            ckpt_path: PPO .zip
            vecnorm_path: VecNormalize stats pickle (saved with stable_baselines3)
            device: 'cpu' or 'cuda'
        """
        from stable_baselines3 import PPO
        self.model = PPO.load(ckpt_path, device=device)
        with open(vecnorm_path, "rb") as f:
            self.vec = pickle.load(f)
        self.vec.training = False
        self.vec.norm_reward = False
        print(f"[GraspPolicy] loaded ckpt={ckpt_path}, vecnorm={vecnorm_path}")

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """obs (6,) float → normalized (6,) float32. batch (1, 6) 내부 처리."""
        norm = self.vec.normalize_obs(obs.astype(np.float32)[None, :])
        return np.asarray(norm[0], dtype=np.float32)

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """obs (6,) → action (3,) in [-1, 1]. deterministic."""
        obs_norm = self.normalize(obs)
        action, _ = self.model.predict(obs_norm, deterministic=True)
        return np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
