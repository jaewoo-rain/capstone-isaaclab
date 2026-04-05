from __future__ import annotations

import torch

from source.example1.tasks.common.franka_base_env import FrankaBaseEnv
from .lift_env_cfg import LiftEnvCfg


class LiftEnv(FrankaBaseEnv):
    cfg: LiftEnvCfg

    def __init__(self, cfg: LiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)

    def _get_rewards(self) -> torch.Tensor:
        obj_pos = self._get_obj_pos()
        left_finger_pos, right_finger_pos = self._get_finger_positions()

        # 집게 가운데점
        finger_center = 0.5 * (left_finger_pos + right_finger_pos)

        # 1) 집게 가운데점과 물체 거리
        dist = torch.norm(obj_pos - finger_center, dim=-1)
        dist_reward = 1.0 / (1.0 + dist**2)

        # 2) 이전보다 얼마나 가까워졌는지
        progress_reward = self._prev_dist - dist

        # 3) 높이 보상
        height_reward = torch.clamp(obj_pos[:, 2] - 0.025, min=0.0)

        # 4) 성공 보상
        lifted = (obj_pos[:, 2] > self.cfg.lift_height_threshold).float()

        # 5) action penalty
        action_penalty = torch.sum(self._actions**2, dim=-1)

        reward = (
            1.5 * dist_reward
            + 4.0 * progress_reward
            + 8.0 * height_reward
            + 20.0 * lifted
            - 0.01 * action_penalty
        )

        # 다음 step용 거리 저장
        self._prev_dist = dist.detach()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_pos = self._get_obj_pos()
        left_finger_pos, right_finger_pos = self._get_finger_positions()

        finger_center = 0.5 * (left_finger_pos + right_finger_pos)
        dist = torch.norm(obj_pos - finger_center, dim=-1)

        # 성공 조건:
        # 1) 물체가 충분히 올라감
        # 2) 집게 가운데점도 물체 근처에 있음
        success = (obj_pos[:, 2] > self.cfg.lift_height_threshold) & (dist < self.cfg.reach_threshold)

        time_out = self.episode_length_buf >= (self.max_episode_length - 1)

        self.extras["success"] = success.float()
        self.extras["mean_dist"] = dist.mean()
        self.extras["mean_obj_height"] = obj_pos[:, 2].mean()

        return success, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        obj_pos = self._get_obj_pos()[env_ids]
        left_finger_pos, right_finger_pos = self._get_finger_positions()

        finger_center = 0.5 * (left_finger_pos[env_ids] + right_finger_pos[env_ids])
        dist = torch.norm(obj_pos - finger_center, dim=-1)

        self._prev_dist[env_ids] = dist