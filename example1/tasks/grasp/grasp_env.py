from __future__ import annotations

import torch

from source.example1.tasks.common.franka_base_env import FrankaBaseEnv
from .grasp_env_cfg import GraspEnvCfg


class GraspEnv(FrankaBaseEnv):
    cfg: GraspEnvCfg

    def __init__(self, cfg: GraspEnvCfg, render_mode: str | None = None, **kwargs):
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

        # 2) 이전 step보다 얼마나 가까워졌는지
        progress_reward = self._prev_dist - dist

        # 3) 물체가 약간 들렸는지
        lifted_small = (obj_pos[:, 2] > self.cfg.grasp_height_threshold).float()

        # 4) action penalty
        action_penalty = torch.sum(self._actions**2, dim=-1)

        reward = (
            2.0 * dist_reward
            + 5.0 * progress_reward
            + 10.0 * lifted_small
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
        # 1) 집게 가운데점이 물체에 충분히 가까움
        # 2) 물체가 약간 들림
        success = (dist < self.cfg.reach_threshold) & (obj_pos[:, 2] > self.cfg.grasp_height_threshold)

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