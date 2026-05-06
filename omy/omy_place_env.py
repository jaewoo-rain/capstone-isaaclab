from __future__ import annotations

import torch

from .omy_base_env import OmyBaseEnv
from .omy_env_cfg import OmyLiftEnvCfg


class OmyPlaceEnv(OmyBaseEnv):
    cfg: OmyLiftEnvCfg

    def __init__(self, cfg: OmyLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 목표 적재 위치 (env 원점 기준)
        self.place_target = torch.tensor([0.60, 0.20, 0.08], device=self.device).repeat(self.num_envs, 1)

    def _get_rewards(self) -> torch.Tensor:
        t = self._get_common_terms()

        obj_pos = t["obj_pos"]
        gripper_joint = t["gripper_joint"]
        is_grasping = t["is_grasping"]
        pre_grasp_ready = t["pre_grasp_ready"]

        env_origins = self.scene.env_origins
        target_pos = env_origins + self.place_target

        obj_to_target = target_pos - obj_pos
        target_dist = torch.norm(obj_to_target, dim=-1)

        obj_height = obj_pos[:, 2]

        # 1. grasp 유지
        grasp_hold_reward = is_grasping.float() * 5.0

        # 2. 목표 위치로 가까워질수록 보상
        place_approach_reward = torch.exp(-10.0 * target_dist) * (1.0 + is_grasping.float()) * 8.0

        # 3. 물체 높이를 어느 정도 유지
        height_keep_reward = torch.clamp(obj_height - 0.05, min=0.0) * 5.0

        # 4. 목표 위치 근처에서 gripper를 열면 보상
        near_target = target_dist < 0.05
        open_reward = near_target.float() * torch.clamp(-gripper_joint + 0.5, min=0.0) * 6.0

        # 5. 성공 판정
        placed = (target_dist < 0.04) & (obj_height > 0.04) & (~is_grasping)
        success = placed.float()
        success_reward = success * 25.0

        # 6. action penalty
        arm_actions = self.actions[:, :6]
        action_penalty = torch.sum(arm_actions**2, dim=-1) * 0.001

        reward = (
            grasp_hold_reward
            + place_approach_reward
            + height_keep_reward
            + open_reward
            + success_reward
            - action_penalty
        )

        self.extras["log"] = {
            "grasp_hold_reward": grasp_hold_reward.mean(),
            "place_approach_reward": place_approach_reward.mean(),
            "height_keep_reward": height_keep_reward.mean(),
            "open_reward": open_reward.mean(),
            "success_rate": success.mean(),
            "target_dist": target_dist.mean(),
            "obj_height": obj_height.mean(),
        }

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._get_common_terms()

        obj_pos = t["obj_pos"]
        is_grasping = t["is_grasping"]

        env_origins = self.scene.env_origins
        target_pos = env_origins + self.place_target
        target_dist = torch.norm(target_pos - obj_pos, dim=-1)

        obj_height = obj_pos[:, 2]

        success = (target_dist < 0.04) & (obj_height > 0.04) & (~is_grasping)
        dropped = obj_height < 0.0

        terminated = success | dropped
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated