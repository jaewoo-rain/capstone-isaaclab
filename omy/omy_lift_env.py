"""
들어올리기 env

역할:

이미 잡은 상태 가정
위로 들어올리는 reward

-> lift skill 학습
"""

from __future__ import annotations

import torch

from .omy_base_env import OmyBaseEnv
from .omy_env_cfg import OmyLiftEnvCfg


class OmyLiftEnv(OmyBaseEnv):
    cfg: OmyLiftEnvCfg

    def _get_rewards(self) -> torch.Tensor:
        t = self._get_common_terms()

        obj_pos = t["obj_pos"]
        gripper_joint = t["gripper_joint"]
        approach_reward = t["approach_reward"]
        is_grasping = t["is_grasping"]
        pre_grasp_ready = t["pre_grasp_ready"]

        obj_height = obj_pos[:, 2]

        # 1. grasp 준비 상태
        pre_grasp_reward = pre_grasp_ready.float() * 1.0

        # 2. grasp 유지 보상
        grasp_hold_reward = is_grasping.float() * 6.0

        # 3. 닫힘 유지 보상
        close_reward = torch.clamp(gripper_joint, min=0.0) * 2.0 * pre_grasp_ready.float()

        # 4. lift 핵심 보상
        # 바닥 근처 0.04 기준에서 얼마나 올라갔는지
        raw_lift = torch.clamp(obj_height - 0.04, min=0.0)

        # grasp 상태일 때 lift 보상 강하게
        lift_reward = raw_lift * (1.0 + 4.0 * is_grasping.float()) * 25.0

        # 5. success reward
        success = (obj_height > self.cfg.lift_height_threshold).float()
        success_reward = success * (self.cfg.success_bonus + 20.0)

        # 6. 접근 보상은 약하게만 유지
        weak_approach_reward = approach_reward * 2.0

        # 7. action penalty
        arm_actions = self.actions[:, :6]
        action_penalty = torch.sum(arm_actions**2, dim=-1) * 0.001

        reward = (
            weak_approach_reward
            + pre_grasp_reward
            + close_reward
            + grasp_hold_reward
            + lift_reward
            + success_reward
            - action_penalty
        )

        self.extras["log"] = {
            "approach_reward": weak_approach_reward.mean(),
            "pre_grasp_reward": pre_grasp_reward.mean(),
            "grasp_hold_reward": grasp_hold_reward.mean(),
            "close_reward": close_reward.mean(),
            "lift_reward": lift_reward.mean(),
            "success_rate": success.mean(),
            "obj_height": obj_height.mean(),
            "gripper_joint": gripper_joint.mean(),
        }

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_height = self.obj_pos_w[:, 2]

        # 너무 떨어지면 실패
        dropped = obj_height < 0.0

        # 충분히 들면 성공 종료
        success = obj_height > self.cfg.lift_height_threshold

        terminated = dropped | success
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated