"""
잡기 전용 env

역할:

접근
정렬
닫기
grasp 판정

👉 grasp skill 학습
"""
from __future__ import annotations

import torch

from source.omy.tasks.common.omy_base_env import OmyBaseEnv
from source.omy.tasks.common.omy_env_cfg import OmyLiftEnvCfg


class OmyGraspEnv(OmyBaseEnv):
    cfg: OmyLiftEnvCfg

    def _get_rewards(self) -> torch.Tensor:
        t = self._get_common_terms()

        obj_pos = t["obj_pos"]
        gripper_joint = t["gripper_joint"]
        xy_dist = t["xy_dist"]
        z_dist = t["z_dist"]
        approach_reward = t["approach_reward"]
        is_grasping = t["is_grasping"]
        pre_grasp_ready = t["pre_grasp_ready"]
        aligned = t["aligned"]

        # -------------------------
        # 1. grasp bonus
        # -------------------------
        grasp_bonus = is_grasping.float() * 4.0
        xy_align_reward = torch.exp(-60.0 * xy_dist**2)
        xy_align = xy_dist < 0.1
        z_align_reward = torch.exp(-60.0 * z_dist**2) *xy_align_reward

        # -------------------------
        # 2. close reward
        # - 정렬이 잘 된 상태에서 gripper를 닫을수록 보상
        # -------------------------
        alignment_score = torch.exp(-60.0 * xy_dist**2) * torch.exp(-60.0 * z_dist**2)
        close_amount = torch.clamp(gripper_joint, min=0.0)
        close_reward = alignment_score * close_amount * aligned

        # -------------------------
        # 3. pre-grasp reward
        # - 잡기 직전 상태 자체에도 보상
        # -------------------------
        pre_grasp_reward = pre_grasp_ready.float() * 2.0

        # -------------------------
        # 4. lift reward
        # - grasp가 된 상태에서 높이가 오르면 추가 보상
        # -------------------------
        obj_height = obj_pos[:, 2]
        raw_lift = torch.clamp(obj_height - 0.04, min=0.0)
        lift_reward = raw_lift * (2.0 * is_grasping.float()) * 12.0

        # -------------------------
        # 5. success reward
        # -------------------------
        success = (obj_height > self.cfg.lift_height_threshold).float()
        success_reward = success * self.cfg.success_bonus

        # -------------------------
        # 6. action penalty
        # - arm만 패널티
        # -------------------------
        arm_actions = self.actions[:, :6]
        action_penalty = torch.sum(arm_actions**2, dim=-1) * 0.001

        # -------------------------
        # 7. final reward
        # -------------------------
        # reward = (
        #     0.1 * approach_reward
        #     + 0.1 * xy_align_reward
        #     # + 0.001 * z_align_reward
        #     # + pre_grasp_reward
        #     + 3 * close_reward
        #     # + grasp_bonus
        #     # + lift_reward
        #     # + success_reward
        #     - action_penalty
        # )
        reward = (
            0.3 * approach_reward
            + 0.8 * xy_align_reward
            + 0.3 * z_align_reward
            + 2.0 * pre_grasp_reward
            + 4.0 * close_reward
            + 8.0 * grasp_bonus
            - action_penalty
        )

        # -------------------------
        # 8. logging
        # -------------------------
        self.reward_log = {
            "approach_reward": float(approach_reward.mean()),
            "xy_align_reward": float(t["xy_align_reward"].mean()),
            "z_align_reward": float(t["z_align_reward"].mean()),
            "pre_grasp_reward": float(pre_grasp_reward.mean()),
            "grasp_bonus": float(grasp_bonus.mean()),
            # "lift_reward": float(lift_reward.mean()),
            # "success_rate": float(success.mean()),
            "xy_dist": float(xy_dist.mean()),
            "z_dist": float(z_dist.mean()),
            # "gripper_joint": float(gripper_joint.mean()),
            "close_reward": float(close_reward.mean()),
        }

        self.extras["log"] = dict(self.reward_log)

        return reward