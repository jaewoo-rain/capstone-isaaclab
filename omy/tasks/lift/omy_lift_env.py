from __future__ import annotations

import torch
import os

from source.omy.tasks.common.omy_base_env import OmyBaseEnv
from source.omy.tasks.common.omy_env_cfg import OmyLiftEnvCfg


class OmyLiftEnv(OmyBaseEnv):
    cfg: OmyLiftEnvCfg

    def __init__(self, cfg: OmyLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.success_state_path = "checkpoints/grasp_success_states.pt"

        if os.path.exists(self.success_state_path):
            self.success_states = torch.load(self.success_state_path)
            self.has_success_states = len(self.success_states["joint_pos"]) > 0
        else:
            self.success_states = None
            self.has_success_states = False

    def _task_specific_reset(self, env_ids: torch.Tensor) -> None:
        if not self.has_success_states:
            print("[WARN] 저장된 grasp 성공 상태가 없어서 기본 reset 사용")
            return

        num_saved = len(self.success_states["joint_pos"])
        rand_idx = torch.randint(0, num_saved, (len(env_ids),), device="cpu")

        joint_pos = torch.stack(
            [self.success_states["joint_pos"][i] for i in rand_idx.tolist()],
            dim=0,
        ).to(self.device)

        joint_vel = torch.stack(
            [self.success_states["joint_vel"][i] for i in rand_idx.tolist()],
            dim=0,
        ).to(self.device)

        object_root_state = torch.stack(
            [self.success_states["object_root_state"][i] for i in rand_idx.tolist()],
            dim=0,
        ).to(self.device)

        # env origin 차이 보정
        env_origins = self.scene.env_origins[env_ids]

        # local -> world
        object_root_state[:, 0:3] += env_origins

        # 저장된 상태가 world 좌표 기준이면, env 별 origin 차이만 맞춰줘야 할 수 있음
        # 가장 안전한 건 object를 env local 기준으로 저장하는 건데,
        # 일단 지금은 num_envs=1로 먼저 검증하는 걸 추천

        self.robot_dof_targets[env_ids] = joint_pos

        # reset 시에는 target 말고 state만 먼저 쓰는 편이 더 안전
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._object.write_root_state_to_sim(object_root_state, env_ids=env_ids)

        self._compute_intermediate_values(env_ids)

    def _get_rewards(self) -> torch.Tensor:
        t = self._get_common_terms()

        obj_pos = t["obj_pos"]
        gripper_joint = t["gripper_joint"]
        is_grasping = t["is_grasping"]

        obj_height = obj_pos[:, 2]

        grasp_hold_reward = is_grasping.float() * 4.0
        close_reward = torch.clamp(gripper_joint, min=0.0) * is_grasping.float()

        raw_lift = torch.clamp(obj_height - 0.06, min=0.0)
        lift_reward = raw_lift * (1.0 + 4.0 * is_grasping.float()) * 25.0

        success = (obj_height > self.cfg.lift_height_threshold).float()
        success_reward = success * (self.cfg.success_bonus + 20.0)

        arm_actions = self.actions[:, :6]
        action_penalty = torch.sum(arm_actions**2, dim=-1) * 0.001

        reward = (
            grasp_hold_reward
            + 0.5 * close_reward
            + lift_reward
            + success_reward
            - action_penalty
        )

        self.reward_log = {
            "grasp_hold_reward": float(grasp_hold_reward.mean()),
            "close_reward": float(close_reward.mean()),
            "lift_reward": float(lift_reward.mean()),
            "success_rate": float(success.mean()),
            "obj_height": float(obj_height.mean()),
            "gripper_joint": float(gripper_joint.mean()),
            "is_grasping": float(is_grasping.float().mean()),
        }
        self.extras["log"] = dict(self.reward_log)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_height = self.obj_pos_w[:, 2]

        success = obj_height > self.cfg.lift_height_threshold
        dropped = obj_height < 0.02

        terminated = success | dropped
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated