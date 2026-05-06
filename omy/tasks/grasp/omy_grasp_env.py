from __future__ import annotations

import os
import torch

from source.omy.tasks.common.omy_base_env import OmyBaseEnv
from source.omy.tasks.common.omy_env_cfg import OmyLiftEnvCfg


class OmyGraspEnv(OmyBaseEnv):
    cfg: OmyLiftEnvCfg

    def __init__(self, cfg: OmyLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.success_state_path = "checkpoints/grasp_success_states.pt"
        self.max_saved_states = 2000
        self._saved_success_count = 0

        # 몇 번 쌓일 때마다 파일 저장할지
        self.save_every_n_successes = 100

        self._grasp_hold_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        os.makedirs("checkpoints", exist_ok=True)

        if os.path.exists(self.success_state_path):
            self.success_states = torch.load(self.success_state_path)
        else:
            self.success_states = {
                "joint_pos": [],
                "joint_vel": [],
                "object_root_state": [],
            }

    def _task_specific_reset(self, env_ids: torch.Tensor) -> None:
        pass

    def _get_rewards(self) -> torch.Tensor:
        t = self._get_common_terms()

        obj_pos = t["obj_pos"]
        gripper_joint = t["gripper_joint"]
        xy_dist = t["xy_dist"]
        z_dist = t["z_dist"]
        approach_reward = t["approach_reward"]
        is_grasping = t["is_grasping"]
        aligned = t["aligned"]
        xy_align_reward = t["xy_align_reward"]
        z_align_reward = t["z_align_reward"]

        grasp_bonus = is_grasping.float() * 4.0

        close_amount = torch.clamp(gripper_joint, min=0.0)
        close_reward = close_amount * aligned

        obj_height = obj_pos[:, 2]
        raw_lift = torch.clamp(obj_height - 0.04, min=0.0)
        lift_reward = raw_lift * (2.0 * is_grasping.float()) * 12.0

        success = (obj_height > self.cfg.lift_height_threshold).float()
        success_reward = success * self.cfg.success_bonus

        arm_actions = self.actions[:, :6]
        action_penalty = torch.sum(arm_actions**2, dim=-1) * 0.001

        reward = (
            # 0.3 * approach_reward
            + 2.0 * xy_align_reward
            + 1.2 * z_align_reward
            + 4.0 * close_reward
            + 8.0 * grasp_bonus
            - action_penalty
        )

        # grasp 유지 카운트
        self._grasp_hold_counter[is_grasping] += 1
        self._grasp_hold_counter[~is_grasping] = 0

        stable_success = self._grasp_hold_counter >= 3
        self._collect_success_states(stable_success)

        self.reward_log = {
            "approach_reward": float(approach_reward.mean()),
            "xy_align_reward": float(t["xy_align_reward"].mean()),
            "z_align_reward": float(t["z_align_reward"].mean()),
            "grasp_bonus": float(grasp_bonus.mean()),
            "xy_dist": float(xy_dist.mean()),
            "z_dist": float(z_dist.mean()),
            "close_reward": float(close_reward.mean()),
        }

        self.extras["log"] = dict(self.reward_log)

        return reward

    def _collect_success_states(self, success_mask: torch.Tensor) -> None:
        success_ids = torch.where(success_mask)[0]

        if len(success_ids) == 0:
            return

        joint_pos = self._robot.data.joint_pos[success_ids].detach().cpu()
        joint_vel = self._robot.data.joint_vel[success_ids].detach().cpu()
        object_root_state = self._object.data.root_state_w[success_ids].detach().cpu()
        env_origins = self.scene.env_origins[success_ids].detach().cpu()

        # world -> local
        object_root_state[:, 0:3] -= env_origins

        for i in range(len(success_ids)):
            self.success_states["joint_pos"].append(joint_pos[i].clone())
            self.success_states["joint_vel"].append(joint_vel[i].clone())
            self.success_states["object_root_state"].append(object_root_state[i].clone())
            self._saved_success_count += 1

        if len(self.success_states["joint_pos"]) > self.max_saved_states:
            keep = self.max_saved_states
            self.success_states["joint_pos"] = self.success_states["joint_pos"][-keep:]
            self.success_states["joint_vel"] = self.success_states["joint_vel"][-keep:]
            self.success_states["object_root_state"] = self.success_states["object_root_state"][-keep:]

        # 파일 저장은 가끔만
        if self._saved_success_count > 0 and self._saved_success_count % self.save_every_n_successes == 0:
            torch.save(self.success_states, self.success_state_path)

    def flush_success_states(self) -> None:
        torch.save(self.success_states, self.success_state_path)