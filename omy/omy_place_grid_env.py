from __future__ import annotations

import torch

from .omy_base_env import OmyBaseEnv
from .omy_env_cfg import OmyLiftEnvCfg


class OmyPlaceGridEnv(OmyBaseEnv):
    cfg: OmyLiftEnvCfg

    def __init__(self, cfg: OmyLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # -------------------------
        # 3x3 grid target 정의
        # env origin 기준 상대좌표
        # -------------------------
        xs = [0.56, 0.62, 0.68]
        ys = [0.14, 0.20, 0.26]
        z = 0.04

        grid = []
        for y in ys:
            for x in xs:
                grid.append([x, y, z])

        self.grid_targets = torch.tensor(grid, dtype=torch.float32, device=self.device)  # (9, 3)

        # env마다 현재 목표 slot index
        self.current_slot = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # 마지막 성공 여부
        self.last_placed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    def _get_current_target_pos(self) -> torch.Tensor:
        """
        env별 현재 목표 slot의 월드 좌표 반환
        shape: (num_envs, 3)
        """
        rel_targets = self.grid_targets[self.current_slot]  # (N, 3)
        return self.scene.env_origins + rel_targets

    # ------------------------------------------------------------------
    # rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        t = self._get_common_terms()

        obj_pos = t["obj_pos"]
        gripper_joint = t["gripper_joint"]
        is_grasping = t["is_grasping"]

        target_pos = self._get_current_target_pos()
        obj_to_target = target_pos - obj_pos
        target_dist = torch.norm(obj_to_target, dim=-1)

        obj_height = obj_pos[:, 2]

        # 목표 위치 위쪽에서 유지되게 유도
        target_xy_dist = torch.norm(obj_to_target[:, :2], dim=-1)
        target_z_dist = torch.abs(obj_to_target[:, 2])

        xy_reward = torch.exp(-40.0 * target_xy_dist**2) * 8.0
        z_reward = torch.exp(-30.0 * target_z_dist**2) * 4.0

        # grasp 유지
        grasp_hold_reward = is_grasping.float() * 5.0

        # 너무 낮게 떨어뜨리지 않고 들고 있게
        height_keep_reward = torch.clamp(obj_height - 0.05, min=0.0) * 4.0

        # 목표 근처에서 gripper를 열도록 유도
        near_target = target_dist < 0.05
        open_reward = near_target.float() * torch.clamp(-gripper_joint + 0.5, min=0.0) * 6.0

        # 성공 판정:
        # 1) 목표 칸 근처
        # 2) 물체가 너무 바닥 밑으로 안 감
        # 3) 더 이상 grasp 중이 아님(놓았음)
        placed = (target_dist < 0.035) & (obj_height > 0.02) & (~is_grasping)

        success_reward = placed.float() * 20.0

        # action penalty
        arm_actions = self.actions[:, :6]
        action_penalty = torch.sum(arm_actions**2, dim=-1) * 0.001

        reward = (
            grasp_hold_reward
            + xy_reward
            + z_reward
            + height_keep_reward
            + open_reward
            + success_reward
            - action_penalty
        )

        self.last_placed = placed.clone()

        self.extras["log"] = {
            "slot_index_mean": self.current_slot.float().mean(),
            "grasp_hold_reward": grasp_hold_reward.mean(),
            "xy_reward": xy_reward.mean(),
            "z_reward": z_reward.mean(),
            "height_keep_reward": height_keep_reward.mean(),
            "open_reward": open_reward.mean(),
            "success_rate": placed.float().mean(),
            "target_dist": target_dist.mean(),
            "obj_height": obj_height.mean(),
        }

        return reward

    # ------------------------------------------------------------------
    # dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_height = self.obj_pos_w[:, 2]

        # 물체를 완전히 놓쳤으면 실패
        dropped = obj_height < 0.0

        # 마지막 슬롯(8번)까지 성공하면 종료
        finished_all = (self.current_slot >= 8) & self.last_placed

        terminated = dropped | finished_all
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------------
    # post step slot update
    # ------------------------------------------------------------------
    def _post_physics_step(self):
        super()._post_physics_step()

        # 이번 step에서 현재 슬롯 배치 성공한 env는 다음 슬롯으로 이동
        success_envs = torch.nonzero(self.last_placed).squeeze(-1)

        if len(success_envs) > 0:
            self.current_slot[success_envs] = torch.clamp(self.current_slot[success_envs] + 1, max=8)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self.current_slot[env_ids_t] = 0
        self.last_placed[env_ids_t] = False