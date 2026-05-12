"""motion1 — Insert yaw 정렬 RL Env (단순 버전).

박스 잡힌 채 셀 위에서 yaw 만 정렬. handoff dataset 에서 시작 상태 random sample.

State (3):  yaw_err, yaw_vel, is_grasping
Action (1): Δyaw  (relative)
- ee_xy 는 cell_xy 에 고정 (학습 X)
- ee_z 고정 (= cfg.ee_fixed_z = 0.26)
- gripper close 고정 (박스 잡힌 상태 유지)
"""
from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
)

from .insert_env_cfg import InsertEnvCfg


# 베이스 EE quat (수직 아래 + finger Y) — motion1 chain 과 동일
BASE_EE_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


class InsertEnv(DirectRLEnv):
    cfg: InsertEnvCfg

    # ------------------------------------------------------------
    def __init__(self, cfg: InsertEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joint / body indices
        self._arm_joint_ids = [self._robot.find_joints(f"joint{i}")[0][0] for i in range(1, 7)]
        self._gripper_joint_ids = [self._robot.find_joints(n)[0][0] for n in self.cfg.gripper_joint_names]
        self._all_joint_ids = self._arm_joint_ids + self._gripper_joint_ids

        self._left_finger_id = self._robot.find_bodies(self.cfg.left_finger_body_name)[0][0]
        self._right_finger_id = self._robot.find_bodies(self.cfg.right_finger_body_name)[0][0]

        if self._robot.is_fixed_base:
            self._l_jac_idx = self._left_finger_id - 1
            self._r_jac_idx = self._right_finger_id - 1
        else:
            self._l_jac_idx = self._left_finger_id
            self._r_jac_idx = self._right_finger_id

        # IK
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls",
        )
        self._ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        # base ee quat
        self._base_ee_quat = torch.tensor(
            [list(BASE_EE_QUAT_WXYZ)], device=self.device, dtype=torch.float
        ).repeat(self.num_envs, 1)

        # internal state (per env)
        # ee_xy 는 cell_xy 에 고정 (학습 X) — _ee_target_xy_w buffer 제거
        self._ee_target_yaw = torch.zeros(self.num_envs, device=self.device)
        self._cell_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._cell_yaw = torch.zeros(self.num_envs, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._aligned_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # handoff dataset 로드
        self._load_handoff_dataset()

        self.reward_log: dict[str, float] = {}

    def _load_handoff_dataset(self):
        path = self.cfg.handoff_dataset_path
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Insert handoff dataset 없음: {path}\n"
                f"먼저 collect_insert_handoff.py 로 수집하세요."
            )
        data = np.load(path)
        self._dataset = {
            "joint_pos":     torch.from_numpy(data["joint_pos"]).to(self.device),
            "box_pos_env":   torch.from_numpy(data["box_pos_env"]).to(self.device),
            "box_quat":      torch.from_numpy(data["box_quat"]).to(self.device),
            "cell_xy":       torch.from_numpy(data["cell_xy"]).to(self.device),
            "cell_yaw":      torch.from_numpy(data["cell_yaw"]).to(self.device),
            "ee_target_yaw": torch.from_numpy(data["ee_target_yaw"]).to(self.device),
        }
        self._dataset_size = self._dataset["joint_pos"].shape[0]
        print(f"[InsertEnv] handoff dataset loaded: {self._dataset_size} samples from {path}")

    # ------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.box)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        actions = actions.clamp(-1.0, 1.0)
        self._actions[:] = actions

        # yaw 만 학습: 비누적, 매 step 현재 ee yaw 기준 재계산
        cur_ee_yaw = self._extract_ee_yaw()
        delta_yaw = actions[:, 0] * self.cfg.action_scale_yaw
        self._ee_target_yaw = (cur_ee_yaw + delta_yaw).clamp(
            self.cfg.ee_yaw_min, self.cfg.ee_yaw_max
        )

    def _apply_action(self) -> None:
        env_origin_z = self.scene.env_origins[:, 2]
        target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        # ee_xy 는 cell_xy 에 고정 (학습 X)
        target_pos_w[:, :2] = self._cell_xy + self.scene.env_origins[:, :2]
        target_pos_w[:, 2] = self.cfg.ee_fixed_z + env_origin_z

        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        yaw_quat = quat_from_angle_axis(self._ee_target_yaw, z_axis)
        target_quat_w = quat_mul(yaw_quat, self._base_ee_quat)

        ee_pos_w = self._grip_center_pos()
        ee_quat_w = self._robot.data.body_quat_w[:, self._left_finger_id]
        cur_arm_q = self._robot.data.joint_pos[:, self._arm_joint_ids]

        J = self._robot.root_physx_view.get_jacobians()
        j_l = J[:, self._l_jac_idx, :, :][:, :, self._arm_joint_ids]
        j_r = J[:, self._r_jac_idx, :, :][:, :, self._arm_joint_ids]
        jac = 0.5 * (j_l + j_r)

        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        tgt_pos_b, tgt_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w
        )
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        self._ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))
        arm_target = self._ik.compute(ee_pos_b, ee_quat_b, jac, cur_arm_q)

        # gripper close 유지 (박스 잡고 있어야)
        cmd = self.cfg.gripper_close_cmd
        ratio = self.cfg.gripper_tip_ratio
        # gripper_names 순서: ["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"]
        gripper_target = torch.tensor(
            [[cmd, cmd * ratio, cmd, cmd * ratio]],
            device=self.device,
        ).expand(self.num_envs, -1)

        full_target = torch.cat([arm_target, gripper_target], dim=-1)
        self._robot.set_joint_position_target(full_target, joint_ids=self._all_joint_ids)

    # ------------------------------------------------------------
    def _grip_center_pos(self) -> torch.Tensor:
        l = self._robot.data.body_pos_w[:, self._left_finger_id]
        r = self._robot.data.body_pos_w[:, self._right_finger_id]
        return 0.5 * (l + r)

    def _grip_center_vel(self) -> torch.Tensor:
        l = self._robot.data.body_lin_vel_w[:, self._left_finger_id]
        r = self._robot.data.body_lin_vel_w[:, self._right_finger_id]
        return 0.5 * (l + r)

    def _is_grasping(self) -> torch.Tensor:
        """finger center ↔ box 거리 < threshold AND box 안 떨어짐."""
        ee_pos = self._grip_center_pos()
        box_pos = self._object.data.root_pos_w
        dist = torch.norm(ee_pos - box_pos, dim=-1)
        env_origin_z = self.scene.env_origins[:, 2]
        box_z_env = self._object.data.root_pos_w[:, 2] - env_origin_z
        not_dropped = box_z_env > self.cfg.box_drop_z_threshold
        close = dist < self.cfg.grasping_dist_threshold
        return close & not_dropped

    def _extract_ee_yaw(self) -> torch.Tensor:
        """현재 ee 의 z 축 yaw 추출 (base_ee_quat=(0,1,0,0) 적용된 상태에서도 표준 공식 작동)."""
        q = self._robot.data.body_quat_w[:, self._left_finger_id]  # (N, 4) wxyz
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _ee_ang_vel_z(self) -> torch.Tensor:
        """양 finger 평균 angular velocity z (yaw rate)."""
        ang_l = self._robot.data.body_ang_vel_w[:, self._left_finger_id, 2]
        ang_r = self._robot.data.body_ang_vel_w[:, self._right_finger_id, 2]
        return 0.5 * (ang_l + ang_r)

    # ------------------------------------------------------------
    def _get_observations(self) -> dict:
        cur_ee_yaw = self._extract_ee_yaw()
        yaw_err = wrap_to_pi(self._cell_yaw - cur_ee_yaw)
        is_grasping = self._is_grasping().float()
        yaw_vel = self._ee_ang_vel_z()

        obs = torch.stack([yaw_err, yaw_vel, is_grasping], dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        cur_ee_yaw = self._extract_ee_yaw()
        yaw_err = wrap_to_pi(self._cell_yaw - cur_ee_yaw)
        is_grasping = self._is_grasping()

        r_yaw = torch.exp(-self.cfg.reward_yaw_align_gain * (yaw_err ** 2))
        yaw_aligned = yaw_err.abs() < self.cfg.align_yaw_threshold
        aligned = yaw_aligned & is_grasping
        r_success = aligned.float() * self.cfg.reward_success_bonus

        total = r_yaw + r_success

        self.reward_log = {
            "r_yaw": float(r_yaw.mean().item()),
            "r_success": float(r_success.mean().item()),
            "yaw_aligned_rate": float(yaw_aligned.float().mean().item()),
            "aligned_rate": float(aligned.float().mean().item()),
            "is_grasping_rate": float(is_grasping.float().mean().item()),
            "yaw_err_abs_mean": float(yaw_err.abs().mean().item()),
        }
        return total

    # ------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cur_ee_yaw = self._extract_ee_yaw()
        yaw_err = wrap_to_pi(self._cell_yaw - cur_ee_yaw)
        is_grasping = self._is_grasping()

        aligned = (yaw_err.abs() < self.cfg.align_yaw_threshold) & is_grasping
        self._aligned_count = torch.where(
            aligned, self._aligned_count + 1, torch.zeros_like(self._aligned_count)
        )
        success = self._aligned_count >= self.cfg.success_hold_steps

        # fail: 박스 떨어뜨림만 (ee_xy 는 cell_xy 고정이라 too_far 불필요)
        env_origin_z = self.scene.env_origins[:, 2]
        box_z_env = self._object.data.root_pos_w[:, 2] - env_origin_z
        dropped = box_z_env < self.cfg.box_drop_z_threshold

        terminated = success | dropped
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # ---- handoff dataset 에서 random sample ----
        idx = torch.randint(0, self._dataset_size, (n,), device=self.device)
        joint_pos_d = self._dataset["joint_pos"][idx]              # (n, 10)
        box_pos_env_d = self._dataset["box_pos_env"][idx]          # (n, 3)
        box_quat_d = self._dataset["box_quat"][idx]                # (n, 4)
        cell_xy_d = self._dataset["cell_xy"][idx]                  # (n, 2)
        cell_yaw_d = self._dataset["cell_yaw"][idx, 0]             # (n,)
        ee_target_yaw_d = self._dataset["ee_target_yaw"][idx, 0]   # (n,)

        # ---- robot joint pose 적용 (handoff joint_pos 그대로 — handoff dataset 자체에
        # ee xy noise 포함되어 있어 추가 noise 불필요) ----
        joint_pos = self._robot.data.default_joint_pos[env_ids_t].clone()
        # joint_pos_d 가 num_joints 와 같다고 가정 (10 = arm6 + gripper4)
        # 직접 indexing
        for i, jid in enumerate(self._arm_joint_ids):
            joint_pos[:, jid] = joint_pos_d[:, i]
        for i, jid in enumerate(self._gripper_joint_ids):
            joint_pos[:, jid] = joint_pos_d[:, 6 + i]
        joint_vel = torch.zeros_like(joint_pos)

        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)

        # ---- 박스 자세 적용 (env-rel → world) ----
        # v10: reset 시 box z 에 +5mm offset (ground penetration 회피, finger 안 박스 위치 미세 조정)
        box_pos_env_d_lifted = box_pos_env_d.clone()
        box_pos_env_d_lifted[:, 2] += 0.01
        box_pos_w = box_pos_env_d_lifted + self.scene.env_origins[env_ids_t]
        self._object.write_root_pose_to_sim(
            torch.cat([box_pos_w, box_quat_d], dim=-1),
            env_ids=env_ids_t,
        )
        self._object.write_root_velocity_to_sim(
            torch.zeros((n, 6), device=self.device), env_ids=env_ids_t,
        )

        # ---- 내부 상태 적용 ----
        # _ee_target_yaw 는 첫 _pre_physics_step 호출에서 actual ee yaw 기준으로 재계산됨
        self._cell_xy[env_ids_t] = cell_xy_d
        self._cell_yaw[env_ids_t] = cell_yaw_d
        self._ee_target_yaw[env_ids_t] = ee_target_yaw_d.clamp(
            self.cfg.ee_yaw_min, self.cfg.ee_yaw_max
        )

        self._aligned_count[env_ids_t] = 0

        self._ik.reset(env_ids_t)
