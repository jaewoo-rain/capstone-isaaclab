"""motion1 — Insert 미세 정렬 RL Env.

박스 잡힌 채 셀 위에서 xy/yaw 정렬. handoff dataset 에서 시작 상태 random sample.

State (7):  slot_rel_x, slot_rel_y, slot_yaw_err, is_grasping, ee_vel_x, ee_vel_y, yaw_vel
Action (3): Δx, Δy, Δyaw  (relative)
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
        self._ee_target_xy_w = torch.zeros(self.num_envs, 2, device=self.device)
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

        ee_pos_w = self._grip_center_pos()
        delta_xy = actions[:, :2] * self.cfg.action_scale_xy
        self._ee_target_xy_w = ee_pos_w[:, :2] + delta_xy

        # yaw 비누적: 매 step 현재 ee yaw 기준 재계산 (xy 와 동일 패턴)
        cur_ee_yaw = self._extract_ee_yaw()
        delta_yaw = actions[:, 2] * self.cfg.action_scale_yaw
        self._ee_target_yaw = (cur_ee_yaw + delta_yaw).clamp(
            self.cfg.ee_yaw_min, self.cfg.ee_yaw_max
        )

    def _apply_action(self) -> None:
        env_origin_z = self.scene.env_origins[:, 2]
        target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        target_pos_w[:, :2] = self._ee_target_xy_w
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
        ee_pos_w = self._grip_center_pos()
        ee_xy_env = ee_pos_w[:, :2] - self.scene.env_origins[:, :2]
        slot_rel_x = self._cell_xy[:, 0] - ee_xy_env[:, 0]
        slot_rel_y = self._cell_xy[:, 1] - ee_xy_env[:, 1]

        # actual ee yaw 기반 (xy 와 동일 패턴)
        cur_ee_yaw = self._extract_ee_yaw()
        slot_yaw_err = wrap_to_pi(self._cell_yaw - cur_ee_yaw)

        is_grasping = self._is_grasping().float()

        ee_vel = self._grip_center_vel()
        yaw_vel = self._ee_ang_vel_z()

        obs = torch.stack(
            [slot_rel_x, slot_rel_y, slot_yaw_err, is_grasping,
             ee_vel[:, 0], ee_vel[:, 1], yaw_vel],
            dim=-1,
        )
        return {"policy": obs}

    # ------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        ee_pos_w = self._grip_center_pos()
        ee_xy_env = ee_pos_w[:, :2] - self.scene.env_origins[:, :2]
        slot_rel = self._cell_xy - ee_xy_env  # (N, 2)
        xy_dist2 = (slot_rel ** 2).sum(dim=-1)

        cur_ee_yaw = self._extract_ee_yaw()
        yaw_err = wrap_to_pi(self._cell_yaw - cur_ee_yaw)

        ee_vel = self._grip_center_vel()
        yaw_vel = self._ee_ang_vel_z()

        is_grasping = self._is_grasping()

        # rewards (grasp 와 동일 — mask 없음. drop 은 termination 으로 자연 페널티)
        # 두 개 exp 합쳐서 — 멀리 가도 작은 신호 (exploration) + 가까이 sharp (정밀 정렬)
        r_xy_align = torch.exp(-self.cfg.reward_xy_align_gain * xy_dist2)
        r_xy_align_close = torch.exp(-self.cfg.reward_xy_align_gain_close * xy_dist2)
        r_yaw_align = torch.exp(-self.cfg.reward_yaw_align_gain * (yaw_err ** 2))
        r_smooth = -self.cfg.reward_smooth_w * (ee_vel[:, 0] ** 2 + ee_vel[:, 1] ** 2 + yaw_vel ** 2)

        # v19: 거리 페널티 — 1cm 초과 거리 비례 페널티 (1cm 안 유도)
        xy_dist = torch.sqrt(xy_dist2 + 1e-9)
        r_far_penalty = -self.cfg.reward_far_penalty_w * torch.clamp(
            xy_dist - self.cfg.reward_far_threshold, min=0.0
        )

        # v20: action L2 penalty — 정책이 작은 action 출력 학습 (fine motor)
        r_action_penalty = -self.cfg.reward_action_penalty_w * (self._actions ** 2).sum(dim=-1)

        # alignment 분리 metric — 학습 진단용 (xy / yaw 어느 쪽 부족인지)
        xy_aligned = (
            (slot_rel[:, 0].abs() < self.cfg.align_xy_threshold) &
            (slot_rel[:, 1].abs() < self.cfg.align_xy_threshold)
        )
        yaw_aligned = yaw_err.abs() < self.cfg.align_yaw_threshold
        aligned = xy_aligned & yaw_aligned & is_grasping
        r_success = aligned.float() * self.cfg.reward_success_bonus

        will_succeed = aligned & ((self._aligned_count + 1) >= self.cfg.success_hold_steps)
        r_success_lump = will_succeed.float() * self.cfg.reward_success_lump

        total = r_xy_align + r_xy_align_close + r_yaw_align + r_smooth + r_success + r_success_lump + r_action_penalty

        # v19+ 진단: 발산/안정성 metric
        ee_speed = torch.sqrt(ee_vel[:, 0] ** 2 + ee_vel[:, 1] ** 2 + 1e-9)
        action_norm = torch.norm(self._actions, dim=-1)
        far_rate = (xy_dist > 0.05).float().mean()        # 5cm 이상 비율
        very_far_rate = (xy_dist > 0.08).float().mean()   # 8cm (발산 직전) 비율
        close_rate = (xy_dist < 0.02).float().mean()      # 2cm 이내 비율
        # box vs ee yaw mismatch (slip 정도)
        box_yaw = self._extract_box_yaw_if_available()
        if box_yaw is not None:
            box_ee_yaw_diff = wrap_to_pi(box_yaw - cur_ee_yaw).abs().mean()
        else:
            box_ee_yaw_diff = torch.tensor(0.0, device=self.device)

        self.reward_log = {
            "r_xy_align": float(r_xy_align.mean().item()),
            "r_xy_align_close": float(r_xy_align_close.mean().item()),
            "r_yaw_align": float(r_yaw_align.mean().item()),
            "r_smooth": float(r_smooth.mean().item()),
            "r_success": float(r_success.mean().item()),
            "r_success_lump": float(r_success_lump.mean().item()),
            "r_far_penalty": float(r_far_penalty.mean().item()),
            "r_action_penalty": float(r_action_penalty.mean().item()),
            # 진단 metric — 어느 정렬 조건 부족인지
            "xy_aligned_rate": float(xy_aligned.float().mean().item()),
            "yaw_aligned_rate": float(yaw_aligned.float().mean().item()),
            "aligned_rate": float(aligned.float().mean().item()),
            "is_grasping_rate": float(is_grasping.float().mean().item()),
            # 평균 거리 (학습 중 줄어드는지)
            "xy_dist_mean": float(torch.sqrt(xy_dist2 + 1e-9).mean().item()),
            "yaw_err_abs_mean": float(yaw_err.abs().mean().item()),
            # v19+ 발산/안정성 진단
            "far_rate_5cm": float(far_rate.item()),
            "very_far_rate_8cm": float(very_far_rate.item()),
            "close_rate_2cm": float(close_rate.item()),
            "ee_speed_mean": float(ee_speed.mean().item()),
            "ee_speed_max": float(ee_speed.max().item()),
            "action_norm_mean": float(action_norm.mean().item()),
            "action_norm_max": float(action_norm.max().item()),
            "box_ee_yaw_diff_rad": float(box_ee_yaw_diff.item()),
        }
        return total

    def _extract_box_yaw_if_available(self) -> torch.Tensor | None:
        """box quat → world Z yaw (있으면)."""
        try:
            q = self._object.data.root_quat_w  # (N, 4) wxyz
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        except Exception:
            return None

    # ------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._grip_center_pos()
        ee_xy_env = ee_pos_w[:, :2] - self.scene.env_origins[:, :2]
        slot_rel = self._cell_xy - ee_xy_env

        cur_ee_yaw = self._extract_ee_yaw()
        yaw_err = wrap_to_pi(self._cell_yaw - cur_ee_yaw)

        is_grasping = self._is_grasping()

        aligned = (
            (slot_rel[:, 0].abs() < self.cfg.align_xy_threshold) &
            (slot_rel[:, 1].abs() < self.cfg.align_xy_threshold) &
            (yaw_err.abs() < self.cfg.align_yaw_threshold) &
            is_grasping
        )
        self._aligned_count = torch.where(
            aligned, self._aligned_count + 1, torch.zeros_like(self._aligned_count)
        )
        success = self._aligned_count >= self.cfg.success_hold_steps

        # fail: 박스 떨어뜨림 또는 ee 가 셀에서 너무 멀어짐
        env_origin_z = self.scene.env_origins[:, 2]
        box_z_env = self._object.data.root_pos_w[:, 2] - env_origin_z
        dropped = box_z_env < self.cfg.box_drop_z_threshold
        too_far = (
            (slot_rel[:, 0].abs() > self.cfg.fail_xy_threshold) |
            (slot_rel[:, 1].abs() > self.cfg.fail_xy_threshold)
        )
        failed = dropped | too_far

        terminated = success | failed
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
        # _ee_target_xy_w / _ee_target_yaw 는 첫 _pre_physics_step 호출에서 actual ee 기준으로
        # 재계산되므로 placeholder 만 두면 됨 (handoff 자세 그대로)
        self._cell_xy[env_ids_t] = cell_xy_d
        self._cell_yaw[env_ids_t] = cell_yaw_d
        self._ee_target_yaw[env_ids_t] = ee_target_yaw_d.clamp(
            self.cfg.ee_yaw_min, self.cfg.ee_yaw_max
        )
        self._ee_target_xy_w[env_ids_t] = cell_xy_d + self.scene.env_origins[env_ids_t, :2]

        self._aligned_count[env_ids_t] = 0

        self._ik.reset(env_ids_t)
