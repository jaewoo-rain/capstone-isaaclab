"""motion1 — Insert RL v2 Env (SAC + HER, coded reset).

박스 잡힌 채 셀 위에서 xy/yaw 미세 정렬.

Obs (flat 13):
  core (5): is_grasping, ee_vel_x, ee_vel_y, yaw_vel, slot_yaw_err
  achieved_goal (4): ee_x_env, ee_y_env, cos(ee_yaw), sin(ee_yaw)
  desired_goal (4): cell_x, cell_y, cos(cell_yaw), sin(cell_yaw)

Action (3): Δx, Δy, Δyaw (cartesian, relative, IK 적용)

Reset: handoff dataset 사용 X. 코드로 직접:
  1. cell xy/yaw random
  2. ee 시작점 = cell + random offset (3~5cm)
  3. IK 로 robot joint 계산 (settle steps)
  4. box 를 grip center 에 매핑 (yaw = cell_yaw)
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import gymnasium as gym
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

from .insert_env_v2_cfg import InsertEnvV2Cfg


# 베이스 EE quat (수직 아래 + finger Y) — motion1 chain / insert v1 과 동일
BASE_EE_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


class InsertEnvV2(DirectRLEnv):
    cfg: InsertEnvV2Cfg

    # obs 구성 (flat → GoalEnvVecWrapper 가 split)
    OBS_CORE_DIM = 5
    GOAL_DIM = 4
    OBS_TOTAL_DIM = OBS_CORE_DIM + GOAL_DIM * 2  # 13

    # ------------------------------------------------------------
    def __init__(self, cfg: InsertEnvV2Cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # gym spaces (flat obs — wrapper 가 dict 로 split)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_TOTAL_DIM,), dtype=np.float32,
        )

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

        # base ee quat (수직 아래)
        self._base_ee_quat = torch.tensor(
            [list(BASE_EE_QUAT_WXYZ)], device=self.device, dtype=torch.float
        ).repeat(self.num_envs, 1)

        # 내부 state
        self._ee_target_xy_w = torch.zeros(self.num_envs, 2, device=self.device)
        self._ee_target_yaw = torch.zeros(self.num_envs, device=self.device)
        self._cell_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._cell_yaw = torch.zeros(self.num_envs, device=self.device)
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._aligned_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # default robot home joint pos (reset 시 초기 자세)
        self._home_joint_pos = self._build_home_joint_pos()

        self.reward_log: dict[str, float] = {}

    # ------------------------------------------------------------
    def _build_home_joint_pos(self) -> torch.Tensor:
        """transport_end 자세 (cell 위 + cell_yaw rotation 적용 가능한 시작점) 의 joint pos.

        motion1 chain 의 home/transport 자세 참고.
        """
        # OMY chain runner 의 HOME_JOINT_POS — 단 transport 자세 (팔 펴짐) 가 아니라
        # 일반 home. reset 시 IK 로 cell 위로 이동시킬 거라 일단 home 으로 시작.
        home = {
            "joint1": 0.0, "joint2": -1.55, "joint3": 2.66,
            "joint4": -1.1, "joint5": 1.6, "joint6": 0.0,
            "rh_r1_joint": self.cfg.gripper_close_cmd,
            "rh_r2": self.cfg.gripper_close_cmd * self.cfg.gripper_tip_ratio,
            "rh_l1": self.cfg.gripper_close_cmd,
            "rh_l2": self.cfg.gripper_close_cmd * self.cfg.gripper_tip_ratio,
        }
        q = torch.zeros(self._robot.num_joints, device=self.device)
        for name, val in home.items():
            jid = self._robot.find_joints(name)[0][0]
            q[jid] = val
        return q

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

        # gripper close 유지
        cmd = self.cfg.gripper_close_cmd
        ratio = self.cfg.gripper_tip_ratio
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
        ee_pos = self._grip_center_pos()
        box_pos = self._object.data.root_pos_w
        dist = torch.norm(ee_pos - box_pos, dim=-1)
        env_origin_z = self.scene.env_origins[:, 2]
        box_z_env = self._object.data.root_pos_w[:, 2] - env_origin_z
        not_dropped = box_z_env > self.cfg.box_drop_z_threshold
        close = dist < self.cfg.grasping_dist_threshold
        return close & not_dropped

    def _extract_ee_yaw(self) -> torch.Tensor:
        q = self._robot.data.body_quat_w[:, self._left_finger_id]
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _ee_ang_vel_z(self) -> torch.Tensor:
        ang_l = self._robot.data.body_ang_vel_w[:, self._left_finger_id, 2]
        ang_r = self._robot.data.body_ang_vel_w[:, self._right_finger_id, 2]
        return 0.5 * (ang_l + ang_r)

    # ------------------------------------------------------------
    def _get_observations(self) -> dict:
        ee_pos_w = self._grip_center_pos()
        ee_xy_env = ee_pos_w[:, :2] - self.scene.env_origins[:, :2]

        cur_ee_yaw = self._extract_ee_yaw()
        slot_yaw_err = wrap_to_pi(self._cell_yaw - cur_ee_yaw)

        is_grasping = self._is_grasping().float()
        ee_vel = self._grip_center_vel()
        yaw_vel = self._ee_ang_vel_z()

        # core (5)
        core = torch.stack(
            [is_grasping, ee_vel[:, 0], ee_vel[:, 1], yaw_vel, slot_yaw_err],
            dim=-1,
        )

        # achieved_goal (4): ee_xy + cos/sin(ee_yaw)
        achieved = torch.stack(
            [ee_xy_env[:, 0], ee_xy_env[:, 1],
             torch.cos(cur_ee_yaw), torch.sin(cur_ee_yaw)],
            dim=-1,
        )

        # desired_goal (4): cell_xy + cos/sin(cell_yaw)
        desired = torch.stack(
            [self._cell_xy[:, 0], self._cell_xy[:, 1],
             torch.cos(self._cell_yaw), torch.sin(self._cell_yaw)],
            dim=-1,
        )

        obs = torch.cat([core, achieved, desired], dim=-1)
        return {"policy": obs}

    # ------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        """SAC live reward — 최소 3 terms."""
        ee_pos_w = self._grip_center_pos()
        ee_xy_env = ee_pos_w[:, :2] - self.scene.env_origins[:, :2]
        xy_dist2 = ((self._cell_xy - ee_xy_env) ** 2).sum(dim=-1)

        cur_ee_yaw = self._extract_ee_yaw()
        yaw_err = wrap_to_pi(self._cell_yaw - cur_ee_yaw)
        is_grasping = self._is_grasping()

        r_xy = torch.exp(-self.cfg.reward_xy_align_gain * xy_dist2)
        r_yaw = torch.exp(-self.cfg.reward_yaw_align_gain * (yaw_err ** 2))

        xy_aligned = torch.sqrt(xy_dist2 + 1e-9) < self.cfg.align_xy_threshold
        yaw_aligned = yaw_err.abs() < self.cfg.align_yaw_threshold
        aligned = xy_aligned & yaw_aligned & is_grasping
        r_success = aligned.float() * self.cfg.reward_success_bonus

        total = r_xy + r_yaw + r_success

        # 진단 metric
        xy_dist = torch.sqrt(xy_dist2 + 1e-9)
        self.reward_log = {
            "rew_xy": float(r_xy.mean().item()),
            "rew_yaw": float(r_yaw.mean().item()),
            "rew_success": float(r_success.mean().item()),
            "rate_xy_aligned": float(xy_aligned.float().mean().item()),
            "rate_yaw_aligned": float(yaw_aligned.float().mean().item()),
            "rate_aligned": float(aligned.float().mean().item()),
            "rate_grasping": float(is_grasping.float().mean().item()),
            "dist_xy_mean": float(xy_dist.mean().item()),
            "dist_yaw_abs_mean": float(yaw_err.abs().mean().item()),
        }
        return total

    # ------------------------------------------------------------
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict | list | None = None,
        **kwargs,
    ) -> np.ndarray:
        """HER relabeling 용 sparse reward.

        achieved/desired = (x, y, cos_yaw, sin_yaw)
        - xy_dist < align_xy_threshold AND yaw_cos_diff > cos(align_yaw_threshold) → 0
        - else → -1
        """
        xy_dist = np.linalg.norm(achieved_goal[..., :2] - desired_goal[..., :2], axis=-1)
        # cos(diff) = cos_a * cos_d + sin_a * sin_d
        cos_diff = (achieved_goal[..., 2] * desired_goal[..., 2]
                    + achieved_goal[..., 3] * desired_goal[..., 3])
        yaw_cos_thresh = float(np.cos(self.cfg.align_yaw_threshold))
        aligned = (xy_dist < self.cfg.align_xy_threshold) & (cos_diff > yaw_cos_thresh)
        return np.where(aligned, 0.0, -1.0).astype(np.float32)

    # ------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._grip_center_pos()
        ee_xy_env = ee_pos_w[:, :2] - self.scene.env_origins[:, :2]
        slot_rel = self._cell_xy - ee_xy_env
        cur_ee_yaw = self._extract_ee_yaw()
        yaw_err = wrap_to_pi(self._cell_yaw - cur_ee_yaw)
        is_grasping = self._is_grasping()

        aligned = (
            (slot_rel[:, 0].abs() < self.cfg.align_xy_threshold)
            & (slot_rel[:, 1].abs() < self.cfg.align_xy_threshold)
            & (yaw_err.abs() < self.cfg.align_yaw_threshold)
            & is_grasping
        )
        self._aligned_count = torch.where(
            aligned, self._aligned_count + 1, torch.zeros_like(self._aligned_count)
        )
        success = self._aligned_count >= self.cfg.success_hold_steps

        env_origin_z = self.scene.env_origins[:, 2]
        box_z_env = self._object.data.root_pos_w[:, 2] - env_origin_z
        dropped = box_z_env < self.cfg.box_drop_z_threshold
        too_far = (
            (slot_rel[:, 0].abs() > self.cfg.fail_xy_threshold)
            | (slot_rel[:, 1].abs() > self.cfg.fail_xy_threshold)
        )
        failed = dropped | too_far

        terminated = success | failed
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """코드 reset (handoff dataset 사용 X).

        1. cell xy + yaw random
        2. ee 시작점 = cell + random offset
        3. robot 자세 = home 으로 초기화 → IK 로 ee 시작점까지 settle
        4. box 를 grip center 에 매핑 (yaw = cell_yaw)
        """
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(env_ids_t)

        # ---- 1. cell random ----
        cell_xy_noise = (torch.rand((n, 2), device=self.device) - 0.5) * 2.0 * self.cfg.cell_xy_noise
        cell_xy = torch.stack(
            [torch.full((n,), self.cfg.cell_center_x, device=self.device),
             torch.full((n,), self.cfg.cell_center_y, device=self.device)],
            dim=-1,
        ) + cell_xy_noise
        cell_yaw = (torch.rand((n,), device=self.device) - 0.5) * 2.0 * self.cfg.cell_yaw_max

        self._cell_xy[env_ids_t] = cell_xy
        self._cell_yaw[env_ids_t] = cell_yaw

        # ---- 2. ee 시작점 = cell + offset ----
        offset_angle = torch.rand((n,), device=self.device) * 2.0 * math.pi
        offset_dist = (torch.rand((n,), device=self.device)
                       * (self.cfg.ee_offset_max - self.cfg.ee_offset_min)
                       + self.cfg.ee_offset_min)
        ee_start_xy = cell_xy.clone()
        ee_start_xy[:, 0] += offset_dist * torch.cos(offset_angle)
        ee_start_xy[:, 1] += offset_dist * torch.sin(offset_angle)

        # ee target yaw (학습 시작 시 cell_yaw 와 일치, 약간 noise)
        yaw_noise = (torch.rand((n,), device=self.device) - 0.5) * 2.0 * 0.3  # ±0.3 rad ≈ ±17°
        ee_start_yaw = (cell_yaw + yaw_noise).clamp(self.cfg.ee_yaw_min, self.cfg.ee_yaw_max)

        # ---- 3. robot 자세: home 으로 set 후 IK settle ----
        home_q = self._home_joint_pos.unsqueeze(0).repeat(n, 1)
        joint_vel = torch.zeros_like(home_q)
        self._robot.write_joint_state_to_sim(home_q, joint_vel, env_ids=env_ids_t)
        self._robot.set_joint_position_target(home_q, env_ids=env_ids_t)

        # _ee_target 초기화 (IK settle 용)
        self._ee_target_xy_w[env_ids_t] = ee_start_xy + self.scene.env_origins[env_ids_t, :2]
        self._ee_target_yaw[env_ids_t] = ee_start_yaw

        # IK settle — _apply_action 호출하기 위해 self._actions 도 0 으로
        self._actions[env_ids_t] = 0.0
        for _ in range(self.cfg.reset_ik_settle_steps):
            self._apply_action_settle(env_ids_t, ee_start_xy, ee_start_yaw)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.cfg.sim.dt)

        # ---- 4. box 를 grip center 에 매핑 ----
        ee_pos_w_after = self._grip_center_pos()[env_ids_t]
        box_pos_w = ee_pos_w_after.clone()
        box_pos_w[:, 2] -= self.cfg.box_z_offset

        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device).expand(n, 3)
        box_quat = quat_from_angle_axis(cell_yaw, z_axis)

        self._object.write_root_pose_to_sim(
            torch.cat([box_pos_w, box_quat], dim=-1),
            env_ids=env_ids_t,
        )
        self._object.write_root_velocity_to_sim(
            torch.zeros((n, 6), device=self.device), env_ids=env_ids_t,
        )

        self._aligned_count[env_ids_t] = 0
        self._ik.reset(env_ids_t)

    def _apply_action_settle(
        self,
        env_ids_t: torch.Tensor,
        ee_start_xy: torch.Tensor,
        ee_start_yaw: torch.Tensor,
    ):
        """Reset settle 용 IK 호출 (모든 env 에 _apply_action 비슷하게 명령)."""
        n = len(env_ids_t)
        target_pos_w_local = torch.zeros((n, 3), device=self.device)
        target_pos_w_local[:, :2] = ee_start_xy + self.scene.env_origins[env_ids_t, :2]
        target_pos_w_local[:, 2] = self.cfg.ee_fixed_z + self.scene.env_origins[env_ids_t, 2]

        # 전체 env 의 target 도 동기화 (다른 env 들 정상 동작)
        target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        target_pos_w[:, :2] = self._ee_target_xy_w
        target_pos_w[:, 2] = self.cfg.ee_fixed_z + self.scene.env_origins[:, 2]
        target_pos_w[env_ids_t] = target_pos_w_local

        target_yaw = self._ee_target_yaw.clone()
        target_yaw[env_ids_t] = ee_start_yaw

        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        yaw_quat = quat_from_angle_axis(target_yaw, z_axis)
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

        cmd = self.cfg.gripper_close_cmd
        ratio = self.cfg.gripper_tip_ratio
        gripper_target = torch.tensor(
            [[cmd, cmd * ratio, cmd, cmd * ratio]],
            device=self.device,
        ).expand(self.num_envs, -1)

        full_target = torch.cat([arm_target, gripper_target], dim=-1)
        self._robot.set_joint_position_target(full_target, joint_ids=self._all_joint_ids)
