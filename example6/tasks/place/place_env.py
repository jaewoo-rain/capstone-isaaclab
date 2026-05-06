"""OMY Place Env 구현

시작 상태: 물체가 그리퍼에 잡혀 20cm 높이에 위치
목표: 타겟 셀에 수직으로 내려놓고 gripper 열어서 안정적으로 안착
Observation: 31차원 flat tensor (25 obs + 3 achieved_goal + 3 desired_goal)
"""
from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply

from .place_env_cfg import PlaceEnvCfg


class PlaceEnv(DirectRLEnv):
    cfg: PlaceEnvCfg

    # observation 구성 (31 = 25 + 3 + 3)
    # - arm joint pos (6) + arm joint vel (6) + gripper close state (1)
    # - end-effector pos (3)
    # - object pos rel (3) + object vel (3)
    # - object to target vector (3)
    # → 합 25
    # + achieved_goal (3, obj pos)
    # + desired_goal (3, target cell center)
    OBS_CORE_DIM = 25
    GOAL_DIM = 3

    def __init__(self, cfg: PlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ----- gym spaces -----
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.observation_space,), dtype=np.float32,
        )

        # ----- joint / body 이름 매핑 -----
        self.arm_joint_names = [f"joint{i}" for i in range(1, 7)]
        self.gripper_joint_names = list(self.cfg.gripper_joint_names)

        self.arm_joint_ids: list[int] = []
        for name in self.arm_joint_names:
            found = self._robot.find_joints(name)[0]
            if len(found) == 0:
                raise RuntimeError(f"OMY arm joint 못 찾음: {name}")
            self.arm_joint_ids.append(int(found[0]))

        self.gripper_joint_ids: list[int] = []
        for name in self.gripper_joint_names:
            found = self._robot.find_joints(name)[0]
            if len(found) == 0:
                raise RuntimeError(f"OMY gripper joint 못 찾음: {name}")
            self.gripper_joint_ids.append(int(found[0]))

        self.main_gripper_joint_id = self.gripper_joint_ids[0]

        left_found = self._robot.find_bodies(self.cfg.left_finger_body_name)[0]
        right_found = self._robot.find_bodies(self.cfg.right_finger_body_name)[0]
        if len(left_found) == 0 or len(right_found) == 0:
            raise RuntimeError("finger body 못 찾음")
        self.left_finger_body_id = int(left_found[0])
        self.right_finger_body_id = int(right_found[0])

        # ----- joint limit -----
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # ----- 버퍼 -----
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device,
        )
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # 타겟 셀 관련
        self.target_cell_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.target_cell_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # 성공 유지 카운터 (안정적 안착 확인용)
        self.success_hold_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # _get_rewards에서 계산한 플래그를 _get_dones에서 재사용 (초기값은 False)
        self._last_success_now = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._last_tilted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # ----- Handoff 데이터셋 로드 -----
        # collect_handoff.py로 example5에서 수집한 상태들
        self._load_handoff_dataset()

        # 그리드 셀 중심 좌표 (환경 원점 기준, z=0 바닥)
        # shape: (num_cells, 3)
        self.cell_centers_local = self._build_cell_centers()

        # reward 로그
        self.reward_log = {}

        self._compute_intermediate_values()

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.object)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 그리드 벽 스폰 (수직 4개 + 수평 4개, corner overlap 허용)
        self._spawn_grid_walls()

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_grid_walls(self):
        """3×3 그리드 벽 생성: 수직 4개 + 수평 4개. corner에서 벽이 겹쳐도 무시."""
        cx = self.cfg.grid_center_x
        cy = self.cfg.grid_center_y
        nx = self.cfg.grid_num_x
        ny = self.cfg.grid_num_y
        px = self.cfg.cell_pitch_x
        py = self.cfg.cell_pitch_y
        t = self.cfg.wall_thickness
        h = self.cfg.wall_height

        # 그리드 전체 크기
        total_x = nx * self.cfg.cell_inner_x + (nx + 1) * t
        total_y = ny * self.cfg.cell_inner_y + (ny + 1) * t

        # 수직 벽(y축을 따라 긴 벽, x 방향 분리): nx+1 개
        # 사용자 요청: "수직 4개 + 수평 4개" → 3×3 그리드의 경계+분리 = 4+4
        vert_size = (t, total_y, h)
        for i in range(nx + 1):
            x = cx - total_x / 2 + t / 2 + i * (self.cfg.cell_inner_x + t)
            wall_cfg = sim_utils.CuboidCfg(
                size=vert_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True, disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
            )
            wall_cfg.func(
                f"/World/envs/env_.*/GridWallV_{i}",
                wall_cfg,
                translation=(x, cy, h / 2),
            )

        # 수평 벽(x축을 따라 긴 벽, y 방향 분리): ny+1 개
        horiz_size = (total_x, t, h)
        for j in range(ny + 1):
            y = cy - total_y / 2 + t / 2 + j * (self.cfg.cell_inner_y + t)
            wall_cfg = sim_utils.CuboidCfg(
                size=horiz_size,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True, disable_gravity=True,
                    kinematic_enabled=True,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
            )
            wall_cfg.func(
                f"/World/envs/env_.*/GridWallH_{j}",
                wall_cfg,
                translation=(cx, y, h / 2),
            )

    def _load_handoff_dataset(self):
        """collect_handoff.py로 수집한 handoff 상태 로드.

        없으면 fallback 값으로 단일 상태 세트 구성.
        """
        path = self.cfg.handoff_dataset_path
        if os.path.exists(path):
            data = np.load(path)
            jp = data["joint_pos"].astype(np.float32)             # (N, num_joints)
            opr = data["obj_pos_rel"].astype(np.float32)          # (N, 3)
            oq = data["obj_quat"].astype(np.float32)              # (N, 4)

            if jp.shape[1] != self._robot.num_joints:
                raise RuntimeError(
                    f"handoff joint dim mismatch: {jp.shape[1]} vs {self._robot.num_joints}"
                )

            self.handoff_joint_pos = torch.from_numpy(jp).to(self.device)
            self.handoff_obj_pos_rel = torch.from_numpy(opr).to(self.device)
            self.handoff_obj_quat = torch.from_numpy(oq).to(self.device)
            print(f"✅ handoff 데이터셋 로드: {path} ({jp.shape[0]} samples)")
        else:
            print(f"⚠️ handoff 데이터셋 없음: {path} — fallback 사용")
            jp_single = torch.zeros(self._robot.num_joints, device=self.device)
            for name, val in self.cfg.fallback_holding_joint_pos.items():
                found = self._robot.find_joints(name)[0]
                if len(found) > 0:
                    jp_single[found[0]] = float(val)
            self.handoff_joint_pos = jp_single.unsqueeze(0)        # (1, num_joints)
            self.handoff_obj_pos_rel = torch.tensor(
                [self.cfg.fallback_obj_pos_rel], device=self.device, dtype=torch.float32,
            )
            self.handoff_obj_quat = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]], device=self.device, dtype=torch.float32,
            )

    def _build_cell_centers(self) -> torch.Tensor:
        """모든 셀의 중심 좌표 (환경 원점 기준, z는 셀 바닥 높이)."""
        cx = self.cfg.grid_center_x
        cy = self.cfg.grid_center_y
        nx = self.cfg.grid_num_x
        ny = self.cfg.grid_num_y
        px = self.cfg.cell_pitch_x
        py = self.cfg.cell_pitch_y

        cells = []
        for j in range(ny):
            for i in range(nx):
                x = cx - (nx - 1) * px / 2 + i * px
                y = cy - (ny - 1) * py / 2 + j * py
                # z: 물체가 셀 바닥에 놓였을 때 중심 높이 (물체 높이 / 2)
                z = self.cfg.object.spawn.size[2] / 2
                cells.append([x, y, z])
        return torch.tensor(cells, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = (
            self.robot_dof_targets
            + self.robot_dof_speed_scales * self.dt
            * self.actions_to_dof(self.actions) * self.cfg.action_scale
        )
        self.robot_dof_targets = torch.clamp(
            targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits,
        )

    def actions_to_dof(self, actions: torch.Tensor) -> torch.Tensor:
        dof_delta = torch.zeros(
            (actions.shape[0], self._robot.num_joints), device=self.device,
        )
        for i, joint_id in enumerate(self.arm_joint_ids):
            dof_delta[:, joint_id] = actions[:, i]

        grip_cmd = actions[:, 6] * 3.0
        for gid in self.gripper_joint_ids:
            dof_delta[:, gid] = grip_cmd
        return dof_delta

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # ------------------------------------------------------------------
    # Observation (31 dim flat tensor)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        # arm joints
        arm_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]
        arm_vel = self._robot.data.joint_vel[:, self.arm_joint_ids]

        arm_lower = self.robot_dof_lower_limits[self.arm_joint_ids]
        arm_upper = self.robot_dof_upper_limits[self.arm_joint_ids]
        arm_pos_scaled = 2.0 * (arm_pos - arm_lower) / (arm_upper - arm_lower + 1e-8) - 1.0
        arm_vel_scaled = arm_vel * self.cfg.dof_velocity_scale

        # gripper close state
        grip_pos = self._robot.data.joint_pos[:, self.main_gripper_joint_id].unsqueeze(-1)
        g_lower = self.robot_dof_lower_limits[self.main_gripper_joint_id].view(1, 1)
        g_upper = self.robot_dof_upper_limits[self.main_gripper_joint_id].view(1, 1)
        gripper_close = (grip_pos - g_lower) / (g_upper - g_lower + 1e-8)

        # end-effector (grip center)
        ee_pos_w = self.grip_center_pos
        ee_pos_rel = ee_pos_w - self.scene.env_origins

        # object
        obj_pos_rel = self.obj_pos_rel
        obj_vel = self._object.data.root_lin_vel_w

        # object → target vector (env 기준)
        target_pos_rel = self.target_cell_pos_w - self.scene.env_origins
        obj_to_target = target_pos_rel - obj_pos_rel

        # core obs (25)
        core = torch.cat(
            [
                arm_pos_scaled,       # 6
                arm_vel_scaled,       # 6
                gripper_close,        # 1
                ee_pos_rel,           # 3
                obj_pos_rel,          # 3
                obj_vel,              # 3
                obj_to_target,        # 3
            ],
            dim=-1,
        )

        # achieved_goal / desired_goal (월드 좌표 기준)
        achieved = self._object.data.root_pos_w.clone()
        desired = self.target_cell_pos_w.clone()

        obs = torch.cat([core, achieved, desired], dim=-1)
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # ------------------------------------------------------------------
    # Rewards — 통합 3-phase (이동 → 정렬 → 삽입)
    # phase 전환은 게이팅(gating)으로 자동: 정렬 reward는 near_cell일 때,
    # 하강 reward는 정렬 완료(aligned)일 때만 활성화.
    # 떨어뜨림(grip 이탈/낙하) 자체에는 패널티 없음 — 넘어지면(_get_dones의 tilted) reset.
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        obj_pos_w = self._object.data.root_pos_w
        obj_quat = self._object.data.root_quat_w
        target_pos_w = self.target_cell_pos_w
        env_origins = self.scene.env_origins

        # --- 거리 (중심-중심) ---
        xy_dist = torch.norm(obj_pos_w[:, :2] - target_pos_w[:, :2], dim=-1)
        z_dist = torch.abs(obj_pos_w[:, 2] - target_pos_w[:, 2])
        in_cell_xy = xy_dist < self.cfg.cell_tolerance

        # --- 끝점 매칭 (yaw 정렬 척도) ---
        # 박스 local +x 방향(길이축) 끝점을 카메라 반대편 끝점으로 정의.
        # object_endpoint_sign으로 ±x 끝 선택 가능.
        half_length = float(self.cfg.object.spawn.size[0]) / 2.0
        local_x = torch.zeros((self.num_envs, 3), device=self.device)
        local_x[:, 0] = self.cfg.object_endpoint_sign
        world_x_dir = quat_apply(obj_quat, local_x)
        obj_endpoint_w = obj_pos_w + half_length * world_x_dir

        half_cell_x = self.cfg.cell_inner_x / 2.0
        cell_endpoint_offset = torch.zeros((1, 3), device=self.device)
        cell_endpoint_offset[0, 0] = self.cfg.cell_endpoint_sign * half_cell_x
        cell_endpoint_w = target_pos_w + cell_endpoint_offset

        endpoint_xy_dist = torch.norm(
            obj_endpoint_w[:, :2] - cell_endpoint_w[:, :2], dim=-1
        )

        # --- 그리퍼/속도/자세 ---
        grip_to_obj_dist = torch.norm(self.grip_center_pos - obj_pos_w, dim=-1)
        obj_vel = torch.norm(self._object.data.root_lin_vel_w, dim=-1)
        stable = obj_vel < self.cfg.stable_vel_threshold

        grip_pos = self._robot.data.joint_pos[:, self.main_gripper_joint_id]
        g_lower = self.robot_dof_lower_limits[self.main_gripper_joint_id]
        g_upper = self.robot_dof_upper_limits[self.main_gripper_joint_id]
        gripper_close_state = (grip_pos - g_lower) / (g_upper - g_lower + 1e-8)
        gripper_open = gripper_close_state < self.cfg.gripper_open_threshold

        local_up = torch.zeros((obj_quat.shape[0], 3), device=self.device)
        local_up[:, 2] = 1.0
        world_up_from_obj = quat_apply(obj_quat, local_up)
        upright_score = world_up_from_obj[:, 2]
        tilted = upright_score < self.cfg.tilt_upright_threshold

        # --- Phase 게이트 ---
        near_cell = xy_dist < self.cfg.near_cell_threshold
        centered = xy_dist < self.cfg.cell_tolerance
        endpoint_matched = endpoint_xy_dist < self.cfg.alignment_tolerance
        aligned = centered & endpoint_matched
        descended = z_dist < 0.05
        should_open = aligned & descended
        success = aligned & descended & stable & gripper_open

        # --- Phase 1: 이동 (move) — 항상 활성 ---
        move_wide = torch.exp(-5.0 * xy_dist**2)         # 멀리서도 gradient
        move_close = torch.exp(-30.0 * xy_dist**2)       # 근접 시 큰 보상
        move_linear = -xy_dist                           # constant pull
        held_up = torch.clamp((obj_pos_w[:, 2] - 0.10) * 5.0, 0.0, 1.0)

        # --- Phase 2: 정렬 (align) — near_cell일 때만 활성 ---
        align_close = torch.exp(-500.0 * endpoint_xy_dist**2) * near_cell.float()
        align_linear = -endpoint_xy_dist * near_cell.float()

        # --- Phase 3: 삽입 (insert/descend) — aligned일 때만 활성 ---
        descend_close = torch.exp(-50.0 * z_dist**2) * aligned.float()
        descend_linear = -z_dist * aligned.float()

        # --- 그리퍼 타이밍: 정렬+하강 전엔 닫고 있고, 그 후엔 열기 ---
        keep_closed = gripper_close_state * (~should_open).float()
        release = gripper_open.float() * should_open.float()

        upright_reward = torch.clamp(upright_score, 0.0, 1.0)
        action_penalty = torch.sum(self.actions ** 2, dim=-1)

        # 지속 성공 보상 (안정 안착 학습 압박)
        sustained_bonus = success.float() * torch.clamp(
            self.success_hold_counter.float() / 10.0, 0.0, 5.0
        )

        reward = (
            100.0 * success.float()
            + 30.0 * sustained_bonus
            # 1. 이동
            + 20.0 * move_wide
            + 10.0 * move_close
            + 15.0 * move_linear
            + 3.0 * held_up
            # 2. 정렬
            + 50.0 * align_close
            + 20.0 * align_linear
            # 3. 삽입(하강)
            + 30.0 * descend_close
            + 15.0 * descend_linear
            # 그리퍼 타이밍
            + 5.0 * keep_closed
            + 30.0 * release
            # 자세/액션
            + 2.0 * upright_reward
            - 0.001 * action_penalty
        )

        # 종료 조건용 플래그 저장
        self._last_tilted = tilted
        self._last_success_now = success

        # =====================================================
        # 로그
        # =====================================================
        obj_pos_rel0 = obj_pos_w[0] - env_origins[0]
        grip_pos_rel0 = self.grip_center_pos[0] - env_origins[0]
        target_pos_rel0 = target_pos_w[0] - env_origins[0]
        arm_joint_pos0 = self._robot.data.joint_pos[0, self.arm_joint_ids]

        self.reward_log["env0_obj_x"] = float(obj_pos_rel0[0])
        self.reward_log["env0_obj_y"] = float(obj_pos_rel0[1])
        self.reward_log["env0_obj_z"] = float(obj_pos_rel0[2])
        self.reward_log["env0_grip_x"] = float(grip_pos_rel0[0])
        self.reward_log["env0_grip_y"] = float(grip_pos_rel0[1])
        self.reward_log["env0_grip_z"] = float(grip_pos_rel0[2])
        self.reward_log["env0_tgt_x"] = float(target_pos_rel0[0])
        self.reward_log["env0_tgt_y"] = float(target_pos_rel0[1])
        self.reward_log["env0_tgt_z"] = float(target_pos_rel0[2])
        self.reward_log["env0_upright"] = float(upright_score[0])
        self.reward_log["env0_grip_close"] = float(gripper_close_state[0])
        self.reward_log["env0_j1"] = float(arm_joint_pos0[0])
        self.reward_log["env0_j2"] = float(arm_joint_pos0[1])
        self.reward_log["env0_j3"] = float(arm_joint_pos0[2])
        self.reward_log["env0_j4"] = float(arm_joint_pos0[3])
        self.reward_log["env0_j5"] = float(arm_joint_pos0[4])
        self.reward_log["env0_j6"] = float(arm_joint_pos0[5])

        # 거리 메트릭
        self.reward_log["dist_xy"] = float(xy_dist.mean())
        self.reward_log["dist_z"] = float(z_dist.mean())
        self.reward_log["dist_endpoint_xy"] = float(endpoint_xy_dist.mean())
        self.reward_log["dist_grip_obj"] = float(grip_to_obj_dist.mean())
        self.reward_log["dist_obj_vel"] = float(obj_vel.mean())

        # phase 비율
        self.reward_log["rate_near_cell"] = float(near_cell.float().mean())
        self.reward_log["rate_centered"] = float(centered.float().mean())
        self.reward_log["rate_endpoint_match"] = float(endpoint_matched.float().mean())
        self.reward_log["rate_aligned"] = float(aligned.float().mean())
        self.reward_log["rate_descended"] = float(descended.float().mean())
        self.reward_log["rate_in_cell"] = float(in_cell_xy.float().mean())
        self.reward_log["rate_stable"] = float(stable.float().mean())
        self.reward_log["rate_grip_open"] = float(gripper_open.float().mean())
        self.reward_log["rate_grip_close_avg"] = float(gripper_close_state.mean())
        self.reward_log["rate_upright_avg"] = float(upright_score.mean())
        self.reward_log["rate_tilted"] = float(tilted.float().mean())
        self.reward_log["rate_success_now"] = float(success.float().mean())
        self.reward_log["rate_success_hold_mean"] = float(self.success_hold_counter.float().mean())

        # 보상 항목 평균
        self.reward_log["rew_move_wide"] = float(move_wide.mean())
        self.reward_log["rew_move_close"] = float(move_close.mean())
        self.reward_log["rew_align_close"] = float(align_close.mean())
        self.reward_log["rew_descend_close"] = float(descend_close.mean())
        self.reward_log["rew_keep_closed"] = float(keep_closed.mean())
        self.reward_log["rew_release"] = float(release.mean())
        self.reward_log["rew_total"] = float(reward.mean())

        return reward

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict | list | None = None,
        **kwargs,  # Sb3VecEnvWrapper가 indices 등을 전달 → 무시
    ) -> np.ndarray:
        """HER relabeling용 reward 계산 (sparse, XY 거리 기준).

        SB3 HER이 env_method로 호출하는 시그니처. numpy 입출력.
        """
        xy_dist = np.linalg.norm(achieved_goal[..., :2] - desired_goal[..., :2], axis=-1)
        return np.where(xy_dist < self.cfg.cell_tolerance, 0.0, -1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # _get_rewards에서 저장한 플래그 재사용
        success_now = self._last_success_now
        tilted = self._last_tilted

        # 연속 유지 카운터
        self.success_hold_counter = torch.where(
            success_now,
            self.success_hold_counter + 1,
            torch.zeros_like(self.success_hold_counter),
        )
        success_stable = self.success_hold_counter >= self.cfg.success_hold_steps

        obj_pos_w = self._object.data.root_pos_w
        fallen_below = obj_pos_w[:, 2] < self.cfg.object_fall_z

        # 그리드 경계 밖 이탈 (공중 이동은 허용)
        obj_pos_rel = obj_pos_w - self.scene.env_origins
        total_x = self.cfg.grid_num_x * self.cfg.cell_inner_x + (self.cfg.grid_num_x + 1) * self.cfg.wall_thickness
        total_y = self.cfg.grid_num_y * self.cfg.cell_inner_y + (self.cfg.grid_num_y + 1) * self.cfg.wall_thickness
        margin = 0.6
        out_of_bounds = (
            (obj_pos_rel[:, 0] < self.cfg.grid_center_x - total_x / 2 - margin)
            | (obj_pos_rel[:, 0] > self.cfg.grid_center_x + total_x / 2 + margin)
            | (obj_pos_rel[:, 1] < self.cfg.grid_center_y - total_y / 2 - margin)
            | (obj_pos_rel[:, 1] > self.cfg.grid_center_y + total_y / 2 + margin)
        )

        # 떨어뜨려도 안 넘어졌으면 다시 잡을 수 있게 — abandoned/on_floor 자체로는 reset 안 함
        terminated = (
            success_stable
            | fallen_below
            | out_of_bounds
            | tilted
        )
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset (handoff state + 타겟 셀 설정)
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(env_ids_t)

        self.success_hold_counter[env_ids_t] = 0

        # ----- handoff 상태 랜덤 샘플링 -----
        num_handoff = self.handoff_joint_pos.shape[0]
        sample_idx = torch.randint(0, num_handoff, (n,), device=self.device, dtype=torch.long)

        # ----- 1) 로봇 자세 + 소량 노이즈 -----
        joint_pos = self.handoff_joint_pos[sample_idx].clone()
        joint_noise = (
            (torch.rand_like(joint_pos) - 0.5) * 2.0 * self.cfg.handoff_joint_noise
        )
        # 그리퍼 관절에는 노이즈 주지 않음 (파지 유지)
        for gid in self.gripper_joint_ids:
            joint_noise[:, gid] = 0.0
        joint_pos = joint_pos + joint_noise
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

        joint_vel = torch.zeros_like(joint_pos)

        self.robot_dof_targets[env_ids_t] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)

        # ----- 2) 물체 위치/회전 (handoff 기록 + 소량 노이즈) -----
        obj_state = self._object.data.default_root_state[env_ids_t].clone()
        env_origins = self.scene.env_origins[env_ids_t]

        base_pos_rel = self.handoff_obj_pos_rel[sample_idx]  # (n, 3)
        base_quat = self.handoff_obj_quat[sample_idx]        # (n, 4)

        noise_xy = (
            (torch.rand((n, 2), device=self.device) - 0.5)
            * 2.0 * self.cfg.handoff_obj_pos_noise_xy
        )
        noise_z = (
            (torch.rand((n,), device=self.device) - 0.5)
            * 2.0 * self.cfg.handoff_obj_pos_noise_z
        )

        obj_state[:, 0] = env_origins[:, 0] + base_pos_rel[:, 0] + noise_xy[:, 0]
        obj_state[:, 1] = env_origins[:, 1] + base_pos_rel[:, 1] + noise_xy[:, 1]
        obj_state[:, 2] = env_origins[:, 2] + base_pos_rel[:, 2] + noise_z
        obj_state[:, 3:7] = base_quat
        obj_state[:, 7:] = 0.0
        self._object.write_root_state_to_sim(obj_state, env_ids=env_ids_t)

        # ----- 3) 타겟 셀 결정 -----
        num_cells = self.cfg.grid_num_x * self.cfg.grid_num_y
        if self.cfg.randomize_target_cell:
            new_idx = torch.randint(0, num_cells, (n,), device=self.device, dtype=torch.long)
        else:
            # row-major 고정 순서 (env_id 기반으로 분산)
            new_idx = (env_ids_t % num_cells).to(torch.long)

        self.target_cell_idx[env_ids_t] = new_idx

        # 월드 좌표로 변환 (env_origins + 로컬 셀 중심)
        cell_local = self.cell_centers_local[new_idx]
        self.target_cell_pos_w[env_ids_t] = env_origins + cell_local

        self._compute_intermediate_values(env_ids_t)

    # ------------------------------------------------------------------
    # 중간값
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        l_pos = self._robot.data.body_pos_w[env_ids, self.left_finger_body_id, :]
        r_pos = self._robot.data.body_pos_w[env_ids, self.right_finger_body_id, :]

        if not hasattr(self, "grip_center_pos"):
            self.grip_center_pos = torch.zeros((self.num_envs, 3), device=self.device)
            self.obj_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
            self.obj_pos_rel = torch.zeros((self.num_envs, 3), device=self.device)

        self.grip_center_pos[env_ids] = 0.5 * (l_pos + r_pos)
        self.obj_pos_w[env_ids] = self._object.data.root_pos_w[env_ids]
        self.obj_pos_rel[env_ids] = self.obj_pos_w[env_ids] - self.scene.env_origins[env_ids]
