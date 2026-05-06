"""OMY Place+Insert Env 구현 (example7)

시작 상태: 물체가 그리퍼에 잡혀 20cm 높이에 위치
목표 (3 phase, 단일 reward 함수에 통합):
  1) 적재 위치 위로 들어올림: obj.xy ≈ target.xy AND obj.z ≥ cell_top + obj_h + 5cm
  2) yaw 정렬: obj 중심 + 카메라쪽 끝점 두 점이 셀 중심 + 대응 끝점과 일치
  3) 삽입+release: 셀 안으로 8cm 들어가면 gripper 열어 release (나머지는 중력 낙하)
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

        # v11: 이전 그리퍼 z 위치 (그리퍼 상승 속도 계산용)
        self._prev_grip_z = torch.zeros(self.num_envs, device=self.device)

        # _get_rewards에서 계산한 플래그를 _get_dones에서 재사용 (초기값은 False)
        self._last_success_now = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._last_tilted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._last_severely_tilted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._last_abandoned = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._last_inserted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._last_xy_aligned_loose = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Episode-level 성공률 추적 (롤링 윈도우)
        from collections import deque
        self.episode_success_history = deque(maxlen=300)
        # 누적 성공 카운터 (학습 시작 이후 총 성공 episode 수, 절대 감소 안 함)
        self.cumulative_success_count: int = 0
        self.cumulative_episode_count: int = 0

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
        """모든 셀의 타겟 좌표 (환경 원점 기준).
        example7_2: z = cfg.lift_target_z (0.30, 셀 위) — 박스가 도달해야 할 위치.
        """
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
                # example7_2: z = lift_target_z (셀 위, 박스가 도달할 높이)
                z = self.cfg.lift_target_z
                cells.append([x, y, z])
        return torch.tensor(cells, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # v9: Action harness — 박스 근처에서 그리퍼 강제 full close
        if hasattr(self, "grip_center_pos") and hasattr(self, "_object"):
            obj_pos_w = self._object.data.root_pos_w
            grip_to_obj = torch.norm(self.grip_center_pos - obj_pos_w, dim=-1)
            near_box = grip_to_obj < 0.15  # 15cm 이내
            self.actions[:, 6] = torch.where(
                near_box, torch.ones_like(self.actions[:, 6]), self.actions[:, 6]
            )

        # v15: IL Bootstrap — 첫 30 step demo action 주입 (arm 위로 들어올리는 동작)
        # SAC replay buffer에 lift trajectory 데모 demonstration 채워서 정책이 lift 가능 학습
        # OMY init joint2=-1.55. action[1]=+0.5 → joint2 += delta (toward 0, 어깨 위로)
        # action[2]=-0.3 → joint3 -= delta (팔꿈치 약간 펴기)
        in_demo = self.episode_length_buf < 30
        if in_demo.any():
            # arm joint 1 (shoulder): 양수 → joint2 toward 0 (위로)
            self.actions[:, 1] = torch.where(in_demo, torch.full_like(self.actions[:, 1], 0.5), self.actions[:, 1])
            # arm joint 2 (elbow): 음수 → 펴기
            self.actions[:, 2] = torch.where(in_demo, torch.full_like(self.actions[:, 2], -0.3), self.actions[:, 2])

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

        # v14: Physical attach harness (training only)
        # 박스가 그리퍼 근처면 박스 위치를 grip_center로 강제 → 박스 절대 안 떨어짐
        # 학습 시 lift+transport 행동만 익히면 됨 (grip 안정성 보장)
        # Deployment에서는 example5_2 grip이 약하면 안 됨, 학습용 가정
        if hasattr(self, "grip_center_pos"):
            obj_pos_w = self._object.data.root_pos_w
            grip_to_obj = torch.norm(self.grip_center_pos - obj_pos_w, dim=-1)
            attach_mask = grip_to_obj < 0.15  # 15cm 이내면 attach

            if attach_mask.any():
                # 박스 위치를 grip_center로 강제 (z는 살짝 아래 = 박스 중심이 그리퍼 중심 약간 아래)
                # grip_center는 두 fingertip의 중점
                # 박스 중심은 fingertip 중간보다 살짝 아래 (박스 잡힌 자세)
                obj_state = self._object.data.default_root_state.clone()
                # 현재 obj_state에서 root pos만 강제 변경
                new_obj_state = self._object.data.root_state_w.clone()
                # 박스 중심 = grip_center - 박스 절반 높이 offset (대략)
                box_target_z = self.grip_center_pos[:, 2] - 0.02  # 그리퍼 중심에서 2cm 아래
                new_obj_state[attach_mask, 0] = self.grip_center_pos[attach_mask, 0]
                new_obj_state[attach_mask, 1] = self.grip_center_pos[attach_mask, 1]
                new_obj_state[attach_mask, 2] = box_target_z[attach_mask]
                # 회전은 정자세로 강제 (upright)
                new_obj_state[attach_mask, 3] = 1.0  # quat w=1, x=y=z=0 (identity)
                new_obj_state[attach_mask, 4] = 0.0
                new_obj_state[attach_mask, 5] = 0.0
                new_obj_state[attach_mask, 6] = 0.0
                # 속도는 그리퍼 따라가게 (대략 0)
                new_obj_state[attach_mask, 7:] = 0.0
                attach_env_ids = attach_mask.nonzero(as_tuple=True)[0]
                self._object.write_root_state_to_sim(new_obj_state[attach_env_ids], env_ids=attach_env_ids)

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
    # Rewards — 6-tier:
    #   T1) 중심까지 거리 비례 (멀리서도 끌어당김)
    #   T2) 중심 정렬되면 추가 보상
    #   T3) 중심 정렬 + 끝점 X축 OR Y축 정렬 시 더 큰 보상
    #   T4) 중심 정렬 + obj_bottom 바닥 닿으면 SUCCESS (큰 보너스)
    #   T5) 그리퍼-물체 가까이 (잡고 있기 유도, 가우시안)
    #   T6) 물체 근처에서 그리퍼 닫혀있으면 보너스 (놓치지 마)
    # 끝점 정렬: |ep_x diff| OR |ep_y diff| 중 최소값으로 측정 (loose)
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        obj_pos_w = self._object.data.root_pos_w
        obj_quat = self._object.data.root_quat_w
        target_pos_w = self.target_cell_pos_w
        env_origins = self.scene.env_origins

        obj_pos_rel = obj_pos_w - env_origins
        target_pos_rel = target_pos_w - env_origins

        # --- 기본 ---
        xy_dist = torch.norm(obj_pos_rel[:, :2] - target_pos_rel[:, :2], dim=-1)
        hover_z = obj_pos_rel[:, 2]
        grip_to_obj_dist = torch.norm(self.grip_center_pos - obj_pos_w, dim=-1)

        obj_vel = torch.norm(self._object.data.root_lin_vel_w, dim=-1)
        stable = obj_vel < self.cfg.stable_vel_threshold

        grip_pos = self._robot.data.joint_pos[:, self.main_gripper_joint_id]
        g_lower = self.robot_dof_lower_limits[self.main_gripper_joint_id]
        g_upper = self.robot_dof_upper_limits[self.main_gripper_joint_id]
        gripper_close_state = (grip_pos - g_lower) / (g_upper - g_lower + 1e-8)

        local_up = torch.zeros_like(obj_pos_w)
        local_up[:, 2] = 1.0
        world_up_from_obj = quat_apply(obj_quat, local_up)
        upright_score = world_up_from_obj[:, 2]
        tilted = upright_score < self.cfg.tilt_upright_threshold

        # --- 끝점 (single-axis: X or Y, 둘 중 더 가까운 축으로 측정) ---
        obj_size_x = self.cfg.object.spawn.size[0]
        obj_height = self.cfg.object.spawn.size[2]

        obj_local_endpoint = torch.zeros_like(obj_pos_w)
        obj_local_endpoint[:, 0] = self.cfg.camera_side_sign * obj_size_x * 0.5
        obj_endpoint_w = obj_pos_w + quat_apply(obj_quat, obj_local_endpoint)
        obj_endpoint_rel = obj_endpoint_w - env_origins

        target_endpoint_rel = target_pos_rel.clone()
        target_endpoint_rel[:, 0] = (
            target_endpoint_rel[:, 0]
            + self.cfg.target_endpoint_sign * obj_size_x * 0.5
        )
        # 단일 축 거리: X or Y 중 더 작은 값 (둘 중 하나만 정렬돼도 OK)
        ep_x_diff = torch.abs(obj_endpoint_rel[:, 0] - target_endpoint_rel[:, 0])
        ep_y_diff = torch.abs(obj_endpoint_rel[:, 1] - target_endpoint_rel[:, 1])
        endpoint_single_axis_diff = torch.minimum(ep_x_diff, ep_y_diff)
        # 전체 XY (호환 로그용)
        endpoint_xy_dist = torch.norm(
            obj_endpoint_rel[:, :2] - target_endpoint_rel[:, :2], dim=-1
        )

        # --- 바닥 / 정렬 판정 ---
        obj_bottom_z = hover_z - obj_height * 0.5
        on_floor = obj_bottom_z < self.cfg.on_floor_touch_threshold  # 바닥 닿음

        xy_aligned = xy_dist < self.cfg.cell_tolerance               # 중심 정렬 (2.5cm)
        xy_aligned_loose = xy_dist < 0.05                             # 셀 안 정렬 (5cm, abandoned 게이팅용)
        endpoint_aligned = endpoint_single_axis_diff < self.cfg.yaw_tolerance  # 끝점 단일축 (4cm)

        # 성공 = 중심 정렬 + 바닥 닿음
        success = xy_aligned & on_floor

        # 실패 종료 (기존 유지)
        abandoned = grip_to_obj_dist > self.cfg.abandoned_dist_threshold

        # 호환성 (로그 + done 재사용)
        deep_enough = on_floor

        # ===========================================================
        # example7_2: Lift + Transport Reward (단순화)
        # 목표: 박스를 (target_xy, target_z=0.30)으로 이동, grip 유지
        # ===========================================================
        target_z = self.cfg.lift_target_z

        # R1: XY 정렬 (셀 xy로 이동)
        r_xy_close = torch.exp(-25.0 * xy_dist ** 2)  # 0~1

        # R2: Z 정렬 (목표 높이 0.30)
        z_diff = hover_z - target_z
        r_z_close = torch.exp(-30.0 * z_diff ** 2)  # 0~1

        # R3: lift progress (단순 ramp, 박스 들기 자체에 보상)
        # v4: 3중 게이트로 push 차단 (closed + near + above)
        lift_progress_raw = torch.clamp((hover_z - 0.06) / 0.24, 0.0, 1.0)
        gripper_closed = (gripper_close_state > 0.4).float()
        grip_near_box = (grip_to_obj_dist < 0.10).float()
        grip_above_box = (self.grip_center_pos[:, 2] > obj_pos_w[:, 2] + 0.02).float()
        real_grasp = grip_near_box * gripper_closed * grip_above_box
        lift_progress = lift_progress_raw * real_grasp

        # v5: 박스 상승 속도 보상 (즉각적 lift 신호)
        # 진짜 잡기 중 박스가 위로 움직이면 +보상 (push exploit 차단됨, real_grasp gated)
        box_z_vel = self._object.data.root_lin_vel_w[:, 2]
        r_rising = torch.clamp(box_z_vel, 0.0, 0.5) * real_grasp  # 박스 위로 0.5m/s까지

        # v11: 그리퍼 자체 상승 속도 보상 (arm 위로 움직이는 행동 학습)
        # 박스 동반 안 해도 그리퍼만 위로 가도 보상 → arm action 학습
        # 단 grip_near_box AND closed AND above 조건 (박스 없이 그냥 arm 들기 차단)
        grip_z = self.grip_center_pos[:, 2]
        grip_z_vel = (grip_z - self._prev_grip_z) / self.dt  # 추정 속도
        self._prev_grip_z = grip_z.clone()
        # gripper rising bonus (박스 근처 + 닫힘 + 위에 있을 때만)
        r_grip_rising = torch.clamp(grip_z_vel, 0.0, 0.5) * gripper_closed * grip_near_box

        # R4: grip 유지 (박스 놓치면 안 됨)
        r_grip_near = torch.exp(-50.0 * grip_to_obj_dist ** 2)

        # R5: upright 유지
        r_upright = torch.clamp(upright_score, 0.0, 1.0)

        # R6: 성공
        # v7: example7_2 단순화 — LIFT ONLY (transport 제외)
        # success = 박스 z >= 0.20 + 진짜 잡기 + upright
        # xy alignment는 example7_2_5에서 학습
        z_aligned = torch.abs(z_diff) < self.cfg.lift_target_z_tolerance
        z_lifted = hover_z >= 0.20  # v7: 박스 20cm 이상 올림
        gripped = (grip_to_obj_dist < 0.10) & (gripper_close_state > 0.4) & (self.grip_center_pos[:, 2] > obj_pos_w[:, 2] + 0.02)
        upright_ok = upright_score > 0.95
        # v7: xy_aligned 제거, lift만 검증
        success = z_lifted & gripped & upright_ok

        # R7: 박스 떨어뜨림 페널티 (10cm 이상 멀어지면)
        dropped = grip_to_obj_dist > 0.10
        r_dropped = dropped.float()

        # R8: gripper open 페널티 (target 미도달 + 그리퍼 열림 → 페널티)
        # v41 grip이 약해서 reset 후 바로 박스 놓치는 문제 해결
        gripper_open = gripper_close_state < 0.3
        z_aligned_loose = torch.abs(z_diff) < 0.05
        at_target = xy_aligned & z_aligned_loose
        r_premature_open = (gripper_open & ~at_target).float()

        # R9: gripper closed 유지 보상 (box 근처에서)
        near_obj_strict = grip_to_obj_dist < 0.10
        closed = gripper_close_state > 0.4
        r_keep_grip = (near_obj_strict & closed).float()

        # action penalty
        action_penalty = torch.sum(self.actions ** 2, dim=-1)

        # 기존 호환 (로그/done용 변수)
        r_endpoint_aligned = (xy_aligned & endpoint_aligned).float()
        r_endpoint_aligned_gated = r_endpoint_aligned * r_upright
        xy_upright_ready = ((r_xy_close * r_upright) > 0.8).float()
        r_not_tilted = (upright_score > self.cfg.bonus_upright_threshold).float()
        r_severe_tilt = tilted.float()
        r_above_when_unaligned = torch.zeros_like(xy_dist)
        r_keep_closed = ((grip_to_obj_dist < 0.06) & (gripper_close_state > 0.5) & ~success).float()
        r_grip_near_obj = r_grip_near
        r_success = success.float()

        reward = (
            # v12: dropped 페널티 완화 (-200→-30), policy가 lift 시도 두려워 안 함
            + 50.0 * r_xy_close
            + 100.0 * r_z_close
            + 800.0 * lift_progress
            + 500.0 * r_rising
            + 500.0 * r_grip_rising           # v12: 300→500 (arm 위로 강화)
            + 30.0 * r_grip_near
            + 80.0 * r_upright
            + 100.0 * r_keep_grip
            + 2000.0 * r_success
            - 30.0 * r_dropped                # v12: 200→30 (lift 시도 두려움 제거)
            - 100.0 * r_premature_open
            - 200.0 * r_severe_tilt
            - 0.1 * action_penalty
        )

        # 종료 조건 재사용
        self._last_tilted = tilted
        self._last_severely_tilted = upright_score < 0.0  # 90° 이상 기울어짐 (옆으로 누움/뒤집힘)
        self._last_abandoned = abandoned
        self._last_success_now = success
        self._last_inserted = on_floor
        self._last_xy_aligned_loose = xy_aligned_loose

        # ===========================================================
        # 로그 — 핵심만
        # ===========================================================
        obj_pos_rel0 = obj_pos_w[0] - env_origins[0]
        grip_pos_rel0 = self.grip_center_pos[0] - env_origins[0]
        target_pos_rel0 = target_pos_w[0] - env_origins[0]
        obj_endpoint_rel0 = obj_endpoint_rel[0]
        target_endpoint_rel0 = target_endpoint_rel[0]
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
        # 끝점 좌표 (방향 일치 검증용)
        self.reward_log["env0_obj_ep_x"] = float(obj_endpoint_rel0[0])
        self.reward_log["env0_obj_ep_y"] = float(obj_endpoint_rel0[1])
        self.reward_log["env0_tgt_ep_x"] = float(target_endpoint_rel0[0])
        self.reward_log["env0_tgt_ep_y"] = float(target_endpoint_rel0[1])
        self.reward_log["env0_upright"] = float(upright_score[0])
        self.reward_log["env0_grip_close"] = float(gripper_close_state[0])
        self.reward_log["env0_j1"] = float(arm_joint_pos0[0])
        self.reward_log["env0_j2"] = float(arm_joint_pos0[1])
        self.reward_log["env0_j3"] = float(arm_joint_pos0[2])
        self.reward_log["env0_j4"] = float(arm_joint_pos0[3])
        self.reward_log["env0_j5"] = float(arm_joint_pos0[4])
        self.reward_log["env0_j6"] = float(arm_joint_pos0[5])

        self.reward_log["dist_xy"] = float(xy_dist.mean())
        self.reward_log["dist_endpoint_xy"] = float(endpoint_xy_dist.mean())
        self.reward_log["dist_endpoint_min"] = float(endpoint_single_axis_diff.mean())
        self.reward_log["dist_obj_bottom_z"] = float(obj_bottom_z.mean())
        self.reward_log["dist_grip_obj"] = float(grip_to_obj_dist.mean())
        self.reward_log["dist_obj_vel"] = float(obj_vel.mean())

        self.reward_log["rate_xy_aligned"] = float(xy_aligned.float().mean())
        self.reward_log["rate_endpoint_aligned"] = float(endpoint_aligned.float().mean())
        self.reward_log["rate_on_floor"] = float(on_floor.float().mean())
        self.reward_log["rate_abandoned"] = float(abandoned.float().mean())
        self.reward_log["rate_tilted"] = float(tilted.float().mean())
        self.reward_log["rate_tilted_on_floor"] = float((tilted & on_floor).float().mean())
        self.reward_log["rate_success_now"] = float(success.float().mean())
        # Episode-level 성공률 (롤링 윈도우 300 episode)
        if len(self.episode_success_history) > 0:
            self.reward_log["rate_episode_success"] = (
                sum(self.episode_success_history) / len(self.episode_success_history)
            )
            self.reward_log["rate_episode_count"] = float(len(self.episode_success_history))
        else:
            self.reward_log["rate_episode_success"] = 0.0
            self.reward_log["rate_episode_count"] = 0.0
        # 누적 성공 카운터
        self.reward_log["rate_cumulative_success"] = float(self.cumulative_success_count)
        self.reward_log["rate_cumulative_episode"] = float(self.cumulative_episode_count)

        self.reward_log["rew_t1_xy_close"] = float(r_xy_close.mean())
        self.reward_log["rew_t1b_z_close"] = float(r_z_close.mean())
        self.reward_log["rew_t3_endpoint"] = float(r_endpoint_aligned.mean())
        self.reward_log["rew_t3_gated"] = float(r_endpoint_aligned_gated.mean())
        self.reward_log["rate_xy_upright_ready"] = float(xy_upright_ready.mean())
        self.reward_log["rew_t4_success"] = float(r_success.mean())
        self.reward_log["rew_t5_grip_near"] = float(r_grip_near_obj.mean())
        self.reward_log["rew_t6_keep_closed"] = float(r_keep_closed.mean())
        self.reward_log["rew_t7_upright"] = float(r_upright.mean())
        self.reward_log["rew_t8_not_tilted"] = float(r_not_tilted.mean())
        self.reward_log["rew_t9_severe_tilt"] = float(r_severe_tilt.mean())
        self.reward_log["rew_t10_above"] = float(r_above_when_unaligned.mean())
        self.reward_log["rew_t11_action_pen"] = float(action_penalty.mean())
        self.reward_log["rew_total"] = float(reward.mean())

        return reward

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: dict | list | None = None,
        **kwargs,  # Sb3VecEnvWrapper가 indices 등을 전달 → 무시
    ) -> np.ndarray:
        """HER relabeling용 sparse reward — 셀 위 정렬 기준 (XY only).

        achieved_goal = obj 현재 pos (3D), desired_goal = 셀 중심 (3D).
        XY 거리만 비교 (z는 자유 — 정렬만 목표).
        """
        diff_xy = achieved_goal[..., :2] - desired_goal[..., :2]
        dist_xy = np.linalg.norm(diff_xy, axis=-1)
        tol = float(self.cfg.cell_tolerance) * 2.0
        return np.where(dist_xy < tol, 0.0, -1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # _get_rewards에서 저장한 플래그 재사용
        success_now = self._last_success_now
        tilted = self._last_tilted
        abandoned = self._last_abandoned

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
        margin = 0.6  # 1x1 grid에서 handoff 위치가 out_of_bounds 되지 않도록 확대
        out_of_bounds = (
            (obj_pos_rel[:, 0] < self.cfg.grid_center_x - total_x / 2 - margin)
            | (obj_pos_rel[:, 0] > self.cfg.grid_center_x + total_x / 2 + margin)
            | (obj_pos_rel[:, 1] < self.cfg.grid_center_y - total_y / 2 - margin)
            | (obj_pos_rel[:, 1] > self.cfg.grid_center_y + total_y / 2 + margin)
        )

        # 기울어짐 reset:
        # - tilted_on_floor: 기울어짐(32°+) AND 바닥 닿음 → reset
        # - severely_tilted: 공중이어도 90° 이상 기울어짐(옆/뒤집힘) → reset
        tilted_on_floor = tilted & self._last_inserted
        severely_tilted = self._last_severely_tilted

        # abandoned는 셀에 들어가지 않은 경우에만 적용
        # (release 동작 시 그리퍼가 박스에서 멀어지는 건 허용)
        # abandoned 비활성 조건: on_floor AND xy 정렬(5cm 이내) — 셀에 정확히 떨어뜨려야만
        # (어디든 던져두고 도망가는 exploit 차단)
        in_cell_on_floor = self._last_inserted & self._last_xy_aligned_loose
        abandoned_active = abandoned & ~in_cell_on_floor

        terminated = (
            success_stable
            | fallen_below
            | out_of_bounds
            | tilted_on_floor    # 기울어짐 + 바닥 닿음
            | severely_tilted    # 90°+ 기울어짐 (공중 포함)
            | abandoned_active   # 그리퍼에서 멀어짐 (단, on_floor 시는 허용)
        )
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # Episode-level 성공률 추적 (terminated/truncated 환경 결과 기록)
        ended = terminated | truncated
        ended_indices = ended.nonzero(as_tuple=True)[0].tolist()
        for idx in ended_indices:
            success_flag = int(success_stable[idx].item())
            self.episode_success_history.append(success_flag)
            self.cumulative_episode_count += 1
            if success_flag:
                self.cumulative_success_count += 1

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
        self._prev_grip_z[env_ids_t] = 0.0  # v11: reset 시 grip z 초기화

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
