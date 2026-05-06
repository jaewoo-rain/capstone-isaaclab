"""OMY Hierarchy Env 구현

Lift policy(example5) → Place policy(example6) 계층 실행을 위한 통합 환경.

- 시작: 물체 바닥(0.45, -0.10, 0.06), 로봇 기본 자세 (lift env 동일)
- 그리드: 3×3 셀 (place env에서 그대로 가져옴)
- Observation:
    * 기본 step 반환: place 형식 31차원 flat (25 + 3 achieved + 3 desired)
    * lift 형식 34차원: get_lift_observation() 별도 호출
- Reward: lift→place 통합 sparse + shaping (재학습 시 사용)
- Done: 물체가 셀에 안정 안착 / 추락 / 시간 초과
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply

from .hier_env_cfg import HierEnvCfg


class HierEnv(DirectRLEnv):
    cfg: HierEnvCfg

    # place 형식: 25 core + 3 achieved + 3 desired = 31
    OBS_CORE_DIM = 25
    GOAL_DIM = 3

    # lift 형식: 34차원
    LIFT_OBS_DIM = 34

    def __init__(self, cfg: HierEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ----- gym spaces -----
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.observation_space,), dtype=np.float32,
        )

        # ----- joint / body 매핑 -----
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
        # lift obs용: arm 6 + gripper 4 = 10개 관절
        self.all_obs_joint_ids = self.arm_joint_ids + self.gripper_joint_ids

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

        # 위치 캐시
        self.obj_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_pos_rel = torch.zeros((self.num_envs, 3), device=self.device)
        self.grip_center_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.grasp_target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_to_grip = torch.zeros((self.num_envs, 3), device=self.device)

        # 타겟 셀
        self.target_cell_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.target_cell_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # 그리드 셀 중심 (환경 원점 기준)
        self.cell_centers_local = self._build_cell_centers()

        # 성공 유지 카운터
        self.success_hold_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # done 플래그 (rewards에서 계산해서 dones에서 재사용)
        self._last_success_now = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._last_tilted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._last_abandoned = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

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

        # 그리드 벽 스폰 (place env와 동일)
        self._spawn_grid_walls()

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _spawn_grid_walls(self):
        cx = self.cfg.grid_center_x
        cy = self.cfg.grid_center_y
        nx = self.cfg.grid_num_x
        ny = self.cfg.grid_num_y
        t = self.cfg.wall_thickness
        h = self.cfg.wall_height

        total_x = nx * self.cfg.cell_inner_x + (nx + 1) * t
        total_y = ny * self.cfg.cell_inner_y + (ny + 1) * t

        # 수직 벽 (nx+1 개)
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

        # 수평 벽 (ny+1 개)
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

    def _build_cell_centers(self) -> torch.Tensor:
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
                z = self.cfg.object.spawn.size[2] / 2
                cells.append([x, y, z])
        return torch.tensor(cells, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Action (lift/place 둘 다 동일)
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

        # gripper는 4개 관절 동일 명령 + ×3 배율 (lift/place 동일)
        grip_cmd = actions[:, 6] * 3.0
        for gid in self.gripper_joint_ids:
            dof_delta[:, gid] = grip_cmd
        return dof_delta

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # ------------------------------------------------------------------
    # 기본 Observation: place 형식 (31차원 flat)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        # arm joints (6개)
        arm_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]
        arm_vel = self._robot.data.joint_vel[:, self.arm_joint_ids]

        arm_lower = self.robot_dof_lower_limits[self.arm_joint_ids]
        arm_upper = self.robot_dof_upper_limits[self.arm_joint_ids]
        arm_pos_scaled = 2.0 * (arm_pos - arm_lower) / (arm_upper - arm_lower + 1e-8) - 1.0
        arm_vel_scaled = arm_vel * self.cfg.dof_velocity_scale

        # gripper close state (1)
        grip_pos = self._robot.data.joint_pos[:, self.main_gripper_joint_id].unsqueeze(-1)
        g_lower = self.robot_dof_lower_limits[self.main_gripper_joint_id].view(1, 1)
        g_upper = self.robot_dof_upper_limits[self.main_gripper_joint_id].view(1, 1)
        gripper_close = (grip_pos - g_lower) / (g_upper - g_lower + 1e-8)

        # end-effector pos rel (3)
        ee_pos_rel = self.grip_center_pos - self.scene.env_origins

        # object (3+3)
        obj_pos_rel = self.obj_pos_rel
        obj_vel = self._object.data.root_lin_vel_w

        # object → target (3)
        target_pos_rel = self.target_cell_pos_w - self.scene.env_origins
        obj_to_target = target_pos_rel - obj_pos_rel

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
        )  # 25

        achieved = self._object.data.root_pos_w.clone()
        desired = self.target_cell_pos_w.clone()

        obs = torch.cat([core, achieved, desired], dim=-1)  # 31
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # ------------------------------------------------------------------
    # Lift 형식 Observation (34차원) — play 스크립트가 lift phase에서 호출
    # ------------------------------------------------------------------
    def get_lift_observation(self) -> torch.Tensor:
        """example5 lift policy가 기대하는 34차원 obs 생성.

        lift_env._get_observations()와 동일한 순서/스케일.
        VecNormalize는 외부에서 적용.
        """
        self._compute_intermediate_values()

        # arm 6 + gripper 4 = 10개 관절
        joint_pos = self._robot.data.joint_pos[:, self.all_obs_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self.all_obs_joint_ids]

        lower = self.robot_dof_lower_limits[self.all_obs_joint_ids]
        upper = self.robot_dof_upper_limits[self.all_obs_joint_ids]

        dof_pos_scaled = 2.0 * (joint_pos - lower) / (upper - lower + 1e-8) - 1.0  # 10
        dof_vel_scaled = joint_vel * self.cfg.dof_velocity_scale                    # 10

        grip_joint_pos = self._robot.data.joint_pos[:, self.main_gripper_joint_id].unsqueeze(-1)
        grip_lower = self.robot_dof_lower_limits[self.main_gripper_joint_id].view(1, 1)
        grip_upper = self.robot_dof_upper_limits[self.main_gripper_joint_id].view(1, 1)
        gripper_close_state = (grip_joint_pos - grip_lower) / (grip_upper - grip_lower + 1e-8)  # 1

        # to_lift_target = lift threshold - obj height
        to_lift_target = (self.cfg.lift_to_place_threshold - self.obj_pos_w[:, 2]).unsqueeze(-1)  # 1

        # finger tip → object 벡터
        l_pos = self._robot.data.body_pos_w[:, self.left_finger_body_id, :]
        r_pos = self._robot.data.body_pos_w[:, self.right_finger_body_id, :]
        left_to_obj_vec = self.obj_pos_w - l_pos    # 3
        right_to_obj_vec = self.obj_pos_w - r_pos   # 3

        obs = torch.cat(
            [
                dof_pos_scaled,       # 10
                dof_vel_scaled,       # 10
                self.obj_pos_rel,     # 3
                self.obj_to_grip,     # 3
                left_to_obj_vec,      # 3
                right_to_obj_vec,     # 3
                gripper_close_state,  # 1
                to_lift_target,       # 1
            ],
            dim=-1,
        )  # 34
        return torch.clamp(obs, -5.0, 5.0)

    # ------------------------------------------------------------------
    # 헬퍼: 현재 물체 높이 (lift→place 전환 판정용)
    # ------------------------------------------------------------------
    def get_object_height(self) -> torch.Tensor:
        return self.obj_pos_w[:, 2].clone()

    # ------------------------------------------------------------------
    # Rewards (재학습용 — play에서는 무시됨)
    # 기본적으로 place env의 stage 2 reward를 사용
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        obj_pos_w = self._object.data.root_pos_w
        target_pos_w = self.target_cell_pos_w

        xy_dist = torch.norm(obj_pos_w[:, :2] - target_pos_w[:, :2], dim=-1)
        z_dist = torch.abs(obj_pos_w[:, 2] - target_pos_w[:, 2])
        in_cell_xy = xy_dist < self.cfg.cell_tolerance

        grip_to_obj_dist = torch.norm(self.grip_center_pos - obj_pos_w, dim=-1)

        obj_vel = torch.norm(self._object.data.root_lin_vel_w, dim=-1)
        stable = obj_vel < self.cfg.stable_vel_threshold

        grip_pos = self._robot.data.joint_pos[:, self.main_gripper_joint_id]
        g_lower = self.robot_dof_lower_limits[self.main_gripper_joint_id]
        g_upper = self.robot_dof_upper_limits[self.main_gripper_joint_id]
        gripper_close_state = (grip_pos - g_lower) / (g_upper - g_lower + 1e-8)
        gripper_open = gripper_close_state < self.cfg.gripper_open_threshold

        obj_quat = self._object.data.root_quat_w
        local_up = torch.zeros((obj_quat.shape[0], 3), device=self.device)
        local_up[:, 2] = 1.0
        world_up_from_obj = quat_apply(obj_quat, local_up)
        upright_score = world_up_from_obj[:, 2]
        tilted = upright_score < self.cfg.tilt_upright_threshold

        on_floor = obj_pos_w[:, 2] < self.cfg.on_floor_z_threshold
        abandoned = grip_to_obj_dist > self.cfg.abandoned_dist_threshold

        # 셀에 안정 안착 = 성공
        success = in_cell_xy & stable & gripper_open & (z_dist < 0.05)

        # 단순 reward (재학습 시 튜닝)
        approach = torch.exp(-30.0 * xy_dist**2)
        height_match = torch.exp(-40.0 * z_dist**2) * in_cell_xy.float()
        action_penalty = torch.sum(self.actions ** 2, dim=-1)

        reward = (
            10.0 * success.float()
            + 1.0 * approach
            + 2.0 * height_match
            - 0.001 * action_penalty
        )

        # done 플래그 저장
        self._last_tilted = tilted
        self._last_abandoned = abandoned
        self._last_success_now = success

        # 로그
        self.reward_log["dist_xy"] = float(xy_dist.mean())
        self.reward_log["dist_z"] = float(z_dist.mean())
        self.reward_log["obj_z"] = float(obj_pos_w[0, 2])
        self.reward_log["rate_in_cell"] = float(in_cell_xy.float().mean())
        self.reward_log["rate_success"] = float(success.float().mean())

        return reward

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        success_now = self._last_success_now

        self.success_hold_counter = torch.where(
            success_now,
            self.success_hold_counter + 1,
            torch.zeros_like(self.success_hold_counter),
        )
        success_stable = self.success_hold_counter >= self.cfg.success_hold_steps

        obj_pos_w = self._object.data.root_pos_w
        fallen_below = obj_pos_w[:, 2] < self.cfg.object_fall_z

        terminated = success_stable | fallen_below
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset (lift env 스타일: 물체 바닥 + 로봇 기본 자세)
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = len(env_ids_t)

        self.success_hold_counter[env_ids_t] = 0

        # ----- 1) 로봇: default 자세 (lift env 동일) -----
        joint_pos = self._robot.data.default_joint_pos[env_ids_t].clone()
        joint_pos = torch.clamp(
            joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits,
        )
        joint_vel = torch.zeros_like(joint_pos)

        self.robot_dof_targets[env_ids_t] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)

        # ----- 2) 물체: 바닥 + 노이즈 (lift env 동일) -----
        obj_state = self._object.data.default_root_state[env_ids_t].clone()
        env_origins = self.scene.env_origins[env_ids_t]

        noise = (
            (torch.rand((n, 2), device=self.device) - 0.5)
            * 2.0 * self.cfg.object_pos_noise
        )

        base_x, base_y, base_z = self.cfg.object.init_state.pos
        obj_state[:, 0] = env_origins[:, 0] + base_x + noise[:, 0]
        obj_state[:, 1] = env_origins[:, 1] + base_y + noise[:, 1]
        obj_state[:, 2] = env_origins[:, 2] + base_z
        obj_state[:, 7:] = 0.0

        self._object.write_root_state_to_sim(obj_state, env_ids=env_ids_t)

        # ----- 3) 타겟 셀 결정 (place env 동일) -----
        num_cells = self.cfg.grid_num_x * self.cfg.grid_num_y
        if self.cfg.randomize_target_cell:
            new_idx = torch.randint(0, num_cells, (n,), device=self.device, dtype=torch.long)
        else:
            new_idx = (env_ids_t % num_cells).to(torch.long)

        self.target_cell_idx[env_ids_t] = new_idx
        cell_local = self.cell_centers_local[new_idx]
        self.target_cell_pos_w[env_ids_t] = env_origins + cell_local

        self._compute_intermediate_values(env_ids_t)

    # ------------------------------------------------------------------
    # 중간값 (lift + place 양쪽 obs에 필요한 값 모두 계산)
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        l_pos = self._robot.data.body_pos_w[env_ids, self.left_finger_body_id, :]
        r_pos = self._robot.data.body_pos_w[env_ids, self.right_finger_body_id, :]

        # 그리퍼 중심
        self.grip_center_pos[env_ids] = 0.5 * (l_pos + r_pos)

        # 물체 위치 (월드 + 환경 원점 기준)
        self.obj_pos_w[env_ids] = self._object.data.root_pos_w[env_ids]
        self.obj_pos_rel[env_ids] = self.obj_pos_w[env_ids] - self.scene.env_origins[env_ids]

        # lift obs용: grasp target = 물체 + z offset, obj_to_grip = target - grip_center
        self.grasp_target_pos[env_ids] = self.obj_pos_w[env_ids].clone()
        self.grasp_target_pos[env_ids, 2] += self.cfg.grasp_target_z_offset

        self.obj_to_grip[env_ids] = self.grasp_target_pos[env_ids] - self.grip_center_pos[env_ids]

    # ------------------------------------------------------------------
    # HER 호환: place env와 같은 시그니처 (재학습 시 사용)
    # ------------------------------------------------------------------
    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info=None,
        **kwargs,
    ) -> np.ndarray:
        xy_dist = np.linalg.norm(
            achieved_goal[..., :2] - desired_goal[..., :2], axis=-1,
        )
        return np.where(xy_dist < self.cfg.cell_tolerance, 0.0, -1.0).astype(np.float32)
