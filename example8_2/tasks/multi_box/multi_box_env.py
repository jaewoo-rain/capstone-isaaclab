"""Multi-box env (example8_2) — 3개 박스 동시 spawn + chain inference 지원.

설계:
- 3개 RigidObject (Object0/1/2)
- active_box_idx 추적
- box ↔ cell 고정 매핑 (cfg.box_to_cell)
- Reset: 처음에 3개 박스 모두 default 위치, active_box=0
- 박스 i 처리 완료 → active_box+=1, robot 자세 reset (다음 박스 잡으러)

추후 추가 필요:
1. example5/7 정책별 obs 변환 (compute_grasp_obs, compute_place_obs)
2. _get_observations에서 active box 정보만
3. _get_rewards (chain inference라 사실상 사용 안 함)
4. _get_dones (전체 3 박스 처리 완료 시 종료)
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply

from .multi_box_env_cfg import MultiBoxEnvCfg


class MultiBoxEnv(DirectRLEnv):
    """3-box deployment env.

    Active box tracking, no actual training (chain inference only).
    """

    cfg: MultiBoxEnvCfg

    def __init__(self, cfg: MultiBoxEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # arm joint ids
        self.arm_joint_names = [f"joint{i}" for i in range(1, 7)]
        self.gripper_joint_names = list(self.cfg.gripper_joint_names)

        self.arm_joint_ids: list[int] = []
        for name in self.arm_joint_names:
            found = self._robot.find_joints(name)[0]
            if len(found) == 0:
                raise RuntimeError(f"OMY arm joint not found: {name}")
            self.arm_joint_ids.append(int(found[0]))

        self.gripper_joint_ids: list[int] = []
        for name in self.gripper_joint_names:
            found = self._robot.find_joints(name)[0]
            if len(found) == 0:
                raise RuntimeError(f"OMY gripper joint not found: {name}")
            self.gripper_joint_ids.append(int(found[0]))

        self.main_gripper_joint_id = self.gripper_joint_ids[0]

        # body ids
        self.left_finger_body_id = int(self._robot.find_bodies(self.cfg.left_finger_body_name)[0][0])
        self.right_finger_body_id = int(self._robot.find_bodies(self.cfg.right_finger_body_name)[0][0])

        # joint limits
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device).clone()
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device).clone()
        # mimic: 4 grip joints clamp [0, 1.135]
        for gid in self.gripper_joint_ids:
            self.robot_dof_lower_limits[gid] = 0.0
            self.robot_dof_upper_limits[gid] = 1.135

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # state buffers
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device,
        )
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # 3개 박스
        self._objects: list[RigidObject] = [self._object_0, self._object_1, self._object_2]
        self.num_boxes = 3

        # Active box index per env
        self.active_box_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # box ↔ cell 매핑
        self.box_to_cell = list(self.cfg.box_to_cell)

        # cell centers (env 기준)
        self.cell_centers_local = self._build_cell_centers()

        # phase (0: GRASP_LIFT, 1: PLACE, 2: DONE)
        self.phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.action_space_dim = self.cfg.action_space
        self.observation_space_dim = self.cfg.observation_space

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object_0 = RigidObject(self.cfg.object_0)
        self._object_1 = RigidObject(self.cfg.object_1)
        self._object_2 = RigidObject(self.cfg.object_2)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object_0"] = self._object_0
        self.scene.rigid_objects["object_1"] = self._object_1
        self.scene.rigid_objects["object_2"] = self._object_2

        # Terrain (PlaceEnv 패턴)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 조명
        import isaaclab.sim as sim_utils
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

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
                z = self.cfg.box_size[2] / 2  # cell bottom
                cells.append([x, y, z])
        return torch.tensor(cells, device=self.device, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public API for chain inference
    # ------------------------------------------------------------------
    def set_active_box(self, env_id: int, box_idx: int):
        """Chain script에서 active box 변경."""
        self.active_box_idx[env_id] = box_idx
        self.phase[env_id] = 0  # GRASP_LIFT 모드

    def get_active_box_pos(self, env_id: int = 0) -> torch.Tensor:
        """현재 active box의 월드 좌표."""
        box_idx = self.active_box_idx[env_id].item()
        return self._objects[box_idx].data.root_pos_w[env_id]

    def get_target_cell_pos(self, env_id: int = 0) -> torch.Tensor:
        """현재 active box의 target cell (월드 좌표)."""
        box_idx = self.active_box_idx[env_id].item()
        cell_idx = self.box_to_cell[box_idx]
        env_origin = self.scene.env_origins[env_id]
        return env_origin + self.cell_centers_local[cell_idx]

    # ------------------------------------------------------------------
    # Action / Step
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = (
            self.robot_dof_targets
            + torch.ones_like(self.robot_dof_targets) * self.dt * self._actions_to_dof(self.actions) * self.cfg.action_scale
        )
        self.robot_dof_targets = torch.clamp(
            targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits,
        )

    def _actions_to_dof(self, actions: torch.Tensor) -> torch.Tensor:
        dof_delta = torch.zeros((actions.shape[0], self._robot.num_joints), device=self.device)
        for i, joint_id in enumerate(self.arm_joint_ids):
            dof_delta[:, joint_id] = actions[:, i]
        grip_cmd = actions[:, 6] * 3.0
        for gid in self.gripper_joint_ids:
            dof_delta[:, gid] = grip_cmd
        return dof_delta

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # ------------------------------------------------------------------
    # Stub observations / rewards (chain inference만, 실제 obs는 외부 변환)
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # Placeholder - chain script에서 직접 raw_env state 사용
        obs = torch.zeros((self.num_envs, self.cfg.observation_space), device=self.device)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 모든 env phase가 DONE이면 종료
        terminated = self.phase >= 2
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # robot default pose
        joint_pos = self._robot.data.default_joint_pos[env_ids_t].clone()
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot_dof_targets[env_ids_t] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)

        # 3개 박스 default 위치로 (initial reset 시)
        for obj in self._objects:
            obj_state = obj.data.default_root_state[env_ids_t].clone()
            obj_state[:, :3] += self.scene.env_origins[env_ids_t]
            obj_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(obj_state, env_ids=env_ids_t)

        # active box = 0, phase = 0
        self.active_box_idx[env_ids_t] = 0
        self.phase[env_ids_t] = 0
