from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .lift_env_cfg import LiftEnvCfg


class LiftEnv(DirectRLEnv):
    cfg: LiftEnvCfg

    def __init__(self, cfg: LiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # -----------------------------
        # joint / body 이름 매핑
        # -----------------------------
        self.arm_joint_names = [f"joint{i}" for i in range(1, 7)]
        self.gripper_joint_names = [
            "rh_r1_joint",
            "rh_r2",
            "rh_l1",
            "rh_l2",
        ]

        # arm joint ids
        self.arm_joint_ids = []
        for name in self.arm_joint_names:
            found = self._robot.find_joints(name)[0]
            if len(found) == 0:
                raise RuntimeError(f"OMY arm joint를 찾지 못함: {name}")
            self.arm_joint_ids.append(int(found[0]))

        # gripper joint ids
        self.gripper_joint_ids = []
        for name in self.gripper_joint_names:
            found = self._robot.find_joints(name)[0]
            if len(found) == 0:
                raise RuntimeError(f"OMY gripper joint를 찾지 못함: {name}")
            self.gripper_joint_ids.append(int(found[0]))

        self.main_gripper_joint_id = self.gripper_joint_ids[0]
        self.control_joint_ids = self.arm_joint_ids + [self.main_gripper_joint_id]

        # finger body ids
        left_found = self._robot.find_bodies("rh_p12_rn_l2")[0]
        if len(left_found) == 0:
            left_found = self._robot.find_bodies("rh_p12_rn_l1")[0]
        if len(left_found) == 0:
            raise RuntimeError("왼쪽 finger body를 찾지 못함")

        right_found = self._robot.find_bodies("rh_p12_rn_r2")[0]
        if len(right_found) == 0:
            right_found = self._robot.find_bodies("rh_p12_rn_r1")[0]
        if len(right_found) == 0:
            raise RuntimeError("오른쪽 finger body를 찾지 못함")

        self.left_finger_body_id = int(left_found[0])
        self.right_finger_body_id = int(right_found[0])

        # camera body id
        camera_found = self._robot.find_bodies("link6")[0]
        if len(camera_found) == 0:
            raise RuntimeError("link6 body를 찾지 못함")
        self.camera_body_id = int(camera_found[0])
        print(f"감시 링크 사용: link6 -> body_id={self.camera_body_id}")

        # spaces
        self.arm_action_dim = 6
        self.gripper_action_dim = 1
        self.total_action_dim = 7

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.total_action_dim,),
            dtype=np.float32,
        )

        # dof_pos(7) + dof_vel(7) + obj_pos_rel(3) + obj_to_grip(3) + grip_state(1) + to_target(1) = 22
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),
            dtype=np.float32,
        )

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        # gripper는 느리게
        for gid in self.gripper_joint_ids:
            self.robot_dof_speed_scales[gid] = 0.1

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )

        self.obj_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_pos_rel = torch.zeros((self.num_envs, 3), device=self.device)
        self.grip_center_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_to_grip = torch.zeros((self.num_envs, 3), device=self.device)

        self.actions = torch.zeros((self.num_envs, self.total_action_dim), device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)

        self.reward_log = {
            "approach_reward": 0.0,
            "grasp_bonus": 0.0,
            "lift_reward": 0.0,
            "success_rate": 0.0,
            "xy_align_reward": 0.0,
            "z_align_reward": 0.0,
            "close_reward": 0.0,
            "camera_penalty": 0.0,
        }

        self._compute_intermediate_values()

    # ------------------------------------------------------------------
    # Scene 설정
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.object)

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

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        targets = (
            self.robot_dof_targets
            + self.robot_dof_speed_scales
            * self.dt
            * self.actions_to_dof(self.actions)
            * self.cfg.action_scale
        )
        self.robot_dof_targets = torch.clamp(
            targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits
        )

    def actions_to_dof(self, actions: torch.Tensor) -> torch.Tensor:
        dof_delta = torch.zeros(
            (actions.shape[0], self._robot.num_joints), device=self.device
        )

        # arm 6축
        for i, joint_id in enumerate(self.arm_joint_ids):
            dof_delta[:, joint_id] = actions[:, i]

        # gripper 1축 -> mimic joint 전부에 동일 명령
        grip_cmd = actions[:, 6]
        for gid in self.gripper_joint_ids:
            dof_delta[:, gid] = grip_cmd

        return dof_delta

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        joint_pos = self._robot.data.joint_pos[:, self.control_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self.control_joint_ids]

        lower = self.robot_dof_lower_limits[self.control_joint_ids]
        upper = self.robot_dof_upper_limits[self.control_joint_ids]

        dof_pos_scaled = (
            2.0 * (joint_pos - lower) / (upper - lower + 1e-8) - 1.0
        )
        dof_vel_scaled = joint_vel * self.cfg.dof_velocity_scale

        # 대표 그리퍼 관절 하나로 닫힘 정도 사용
        grip_joint_pos = self._robot.data.joint_pos[:, self.main_gripper_joint_id].unsqueeze(-1)
        grip_lower = self.robot_dof_lower_limits[self.main_gripper_joint_id].view(1, 1)
        grip_upper = self.robot_dof_upper_limits[self.main_gripper_joint_id].view(1, 1)

        gripper_close_state = (
            (grip_joint_pos - grip_lower) / (grip_upper - grip_lower + 1e-8)
        )

        to_lift_target = (
            self.cfg.lift_height_threshold - self.obj_pos_w[:, 2]
        ).unsqueeze(-1)

        obs = torch.cat(
            [
                dof_pos_scaled,
                dof_vel_scaled,
                self.obj_pos_rel,
                self.obj_to_grip,
                gripper_close_state,
                to_lift_target,
            ],
            dim=-1,
        )

        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        obj_pos = self.obj_pos_w
        grip_pos = self.grip_center_pos

        l_pos = self._robot.data.body_pos_w[:, self.left_finger_body_id, :]
        r_pos = self._robot.data.body_pos_w[:, self.right_finger_body_id, :]
        camera_pos = self._robot.data.body_pos_w[:, self.camera_body_id, :]

        grip_joint_pos = self._robot.data.joint_pos[:, self.main_gripper_joint_id]
        grip_lower = self.robot_dof_lower_limits[self.main_gripper_joint_id]
        grip_upper = self.robot_dof_upper_limits[self.main_gripper_joint_id]

        gripper_close_state = (
            (grip_joint_pos - grip_lower) / (grip_upper - grip_lower + 1e-8)
        )

        obj_to_grip = obj_pos - grip_pos
        dist = torch.norm(obj_to_grip, dim=-1)

        xy_dist = torch.norm(obj_to_grip[:, :2], dim=-1)
        z_dist = torch.abs(obj_to_grip[:, 2])

        # 1) 접근 진행 보상
        approach_reward = torch.clamp(self._prev_dist - dist, min=0.0)
        self._prev_dist = dist.clone()

        # 2) 정렬 보상
        xy_align_reward = torch.exp(-40.0 * xy_dist**2)
        z_align_reward = torch.exp(-60.0 * z_dist**2)

        xy_aligned = xy_dist < 0.05
        z_aligned = z_dist < 0.05
        aligned = xy_aligned & z_aligned

        # 3) 양옆 finger 배치
        left_is_left = l_pos[:, 1] < obj_pos[:, 1]
        right_is_right = r_pos[:, 1] > obj_pos[:, 1]

        left_to_obj = torch.norm(obj_pos - l_pos, dim=-1)
        right_to_obj = torch.norm(obj_pos - r_pos, dim=-1)

        fingers_near = (left_to_obj < 0.08) & (right_to_obj < 0.08)
        side_ok = left_is_left & right_is_right

        pre_grasp_ready = aligned & fingers_near & side_ok

        # 4) grasp 판정
        closed_enough = gripper_close_state > 0.55
        is_grasping = pre_grasp_ready & closed_enough
        grasp_bonus = is_grasping.float()

        alignment_score = (
            torch.exp(-40.0 * xy_dist**2)
            * torch.exp(-60.0 * z_dist**2)
            * is_grasping.float()
        )
        close_reward = alignment_score * gripper_close_state

        # 5) lift 보상
        obj_height = obj_pos[:, 2]
        lift_reward = torch.clamp(obj_height - 0.022, min=0.0) * is_grasping.float()

        # 6) 성공 보너스
        success = (obj_height > self.cfg.lift_height_threshold).float()
        success_reward = success

        # 7) 카메라 바닥 접근 패널티
        camera_height = camera_pos[:, 2]
        camera_penalty = torch.clamp(
            self.cfg.camera_min_height - camera_height,
            min=0.0,
        )

        # 8) 액션 패널티
        action_penalty = torch.sum(self.actions ** 2, dim=-1)

        # 9) 최종 reward
        reward = (
            + 0.15 * xy_align_reward
            + 0.12 * z_align_reward
            + 10.0 * close_reward
            + 30.0 * lift_reward
            + 1000.0 * success_reward
            - 0.001 * action_penalty
            - 50 * camera_penalty
        )

        self.extras["log"] = {
            "approach_reward": approach_reward.mean(),
            "xy_align_reward": xy_align_reward.mean(),
            "z_align_reward": z_align_reward.mean(),
            "grasp_bonus": grasp_bonus.mean(),
            "lift_reward": lift_reward.mean(),
            "success_rate": success.mean(),
            "xy_dist": xy_dist.mean(),
            "z_dist": z_dist.mean(),
            "gripper_width": gripper_close_state.mean(),
            "close_reward": close_reward.mean(),
            "camera_penalty": camera_penalty.mean(),
            "camera_height": camera_height.mean(),
        }

        self.reward_log["approach_reward"] = float(approach_reward.mean())
        self.reward_log["xy_align_reward"] = float(xy_align_reward.mean())
        self.reward_log["z_align_reward"] = float(z_align_reward.mean())
        self.reward_log["grasp_bonus"] = float(grasp_bonus.mean())
        self.reward_log["lift_reward"] = float(lift_reward.mean())
        self.reward_log["success_rate"] = float(success.mean())
        self.reward_log["close_reward"] = float(close_reward.mean())
        self.reward_log["camera_penalty"] = float(camera_penalty.mean())

        return reward

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_height = self.obj_pos_w[:, 2]

        terminated = (
            (obj_height > self.cfg.lift_height_threshold)
            | (obj_height < -0.1)
        )
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._prev_dist[env_ids_t] = 0.0

        joint_pos = self._robot.data.default_joint_pos[env_ids_t].clone()
        joint_pos = torch.clamp(
            joint_pos,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )
        joint_vel = torch.zeros_like(joint_pos)

        self.robot_dof_targets[env_ids_t] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)

        obj_state = self._object.data.default_root_state[env_ids_t].clone()
        noise = (
            (torch.rand((len(env_ids_t), 2), device=self.device) - 0.5)
            * 2.0
            * self.cfg.object_pos_noise
        )
        env_origins = self.scene.env_origins[env_ids_t]

        obj_state[:, 0] = env_origins[:, 0] + 0.50 + noise[:, 0]
        obj_state[:, 1] = env_origins[:, 1] + 0.00 + noise[:, 1]
        obj_state[:, 2] = env_origins[:, 2] + 0.022
        obj_state[:, 7:] = 0.0
        self._object.write_root_state_to_sim(obj_state, env_ids=env_ids_t)

        self._compute_intermediate_values(env_ids_t)

    # ------------------------------------------------------------------
    # 중간값 계산
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        l_pos = self._robot.data.body_pos_w[env_ids, self.left_finger_body_id, :]
        r_pos = self._robot.data.body_pos_w[env_ids, self.right_finger_body_id, :]

        self.grip_center_pos[env_ids] = 0.5 * (l_pos + r_pos)
        self.obj_pos_w[env_ids] = self._object.data.root_pos_w[env_ids]
        self.obj_pos_rel[env_ids] = (
            self.obj_pos_w[env_ids] - self.scene.env_origins[env_ids]
        )
        self.obj_to_grip[env_ids] = (
            self.obj_pos_w[env_ids] - self.grip_center_pos[env_ids]
        )