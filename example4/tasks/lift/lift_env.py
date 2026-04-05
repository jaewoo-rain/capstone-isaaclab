
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

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.observation_space,), dtype=np.float32
        )

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        lf_idx = self._robot.find_joints("panda_finger_joint1")[0][0]
        rf_idx = self._robot.find_joints("panda_finger_joint2")[0][0]

        # 집게 속도
        self.robot_dof_speed_scales[lf_idx] = 0.1 
        self.robot_dof_speed_scales[rf_idx] = 0.1

        self.left_finger_joint_id  = lf_idx
        self.right_finger_joint_id = rf_idx
        self.left_finger_body_id   = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_body_id  = self._robot.find_bodies("panda_rightfinger")[0][0]

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )

        self.obj_pos_w       = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_pos_rel     = torch.zeros((self.num_envs, 3), device=self.device)
        self.grip_center_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_to_grip     = torch.zeros((self.num_envs, 3), device=self.device)

        self.actions    = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)

        # ── reward 세부 항목 공유 버퍼 (callback에서 직접 읽음) ──
        self.reward_log = {
            # "dist_reward":     0.0,
            "approach_reward": 0.0,
            "grasp_bonus":     0.0,
            "lift_reward":     0.0,
            "success_rate":    0.0,
        }

        self._compute_intermediate_values()

    # ------------------------------------------------------------------
    # Scene 설정
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot  = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.object)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        self.cfg.terrain.num_envs    = self.scene.cfg.num_envs
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
        dof_delta[:, :7]                         = actions[:, :7]
        dof_delta[:, self.left_finger_joint_id]  = actions[:, 7]
        dof_delta[:, self.right_finger_joint_id] = actions[:, 7]
        return dof_delta

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits)
            - 1.0
        )
        dof_vel_scaled = self._robot.data.joint_vel * self.cfg.dof_velocity_scale

        gripper_width = (
            self._robot.data.joint_pos[:, self.left_finger_joint_id]
            + self._robot.data.joint_pos[:, self.right_finger_joint_id]
        ).unsqueeze(-1)

        to_lift_target = (
            self.cfg.lift_height_threshold - self.obj_pos_w[:, 2]
        ).unsqueeze(-1)

        obs = torch.cat(
            [
                dof_pos_scaled,
                dof_vel_scaled,
                self.obj_pos_rel,
                self.obj_to_grip,
                gripper_width,
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

        # --------------------------------------------------
        # 1) 기본 위치 정보
        # --------------------------------------------------
        obj_pos = self.obj_pos_w
        grip_pos = self.grip_center_pos

        l_pos = self._robot.data.body_pos_w[:, self.left_finger_body_id, :]
        r_pos = self._robot.data.body_pos_w[:, self.right_finger_body_id, :]

        gripper_width = (
            self._robot.data.joint_pos[:, self.left_finger_joint_id]
            + self._robot.data.joint_pos[:, self.right_finger_joint_id]
        )

        obj_to_grip = obj_pos - grip_pos
        dist = torch.norm(obj_to_grip, dim=-1)

        # xy / z 분리
        xy_dist = torch.norm(obj_to_grip[:, :2], dim=-1)
        z_dist = torch.abs(obj_to_grip[:, 2])

        # --------------------------------------------------
        # 2) 거리 보상
        # - 전체적으로 물체 중심 쪽으로 가게 함
        # --------------------------------------------------
        # dist_reward = 1.0 / (1.0 + 8.0 * dist**2)

        # --------------------------------------------------
        # 3) 접근 진행 보상
        # - 이전 step보다 가까워졌을 때만 보상
        # --------------------------------------------------
        approach_reward = torch.clamp(self._prev_dist - dist, min=0.0) * 8.0
        self._prev_dist = dist.clone()

        # --------------------------------------------------
        # 4) 정렬 보상
        # - xy 중심 정렬
        # - z 높이 정렬
        # --------------------------------------------------
        xy_align_reward = torch.exp(-40.0 * xy_dist**2)
        z_align_reward = torch.exp(-60.0 * z_dist**2)

        # 정렬 완료 판정
        xy_aligned = xy_dist < 0.05
        z_aligned = z_dist < 0.05
        aligned = xy_aligned & z_aligned

        # --------------------------------------------------
        # 5) 손가락 양옆 배치 체크
        # - 현재 네 환경에서는 panda hand가 물체를 좌우로 집는 구조라고 가정
        # - y축 기준으로 물체를 사이에 두고 양쪽 finger가 있어야 함
        # --------------------------------------------------
        left_is_left = l_pos[:, 1] < obj_pos[:, 1]
        right_is_right = r_pos[:, 1] > obj_pos[:, 1]

        # 양쪽 finger가 물체 가까이 있는지도 같이 확인
        left_to_obj = torch.norm(obj_pos - l_pos, dim=-1)
        right_to_obj = torch.norm(obj_pos - r_pos, dim=-1)

        fingers_near = (left_to_obj < 0.05) & (right_to_obj < 0.05)

        side_ok = left_is_left & right_is_right

        # if aligned:
        #     print("===============================")
        #     print(f"aligned = ${aligned}")
        # if fingers_near:
        #     print("===============================")
        #     print(f"fingers_near = ${fingers_near}")

        pre_grasp_ready = aligned & fingers_near & side_ok

        # pre-grasp 정렬 보상
        # pre_grasp_reward = pre_grasp_ready.float() * 2.0

        # --------------------------------------------------
        # 6) 진짜 grasp 보상
        # - 정렬된 상태 + 양옆 finger 배치 + gripper가 닫힘
        # --------------------------------------------------
        closed_enough = gripper_width < 0.055
        is_grasping = pre_grasp_ready & closed_enough
        grasp_bonus = is_grasping.float() * 4.0

        # 닫는 과정에서 리워드
        # close_reward = pre_grasp_ready.float() * torch.clamp(0.08 - gripper_width, min=0.0) * 10.0

        alignment_score = torch.exp(-40.0 * xy_dist**2) * torch.exp(-60.0 * z_dist**2)

        close_reward = alignment_score * torch.clamp(0.08 - gripper_width, min=0.0) * 20.0


        # --------------------------------------------------
        # 7) 너무 일찍 닫는 행동 패널티
        # - 정렬 안 됐는데 닫으면 패널티
        # --------------------------------------------------
        # premature_close = (~pre_grasp_ready) & (gripper_width < 0.055)
        # close_penalty = premature_close.float() * 1.5

        # --------------------------------------------------
        # 8) lift 보상
        # - 그냥 들어올린 것보다
        # - "잡은 뒤 들어올림"에 더 의미를 둠
        # --------------------------------------------------
        obj_height = obj_pos[:, 2]

        raw_lift = torch.clamp(obj_height - 0.04, min=0.0)
        lift_reward = raw_lift * (1.0 + 2.0 * is_grasping.float()) * 12.0

        # --------------------------------------------------
        # 9) 성공 보너스
        # --------------------------------------------------
        success = (obj_height > self.cfg.lift_height_threshold).float()
        success_reward = success * self.cfg.success_bonus

        # --------------------------------------------------
        # 10) 액션 패널티
        # --------------------------------------------------
        # action_penalty = torch.sum(self.actions**2, dim=-1) * 0.001
        # 집게 빼고 패널티 부여
        action_penalty = torch.sum(self.actions[:, :7]**2, dim=-1) * 0.001



        # --------------------------------------------------
        # 11) 최종 reward
        # --------------------------------------------------
        reward = (
            # 1.2 * dist_reward
            + approach_reward
            # + 1.5 * xy_align_reward
            # + 1.2 * z_align_reward
            # + pre_grasp_reward
            + grasp_bonus
            + lift_reward
            + success_reward
            + close_reward
            # - close_penalty
            - action_penalty
        )

        # --------------------------------------------------
        # 12) 로그
        # --------------------------------------------------
        self.extras["log"] = {
            # "dist_reward": dist_reward.mean(),
            "approach_reward": approach_reward.mean(),
            "xy_align_reward": xy_align_reward.mean(),
            "z_align_reward": z_align_reward.mean(),
            # "pre_grasp_reward": pre_grasp_reward.mean(),
            "grasp_bonus": grasp_bonus.mean(),
            "lift_reward": lift_reward.mean(),
            # "close_penalty": close_penalty.mean(),
            "success_rate": success.mean(),
            "xy_dist": xy_dist.mean(),
            "z_dist": z_dist.mean(),
            "gripper_width": gripper_width.mean(),
        }

        # self.reward_log["dist_reward"] = float(dist_reward.mean())
        self.reward_log["approach_reward"] = float(approach_reward.mean())
        self.reward_log["xy_align_reward"] = float(xy_align_reward.mean())
        self.reward_log["z_align_reward"] = float(z_align_reward.mean())
        # self.reward_log["pre_grasp_reward"] = float(pre_grasp_reward.mean())
        self.reward_log["grasp_bonus"] = float(grasp_bonus.mean())
        self.reward_log["lift_reward"] = float(lift_reward.mean())
        self.reward_log["success_rate"] = float(success.mean())
        return reward

    # ------------------------------------------------------------------
    # Dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_height = self.obj_pos_w[:, 2]
        terminated = (obj_height > self.cfg.lift_height_threshold) | (obj_height < -0.1)
        truncated  = self.episode_length_buf >= self.max_episode_length - 1
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
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        self.robot_dof_targets[env_ids_t] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)

        obj_state = self._object.data.default_root_state[env_ids_t].clone()
        noise = (torch.rand((len(env_ids_t), 2), device=self.device) - 0.5) * 2.0 * self.cfg.object_pos_noise
        env_origins = self.scene.env_origins[env_ids_t]
        obj_state[:, 0] = env_origins[:, 0] + 0.5 + noise[:, 0]
        obj_state[:, 1] = env_origins[:, 1] + 0.0 + noise[:, 1]
        obj_state[:, 2] = env_origins[:, 2] + 0.02
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
        self.obj_pos_w[env_ids]       = self._object.data.root_pos_w[env_ids]
        self.obj_pos_rel[env_ids]     = self.obj_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.obj_to_grip[env_ids]     = self.obj_pos_w[env_ids] - self.grip_center_pos[env_ids]