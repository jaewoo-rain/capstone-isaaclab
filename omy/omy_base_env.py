"""
공통 로직 담당

역할:

action → joint 변환
observation 구성
EE / object / finger 계산
reset 기본 구조

-> 모든 env의 뿌리
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera

from .omy_env_cfg import OmyLiftEnvCfg


class OmyBaseEnv(DirectRLEnv):
    cfg: OmyLiftEnvCfg

    def __init__(self, cfg: OmyLiftEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.cfg.observation_space,), dtype=np.float32
        )

        # joint limits
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)

        # joint ids
        self.arm_joint_ids = [self._robot.find_joints(f"joint{i}")[0][0] for i in range(1, 7)]

        self.gripper_master_joint_id = self._robot.find_joints(self.cfg.gripper_master_joint_name)[0][0]

        self.left_finger_joint_id = self._robot.find_joints("rh_l1")[0][0]
        self.right_finger_joint_id = self._robot.find_joints("rh_r1_joint")[0][0]
        self.left_tip_joint_id = self._robot.find_joints("rh_l2")[0][0]
        self.right_tip_joint_id = self._robot.find_joints("rh_r2")[0][0]

        # body ids
        self.ee_body_id = self._robot.find_bodies(self.cfg.ee_body_name)[0][0]

        self.left_finger_body_id = self._robot.find_bodies(self.cfg.left_finger_body_name)[0][0]
        self.right_finger_body_id = self._robot.find_bodies(self.cfg.right_finger_body_name)[0][0]

        self.left_tip_body_id = self._robot.find_bodies(self.cfg.left_tip_body_name)[0][0]
        self.right_tip_body_id = self._robot.find_bodies(self.cfg.right_tip_body_name)[0][0]

        # dt
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # speed scale
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self.gripper_master_joint_id] = 0.2

        # targets
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints), device=self.device
        )

        # buffers
        self.obj_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_pos_rel = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_to_ee = torch.zeros((self.num_envs, 3), device=self.device)

        self.left_finger_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_finger_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_tip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_tip_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)

        self.reward_log = {
            "approach_reward": 0.0,
            "grasp_bonus": 0.0,
            "lift_reward": 0.0,
            "success_rate": 0.0,
            "close_reward": 0.0,
        }

        self._compute_intermediate_values()

    # ------------------------------------------------------------------
    # scene
    # ------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.object)
        self._camera = Camera(self.cfg.camera)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object
        self.scene.sensors["camera"] = self._camera

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # action
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

        # gripper 1축 -> mimic 구조상 대표 조인트만 제어
        dof_delta[:, self.gripper_master_joint_id] = actions[:, 6]

        return dof_delta

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.robot_dof_targets)

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        dof_pos_scaled = (
            2.0
            * (self._robot.data.joint_pos - self.robot_dof_lower_limits)
            / (self.robot_dof_upper_limits - self.robot_dof_lower_limits + 1e-8)
            - 1.0
        )

        dof_vel_scaled = self._robot.data.joint_vel * self.cfg.dof_velocity_scale

        gripper_joint = self._robot.data.joint_pos[:, self.gripper_master_joint_id].unsqueeze(-1)

        to_lift_target = (
            self.cfg.lift_height_threshold - self.obj_pos_w[:, 2]
        ).unsqueeze(-1)

        obs = torch.cat(
            [
                dof_pos_scaled,      # 10
                dof_vel_scaled,      # 10
                self.obj_pos_rel,    # 3
                self.obj_to_ee,      # 3
                gripper_joint,       # 1
                to_lift_target,      # 1
            ],
            dim=-1,
        )

        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._prev_dist[env_ids_t] = 0.0

        # default joint pose
        joint_pos = self._robot.data.default_joint_pos[env_ids_t].clone()
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        self.robot_dof_targets[env_ids_t] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)

        # object reset
        obj_state = self._object.data.default_root_state[env_ids_t].clone()
        noise = (torch.rand((len(env_ids_t), 2), device=self.device) - 0.5) * 2.0 * self.cfg.object_pos_noise
        env_origins = self.scene.env_origins[env_ids_t]

        obj_state[:, 0] = env_origins[:, 0] + 0.45 + noise[:, 0]
        obj_state[:, 1] = env_origins[:, 1] - 0.10 + noise[:, 1]
        obj_state[:, 2] = env_origins[:, 2] + 0.02
        obj_state[:, 7:] = 0.0

        self._object.write_root_state_to_sim(obj_state, env_ids=env_ids_t)

        self._compute_intermediate_values(env_ids_t)

    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.ee_pos_w[env_ids] = self._robot.data.body_pos_w[env_ids, self.ee_body_id, :]
        self.obj_pos_w[env_ids] = self._object.data.root_pos_w[env_ids]
        self.obj_pos_rel[env_ids] = self.obj_pos_w[env_ids] - self.scene.env_origins[env_ids]
        self.obj_to_ee[env_ids] = self.obj_pos_w[env_ids] - self.ee_pos_w[env_ids]

        self.left_finger_pos[env_ids] = self._robot.data.body_pos_w[env_ids, self.left_finger_body_id, :]
        self.right_finger_pos[env_ids] = self._robot.data.body_pos_w[env_ids, self.right_finger_body_id, :]
        self.left_tip_pos[env_ids] = self._robot.data.body_pos_w[env_ids, self.left_tip_body_id, :]
        self.right_tip_pos[env_ids] = self._robot.data.body_pos_w[env_ids, self.right_tip_body_id, :]

    def _get_ee_pos(self) -> torch.Tensor:
        return self.ee_pos_w

    def _get_obj_pos(self) -> torch.Tensor:
        return self.obj_pos_w

    def _get_obj_height(self) -> torch.Tensor:
        return self.obj_pos_w[:, 2]

    def _get_gripper_joint(self) -> torch.Tensor:
        return self._robot.data.joint_pos[:, self.gripper_master_joint_id]

    def _get_finger_center(self) -> torch.Tensor:
        return 0.5 * (self.left_finger_pos + self.right_finger_pos)

    def _get_tip_center(self) -> torch.Tensor:
        return 0.5 * (self.left_tip_pos + self.right_tip_pos)

    def _get_common_terms(self):
        self._compute_intermediate_values()

        obj_pos = self.obj_pos_w
        ee_pos = self.ee_pos_w

        finger_center = self._get_finger_center()
        tip_center = self._get_tip_center()

        gripper_joint = self._get_gripper_joint()

        obj_to_ee = obj_pos - ee_pos
        dist = torch.norm(obj_to_ee, dim=-1)

        obj_to_tip = obj_pos - tip_center
        xy_dist = torch.norm(obj_to_tip[:, :2], dim=-1)
        z_dist = torch.abs(obj_to_tip[:, 2])

        approach_reward = torch.clamp(self._prev_dist - dist, min=0.0)
        self._prev_dist = dist.clone()

        xy_align_reward = torch.exp(-40.0 * xy_dist**2)
        z_align_reward = torch.exp(-60.0 * z_dist**2)

        xy_aligned = xy_dist < 0.05
        z_aligned = z_dist < 0.05
        aligned = xy_aligned & z_aligned

        left_is_left = self.left_tip_pos[:, 1] < obj_pos[:, 1]
        right_is_right = self.right_tip_pos[:, 1] > obj_pos[:, 1]

        left_to_obj = torch.norm(obj_pos - self.left_tip_pos, dim=-1)
        right_to_obj = torch.norm(obj_pos - self.right_tip_pos, dim=-1)

        fingers_near = (left_to_obj < 0.05) & (right_to_obj < 0.05)
        side_ok = left_is_left & right_is_right

        pre_grasp_ready = aligned & fingers_near & side_ok

        # gripper master joint가 커질수록 닫히는 방향이라고 가정
        # 조정 : 바꿔야함
        closed_enough = gripper_joint > 0.3
        is_grasping = pre_grasp_ready & closed_enough

        return {
            "obj_pos": obj_pos,
            "ee_pos": ee_pos,
            "finger_center": finger_center,
            "tip_center": tip_center,
            "gripper_joint": gripper_joint,
            "dist": dist,
            "xy_dist": xy_dist,
            "z_dist": z_dist,
            "approach_reward": approach_reward,
            "xy_align_reward": xy_align_reward,
            "z_align_reward": z_align_reward,
            "pre_grasp_ready": pre_grasp_ready,
            "is_grasping": is_grasping,
        }

    def _get_camera_rgb(self) -> torch.Tensor | None:
        if "rgb" in self._camera.data.output:
            return self._camera.data.output["rgb"]
        return None

    # ------------------------------------------------------------------
    # dones
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        obj_height = self.obj_pos_w[:, 2]
        terminated = (obj_height > self.cfg.lift_height_threshold) | (obj_height < -0.1)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------------
    # rewards
    # 자식 클래스에서 override
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        raise NotImplementedError