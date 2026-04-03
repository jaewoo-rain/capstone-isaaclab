from __future__ import annotations

from collections.abc import Sequence

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.sim.spawners.from_files import spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .grasp_franka_env_cfg import GraspFrankaEnvCfg


class GraspFrankaEnv(DirectRLEnv):
    """Franka grasp + lift Direct RL environment.

    Grasp RL (SAC/PPO) 기준 핵심 구성:
    - State  : robot joint state + gripper state + end-effector pos + object pos + relative vector
    - Action : continuous joint delta + gripper open/close scalar
    - Reward : reach + grasp + lift - penalties
    """

    cfg: GraspFrankaEnvCfg

    def __init__(self, cfg: GraspFrankaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # -----------------------------
        # joint / body indices
        # -----------------------------
        self.arm_joint_ids = [self._robot.find_joints(name)[0][0] for name in self.cfg.arm_joint_names]
        self.left_finger_id = self._robot.find_joints(self.cfg.left_finger_joint_name)[0][0]
        self.right_finger_id = self._robot.find_joints(self.cfg.right_finger_joint_name)[0][0]
        self.ee_body_id = self._robot.find_bodies(self.cfg.ee_body_name)[0][0]

        self.num_arm_dofs = len(self.arm_joint_ids)

        # -----------------------------
        # buffers
        # -----------------------------
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.arm_action = torch.zeros((self.num_envs, self.num_arm_dofs), device=self.device)
        self.gripper_action = torch.zeros((self.num_envs, 1), device=self.device)

        self.joint_targets = self._robot.data.default_joint_pos.clone()

        self.ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_to_ee = torch.zeros((self.num_envs, 3), device=self.device)
        self.grasped = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    # ---------------------------------------------------------------------
    # scene setup
    # ---------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self._object = RigidObject(self.cfg.object_cfg)

        self.scene = InteractiveScene(self.cfg.scene)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------------------------------------------------
    # action
    # ---------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.arm_action = actions[:, : self.num_arm_dofs]
        self.gripper_action = actions[:, -1:].clone()

    def _apply_action(self) -> None:
        # continuous joint-delta control
        current_targets = self.joint_targets[:, self.arm_joint_ids]
        new_targets = current_targets + self.cfg.arm_action_scale * self.arm_action
        self.joint_targets[:, self.arm_joint_ids] = new_targets

        # gripper scalar: positive -> close, non-positive -> open
        close_mask = self.gripper_action.squeeze(-1) > 0.0
        open_mask = ~close_mask

        self.joint_targets[close_mask, self.left_finger_id] = self.cfg.gripper_close_target
        self.joint_targets[close_mask, self.right_finger_id] = self.cfg.gripper_close_target

        self.joint_targets[open_mask, self.left_finger_id] = self.cfg.gripper_open_target
        self.joint_targets[open_mask, self.right_finger_id] = self.cfg.gripper_open_target

        self._robot.set_joint_position_target(self.joint_targets)

    # ---------------------------------------------------------------------
    # observations
    # State = joint_pos + joint_vel + gripper_pos + gripper_vel + ee_pos + obj_pos + obj_to_ee + obj_lin_vel
    # total = 7 + 7 + 2 + 2 + 3 + 3 + 3 + 3 = 30
    # ---------------------------------------------------------------------
    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        joint_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self.arm_joint_ids]

        finger_pos = self._robot.data.joint_pos[:, [self.left_finger_id, self.right_finger_id]]
        finger_vel = self._robot.data.joint_vel[:, [self.left_finger_id, self.right_finger_id]]

        obj_lin_vel = self._object.data.root_lin_vel_w

        obs = torch.cat(
            [
                joint_pos,       # 7
                joint_vel,       # 7
                finger_pos,      # 2
                finger_vel,      # 2
                self.ee_pos_w,   # 3
                self.obj_pos_w,  # 3
                self.obj_to_ee,  # 3
                obj_lin_vel,     # 3
            ],
            dim=-1,
        )

        return {"policy": obs}

    # ---------------------------------------------------------------------
    # rewards
    # Reward = reach + grasp + lift - action_penalty - joint_vel_penalty
    # ---------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        dist = torch.norm(self.obj_to_ee, dim=-1)

        # 1) reach
        reach_reward = torch.exp(-6.0 * dist)

        # 2) grasp
        left_finger_pos = self._robot.data.joint_pos[:, self.left_finger_id]
        right_finger_pos = self._robot.data.joint_pos[:, self.right_finger_id]
        finger_mean = 0.5 * (left_finger_pos + right_finger_pos)

        gripper_closed = finger_mean < 0.025
        near_object = dist < 0.04

        self.grasped = gripper_closed & near_object
        grasp_reward = self.cfg.rew_grasp * self.grasped.float()

        # 3) lift
        obj_height = self.obj_pos_w[:, 2]
        lifted_height = torch.clamp(obj_height - self.cfg.object_z, min=0.0)
        lift_reward = torch.where(
            self.grasped,
            self.cfg.rew_lift * lifted_height,
            torch.zeros_like(lifted_height),
        )

        action_penalty = (-self.cfg.rew_action_penalty) * torch.sum(self.actions**2, dim=-1)
        joint_vel_penalty = (-self.cfg.rew_joint_vel_penalty) * torch.sum(
            self._robot.data.joint_vel[:, self.arm_joint_ids] ** 2,
            dim=-1,
        )

        reward = (
            self.cfg.rew_reach * reach_reward
            + grasp_reward
            + lift_reward
            - action_penalty
            - joint_vel_penalty
        )
        return reward

    # ---------------------------------------------------------------------
    # dones
    # Grasp RL 초기 단계에서는 너무 공격적인 종료를 피함
    # ---------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        obj_height = self.obj_pos_w[:, 2]
        success = obj_height > self.cfg.success_lift_height
        object_fallen = obj_height < 0.0

        terminated = success | object_fallen
        return terminated, time_out

    # ---------------------------------------------------------------------
    # reset
    # object는 각 env origin 기준 local randomization
    # ---------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # -----------------------------
        # robot reset
        # -----------------------------
        default_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        default_joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        noise = sample_uniform(
            -0.05,
            0.05,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = default_joint_pos + noise
        joint_vel = default_joint_vel

        self.joint_targets[env_ids] = joint_pos
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        # -----------------------------
        # object reset
        # -----------------------------
        object_default_state = self._object.data.default_root_state[env_ids].clone()
        env_origins = self.scene.env_origins[env_ids]

        rand_x = sample_uniform(
            self.cfg.object_x_range[0],
            self.cfg.object_x_range[1],
            (len(env_ids),),
            self.device,
        )
        rand_y = sample_uniform(
            self.cfg.object_y_range[0],
            self.cfg.object_y_range[1],
            (len(env_ids),),
            self.device,
        )

        # world pos = env origin + local offset
        object_default_state[:, 0] = env_origins[:, 0] + rand_x
        object_default_state[:, 1] = env_origins[:, 1] + rand_y
        object_default_state[:, 2] = env_origins[:, 2] + self.cfg.object_z

        # keep quaternion as default, zero out linear/angular velocities
        object_default_state[:, 7:13] = 0.0

        self._object.write_root_state_to_sim(object_default_state, env_ids=env_ids)

        self._compute_intermediate_values()

    # ---------------------------------------------------------------------
    # intermediate values
    # ---------------------------------------------------------------------
    def _compute_intermediate_values(self):
        self.ee_pos_w = self._robot.data.body_pos_w[:, self.ee_body_id, :]
        self.obj_pos_w = self._object.data.root_pos_w
        self.obj_to_ee = self.obj_pos_w - self.ee_pos_w