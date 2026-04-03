from __future__ import annotations

from collections.abc import Sequence

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.sim.spawners.from_files import spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .good_robot_franka_env_cfg import GoodRobotFrankaEnvCfg


class GoodRobotFrankaEnv(DirectRLEnv):
    """
    Good Robot 논문 스타일의 multi-step pick-and-place 환경.

    핵심 아이디어:
    1) 단순히 최종 성공만 보는 것이 아니라
       reach -> grasp -> lift -> transport -> place -> release -> stable
       단계별 진행(progress)을 학습시킨다.
    2) 큰 실패 패널티보다 stage completion / progress reward 위주로 설계한다.
    3) 중간 단계가 무너지지 않도록 이전 단계 달성 여부를 버퍼로 유지한다.
    """

    cfg: GoodRobotFrankaEnvCfg

    def __init__(self, cfg: GoodRobotFrankaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---------------------------------------------------------
        # 1. joint / finger index
        # ---------------------------------------------------------
        self.arm_joint_ids = [self._robot.find_joints(name)[0][0] for name in self.cfg.arm_joint_names]

        self.left_finger_id = self._robot.find_joints(self.cfg.left_finger_joint_name)[0][0]
        self.right_finger_id = self._robot.find_joints(self.cfg.right_finger_joint_name)[0][0]

        self.left_finger_body_id = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_body_id = self._robot.find_bodies("panda_rightfinger")[0][0]

        self.num_arm_dofs = len(self.arm_joint_ids)

        # ---------------------------------------------------------
        # 2. action buffers
        # ---------------------------------------------------------
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.arm_action = torch.zeros((self.num_envs, self.num_arm_dofs), device=self.device)
        self.gripper_action = torch.zeros((self.num_envs, 1), device=self.device)
        self.joint_targets = self._robot.data.default_joint_pos.clone()

        # ---------------------------------------------------------
        # 3. state buffers
        # ---------------------------------------------------------
        self.ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.source_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_pad_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_place_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        self.obj_to_ee = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_to_target = torch.zeros((self.num_envs, 3), device=self.device)

        self.grasped = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # ---------------------------------------------------------
        # stage progress buffers
        # Good Robot 스타일 핵심
        # ---------------------------------------------------------
        self.stage_reached = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.stage_grasped = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.stage_lifted = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.stage_transported = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.stage_placed = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.stage_released = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        self.stable_counter = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    # -------------------------------------------------
    # scene
    # -------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        self._source_object = RigidObject(self.cfg.source_object_cfg)
        self._target_object = RigidObject(self.cfg.target_object_cfg)

        self.scene = InteractiveScene(self.cfg.scene)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["source_object"] = self._source_object
        self.scene.rigid_objects["target_object"] = self._target_object

        spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # -------------------------------------------------
    # action input
    # -------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self.arm_action = actions[:, : self.num_arm_dofs]
        self.gripper_action = actions[:, -1:].clone()

    # -------------------------------------------------
    # apply action
    # -------------------------------------------------
    def _apply_action(self) -> None:
        current_targets = self.joint_targets[:, self.arm_joint_ids]
        new_targets = current_targets + self.cfg.arm_action_scale * self.arm_action
        self.joint_targets[:, self.arm_joint_ids] = new_targets

        close_mask = self.gripper_action.squeeze(-1) > 0.0
        open_mask = ~close_mask

        self.joint_targets[close_mask, self.left_finger_id] = self.cfg.gripper_close_target
        self.joint_targets[close_mask, self.right_finger_id] = self.cfg.gripper_close_target

        self.joint_targets[open_mask, self.left_finger_id] = self.cfg.gripper_open_target
        self.joint_targets[open_mask, self.right_finger_id] = self.cfg.gripper_open_target

        self._robot.set_joint_position_target(self.joint_targets)

    # -------------------------------------------------
    # observations
    # -------------------------------------------------
    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        joint_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self.arm_joint_ids]

        finger_pos = self._robot.data.joint_pos[:, [self.left_finger_id, self.right_finger_id]]
        finger_vel = self._robot.data.joint_vel[:, [self.left_finger_id, self.right_finger_id]]

        obj_lin_vel = self._source_object.data.root_lin_vel_w

        progress_flags = torch.stack(
            [
                self.stage_reached.float(),
                self.stage_grasped.float(),
                self.stage_lifted.float(),
                self.stage_transported.float(),
                self.stage_placed.float(),
                self.stage_released.float(),
            ],
            dim=-1,
        )

        obs = torch.cat(
            [
                joint_pos,               # 7
                joint_vel,               # 7
                finger_pos,              # 2
                finger_vel,              # 2
                self.ee_pos_w,           # 3
                self.source_pos_w,       # 3
                self.target_pad_pos_w,   # 3
                self.obj_to_ee,          # 3
                self.obj_to_target,      # 3
                obj_lin_vel,             # 3
                progress_flags,          # 6
            ],
            dim=-1,
        )
        return {"policy": obs}

    # -------------------------------------------------
    # rewards
    # -------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        dist_obj = torch.norm(self.obj_to_ee, dim=-1)
        dist_target = torch.norm(self.obj_to_target, dim=-1)
        xy_dist = torch.norm(self.obj_to_target[:, :2], dim=-1)
        z_dist = torch.abs(self.obj_to_target[:, 2])

        left_finger_pos = self._robot.data.joint_pos[:, self.left_finger_id]
        right_finger_pos = self._robot.data.joint_pos[:, self.right_finger_id]
        finger_mean = 0.5 * (left_finger_pos + right_finger_pos)

        gripper_closed = finger_mean < 0.025
        gripper_open = finger_mean > 0.03
        near_object = dist_obj < self.cfg.reach_threshold

        self.grasped = gripper_closed & near_object

        # -------------------------------------------------
        # current conditions
        # -------------------------------------------------
        reached_now = near_object
        grasped_now = self.grasped
        lifted_now = self.grasped & (self.source_pos_w[:, 2] > self.cfg.lift_height_threshold)
        transported_now = lifted_now & (xy_dist < self.cfg.transport_xy_threshold)
        placed_now = (xy_dist < self.cfg.place_xy_threshold) & (z_dist < self.cfg.place_z_threshold)
        released_now = placed_now & gripper_open

        stable_now = released_now & placed_now & (
            torch.norm(self._source_object.data.root_lin_vel_w, dim=-1) < 0.05
        )

        # -------------------------------------------------
        # progress preservation
        # 이전 단계를 달성한 env는 유지
        # -------------------------------------------------
        new_stage_reached = self.stage_reached | reached_now
        new_stage_grasped = self.stage_grasped | (new_stage_reached & grasped_now)
        new_stage_lifted = self.stage_lifted | (new_stage_grasped & lifted_now)
        new_stage_transported = self.stage_transported | (new_stage_lifted & transported_now)
        new_stage_placed = self.stage_placed | (new_stage_transported & placed_now)
        new_stage_released = self.stage_released | (new_stage_placed & released_now)

        # stage bonus는 "새로 달성했을 때만"
        stage_bonus = (
            (new_stage_reached & ~self.stage_reached).float()
            + (new_stage_grasped & ~self.stage_grasped).float()
            + (new_stage_lifted & ~self.stage_lifted).float()
            + (new_stage_transported & ~self.stage_transported).float()
            + (new_stage_placed & ~self.stage_placed).float()
            + (new_stage_released & ~self.stage_released).float()
        ) * self.cfg.rew_stage_bonus

        self.stage_reached = new_stage_reached
        self.stage_grasped = new_stage_grasped
        self.stage_lifted = new_stage_lifted
        self.stage_transported = new_stage_transported
        self.stage_placed = new_stage_placed
        self.stage_released = new_stage_released

        # -------------------------------------------------
        # dense progress rewards
        # -------------------------------------------------
        reach_reward = self.cfg.rew_reach * torch.exp(-6.0 * dist_obj)

        grasp_reward = self.cfg.rew_grasp * self.stage_grasped.float()

        lift_gap = torch.clamp(self.cfg.lift_height_threshold - self.source_pos_w[:, 2], min=0.0)
        lift_reward = torch.where(
            self.stage_grasped,
            self.cfg.rew_lift * torch.exp(-25.0 * lift_gap),
            torch.zeros_like(lift_gap),
        )

        transport_reward = torch.where(
            self.stage_lifted,
            self.cfg.rew_transport * torch.exp(-5.0 * dist_target),
            torch.zeros_like(dist_target),
        )

        place_reward = torch.where(
            self.stage_transported,
            self.cfg.rew_place * (torch.exp(-20.0 * xy_dist) + torch.exp(-20.0 * z_dist)) * 0.5,
            torch.zeros_like(xy_dist),
        )

        release_reward = self.cfg.rew_release * self.stage_released.float()

        self.stable_counter = torch.where(
            stable_now & self.stage_released,
            self.stable_counter + 1,
            torch.zeros_like(self.stable_counter),
        )

        stable_reward = self.cfg.rew_stable * (
            self.stable_counter >= self.cfg.stable_steps_required
        ).float()

        # -------------------------------------------------
        # penalties
        # Good Robot 느낌으로 "큰 실패 패널티"는 두지 않음
        # -------------------------------------------------
        action_penalty = self.cfg.rew_action_penalty * torch.sum(self.actions ** 2, dim=-1)
        joint_vel_penalty = self.cfg.rew_joint_vel_penalty * torch.sum(
            self._robot.data.joint_vel[:, self.arm_joint_ids] ** 2,
            dim=-1,
        )

        reward = (
            reach_reward
            + grasp_reward
            + lift_reward
            + transport_reward
            + place_reward
            + release_reward
            + stable_reward
            + stage_bonus
            - action_penalty
            - joint_vel_penalty
        )

        return reward

    # -------------------------------------------------
    # dones
    # -------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        xy_dist = torch.norm(self.obj_to_target[:, :2], dim=-1)
        z_dist = torch.abs(self.obj_to_target[:, 2])

        success = (
            (xy_dist < self.cfg.place_xy_threshold)
            & (z_dist < self.cfg.place_z_threshold)
            & (self.stable_counter >= self.cfg.stable_steps_required)
        )

        object_fallen = self.source_pos_w[:, 2] < 0.0

        terminated = success | object_fallen
        return terminated, time_out

    # -------------------------------------------------
    # reset
    # -------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # robot reset
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

        env_origins = self.scene.env_origins[env_ids]

        # source object reset
        source_state = self._source_object.data.default_root_state[env_ids].clone()
        source_state[:, 0] = env_origins[:, 0] + sample_uniform(
            self.cfg.source_x_range[0], self.cfg.source_x_range[1], (len(env_ids),), self.device
        )
        source_state[:, 1] = env_origins[:, 1] + sample_uniform(
            self.cfg.source_y_range[0], self.cfg.source_y_range[1], (len(env_ids),), self.device
        )
        source_state[:, 2] = env_origins[:, 2] + self.cfg.source_object_z
        source_state[:, 7:13] = 0.0
        self._source_object.write_root_state_to_sim(source_state, env_ids=env_ids)

        # target object reset
        target_state = self._target_object.data.default_root_state[env_ids].clone()
        target_state[:, 0] = env_origins[:, 0] + sample_uniform(
            self.cfg.target_x_range[0], self.cfg.target_x_range[1], (len(env_ids),), self.device
        )
        target_state[:, 1] = env_origins[:, 1] + sample_uniform(
            self.cfg.target_y_range[0], self.cfg.target_y_range[1], (len(env_ids),), self.device
        )
        target_state[:, 2] = env_origins[:, 2] + 0.005
        target_state[:, 7:13] = 0.0
        self._target_object.write_root_state_to_sim(target_state, env_ids=env_ids)

        # progress reset
        self.stage_reached[env_ids] = False
        self.stage_grasped[env_ids] = False
        self.stage_lifted[env_ids] = False
        self.stage_transported[env_ids] = False
        self.stage_placed[env_ids] = False
        self.stage_released[env_ids] = False
        self.stable_counter[env_ids] = 0

        self._compute_intermediate_values()

    # -------------------------------------------------
    # intermediate values
    # -------------------------------------------------
    def _compute_intermediate_values(self):
        left_pos = self._robot.data.body_pos_w[:, self.left_finger_body_id, :]
        right_pos = self._robot.data.body_pos_w[:, self.right_finger_body_id, :]
        self.ee_pos_w = 0.5 * (left_pos + right_pos)

        self.source_pos_w = self._source_object.data.root_pos_w
        self.target_pad_pos_w = self._target_object.data.root_pos_w

        self.target_place_pos_w = self.target_pad_pos_w.clone()
        self.target_place_pos_w[:, 2] = self.cfg.target_place_z

        self.obj_to_ee = self.source_pos_w - self.ee_pos_w
        self.obj_to_target = self.source_pos_w - self.target_place_pos_w