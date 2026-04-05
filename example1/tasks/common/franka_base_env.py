from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .franka_base_env_cfg import FrankaBaseEnvCfg


class FrankaBaseEnv(DirectRLEnv):
    cfg: FrankaBaseEnvCfg

    def __init__(self, cfg: FrankaBaseEnvCfg, render_mode: str | None = None, **kwargs):
        self._robot = None
        self._object = None
        self._actions = None

        super().__init__(cfg, render_mode, **kwargs)

        # Franka joint ids
        self._arm_joint_ids = torch.arange(7, device=self.device)
        self._finger_joint_ids = torch.tensor([7, 8], device=self.device)

        # body ids
        # body names may vary slightly by version; these are common Franka names
        self._ee_body_idx = self._robot.find_bodies("panda_hand")[0][0]
        self._left_finger_body_idx = self._robot.find_bodies("panda_leftfinger")[0][0]
        self._right_finger_body_idx = self._robot.find_bodies("panda_rightfinger")[0][0]

        self._actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.object)

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        if self.cfg.ground is not None:
            self.cfg.ground.func(
                "/World/defaultGroundPlane",
                self.cfg.ground,
            )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self):
        joint_pos = self._robot.data.joint_pos.clone()

        # arm: delta joint control
        joint_pos[:, :7] += self._actions[:, :7] * self.cfg.arm_action_scale

        # clamp by soft limits if available
        lower_limits = self._robot.data.soft_joint_pos_limits[:, :, 0]
        upper_limits = self._robot.data.soft_joint_pos_limits[:, :, 1]
        joint_pos = torch.clamp(joint_pos, lower_limits, upper_limits)

        # gripper
        close_mask = self._actions[:, 7] > 0.0
        joint_pos[close_mask, 7] = self.cfg.gripper_close_pos
        joint_pos[close_mask, 8] = self.cfg.gripper_close_pos
        joint_pos[~close_mask, 7] = self.cfg.gripper_open_pos
        joint_pos[~close_mask, 8] = self.cfg.gripper_open_pos

        self._robot.set_joint_position_target(joint_pos)

    def _get_observations(self) -> dict:
        joint_pos = self._robot.data.joint_pos[:, :7]
        joint_vel = self._robot.data.joint_vel[:, :7]

        obj_pos = self._object.data.root_pos_w
        ee_pos = self._robot.data.body_pos_w[:, self._ee_body_idx, :]
        rel_obj_pos = obj_pos - ee_pos

        obs = torch.cat(
            [
                joint_pos,
                joint_vel,
                rel_obj_pos,
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)

        # -----------------------------
        # robot reset
        # -----------------------------
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        joint_pos[:, 0] = 0.0
        joint_pos[:, 1] = -0.6
        joint_pos[:, 2] = 0.0
        joint_pos[:, 3] = -2.0
        joint_pos[:, 4] = 0.0
        joint_pos[:, 5] = 1.6
        joint_pos[:, 6] = 0.8
        joint_pos[:, 7] = self.cfg.gripper_open_pos
        joint_pos[:, 8] = self.cfg.gripper_open_pos

        joint_vel[:] = 0.0

        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        # -----------------------------
        # object reset
        # 각 env origin 기준으로 로봇 앞에 배치
        # -----------------------------
        obj_state = self._object.data.default_root_state[env_ids].clone()
        env_origins = self.scene.env_origins[env_ids]

        obj_state[:, 0] = env_origins[:, 0] + 0.55
        obj_state[:, 1] = env_origins[:, 1] + 0.00
        obj_state[:, 2] = env_origins[:, 2] + 0.025

        obj_state[:, 0] += torch.empty(len(env_ids), device=self.device).uniform_(
            -self.cfg.object_pos_noise_x, self.cfg.object_pos_noise_x
        )
        obj_state[:, 1] += torch.empty(len(env_ids), device=self.device).uniform_(
            -self.cfg.object_pos_noise_y, self.cfg.object_pos_noise_y
        )

        # quaternion
        obj_state[:, 3] = 1.0
        obj_state[:, 4] = 0.0
        obj_state[:, 5] = 0.0
        obj_state[:, 6] = 0.0

        # zero velocity
        obj_state[:, 7:] = 0.0

        self._object.write_root_state_to_sim(obj_state, env_ids=env_ids)

    # helper methods
    def _get_obj_pos(self):
        return self._object.data.root_pos_w

    def _get_ee_pos(self):
        return self._robot.data.body_pos_w[:, self._ee_body_idx, :]

    def _get_finger_positions(self):
        left_finger_pos = self._robot.data.body_pos_w[:, self._left_finger_body_idx, :]
        right_finger_pos = self._robot.data.body_pos_w[:, self._right_finger_body_idx, :]
        return left_finger_pos, right_finger_pos