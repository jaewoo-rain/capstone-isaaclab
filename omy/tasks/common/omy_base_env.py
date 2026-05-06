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
from isaaclab.utils.math import quat_apply

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

        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        # self._camera = None
        # if self.cfg.use_camera:
        #     self._camera = Camera(self.cfg.camera)
        #     self.scene.sensors["camera"] = self._camera

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
    # def _reset_idx(self, env_ids: Sequence[int] | None):
    #     if env_ids is None:
    #         env_ids = self._robot._ALL_INDICES

    #     super()._reset_idx(env_ids)
    #     env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    #     self._prev_dist[env_ids_t] = 0.0

    #     # default joint pose
    #     joint_pos = self._robot.data.default_joint_pos[env_ids_t].clone()
    #     joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
    #     joint_vel = torch.zeros_like(joint_pos)

    #     self.robot_dof_targets[env_ids_t] = joint_pos
    #     self._robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
    #     self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)

    #     # object reset
    #     # OmyBaseEnv._reset_idx()

    #     obj_state = self._object.data.default_root_state[env_ids_t].clone()
    #     noise = (torch.rand((len(env_ids_t), 2), device=self.device) - 0.5) * 2.0 * self.cfg.object_pos_noise
    #     env_origins = self.scene.env_origins[env_ids_t]

    #     base_x, base_y, base_z = self.cfg.object.init_state.pos

    #     # obj_state[:, 0] = env_origins[:, 0] + base_x + noise[:, 0]
    #     # obj_state[:, 1] = env_origins[:, 1] + base_y + noise[:, 1]

    #     obj_state[:, 0] = env_origins[:, 0] + base_x
    #     obj_state[:, 1] = env_origins[:, 1] + base_y
    #     obj_state[:, 2] = env_origins[:, 2] + base_z
    #     obj_state[:, 7:] = 0.0

    #     self._object.write_root_state_to_sim(obj_state, env_ids=env_ids_t)

    #     self._compute_intermediate_values(env_ids_t)
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        super()._reset_idx(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._prev_dist[env_ids_t] = 0.0

        # 1. 기본 로봇 reset
        joint_pos = self._robot.data.default_joint_pos[env_ids_t].clone()
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)

        self.robot_dof_targets[env_ids_t] = joint_pos
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)

        # 2. 기본 물체 reset
        obj_state = self._object.data.default_root_state[env_ids_t].clone()
        env_origins = self.scene.env_origins[env_ids_t]
        base_x, base_y, base_z = self.cfg.object.init_state.pos

        obj_state[:, 0] = env_origins[:, 0] + base_x
        obj_state[:, 1] = env_origins[:, 1] + base_y
        obj_state[:, 2] = env_origins[:, 2] + base_z
        obj_state[:, 7:] = 0.0

        self._object.write_root_state_to_sim(obj_state, env_ids=env_ids_t)

        # 3. task별 reset 덮어쓰기
        self._task_specific_reset(env_ids_t)

        self._compute_intermediate_values(env_ids_t)

    def _task_specific_reset(self, env_ids: torch.Tensor) -> None:
        pass
    # ------------------------------------------------------------------
    # helper
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.ee_pos_w[env_ids] = self._robot.data.body_pos_w[env_ids, self.ee_body_id, :]
        self.obj_pos_w[env_ids] = self._object.data.root_pos_w[env_ids]
        self.obj_pos_rel[env_ids] = self.obj_pos_w[env_ids] - self.scene.env_origins[env_ids]

        # self.obj_to_ee[env_ids] = self.obj_pos_w[env_ids] - self.ee_pos_w[env_ids]
        obj_pos = self.obj_pos_w[env_ids]
        obj_quat = self._object.data.root_quat_w[env_ids]

        local_offset = torch.zeros((obj_pos.shape[0], 3), device=self.device)
        local_offset[:, 2] = self.cfg.grasp_target_z_offset

        world_offset = quat_apply(obj_quat, local_offset)
        grasp_target_pos = obj_pos + world_offset

        self.obj_to_ee[env_ids] = grasp_target_pos - self.ee_pos_w[env_ids]


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
        # 최신 상태 (EE, object, finger 위치 등) 갱신
        self._compute_intermediate_values()

        # -------------------------
        # 1. 기본 위치 정보
        # -------------------------
        obj_pos = self.obj_pos_w        # 물체의 월드 좌표 (x, y, z)
        ee_pos = self.ee_pos_w          # 로봇 end-effector 위치

        obj_quat = self._object.data.root_quat_w

        local_offset = torch.zeros((obj_pos.shape[0], 3), device=self.device)
        local_offset[:, 2] = self.cfg.grasp_target_z_offset

        world_offset = quat_apply(obj_quat, local_offset)
        grasp_target_pos = obj_pos + world_offset

        # -------------------------
        # 2. 그리퍼 중심 / 끝 위치
        # -------------------------
        finger_center = self._get_finger_center()   # 손가락 두 개의 중간 지점
        tip_center = self._get_tip_center()         # 손가락 끝(tip) 기준 중심

        # -------------------------
        # 3. 그리퍼 상태
        # -------------------------
        gripper_joint = self._get_gripper_joint()   # gripper 열림/닫힘 정도 (joint 값)

        # if torch.rand(1).item() < 0.001:  # 그리퍼 얼마나 닫히는지 체크
        #     print("gripper range:", gripper_joint.min().item(), gripper_joint.max().item())

        # -------------------------
        # 4. EE ↔ Object 거리
        # -------------------------
        target_to_ee = grasp_target_pos - ee_pos
        dist = torch.norm(target_to_ee, dim=-1)


        # # -------------------------
        # # 5. Tip 기준 정렬 거리
        # # -------------------------
        # 잡을 위치 기준 재정의
        target_to_tip = grasp_target_pos - tip_center
        xy_dist = torch.norm(target_to_tip[:, :2], dim=-1)
        z_dist = torch.abs(target_to_tip[:, 2])

        # -------------------------
        # 6. 접근 보상 (progress 기반)
        # -------------------------
        # 이전 step보다 가까워졌으면 +reward
        approach_reward = torch.clamp(self._prev_dist - dist, min=0.0)

        # 현재 거리 저장 (다음 step에서 비교용)
        self._prev_dist = dist.clone()

        # -------------------------
        # 7. 정렬 보상 (continuous)
        # -------------------------
        # 거리가 가까울수록 1에 가까워지는 exp 함수
        xy_align_reward = torch.exp(-60.0 * xy_dist**2)
        z_align_reward = torch.exp(-60.0 * z_dist**2) * xy_align_reward

        # -------------------------
        # 8. 정렬 조건 (binary)
        # -------------------------
        xy_aligned = xy_dist < 0.05     # XY 기준 충분히 가까움
        z_aligned = z_dist < 0.08       # Z 기준 충분히 가까움

        aligned = xy_aligned & z_aligned   # 둘 다 만족해야 정렬 완료

        # -------------------------
        # 9. 손가락 위치 관계 체크
        # -------------------------
        # 물체 기준으로 왼쪽/오른쪽에 finger가 있는지 확인

        left_is_left = self.left_tip_pos[:, 1] < obj_pos[:, 1]
        right_is_right = self.right_tip_pos[:, 1] > obj_pos[:, 1]

        # 👉 의미:
        # 물체 기준 Y축으로
        # 왼손은 왼쪽, 오른손은 오른쪽에 있어야 제대로 잡는 구조

        # -------------------------
        # 10. finger ↔ object 거리
        # -------------------------
        left_to_obj = torch.norm(obj_pos - self.left_tip_pos, dim=-1)
        right_to_obj = torch.norm(obj_pos - self.right_tip_pos, dim=-1)

        # 양쪽 finger가 모두 물체 근처에 있어야 함
        fingers_near = (left_to_obj < 0.05) & (right_to_obj < 0.05)

        # 좌우 위치 조건까지 만족해야 함
        side_ok = left_is_left & right_is_right

        # -------------------------
        # 11. pre-grasp 상태 정의
        # -------------------------
        # "잡기 직전 상태"
        # 조건:
        # - 정렬됨
        # - 손가락 둘 다 근처
        # - 좌우 위치 올바름
        pre_grasp_ready = aligned & side_ok

        # -------------------------
        # 12. 그리퍼 닫힘 조건
        # -------------------------
        # joint 값이 클수록 닫힌 상태라고 가정
        closed_enough = gripper_joint > 0.3
        closed_enough = (left_to_obj < 0.05) & (right_to_obj < 0.05)


        # -------------------------
        # 13. grasp 판정
        # -------------------------
        # 잡았다 = pre-grasp 상태 + 그리퍼 닫힘
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
            "aligned": aligned,
        }

    # def _get_camera_rgb(self) -> torch.Tensor | None:
    #     if self._camera is None:
    #         return None
    #     if "rgb" in self._camera.data.output:
    #         return self._camera.data.output["rgb"]
    #     return None

    # ------------------------------------------------------------------
    # dones
    # ------------------------------------------------------------------
    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     obj_height = self.obj_pos_w[:, 2]
    #     terminated = (obj_height > self.cfg.lift_height_threshold) | (obj_height < -0.1)
    #     truncated = self.episode_length_buf >= self.max_episode_length - 1
    #     return terminated, truncated
    # 잡기 전용
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._get_common_terms()
        obj_height = self.obj_pos_w[:, 2]

        obj_quat = self._object.data.root_quat_w
        local_up = torch.zeros((obj_quat.shape[0], 3), device=self.device)
        local_up[:, 2] = 1.0

        world_up_from_obj = quat_apply(obj_quat, local_up)
        upright_score = world_up_from_obj[:, 2]
        fallen = upright_score < 0.3

        terminated = t["is_grasping"] | fallen |(obj_height < -0.1)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------------
    # rewards
    # 자식 클래스에서 override
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        raise NotImplementedError