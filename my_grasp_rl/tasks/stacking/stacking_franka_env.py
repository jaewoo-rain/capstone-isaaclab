from __future__ import annotations

from collections.abc import Sequence

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.scene import InteractiveScene
from isaaclab.sim.spawners.from_files import spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .stacking_franka_env_cfg import StackingFrankaEnvCfg


class StackingFrankaEnv(DirectRLEnv):
    """
    stacking task용 RL 환경.

    목표:
    - robot이 '쌓을 블록(stack object)'을 집고
    - '기준 블록(base object)' 위 목표 위치로 옮긴 뒤
    - 정렬해서 놓고
    - 일정 시간 안정적으로 유지되면 성공

    즉, 단순 grasp가 아니라
    grasp -> move -> align -> release -> stable
    전체를 하나의 task로 묶은 환경
    """

    cfg: StackingFrankaEnvCfg

    def __init__(self, cfg: StackingFrankaEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---------------------------------------------------------
        # 1. 로봇의 관절 / finger body 인덱스 찾기
        # ---------------------------------------------------------
        # 관절 이름을 cfg에서 받아와 실제 articulation 내부 index로 바꿈
        self.arm_joint_ids = [self._robot.find_joints(name)[0][0] for name in self.cfg.arm_joint_names]

        # 그리퍼 관절 index
        self.left_finger_id = self._robot.find_joints(self.cfg.left_finger_joint_name)[0][0]
        self.right_finger_id = self._robot.find_joints(self.cfg.right_finger_joint_name)[0][0]

        # finger body index
        # 이걸 쓰는 이유:
        # "손목"이 아니라 "실제 집는 손가락 중심"을 EE로 쓰기 위해
        self.left_finger_body_id = self._robot.find_bodies("panda_leftfinger")[0][0]
        self.right_finger_body_id = self._robot.find_bodies("panda_rightfinger")[0][0]

        self.num_arm_dofs = len(self.arm_joint_ids)

        # ---------------------------------------------------------
        # 2. action / 상태 저장용 buffer
        # ---------------------------------------------------------
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # arm 관절 제어값
        self.arm_action = torch.zeros((self.num_envs, self.num_arm_dofs), device=self.device)

        # gripper open/close 제어값
        self.gripper_action = torch.zeros((self.num_envs, 1), device=self.device)

        # joint target buffer
        self.joint_targets = self._robot.data.default_joint_pos.clone()

        # ---------------------------------------------------------
        # 3. 중간 계산값 저장용 buffer
        # ---------------------------------------------------------
        # ee(손가락 중심) 위치
        self.ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # 파란 블록(base)
        self.base_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # 빨간 블록(stack)
        self.stack_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # 빨간 블록이 올라가야 하는 목표 위치
        self.target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # 빨간 블록과 손가락 중심 사이 상대 위치
        self.obj_to_ee = torch.zeros((self.num_envs, 3), device=self.device)

        # 빨간 블록과 목표 위치 사이 상대 위치
        self.obj_to_target = torch.zeros((self.num_envs, 3), device=self.device)

        # grasp 상태 여부
        self.grasped = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # release가 제대로 됐는지
        self.release_ok = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # 안정적으로 쌓인 상태가 몇 step 유지됐는지 카운터
        self.stable_counter = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

    # -------------------------------------------------
    # scene 생성
    # -------------------------------------------------
    def _setup_scene(self):
        # robot 생성
        self._robot = Articulation(self.cfg.robot_cfg)

        # 아래 기준 블록(파란색)
        self._base_object = RigidObject(self.cfg.base_object_cfg)

        # 위에 쌓을 블록(빨간색)
        self._stack_object = RigidObject(self.cfg.stack_object_cfg)

        # scene 등록
        self.scene = InteractiveScene(self.cfg.scene)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["base_object"] = self._base_object
        self.scene.rigid_objects["stack_object"] = self._stack_object

        # 바닥
        spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground)

        # 병렬 환경 복제
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

        # 조명
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # -------------------------------------------------
    # action 입력 받기
    # -------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 전체 action 저장
        self.actions = actions.clone()

        # 앞쪽 7개 = arm joint delta
        self.arm_action = actions[:, : self.num_arm_dofs]

        # 마지막 1개 = gripper open/close
        self.gripper_action = actions[:, -1:].clone()

    # -------------------------------------------------
    # 실제 action 적용
    # -------------------------------------------------
    def _apply_action(self) -> None:
        # 현재 joint target에서 delta만큼 더함
        current_targets = self.joint_targets[:, self.arm_joint_ids]
        new_targets = current_targets + self.cfg.arm_action_scale * self.arm_action
        self.joint_targets[:, self.arm_joint_ids] = new_targets

        # gripper action 해석
        # > 0 이면 닫기, <= 0 이면 열기
        close_mask = self.gripper_action.squeeze(-1) > 0.0
        open_mask = ~close_mask

        self.joint_targets[close_mask, self.left_finger_id] = self.cfg.gripper_close_target
        self.joint_targets[close_mask, self.right_finger_id] = self.cfg.gripper_close_target

        self.joint_targets[open_mask, self.left_finger_id] = self.cfg.gripper_open_target
        self.joint_targets[open_mask, self.right_finger_id] = self.cfg.gripper_open_target

        # 최종 target 적용
        self._robot.set_joint_position_target(self.joint_targets)

    # -------------------------------------------------
    # observation 만들기
    # -------------------------------------------------
    def _get_observations(self) -> dict:
        # 중간 계산값 먼저 갱신
        self._compute_intermediate_values()

        # arm joint 상태
        joint_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self.arm_joint_ids]

        # finger 상태
        finger_pos = self._robot.data.joint_pos[:, [self.left_finger_id, self.right_finger_id]]
        finger_vel = self._robot.data.joint_vel[:, [self.left_finger_id, self.right_finger_id]]

        # 빨간 블록 속도
        stack_obj_lin_vel = self._stack_object.data.root_lin_vel_w

        # 최종 observation
        obs = torch.cat(
            [
                joint_pos,          # 로봇 자세
                joint_vel,          # 로봇 속도
                finger_pos,         # 그리퍼 상태
                finger_vel,         # 그리퍼 속도
                self.ee_pos_w,      # 손가락 중심 위치
                self.stack_pos_w,   # 빨간 블록 위치
                self.base_pos_w,    # 파란 블록 위치
                self.obj_to_ee,     # 빨간 블록이 손가락에서 얼마나 떨어졌는지
                self.obj_to_target, # 빨간 블록이 목표 위치에서 얼마나 떨어졌는지
                stack_obj_lin_vel,  # 빨간 블록 움직임 속도
            ],
            dim=-1,
        )
        return {"policy": obs}

    # -------------------------------------------------
    # reward 계산
    # -------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        # 1) 빨간 블록과 손가락 중심 사이 거리
        dist_obj = torch.norm(self.obj_to_ee, dim=-1)

        # 2) 빨간 블록과 목표 위치 사이 거리
        dist_target = torch.norm(self.obj_to_target, dim=-1)

        # 3) xy 평면에서 정렬 오차
        xy_dist = torch.norm(self.obj_to_target[:, :2], dim=-1)

        # 4) z축 높이 오차
        z_dist = torch.abs(self.obj_to_target[:, 2])

        # finger 평균 닫힘 정도
        left_finger_pos = self._robot.data.joint_pos[:, self.left_finger_id]
        right_finger_pos = self._robot.data.joint_pos[:, self.right_finger_id]
        finger_mean = 0.5 * (left_finger_pos + right_finger_pos)

        # -------------------------------------------------
        # grasp 조건 근사
        # -------------------------------------------------
        # 완전한 contact 기반은 아니고 heuristic
        gripper_closed = finger_mean < 0.025
        gripper_open = finger_mean > 0.03
        near_object = dist_obj < 0.04

        # 가까이 있고 닫혀 있으면 grasp로 간주
        self.grasped = gripper_closed & near_object

        # -------------------------------------------------
        # reward 항목들
        # -------------------------------------------------

        # 빨간 블록으로 접근
        reach_obj_reward = torch.exp(-6.0 * dist_obj)

        # 잡기 성공
        grasp_reward = self.cfg.rew_grasp * self.grasped.float()

        # 잡은 뒤 목표 위치로 옮기기
        move_target_reward = torch.where(
            self.grasped,
            torch.exp(-5.0 * dist_target),
            torch.zeros_like(dist_target),
        )

        # 목표 위치와 xy 정렬
        align_xy_reward = torch.where(
            self.grasped,
            torch.exp(-20.0 * xy_dist),
            torch.zeros_like(xy_dist),
        )

        # 목표 높이와 z 정렬
        align_z_reward = torch.where(
            self.grasped,
            torch.exp(-20.0 * z_dist),
            torch.zeros_like(z_dist),
        )

        # -------------------------------------------------
        # release 보상
        # -------------------------------------------------
        # 목표 위치 근처에서 gripper를 열면 보상
        release_zone = (xy_dist < 0.015) & (z_dist < 0.02)
        self.release_ok = release_zone & gripper_open

        release_reward = self.cfg.rew_release * self.release_ok.float()

        # -------------------------------------------------
        # stable 보상
        # -------------------------------------------------
        # release 후에도 계속 목표 근처에 있으면 안정적 stack으로 판단
        stable_now = self.release_ok & (xy_dist < self.cfg.success_xy_threshold) & (z_dist < self.cfg.success_z_threshold)

        self.stable_counter = torch.where(
            stable_now,
            self.stable_counter + 1,
            torch.zeros_like(self.stable_counter),
        )

        stable_reward = self.cfg.rew_stable * (self.stable_counter >= self.cfg.stable_steps_required).float()

        # -------------------------------------------------
        # penalty
        # -------------------------------------------------
        action_penalty = (-self.cfg.rew_action_penalty) * torch.sum(self.actions**2, dim=-1)
        joint_vel_penalty = (-self.cfg.rew_joint_vel_penalty) * torch.sum(
            self._robot.data.joint_vel[:, self.arm_joint_ids] ** 2,
            dim=-1,
        )

        # 최종 reward 합산
        reward = (
            self.cfg.rew_reach_obj * reach_obj_reward
            + grasp_reward
            + self.cfg.rew_move_target * move_target_reward
            + self.cfg.rew_align_xy * align_xy_reward
            + self.cfg.rew_align_z * align_z_reward
            + release_reward
            + stable_reward
            - action_penalty
            - joint_vel_penalty
        )

        return reward

    # -------------------------------------------------
    # 종료 조건
    # -------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # time limit 종료
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 목표 위치와의 오차
        xy_dist = torch.norm(self.obj_to_target[:, :2], dim=-1)
        z_dist = torch.abs(self.obj_to_target[:, 2])

        # 성공 조건:
        # - xy/z 정렬 오차가 작고
        # - stable counter가 충분히 쌓였을 때
        success = (
            (xy_dist < self.cfg.success_xy_threshold)
            & (z_dist < self.cfg.success_z_threshold)
            & (self.stable_counter >= self.cfg.stable_steps_required)
        )

        # 물체가 바닥 아래로 떨어지면 실패
        object_fallen = self.stack_pos_w[:, 2] < 0.0

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

        env_origins = self.scene.env_origins[env_ids]

        # -----------------------------
        # 파란 블록 reset
        # -----------------------------
        base_state = self._base_object.data.default_root_state[env_ids].clone()
        base_state[:, 0] = env_origins[:, 0] + sample_uniform(
            self.cfg.base_x_range[0], self.cfg.base_x_range[1], (len(env_ids),), self.device
        )
        base_state[:, 1] = env_origins[:, 1] + sample_uniform(
            self.cfg.base_y_range[0], self.cfg.base_y_range[1], (len(env_ids),), self.device
        )
        base_state[:, 2] = env_origins[:, 2] + self.cfg.object_z
        base_state[:, 7:13] = 0.0
        self._base_object.write_root_state_to_sim(base_state, env_ids=env_ids)

        # -----------------------------
        # 빨간 블록 reset
        # -----------------------------
        stack_state = self._stack_object.data.default_root_state[env_ids].clone()
        stack_state[:, 0] = env_origins[:, 0] + sample_uniform(
            self.cfg.stack_x_range[0], self.cfg.stack_x_range[1], (len(env_ids),), self.device
        )
        stack_state[:, 1] = env_origins[:, 1] + sample_uniform(
            self.cfg.stack_y_range[0], self.cfg.stack_y_range[1], (len(env_ids),), self.device
        )
        stack_state[:, 2] = env_origins[:, 2] + self.cfg.object_z
        stack_state[:, 7:13] = 0.0
        self._stack_object.write_root_state_to_sim(stack_state, env_ids=env_ids)

        # stable counter 초기화
        self.stable_counter[env_ids] = 0
        self._compute_intermediate_values()

    # -------------------------------------------------
    # 중간 계산
    # -------------------------------------------------
    def _compute_intermediate_values(self):
        # finger 두 개의 평균을 EE 중심으로 사용
        left_pos = self._robot.data.body_pos_w[:, self.left_finger_body_id, :]
        right_pos = self._robot.data.body_pos_w[:, self.right_finger_body_id, :]
        self.ee_pos_w = 0.5 * (left_pos + right_pos)

        # base / stack 위치
        self.base_pos_w = self._base_object.data.root_pos_w
        self.stack_pos_w = self._stack_object.data.root_pos_w

        # 목표 위치 = 파란 블록 위쪽
        self.target_pos_w = self.base_pos_w.clone()
        self.target_pos_w[:, 2] = self.base_pos_w[:, 2] + self.cfg.cube_size

        # 빨간 블록이 손가락 중심과 얼마나 떨어졌는지
        self.obj_to_ee = self.stack_pos_w - self.ee_pos_w

        # 빨간 블록이 목표 위치와 얼마나 떨어졌는지
        self.obj_to_target = self.stack_pos_w - self.target_pos_w