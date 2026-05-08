"""motion1 — Grasp 미세 정렬 RL Env.

DirectRLEnv 상속. 박스 위 PRE_GRASP_Z 에서 ee 가 박스 xy/yaw 에 정렬.

State (6):  obj_rel_x, obj_rel_y, obj_yaw_err, ee_vel_x, ee_vel_y, yaw_vel
Action (3): Δx, Δy, Δyaw  (relative, ee_target 에 누적 — IK 가 풀음)
- ee_z 고정 (= cfg.ee_fixed_z)
- gripper open 고정 (학습 안 함)

Reward, 종료 조건은 cfg 참고.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import (
    quat_from_angle_axis,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)

from .grasp_env_cfg import GraspEnvCfg


# 베이스 EE quat (world frame, 수직 아래 + finger Y) — motion1/play_motion_chain.py 와 동일
BASE_EE_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """각도를 [-π, π] 로 wrap."""
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


class GraspEnv(DirectRLEnv):
    cfg: GraspEnvCfg

    # ------------------------------------------------------------
    # 초기화
    # ------------------------------------------------------------
    def __init__(self, cfg: GraspEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joint indices
        self._arm_joint_ids = [self._robot.find_joints(f"joint{i}")[0][0] for i in range(1, 7)]
        self._gripper_joint_ids = [self._robot.find_joints(n)[0][0] for n in self.cfg.gripper_joint_names]
        self._all_joint_ids = self._arm_joint_ids + self._gripper_joint_ids

        # finger body indices
        self._left_finger_id = self._robot.find_bodies(self.cfg.left_finger_body_name)[0][0]
        self._right_finger_id = self._robot.find_bodies(self.cfg.right_finger_body_name)[0][0]

        # jacobian indices (fixed-base → body_id - 1)
        if self._robot.is_fixed_base:
            self._l_jac_idx = self._left_finger_id - 1
            self._r_jac_idx = self._right_finger_id - 1
        else:
            self._l_jac_idx = self._left_finger_id
            self._r_jac_idx = self._right_finger_id

        # IK controller
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls",
        )
        self._ik = DifferentialIKController(ik_cfg, num_envs=self.num_envs, device=self.device)

        # base ee quat (한번 만들어둠)
        self._base_ee_quat = torch.tensor(
            [list(BASE_EE_QUAT_WXYZ)], device=self.device, dtype=torch.float
        ).repeat(self.num_envs, 1)  # (N, 4)

        # 학습 상태 (per env)
        self._ee_target_xy_w = torch.zeros(self.num_envs, 2, device=self.device)  # world xy target
        self._ee_target_yaw = torch.zeros(self.num_envs, device=self.device)      # ee yaw target
        self._prev_ee_target_yaw = torch.zeros(self.num_envs, device=self.device) # for yaw_vel
        self._box_yaw = torch.zeros(self.num_envs, device=self.device)            # cached
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # fallback grasp pose (motion-only 와 동일 — example7 fallback_holding_joint_pos)
        # 이 자세에서 ee 는 약 (0.46, -0.30, 0.75) world. 박스 위 자연스러운 시작점.
        self._fallback_arm_pos = torch.tensor(
            [0.0, 0.06, 1.98, -1.02, 1.26, -0.13],
            device=self.device, dtype=torch.float
        )  # (6,)

        # reward log (TrainCallback 이 읽음)
        self.reward_log: dict[str, float] = {}

    # ------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object = RigidObject(self.cfg.box)
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        # ground
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone & light
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------
    # Action / Apply
    # ------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """action: (N, 3) in [-1, 1]. ee_target xy 와 yaw 에 누적 적용 (relative)."""
        actions = actions.clamp(-1.0, 1.0)
        self._actions[:] = actions

        # relative 업데이트: 현재 actual ee xy 기준 + delta
        ee_pos_w = self._grip_center_pos()  # (N, 3)
        delta_xy = actions[:, :2] * self.cfg.action_scale_xy
        self._ee_target_xy_w = ee_pos_w[:, :2] + delta_xy

        # yaw target: prev + delta, clip
        self._prev_ee_target_yaw = self._ee_target_yaw.clone()
        delta_yaw = actions[:, 2] * self.cfg.action_scale_yaw
        self._ee_target_yaw = (self._ee_target_yaw + delta_yaw).clamp(
            self.cfg.ee_yaw_min, self.cfg.ee_yaw_max
        )

    def _apply_action(self) -> None:
        """IK 풀어서 joint position target 적용. gripper open 고정."""
        # ee target pose (world frame)
        # xy = self._ee_target_xy_w, z = ee_fixed_z (env_origin_z 더해줌)
        env_origin_z = self.scene.env_origins[:, 2]
        target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        target_pos_w[:, :2] = self._ee_target_xy_w
        target_pos_w[:, 2] = self.cfg.ee_fixed_z + env_origin_z

        # ee quat = R_z(ee_target_yaw) ⊗ base_ee_quat  (yaw 만 회전)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, 3)
        yaw_quat = quat_from_angle_axis(self._ee_target_yaw, z_axis)  # (N, 4) wxyz
        target_quat_w = quat_mul(yaw_quat, self._base_ee_quat)        # (N, 4) wxyz

        # current ee pose (world)
        ee_pos_w = self._grip_center_pos()
        ee_quat_w = self._robot.data.body_quat_w[:, self._left_finger_id]  # (N, 4)

        # current arm joint pos
        cur_arm_q = self._robot.data.joint_pos[:, self._arm_joint_ids]  # (N, 6)

        # jacobian (양 finger 평균)
        J = self._robot.root_physx_view.get_jacobians()
        j_l = J[:, self._l_jac_idx, :, :][:, :, self._arm_joint_ids]
        j_r = J[:, self._r_jac_idx, :, :][:, :, self._arm_joint_ids]
        jac = 0.5 * (j_l + j_r)  # (N, 6, 6)

        # world → root frame
        root_pos_w = self._robot.data.root_pos_w
        root_quat_w = self._robot.data.root_quat_w
        tgt_pos_b, tgt_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w
        )
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        # IK
        self._ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))
        arm_target = self._ik.compute(ee_pos_b, ee_quat_b, jac, cur_arm_q)  # (N, 6)

        # gripper open 고정 (4 joint 모두 0)
        gripper_target = torch.zeros(self.num_envs, 4, device=self.device)

        # 적용
        full_target = torch.cat([arm_target, gripper_target], dim=-1)  # (N, 10)
        self._robot.set_joint_position_target(full_target, joint_ids=self._all_joint_ids)

    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _grip_center_pos(self) -> torch.Tensor:
        """양 finger 평균 world pos (N, 3)."""
        l = self._robot.data.body_pos_w[:, self._left_finger_id]
        r = self._robot.data.body_pos_w[:, self._right_finger_id]
        return 0.5 * (l + r)

    def _grip_center_vel(self) -> torch.Tensor:
        """양 finger 평균 world lin vel (N, 3)."""
        l = self._robot.data.body_lin_vel_w[:, self._left_finger_id]
        r = self._robot.data.body_lin_vel_w[:, self._right_finger_id]
        return 0.5 * (l + r)

    def _box_yaw_from_quat(self) -> torch.Tensor:
        """박스 yaw (world frame). quat wxyz → yaw 추출."""
        q = self._object.data.root_quat_w  # (N, 4) wxyz
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    # ------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------
    def _get_observations(self) -> dict:
        # 박스 / ee actual pose (env-rel xy)
        box_pos_w = self._object.data.root_pos_w
        ee_pos_w = self._grip_center_pos()
        box_xy_env = box_pos_w[:, :2] - self.scene.env_origins[:, :2]
        ee_xy_env = ee_pos_w[:, :2] - self.scene.env_origins[:, :2]

        obj_rel_x = box_xy_env[:, 0] - ee_xy_env[:, 0]
        obj_rel_y = box_xy_env[:, 1] - ee_xy_env[:, 1]

        # yaw err
        box_yaw = self._box_yaw_from_quat()
        self._box_yaw = box_yaw  # cache (reset 에서 random spawn 한 값 그대로일 가능성)
        obj_yaw_err = wrap_to_pi(box_yaw - self._ee_target_yaw)

        # ee vel (xy) — 시뮬값
        ee_vel = self._grip_center_vel()
        ee_vel_x = ee_vel[:, 0]
        ee_vel_y = ee_vel[:, 1]

        # yaw vel = ee_target_yaw 의 변화율
        sim_dt = self.cfg.sim.dt * self.cfg.decimation  # control dt = 1/60
        yaw_vel = (self._ee_target_yaw - self._prev_ee_target_yaw) / max(sim_dt, 1e-6)

        obs = torch.stack(
            [obj_rel_x, obj_rel_y, obj_yaw_err, ee_vel_x, ee_vel_y, yaw_vel], dim=-1
        )  # (N, 6)
        return {"policy": obs}

    # ------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # obs 와 동일 계산. 한 step 안에서 같이 묶어도 되지만 명료성 위해 분리.
        box_pos_w = self._object.data.root_pos_w
        ee_pos_w = self._grip_center_pos()
        obj_rel_xy = (box_pos_w[:, :2] - self.scene.env_origins[:, :2]) - (ee_pos_w[:, :2] - self.scene.env_origins[:, :2])
        xy_dist2 = (obj_rel_xy ** 2).sum(dim=-1)

        box_yaw = self._box_yaw_from_quat()
        yaw_err = wrap_to_pi(box_yaw - self._ee_target_yaw)

        ee_vel = self._grip_center_vel()
        sim_dt = self.cfg.sim.dt * self.cfg.decimation
        yaw_vel = (self._ee_target_yaw - self._prev_ee_target_yaw) / max(sim_dt, 1e-6)

        # rewards
        r_xy_align = torch.exp(-self.cfg.reward_xy_align_gain * xy_dist2)
        r_yaw_align = torch.exp(-self.cfg.reward_yaw_align_gain * (yaw_err ** 2))
        r_smooth = -self.cfg.reward_smooth_w * (ee_vel[:, 0] ** 2 + ee_vel[:, 1] ** 2 + yaw_vel ** 2)

        aligned = (
            (obj_rel_xy[:, 0].abs() < self.cfg.align_xy_threshold) &
            (obj_rel_xy[:, 1].abs() < self.cfg.align_xy_threshold) &
            (yaw_err.abs() < self.cfg.align_yaw_threshold)
        )
        r_success = aligned.float() * self.cfg.reward_success_bonus

        total = r_xy_align + r_yaw_align + r_smooth + r_success

        # log (callback 이 읽음, env 0 기준)
        self.reward_log = {
            "r_xy_align": float(r_xy_align.mean().item()),
            "r_yaw_align": float(r_yaw_align.mean().item()),
            "r_smooth": float(r_smooth.mean().item()),
            "r_success": float(r_success.mean().item()),
            "aligned_rate": float(aligned.float().mean().item()),
        }
        return total

    # ------------------------------------------------------------
    # Done
    # ------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        box_pos_w = self._object.data.root_pos_w
        ee_pos_w = self._grip_center_pos()
        obj_rel_xy = (box_pos_w[:, :2] - self.scene.env_origins[:, :2]) - (ee_pos_w[:, :2] - self.scene.env_origins[:, :2])

        box_yaw = self._box_yaw_from_quat()
        yaw_err = wrap_to_pi(box_yaw - self._ee_target_yaw)

        aligned = (
            (obj_rel_xy[:, 0].abs() < self.cfg.align_xy_threshold) &
            (obj_rel_xy[:, 1].abs() < self.cfg.align_xy_threshold) &
            (yaw_err.abs() < self.cfg.align_yaw_threshold)
        )
        failed = (
            (obj_rel_xy[:, 0].abs() > self.cfg.fail_xy_threshold) |
            (obj_rel_xy[:, 1].abs() > self.cfg.fail_xy_threshold)
        )

        terminated = aligned | failed
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    # ------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # ----- 박스 random spawn -----
        # xy noise ±box_spawn_xy_noise
        noise_xy = sample_uniform(
            -self.cfg.box_spawn_xy_noise, self.cfg.box_spawn_xy_noise,
            (n, 2), device=self.device,
        )
        box_x = self.cfg.box_spawn_xy[0] + noise_xy[:, 0]
        box_y = self.cfg.box_spawn_xy[1] + noise_xy[:, 1]
        box_z = torch.full((n,), self.cfg.box_spawn_z, device=self.device)

        # yaw ±box_spawn_yaw_max → quat (z축 회전, isaaclab 표준 (w,x,y,z) 형식)
        box_yaw = sample_uniform(
            -self.cfg.box_spawn_yaw_max, self.cfg.box_spawn_yaw_max,
            (n,), device=self.device,
        )
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(n, 3)
        box_quat = quat_from_angle_axis(box_yaw, z_axis)  # (n, 4) wxyz

        # 디버그 — 첫 reset 시 yaw 값 출력 (random 동작 확인용)
        if not hasattr(self, "_reset_debug_done"):
            self._reset_debug_done = True
            print(f"[GraspEnv reset DEBUG] box_yaw sample (deg): "
                  f"{(box_yaw[:5] * 180.0 / 3.14159).tolist()}")
            print(f"[GraspEnv reset DEBUG] box_quat[0] (wxyz): {box_quat[0].tolist()}")

        # 박스 root state 작성 (world = env_origin + env-rel)
        box_pos_w = torch.stack([box_x, box_y, box_z], dim=-1) + self.scene.env_origins[env_ids_t]
        box_root_state = torch.zeros(n, 13, device=self.device)
        box_root_state[:, 0:3] = box_pos_w
        box_root_state[:, 3:7] = box_quat
        # 7~9 lin vel = 0, 10~12 ang vel = 0
        self._object.write_root_pose_to_sim(box_root_state[:, 0:7], env_ids=env_ids_t)
        self._object.write_root_velocity_to_sim(box_root_state[:, 7:13], env_ids=env_ids_t)

        # ----- robot reset (fallback grasp pose + small joint noise) -----
        joint_pos = self._robot.data.default_joint_pos[env_ids_t].clone()
        for i, jid in enumerate(self._arm_joint_ids):
            joint_pos[:, jid] = self._fallback_arm_pos[i]
        # small noise
        joint_pos[:, self._arm_joint_ids] += sample_uniform(
            -0.05, 0.05, (n, 6), device=self.device,
        )
        # gripper open
        for jid in self._gripper_joint_ids:
            joint_pos[:, jid] = 0.0
        joint_vel = torch.zeros_like(joint_pos)

        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)

        # ----- ee_target / yaw 초기화 (현재 actual ee 위치 기준) -----
        # write_joint_state 후 body_pos 가 즉시 갱신되지 않을 수 있음 → next step 후 자연스럽게 됨.
        # 시작 시 ee_target = (현재 자세의 추정 ee xy) — 우선 박스 spawn xy 근처로 두고 학습이 알아서.
        # 여기서는 fallback pose 의 대략 ee xy 위치 (0.46, -0.30) 를 시작점으로.
        # (정확한 값은 첫 _pre_physics_step 에서 grip_center_pos 로 보정됨)
        ee_world_xy_default = torch.tensor([0.46, -0.30], device=self.device).expand(n, 2)
        self._ee_target_xy_w[env_ids_t] = ee_world_xy_default + self.scene.env_origins[env_ids_t, :2]
        self._ee_target_yaw[env_ids_t] = 0.0
        self._prev_ee_target_yaw[env_ids_t] = 0.0

        # IK 컨트롤러 reset
        self._ik.reset(env_ids_t)
