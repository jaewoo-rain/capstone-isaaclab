from __future__ import annotations

from collections.abc import Sequence

import cv2
import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import Camera

from source.omy.vision.camera.yolo_wrapper import YoloWrapper
from source.omy.vision.camera.tracker_manager import TrackerManager
from source.omy.vision.camera.depth_projector import intrinsics_from_fov, sample_depth_at_bbox_center, pixel_to_camera_xyz, camera_to_world
from source.omy.vision.camera.target_selector import TargetSelector
from source.omy.vision.camera.vision_buffer import VisionBuffer
from source.omy.vision.camera.vision_types import torch_to_numpy_image
from .omy_vision_env_cfg import OmyVisionEnvCfg


class OmyBaseVisionEnv(DirectRLEnv):
    cfg: OmyVisionEnvCfg

    def __init__(self, cfg: OmyVisionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.action_space,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.cfg.observation_space,), dtype=np.float32)

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.arm_joint_ids = [self._robot.find_joints(f'joint{i}')[0][0] for i in range(1, 7)]
        self.gripper_master_joint_id = self._robot.find_joints(self.cfg.gripper_master_joint_name)[0][0]
        self.ee_body_id = self._robot.find_bodies(self.cfg.ee_body_name)[0][0]
        self.left_tip_body_id = self._robot.find_bodies(self.cfg.left_tip_body_name)[0][0]
        self.right_tip_body_id = self._robot.find_bodies(self.cfg.right_tip_body_name)[0][0]

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        self.robot_dof_speed_scales[self.gripper_master_joint_id] = 0.2
        self.robot_dof_targets = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)

        self.ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.left_tip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.right_tip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.obj_pos_w = torch.zeros((self.num_envs, 3, 3), device=self.device)
        self.obj_quat_w = torch.zeros((self.num_envs, 3, 4), device=self.device)
        self.slot_pos_w = torch.zeros((self.num_envs, self.cfg.slot_grid_rows * self.cfg.slot_grid_cols, 3), device=self.device)
        self.slot_quat_w = torch.zeros((self.num_envs, self.cfg.slot_grid_rows * self.cfg.slot_grid_cols, 4), device=self.device)
        self.vision_target_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.vision_target_valid = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.vision_target_stale = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.vision_miss_count = torch.zeros((self.num_envs,), device=self.device, dtype=torch.long)
        self.current_slot_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self.current_slot_valid = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self.reward_log = {}
        self.extras['vision'] = {}

        self.yolo = None
        if getattr(self.cfg, "enable_yolo", True):
            self.yolo = YoloWrapper(
                model_path=self.cfg.yolo_model_path,
                names_override={0: "object", 1: "place_slot"},
                conf=self.cfg.yolo_conf,
                iou=self.cfg.yolo_iou,
                tracker_cfg=self.cfg.yolo_tracker_cfg,
                device=self.cfg.yolo_device,
                imgsz=self.cfg.yolo_imgsz,
            )

        if self.yolo is not None:
            class_names = self.yolo.names()
        else:
            class_names = {0: "object", 1: "place_slot"}

        self.tracker = TrackerManager(class_names=class_names)

        self.selector = TargetSelector(object_class_name='object', slot_class_name='place_slot')
        self.buffers = [VisionBuffer(max_stale_frames=self.cfg.max_stale_frames) for _ in range(self.num_envs)]
        self.K = intrinsics_from_fov(self.cfg.camera_width, self.cfg.camera_height, self.cfg.camera_hfov_deg, self.cfg.camera_vfov_deg)
        self._compute_intermediate_values()

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._object_a = RigidObject(self.cfg.object_a)
        self._object_b = RigidObject(self.cfg.object_b)
        self._object_c = RigidObject(self.cfg.object_c)
        self.scene.articulations['robot'] = self._robot
        self.scene.rigid_objects['object_a'] = self._object_a
        self.scene.rigid_objects['object_b'] = self._object_b
        self.scene.rigid_objects['object_c'] = self._object_c
        self._camera = Camera(self.cfg.camera) if self.cfg.use_camera else None
        if self._camera is not None:
            self.scene.sensors['camera'] = self._camera
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == 'cpu':
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85))
        light_cfg.func('/World/Light', light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone().clamp(-1.0, 1.0)
        dof_delta = torch.zeros((actions.shape[0], self._robot.num_joints), device=self.device)
        for i, joint_id in enumerate(self.arm_joint_ids):
            dof_delta[:, joint_id] = self.actions[:, i]
        dof_delta[:, self.gripper_master_joint_id] = self.actions[:, 6]
        targets = self.robot_dof_targets + self.robot_dof_speed_scales * self.dt * dof_delta * self.cfg.action_scale
        self.robot_dof_targets = torch.clamp(targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)

    def _apply_action(self) -> None:
        self._robot.set_joint_position_target(self.robot_dof_targets)

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        self._run_vision_step()
        dof_pos_scaled = 2.0 * (self._robot.data.joint_pos - self.robot_dof_lower_limits) / (self.robot_dof_upper_limits - self.robot_dof_lower_limits + 1e-8) - 1.0
        dof_vel_scaled = self._robot.data.joint_vel * self.cfg.dof_velocity_scale
        gripper_joint = self._robot.data.joint_pos[:, self.gripper_master_joint_id].unsqueeze(-1)
        target_pos = self.vision_target_pos_w
        obj_to_ee = target_pos - self.ee_pos_w
        to_lift_target = (self.cfg.lift_height_threshold - target_pos[:, 2]).unsqueeze(-1)
        obs = torch.cat([dof_pos_scaled, dof_vel_scaled, target_pos, obj_to_ee, gripper_joint, to_lift_target, self.vision_target_valid.float().unsqueeze(-1), self.vision_target_stale.float().unsqueeze(-1), self.current_slot_pos_w, self.current_slot_valid.float().unsqueeze(-1)], dim=-1)
        return {'policy': torch.clamp(obs, -5.0, 5.0)}

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._prev_dist[env_ids_t] = 0.0
        joint_pos = torch.clamp(self._robot.data.default_joint_pos[env_ids_t].clone(), self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        self.robot_dof_targets[env_ids_t] = joint_pos
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)
        self._reset_objects(env_ids_t)
        self._reset_slots(env_ids_t)
        for env_id in env_ids_t.tolist():
            self.buffers[env_id].reset()
        self.vision_target_valid[env_ids_t] = False
        self.vision_target_stale[env_ids_t] = False
        self.vision_miss_count[env_ids_t] = 0
        self.current_slot_valid[env_ids_t] = False
        self._compute_intermediate_values(env_ids_t)

    def _reset_objects(self, env_ids_t: torch.Tensor):
        env_origins = self.scene.env_origins[env_ids_t]
        base_positions = torch.tensor([[0.42, -0.15, self.cfg.object_size_xyz[2] * 0.5], [0.47, 0.00, self.cfg.object_size_xyz[2] * 0.5], [0.52, 0.15, self.cfg.object_size_xyz[2] * 0.5]], device=self.device)
        for idx, obj in enumerate([self._object_a, self._object_b, self._object_c]):
            obj_state = obj.data.default_root_state[env_ids_t].clone()
            noise_xy = (torch.rand((len(env_ids_t), 2), device=self.device) - 0.5) * 2.0 * self.cfg.object_pos_noise
            obj_state[:, 0] = env_origins[:, 0] + base_positions[idx, 0] + noise_xy[:, 0]
            obj_state[:, 1] = env_origins[:, 1] + base_positions[idx, 1] + noise_xy[:, 1]
            obj_state[:, 2] = env_origins[:, 2] + base_positions[idx, 2]
            obj_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
            obj_state[:, 7:] = 0.0
            obj.write_root_state_to_sim(obj_state, env_ids=env_ids_t)

    def _reset_slots(self, env_ids_t: torch.Tensor):
        for env_id in env_ids_t.tolist():
            origin = self.scene.env_origins[env_id]
            k = 0
            for r in range(self.cfg.slot_grid_rows):
                for c in range(self.cfg.slot_grid_cols):
                    self.slot_pos_w[env_id, k] = torch.tensor([origin[0] + self.cfg.slot_origin_xy[0] + r * self.cfg.slot_spacing_x, origin[1] + self.cfg.slot_origin_xy[1] + c * self.cfg.slot_spacing_y, origin[2] + 0.005], device=self.device)
                    self.slot_quat_w[env_id, k] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                    k += 1

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self.ee_pos_w[env_ids] = self._robot.data.body_pos_w[env_ids, self.ee_body_id, :]
        self.left_tip_pos[env_ids] = self._robot.data.body_pos_w[env_ids, self.left_tip_body_id, :]
        self.right_tip_pos[env_ids] = self._robot.data.body_pos_w[env_ids, self.right_tip_body_id, :]
        for obj_idx, obj in enumerate([self._object_a, self._object_b, self._object_c]):
            self.obj_pos_w[env_ids, obj_idx] = obj.data.root_pos_w[env_ids]
            self.obj_quat_w[env_ids, obj_idx] = obj.data.root_quat_w[env_ids]

    def _get_camera_rgb(self):
        return None if self._camera is None else self._camera.data.output.get('rgb', None)

    def _get_camera_depth(self):
        if self._camera is None:
            return None
        return self._camera.data.output.get('distance_to_camera', self._camera.data.output.get('depth', None))

    def _get_camera_pose_for_env(self, env_id: int):
        data = self._camera.data
        cam_pos = None
        cam_quat = None
        for name in ['pos_w', 'position_w', 'body_pos_w']:
            if hasattr(data, name):
                val = getattr(data, name)
                if isinstance(val, torch.Tensor):
                    cam_pos = val[env_id].detach().cpu().numpy() if val.ndim == 2 else val[env_id, 0].detach().cpu().numpy()
                    break
        for name in ['quat_w', 'orientation_w', 'body_quat_w']:
            if hasattr(data, name):
                val = getattr(data, name)
                if isinstance(val, torch.Tensor):
                    cam_quat = val[env_id].detach().cpu().numpy() if val.ndim == 2 else val[env_id, 0].detach().cpu().numpy()
                    break
        if cam_pos is None:
            cam_pos = self.ee_pos_w[env_id].detach().cpu().numpy() + np.array([0.0, -0.08, 0.08], dtype=np.float32)
        if cam_quat is None:
            cam_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return cam_pos.astype(np.float32), cam_quat.astype(np.float32)

    def _run_vision_step(self):
        if self.yolo is None:
            self.vision_target_valid[:] = False
            self.vision_target_stale[:] = False
            self.current_slot_valid[:] = False
            return

        rgb = self._get_camera_rgb()
        depth = self._get_camera_depth()


        if rgb is None:
            self.vision_target_valid[:] = False
            self.vision_target_stale[:] = True
            return
        rgb_cpu = rgb.detach().cpu()
        depth_cpu = depth.detach().cpu() if depth is not None else None
        for env_id in range(self.num_envs):
            frame_rgb = torch_to_numpy_image(rgb_cpu[env_id])
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            detections = self.tracker.parse_ultralytics_result(self.yolo.track(frame_bgr)[0])
            depth_np = None
            if depth_cpu is not None:
                d = depth_cpu[env_id]
                depth_np = (d[..., 0] if d.ndim == 3 else d).numpy().astype(np.float32)
            cam_pos_w, cam_quat_w = self._get_camera_pose_for_env(env_id)
            ee_pos_np = self.ee_pos_w[env_id].detach().cpu().numpy()
            for det in detections:
                if depth_np is None:
                    continue
                depth_m = sample_depth_at_bbox_center(depth_np, det.bbox_xyxy, kernel=5)
                if depth_m is None or not (self.cfg.camera_min_depth_m <= depth_m <= self.cfg.camera_max_depth_m):
                    continue
                det.depth_m = depth_m
                det.pos_cam = pixel_to_camera_xyz(det.center_uv[0], det.center_uv[1], depth_m, self.K)
                det.pos_robot = camera_to_world(det.pos_cam, cam_pos_w, cam_quat_w)
            pick_sel = self.selector.select_for_pick(detections, ee_pos_np)
            place_sel = self.selector.select_empty_slot(detections, ee_pos_np)
            vision_state = self.buffers[env_id].update(pick_sel.target, detections, debug_image=self.tracker.draw(frame_bgr, detections))
            if vision_state.target_detection is not None and vision_state.target_detection.pos_robot is not None:
                self.vision_target_pos_w[env_id] = torch.tensor(vision_state.target_detection.pos_robot, device=self.device)
                self.vision_target_valid[env_id] = True
            else:
                self.vision_target_valid[env_id] = False
            self.vision_target_stale[env_id] = bool(vision_state.is_stale)
            self.vision_miss_count[env_id] = int(vision_state.miss_count)
            if place_sel.target is not None and place_sel.target.pos_robot is not None:
                self.current_slot_pos_w[env_id] = torch.tensor(place_sel.target.pos_robot, device=self.device)
                self.current_slot_valid[env_id] = True
            else:
                self.current_slot_valid[env_id] = False
            self.extras['vision'][f'env_{env_id}'] = {'has_detection': bool(vision_state.has_detection), 'is_stale': bool(vision_state.is_stale), 'miss_count': int(vision_state.miss_count), 'num_dets': len(detections)}

    def _get_common_terms(self):
        self._compute_intermediate_values()
        obj_pos = self.vision_target_pos_w
        ee_pos = self.ee_pos_w
        gripper_joint = self._robot.data.joint_pos[:, self.gripper_master_joint_id]
        tip_center = 0.5 * (self.left_tip_pos + self.right_tip_pos)
        obj_to_ee = obj_pos - ee_pos
        dist = torch.norm(obj_to_ee, dim=-1)
        obj_to_tip = obj_pos - tip_center
        xy_dist = torch.norm(obj_to_tip[:, :2], dim=-1)
        z_dist = torch.abs(obj_to_tip[:, 2])
        approach_reward = torch.clamp(self._prev_dist - dist, min=0.0)
        self._prev_dist = dist.clone()
        xy_align_reward = torch.exp(-40.0 * xy_dist**2)
        z_align_reward = torch.exp(-60.0 * z_dist**2)
        aligned = (xy_dist < 0.06) & (z_dist < 0.07)
        left_is_left = self.left_tip_pos[:, 1] < obj_pos[:, 1]
        right_is_right = self.right_tip_pos[:, 1] > obj_pos[:, 1]
        left_to_obj = torch.norm(obj_pos - self.left_tip_pos, dim=-1)
        right_to_obj = torch.norm(obj_pos - self.right_tip_pos, dim=-1)
        fingers_near = (left_to_obj < 0.08) & (right_to_obj < 0.08)
        pre_grasp_ready = aligned & fingers_near & left_is_left & right_is_right
        is_grasping = pre_grasp_ready & (gripper_joint > 0.3)
        return {'obj_pos': obj_pos, 'ee_pos': ee_pos, 'gripper_joint': gripper_joint, 'dist': dist, 'xy_dist': xy_dist, 'z_dist': z_dist, 'approach_reward': approach_reward, 'xy_align_reward': xy_align_reward, 'z_align_reward': z_align_reward, 'pre_grasp_ready': pre_grasp_ready, 'is_grasping': is_grasping, 'aligned': aligned, 'vision_ok': self.vision_target_valid, 'vision_stale': self.vision_target_stale}

    def _get_dones(self):
        target_height = self.vision_target_pos_w[:, 2]
        terminated = (target_height > self.cfg.lift_height_threshold) | (target_height < -0.1) | (self.vision_miss_count > (self.cfg.max_stale_frames + 20))
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self):
        raise NotImplementedError
