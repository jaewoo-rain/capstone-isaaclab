"""motion2 — Sim adapter (IsaacLab implementation).

BaseAdapter 의 IsaacLab 기반 구현. ChainStateMachine 가 이 adapter 호출하면
play_motion_chain_with_grasp_insert_camera.py 와 동일 동작 (sim 검증용).

real 로 가려면 real_adapter.py 를 작성. ChainStateMachine 는 변경 X.
"""
from __future__ import annotations

import math
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply, quat_from_angle_axis, quat_mul, subtract_frame_transforms,
)

from source.omy.omy_robot_cfg import OMY_OFF_SELF_COLLISION_CFG

from .base_adapter import BaseAdapter, CamData, EePose, BoxGtPose


# ===== 박스/셀 기하 (chain runner 와 동일) =====
BOX_SPAWN = (0.30, -0.10, 0.07)
BOX_SIZE = (0.139, 0.044, 0.118)
BOX_MASS = 0.3

CELL_CENTER_X = 0.30
CELL_CENTER_Y = -0.30
CELL_INNER_X = 0.16
CELL_INNER_Y = 0.065
WALL_THICKNESS = 0.008
WALL_HEIGHT = 0.12
_V_WALL_SIZE = (WALL_THICKNESS, CELL_INNER_Y + 2 * WALL_THICKNESS, WALL_HEIGHT)
_H_WALL_SIZE = (CELL_INNER_X + 2 * WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT)
_WALL_Z = WALL_HEIGHT / 2

TOP_CAM_POS = (0.30, -0.20, 0.80)
TOP_CAM_ROT = (0.0, 1.0, 0.0, 0.0)
WRIST_CAM_POS_LINK6 = (0.0, -0.1, 0.084)
WRIST_CAM_ROT = (0.0, 0.0, 0.7071068, -0.7071068)

# 박스/셀 random spawn 범위
BOX_SPAWN_XY_NOISE = 0.00
BOX_SPAWN_YAW_MAX = math.pi / 2
CELL_SPAWN_XY_NOISE = 0.00
CELL_SPAWN_YAW_MAX = math.pi / 2

HOME_JOINT_POS = {
    "joint1": 0.0, "joint2": -1.55, "joint3": 2.66,
    "joint4": -1.1, "joint5": 1.6, "joint6": 0.0,
    "rh_r1_joint": 0.0, "rh_r2": 0.0, "rh_l1": 0.0, "rh_l2": 0.0,
}


@configclass
class SimSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))
    robot = OMY_OFF_SELF_COLLISION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, disable_gravity=False,
                max_depenetration_velocity=5.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.7, 0.72), metallic=0.85, roughness=0.12),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                static_friction=3.0, dynamic_friction=3.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_SPAWN))

    wall_v_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_VL",
        spawn=sim_utils.CuboidCfg(
            size=_V_WALL_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(CELL_CENTER_X - CELL_INNER_X / 2 - WALL_THICKNESS / 2,
                 CELL_CENTER_Y, _WALL_Z)))
    wall_v_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_VR",
        spawn=sim_utils.CuboidCfg(
            size=_V_WALL_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(CELL_CENTER_X + CELL_INNER_X / 2 + WALL_THICKNESS / 2,
                 CELL_CENTER_Y, _WALL_Z)))
    wall_h_front: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_HF",
        spawn=sim_utils.CuboidCfg(
            size=_H_WALL_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(CELL_CENTER_X,
                 CELL_CENTER_Y - CELL_INNER_Y / 2 - WALL_THICKNESS / 2, _WALL_Z)))
    wall_h_back: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_HB",
        spawn=sim_utils.CuboidCfg(
            size=_H_WALL_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(CELL_CENTER_X,
                 CELL_CENTER_Y + CELL_INNER_Y / 2 + WALL_THICKNESS / 2, _WALL_Z)))

    top_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TopCam",
        update_period=0.0, height=480, width=640,
        data_types=["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0)),
        offset=CameraCfg.OffsetCfg(pos=TOP_CAM_POS, rot=TOP_CAM_ROT, convention="ros"),
        colorize_instance_id_segmentation=False,
        update_latest_camera_pose=True)
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/WristCam",
        update_period=0.0, height=240, width=320,
        data_types=["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=11.0, focus_distance=0.25, horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),
                                   convention="world"),
        colorize_instance_id_segmentation=False,
        update_latest_camera_pose=True)


class SimAdapter(BaseAdapter):
    """IsaacLab 기반 sim 환경 adapter."""

    def __init__(self, sim, scene, gripper_close: float = 0.8):
        self.sim = sim
        self.scene = scene
        self.device = sim.device
        self._dt = sim.get_physics_dt()
        self._gripper_close = gripper_close

        self.robot = scene["robot"]
        self.box = scene["box"]
        self.top_cam_sensor = scene["top_cam"]
        self.wrist_cam_sensor = scene["wrist_cam"]

        # joint / body ids
        self.arm_ids = [self.robot.find_joints(f"joint{i}")[0][0] for i in range(1, 7)]
        gripper_names = ["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"]
        self.gripper_ids = [self.robot.find_joints(n)[0][0] for n in gripper_names]
        self.all_joint_ids = self.arm_ids + self.gripper_ids
        self.left_id = self.robot.find_bodies("rh_p12_rn_l2")[0][0]
        self.right_id = self.robot.find_bodies("rh_p12_rn_r2")[0][0]
        self.link6_id = self.robot.find_bodies("link6")[0][0]
        if self.robot.is_fixed_base:
            self.l_jac, self.r_jac = self.left_id - 1, self.right_id - 1
        else:
            self.l_jac, self.r_jac = self.left_id, self.right_id

        # IK
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose", use_relative_mode=False, ik_method="dls")
        self.ik = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=self.device)

        self.env_origin = scene.env_origins[0]

        # home pose 캐싱
        self._home_q = torch.zeros((scene.num_envs, self.robot.num_joints), device=self.device)
        for n, v in HOME_JOINT_POS.items():
            jid = self.robot.find_joints(n)[0][0]
            self._home_q[:, jid] = v
        self._joint_vel = torch.zeros_like(self._home_q)

        # home ee pose (reset_to_home 호출 후 설정)
        self._home_ee_pos = None
        self._home_ee_quat = None

        # 손목 cam offset
        self._wrist_offset_pos = torch.tensor(
            [list(WRIST_CAM_POS_LINK6)], device=self.device, dtype=torch.float32)
        self._wrist_offset_quat = torch.tensor(
            [list(WRIST_CAM_ROT)], device=self.device, dtype=torch.float32)

        # 셀 wall offsets (cell-local)
        self._cell_wall_offsets = [
            ("wall_v_left",  -(CELL_INNER_X / 2 + WALL_THICKNESS / 2), 0.0),
            ("wall_v_right", +(CELL_INNER_X / 2 + WALL_THICKNESS / 2), 0.0),
            ("wall_h_front", 0.0, -(CELL_INNER_Y / 2 + WALL_THICKNESS / 2)),
            ("wall_h_back",  0.0, +(CELL_INNER_Y / 2 + WALL_THICKNESS / 2)),
        ]

        # 마지막 spawn yaw/xy 캐싱 (get_box_gt 용)
        self._last_box_spawn = (BOX_SPAWN[0], BOX_SPAWN[1], 0.0)
        self._last_cell_spawn = (CELL_CENTER_X, CELL_CENTER_Y, 0.0)

    # ===== private helpers =====
    def _grip_center_pos(self) -> torch.Tensor:
        return 0.5 * (self.robot.data.body_pos_w[:, self.left_id]
                      + self.robot.data.body_pos_w[:, self.right_id])

    def _grip_center_quat(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.left_id]

    def _grip_center_vel(self) -> torch.Tensor:
        return 0.5 * (self.robot.data.body_lin_vel_w[:, self.left_id]
                      + self.robot.data.body_lin_vel_w[:, self.right_id])

    def _update_wrist_cam_pose(self):
        link6_pos_w = self.robot.data.body_pos_w[:, self.link6_id]
        link6_quat_w = self.robot.data.body_quat_w[:, self.link6_id]
        offset_pos_world = quat_apply(
            link6_quat_w, self._wrist_offset_pos.expand_as(link6_pos_w))
        cam_pos = link6_pos_w + offset_pos_world
        cam_quat_ros = quat_mul(
            link6_quat_w, self._wrist_offset_quat.expand(link6_quat_w.shape))
        self.wrist_cam_sensor.set_world_poses(cam_pos, cam_quat_ros, convention="ros")

    def _cam_data(self, sensor) -> CamData:
        sensor.update(dt=self._dt, force_recompute=True)
        rgb_t = sensor.data.output["rgb"][0]
        rgb = rgb_t.cpu().numpy()
        if rgb.shape[-1] >= 3:
            rgb = rgb[..., :3]
        depth_t = sensor.data.output["distance_to_image_plane"][0].squeeze(-1)
        depth = depth_t.cpu().numpy().astype(np.float32)
        K = sensor.data.intrinsic_matrices[0].cpu().numpy().astype(np.float32)
        pos_w = sensor.data.pos_w[0].cpu().numpy().astype(np.float32)
        quat_w = sensor.data.quat_w_world[0].cpu().numpy().astype(np.float32)
        return CamData(rgb=rgb, depth=depth, K=K, pos_w=pos_w, quat_w_world=quat_w)

    # ===== BaseAdapter API =====
    def get_top_cam(self) -> CamData:
        # top cam 의 view 가 spawn 직후 안정화되도록 1 step 진행
        return self._cam_data(self.top_cam_sensor)

    def get_wrist_cam(self) -> CamData:
        self._update_wrist_cam_pose()
        return self._cam_data(self.wrist_cam_sensor)

    def get_ee_pose(self) -> EePose:
        pos_w = self._grip_center_pos()[0].cpu().numpy().astype(np.float32)
        quat_w = self._grip_center_quat()[0].cpu().numpy().astype(np.float32)
        lin = self._grip_center_vel()[0].cpu().numpy().astype(np.float32)
        ang_l = float(self.robot.data.body_ang_vel_w[0, self.left_id, 2].item())
        ang_r = float(self.robot.data.body_ang_vel_w[0, self.right_id, 2].item())
        return EePose(pos_w=pos_w, quat_w=quat_w, lin_vel=lin,
                      ang_vel_z=0.5 * (ang_l + ang_r))

    def set_ee_target(self, target_pos: np.ndarray, target_quat: np.ndarray,
                      gripper_value: float) -> None:
        target_pos_t = torch.tensor(
            target_pos, device=self.device, dtype=torch.float32).unsqueeze(0)
        target_quat_t = torch.tensor(
            target_quat, device=self.device, dtype=torch.float32).unsqueeze(0)
        tip_ratio = 2.3
        gripper_target = torch.tensor(
            [[gripper_value, gripper_value * tip_ratio,
              gripper_value, gripper_value * tip_ratio]],
            device=self.device).expand(self.scene.num_envs, -1)

        ee_pos_w = self._grip_center_pos()
        ee_quat_w = self._grip_center_quat()
        cur_arm_q = self.robot.data.joint_pos[:, self.arm_ids]
        J = self.robot.root_physx_view.get_jacobians()
        j_l = J[:, self.l_jac, :, :][:, :, self.arm_ids]
        j_r = J[:, self.r_jac, :, :][:, :, self.arm_ids]
        jac = 0.5 * (j_l + j_r)

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        tgt_pos_b, tgt_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_t, target_quat_t)
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        self.ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))
        arm_target = self.ik.compute(ee_pos_b, ee_quat_b, jac, cur_arm_q)
        full_target = torch.cat([arm_target, gripper_target], dim=-1)
        self.robot.set_joint_position_target(full_target, joint_ids=self.all_joint_ids)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self._dt)

    def reset_to_home(self) -> None:
        self.robot.write_joint_state_to_sim(self._home_q, self._joint_vel)
        self.robot.set_joint_position_target(self._home_q)
        self.robot.reset()
        self.step(60)
        if self._home_ee_pos is None:
            self._home_ee_pos = (self._grip_center_pos()[0].cpu().numpy().astype(np.float32))
            self._home_ee_quat = (self._grip_center_quat()[0].cpu().numpy().astype(np.float32))

    def spawn_random_box(self) -> tuple[float, float, float]:
        nx = float(torch.empty(1).uniform_(-BOX_SPAWN_XY_NOISE, BOX_SPAWN_XY_NOISE).item())
        ny = float(torch.empty(1).uniform_(-BOX_SPAWN_XY_NOISE, BOX_SPAWN_XY_NOISE).item())
        bx = BOX_SPAWN[0] + nx
        by = BOX_SPAWN[1] + ny
        yaw = float(torch.empty(1).uniform_(-BOX_SPAWN_YAW_MAX, BOX_SPAWN_YAW_MAX).item())
        pos_w = torch.tensor(
            [[bx + self.env_origin[0].item(),
              by + self.env_origin[1].item(),
              BOX_SPAWN[2] + self.env_origin[2].item()]],
            device=self.device, dtype=torch.float)
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=self.device, dtype=torch.float)
        box_quat = quat_from_angle_axis(
            torch.tensor([yaw], device=self.device, dtype=torch.float), z_axis)
        pose = torch.cat([pos_w, box_quat], dim=-1)
        self.box.write_root_pose_to_sim(pose)
        self.box.write_root_velocity_to_sim(
            torch.zeros((1, 6), device=self.device, dtype=torch.float))
        self._last_box_spawn = (bx, by, yaw)
        return bx, by, yaw

    def spawn_random_cell(self) -> tuple[float, float, float]:
        nx = float(torch.empty(1).uniform_(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE).item())
        ny = float(torch.empty(1).uniform_(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE).item())
        cx = CELL_CENTER_X + nx
        cy = CELL_CENTER_Y + ny
        cyaw = float(torch.empty(1).uniform_(-CELL_SPAWN_YAW_MAX, CELL_SPAWN_YAW_MAX).item())
        cos_y = math.cos(cyaw); sin_y = math.sin(cyaw)
        half = cyaw / 2.0
        cell_quat = torch.tensor(
            [[math.cos(half), 0.0, 0.0, math.sin(half)]],
            device=self.device, dtype=torch.float)
        for name, dx_loc, dy_loc in self._cell_wall_offsets:
            wx_local = cos_y * dx_loc - sin_y * dy_loc
            wy_local = sin_y * dx_loc + cos_y * dy_loc
            pos_w = torch.tensor(
                [[cx + wx_local + self.env_origin[0].item(),
                  cy + wy_local + self.env_origin[1].item(),
                  _WALL_Z + self.env_origin[2].item()]],
                device=self.device, dtype=torch.float)
            pose = torch.cat([pos_w, cell_quat], dim=-1)
            self.scene[name].write_root_pose_to_sim(pose)
            self.scene[name].write_root_velocity_to_sim(
                torch.zeros((1, 6), device=self.device, dtype=torch.float))
        self._last_cell_spawn = (cx, cy, cyaw)
        return cx, cy, cyaw

    def get_box_gt(self) -> BoxGtPose | None:
        pos_w = self.box.data.root_pos_w[0] - self.env_origin
        quat = self.box.data.root_quat_w[0].cpu().numpy()
        w, x, y, z = quat
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return BoxGtPose(xy=(float(pos_w[0]), float(pos_w[1])), yaw=yaw)

    def get_cell_gt(self) -> BoxGtPose | None:
        wall_xy = torch.stack([
            self.scene["wall_v_left"].data.root_pos_w[0, :2],
            self.scene["wall_v_right"].data.root_pos_w[0, :2],
            self.scene["wall_h_front"].data.root_pos_w[0, :2],
            self.scene["wall_h_back"].data.root_pos_w[0, :2],
        ]).mean(dim=0) - self.env_origin[:2]
        quat = self.scene["wall_v_left"].data.root_quat_w[0].cpu().numpy()
        w, x, y, z = quat
        yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return BoxGtPose(xy=(float(wall_xy[0]), float(wall_xy[1])), yaw=yaw)

    def get_base_ee_quat(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    def get_home_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._home_ee_pos is None:
            self.reset_to_home()
        return self._home_ee_pos, self._home_ee_quat

    @property
    def control_dt(self) -> float:
        return self._dt
