"""motion1 — YOLO 박스 detection 데이터셋 수집 (Step 7).

매 iteration sim 에서 random scene 만들고 천장 cam + 손목 cam 각각 RGB + 박스 mask 캡처.
박스 mask → axis-aligned bbox → YOLO label 형식 (class_id cx cy w h, normalized).

설정:
- 박스 spawn: 위치 (BOX_SPAWN, xy ±10cm noise), yaw ±π/2
- 셀 spawn: xy ±10cm noise (background 다양화), yaw ±π/2
- ee pose: 박스 위 (±5cm) + z 0.15~0.35m + yaw ±π/2 (chain runner 분포)
- 조명 intensity: 1000~4000 random
- 박스 material: metallic 0.85, roughness 0.12 (chain runner 와 동일)

저장:
    source/motion1/yolo_dataset/
        images/train/{cam}_{idx:06d}.jpg
        images/val/{cam}_{idx:06d}.jpg
        labels/train/{cam}_{idx:06d}.txt
        labels/val/{cam}_{idx:06d}.txt
        data.yaml

실행 (headless 권장):
    ./isaaclab.sh -p source/motion1/scripts/collect_yolo_dataset.py \
        --headless --enable_cameras --target 1500
"""
from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

# -------------------- argparse / app launcher --------------------
parser = argparse.ArgumentParser(description="motion1 YOLO 박스 detection 데이터셋 수집")
parser.add_argument("--target", type=int, default=1500,
                    help="총 데이터셋 이미지 개수 (천장+손목 합산). default 1500")
parser.add_argument("--output_dir", type=str,
                    default="source/motion1/yolo_dataset",
                    help="데이터셋 저장 root")
parser.add_argument("--val_ratio", type=float, default=0.2,
                    help="val split ratio. default 0.2 (20%)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--min_bbox_px", type=int, default=200,
                    help="bbox area 최소 픽셀 수 (이보다 작으면 skip)")
parser.add_argument("--settle_steps", type=int, default=30,
                    help="ee pose 변경 후 sim settle step")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------- imports (after app start) --------------------
import os
import random
import shutil
import yaml
import numpy as np
import torch
import cv2

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

# -------------------- constants (chain runner 와 동일) --------------------
BOX_SPAWN = (0.30, -0.10, 0.07)
BOX_SIZE = (0.139, 0.044, 0.118)
BOX_MASS = 0.3

CELL_CENTER_X = 0.30
CELL_CENTER_Y = -0.30
CELL_INNER_X = 0.16
CELL_INNER_Y = 0.065
WALL_THICKNESS = 0.008
WALL_HEIGHT = 0.12

# 데이터셋 수집 시 더 다양한 박스 / 셀 분포
BOX_SPAWN_XY_NOISE = 0.10        # ±10cm
BOX_SPAWN_YAW_MAX  = math.pi / 2  # ±π/2
CELL_SPAWN_XY_NOISE = 0.10
CELL_SPAWN_YAW_MAX  = math.pi / 2

# ee pose random 범위 (chain runner 분포)
EE_XY_NOISE = 0.05          # 박스 위 ±5cm
EE_Z_MIN = 0.15
EE_Z_MAX = 0.35
EE_YAW_MAX = math.pi / 2     # ±π/2

# 조명 intensity range
LIGHT_INTENSITY_MIN = 800.0
LIGHT_INTENSITY_MAX = 4000.0

# 카메라 cfg (chain runner 와 동일)
TOP_CAM_POS_ENV = (0.30, -0.20, 0.80)
TOP_CAM_ROT_ROS = (0.0, 1.0, 0.0, 0.0)
TOP_CAM_W = 640
TOP_CAM_H = 480
WRIST_CAM_POS = (0.0, -0.1, 0.084)
WRIST_CAM_ROT = (0.0, 0.0, 0.7071068, -0.7071068)
WRIST_CAM_W = 320
WRIST_CAM_H = 240

_V_WALL_SIZE = (WALL_THICKNESS, CELL_INNER_Y + 2 * WALL_THICKNESS, WALL_HEIGHT)
_H_WALL_SIZE = (CELL_INNER_X + 2 * WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT)
_WALL_Z = WALL_HEIGHT / 2

YOLO_CLASS_NAMES = ["box", "cell"]  # box=0, cell=1
YOLO_CLASS_ID_BOX = 0
YOLO_CLASS_ID_CELL = 1


# -------------------- Scene cfg --------------------
@configclass
class CollectSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = OMY_OFF_SELF_COLLISION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.7, 0.72), metallic=0.85, roughness=0.12),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                static_friction=3.0, dynamic_friction=3.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_SPAWN),
    )

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
                 CELL_CENTER_Y, _WALL_Z)),
    )
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
                 CELL_CENTER_Y, _WALL_Z)),
    )
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
                 CELL_CENTER_Y - CELL_INNER_Y / 2 - WALL_THICKNESS / 2, _WALL_Z)),
    )
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
                 CELL_CENTER_Y + CELL_INNER_Y / 2 + WALL_THICKNESS / 2, _WALL_Z)),
    )

    # 천장 cam
    top_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/TopCam",
        update_period=0.0,
        height=TOP_CAM_H, width=TOP_CAM_W,
        data_types=["rgb", "instance_id_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0)),
        offset=CameraCfg.OffsetCfg(pos=TOP_CAM_POS_ENV, rot=TOP_CAM_ROT_ROS, convention="ros"),
        colorize_instance_id_segmentation=False,
        update_latest_camera_pose=True,
    )

    # 손목 cam (외부 prim, manual pose set 으로 link6 추적)
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/WristCam",
        update_period=0.0,
        height=WRIST_CAM_H, width=WRIST_CAM_W,
        data_types=["rgb", "instance_id_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=11.0, focus_distance=0.25, horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0)),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0),
                                   convention="world"),
        colorize_instance_id_segmentation=False,
        update_latest_camera_pose=True,
    )


# -------------------- helpers --------------------
def grip_center_pos(robot, l_id, r_id):
    return 0.5 * (robot.data.body_pos_w[:, l_id] + robot.data.body_pos_w[:, r_id])

def grip_center_quat(robot, l_id):
    return robot.data.body_quat_w[:, l_id]

def grip_center_jacobian(robot, l_jac_idx, r_jac_idx, joint_ids):
    J = robot.root_physx_view.get_jacobians()
    j_l = J[:, l_jac_idx, :, :][:, :, joint_ids]
    j_r = J[:, r_jac_idx, :, :][:, :, joint_ids]
    return 0.5 * (j_l + j_r)


def mask_to_polygon_yolo(mask: np.ndarray, min_area_px: int,
                         max_points: int = 64) -> list[tuple[float, float]] | None:
    """박스 mask (H,W bool) → YOLO seg polygon [(x_norm, y_norm), ...].

    None if area < min_area_px or contour 부적합.
    """
    if not mask.any():
        return None
    if int(mask.sum()) < min_area_px:
        return None
    H, W = mask.shape
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < min_area_px:
        return None
    # simplify
    epsilon = 0.005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    pts = approx.reshape(-1, 2)
    # 점 너무 많으면 sample
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    # YOLO seg 는 최소 3 점 필요
    if len(pts) < 3:
        return None
    poly = [(float(x) / W, float(y) / H) for x, y in pts]
    return poly


def write_yolo_seg_label(path: str,
                         items: list[tuple[int, list[tuple[float, float]]]]):
    """items: list of (class_id, polygon). polygon = [(x_norm, y_norm), ...]."""
    with open(path, "w") as f:
        for cls_id, poly in items:
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            f.write(f"{cls_id} {coords}\n")


def set_dome_light_intensity(stage_or_helper, value: float):
    """dome light intensity 변경. omni.usd 의 stage 에 직접 attribute set."""
    try:
        import omni.usd
        from pxr import UsdLux
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath("/World/Light")
        if not prim.IsValid():
            return
        light = UsdLux.DomeLight(prim)
        light.GetIntensityAttr().Set(value)
    except Exception as e:
        print(f"[WARN] dome light intensity set failed: {e}")


# -------------------- main collection loop --------------------
def run_collection(sim, scene):
    robot = scene["robot"]
    box   = scene["box"]
    top_cam = scene["top_cam"]
    wrist_cam = scene["wrist_cam"]
    device = sim.device
    dt = sim.get_physics_dt()

    random.seed(args_cli.seed)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    arm_names = [f"joint{i}" for i in range(1, 7)]
    gripper_names = ["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"]
    arm_ids = [robot.find_joints(n)[0][0] for n in arm_names]
    gripper_ids = [robot.find_joints(n)[0][0] for n in gripper_names]
    all_joint_ids = arm_ids + gripper_ids
    left_id = robot.find_bodies("rh_p12_rn_l2")[0][0]
    right_id = robot.find_bodies("rh_p12_rn_r2")[0][0]
    link6_id = robot.find_bodies("link6")[0][0]
    if robot.is_fixed_base:
        l_jac, r_jac = left_id - 1, right_id - 1
    else:
        l_jac, r_jac = left_id, right_id

    ik_cfg = DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=device)

    # Home reset
    HOME_JOINT_POS = {
        "joint1": 0.0, "joint2": -1.55, "joint3": 2.66,
        "joint4": -1.1, "joint5": 1.6, "joint6": 0.0,
        "rh_r1_joint": 0.0, "rh_r2": 0.0, "rh_l1": 0.0, "rh_l2": 0.0,
    }
    home_q = torch.zeros((scene.num_envs, robot.num_joints), device=device)
    for n, v in HOME_JOINT_POS.items():
        jid = robot.find_joints(n)[0][0]
        home_q[:, jid] = v
    joint_vel = torch.zeros_like(home_q)
    robot.write_joint_state_to_sim(home_q, joint_vel)
    robot.set_joint_position_target(home_q)
    robot.reset()
    box.reset()
    for _ in range(60):
        scene.write_data_to_sim(); sim.step(); scene.update(dt)

    env_origin = scene.env_origins[0]
    base_ee_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)

    # 손목 cam manual pose
    _wrist_offset_pos = torch.tensor([list(WRIST_CAM_POS)], device=device, dtype=torch.float32)
    _wrist_offset_quat = torch.tensor([list(WRIST_CAM_ROT)], device=device, dtype=torch.float32)

    def _update_wrist_cam_pose():
        link6_pos_w = robot.data.body_pos_w[:, link6_id]
        link6_quat_w = robot.data.body_quat_w[:, link6_id]
        offset_pos_world = quat_apply(link6_quat_w, _wrist_offset_pos.expand_as(link6_pos_w))
        cam_pos_w_t = link6_pos_w + offset_pos_world
        cam_quat_w_ros = quat_mul(link6_quat_w, _wrist_offset_quat.expand(link6_quat_w.shape))
        wrist_cam.set_world_poses(cam_pos_w_t, cam_quat_w_ros, convention="ros")

    # 셀 wall 위치 갱신
    _CELL_WALL_OFFSETS = [
        ("wall_v_left",  -(CELL_INNER_X / 2 + WALL_THICKNESS / 2), 0.0),
        ("wall_v_right", +(CELL_INNER_X / 2 + WALL_THICKNESS / 2), 0.0),
        ("wall_h_front", 0.0, -(CELL_INNER_Y / 2 + WALL_THICKNESS / 2)),
        ("wall_h_back",  0.0, +(CELL_INNER_Y / 2 + WALL_THICKNESS / 2)),
    ]

    def update_cell_walls(cx_: float, cy_: float, cyaw_: float):
        cos_y = math.cos(cyaw_); sin_y = math.sin(cyaw_)
        half = cyaw_ / 2.0
        cell_quat = torch.tensor(
            [[math.cos(half), 0.0, 0.0, math.sin(half)]],
            device=device, dtype=torch.float)
        for name, dx_loc, dy_loc in _CELL_WALL_OFFSETS:
            wx_local = cos_y * dx_loc - sin_y * dy_loc
            wy_local = sin_y * dx_loc + cos_y * dy_loc
            wall_pos_w = torch.tensor(
                [[cx_ + wx_local + env_origin[0].item(),
                  cy_ + wy_local + env_origin[1].item(),
                  _WALL_Z + env_origin[2].item()]],
                device=device, dtype=torch.float)
            pose = torch.cat([wall_pos_w, cell_quat], dim=-1)
            scene[name].write_root_pose_to_sim(pose)
            scene[name].write_root_velocity_to_sim(
                torch.zeros((1, 6), device=device, dtype=torch.float))

    def set_box_pose(bx: float, by: float, byaw: float):
        box_pos_w = torch.tensor(
            [[bx + env_origin[0].item(),
              by + env_origin[1].item(),
              BOX_SPAWN[2] + env_origin[2].item()]], device=device, dtype=torch.float)
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float)
        box_quat = quat_from_angle_axis(
            torch.tensor([byaw], device=device, dtype=torch.float), z_axis)
        pose = torch.cat([box_pos_w, box_quat], dim=-1)
        vel = torch.zeros((1, 6), device=device, dtype=torch.float)
        box.write_root_pose_to_sim(pose)
        box.write_root_velocity_to_sim(vel)

    # IK 한 번 호출로 ee pose 변경 후 sim settle
    def move_ee_to(target_pos_env, target_yaw: float, settle_steps: int = 30):
        z_axis_t = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float)
        yaw_q = quat_from_angle_axis(
            torch.tensor([target_yaw], device=device, dtype=torch.float), z_axis_t)
        target_quat_w = quat_mul(yaw_q, base_ee_quat)
        target_pos_w = target_pos_env.unsqueeze(0) + env_origin.unsqueeze(0)
        gripper_target = torch.zeros((scene.num_envs, 4), device=device)
        for _ in range(settle_steps):
            ee_pos_w = grip_center_pos(robot, left_id, right_id)
            ee_quat_w = grip_center_quat(robot, left_id)
            cur_arm_q = robot.data.joint_pos[:, arm_ids]
            jac = grip_center_jacobian(robot, l_jac, r_jac, arm_ids)
            root_pos_w = robot.data.root_pos_w
            root_quat_w = robot.data.root_quat_w
            tgt_pos_b, tgt_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, target_pos_w, target_quat_w)
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
            ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))
            arm_target = ik.compute(ee_pos_b, ee_quat_b, jac, cur_arm_q)
            full_target = torch.cat([arm_target, gripper_target], dim=-1)
            robot.set_joint_position_target(full_target, joint_ids=all_joint_ids)
            scene.write_data_to_sim(); sim.step(); scene.update(dt)

    # 박스 + 셀(4 wall) instance ID 캐싱
    _box_ids_top: list[int] = []
    _box_ids_wrist: list[int] = []
    _cell_ids_top: list[int] = []
    _cell_ids_wrist: list[int] = []

    def find_ids(cam, cache_box: list[int], cache_cell: list[int]):
        if cache_box and cache_cell:
            return
        info = cam.data.info[0].get("instance_id_segmentation_fast", {})
        raw = info.get("idToLabels", {})
        for k, v in raw.items():
            try:
                sv = str(v)
                kid = int(k)
            except (ValueError, TypeError):
                continue
            if "CellWall" in sv:
                if kid not in cache_cell:
                    cache_cell.append(kid)
            elif "/Box" in sv:
                if kid not in cache_box:
                    cache_box.append(kid)

    # 출력 dir
    root = args_cli.output_dir
    if os.path.exists(root):
        shutil.rmtree(root)
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)

    # ============= main loop =============
    target_per_cam = args_cli.target // 2
    val_per_cam = int(target_per_cam * args_cli.val_ratio)
    train_per_cam = target_per_cam - val_per_cam

    saved_top = 0
    saved_wrist = 0
    skipped = 0
    iter_count = 0

    def save_capture(cam, cam_tag: str, saved_count: int) -> bool:
        """현재 cam frame 캡처 → 박스/셀 mask 추출 → image + label 저장.
        Returns True if saved (둘 중 하나라도 valid mask)."""
        cam.update(dt=dt, force_recompute=True)
        if "rgb" not in cam.data.output or "instance_id_segmentation_fast" not in cam.data.output:
            return False
        rgb = cam.data.output["rgb"][0].cpu().numpy()
        if rgb.shape[-1] >= 3:
            rgb = rgb[..., :3]
        seg = cam.data.output["instance_id_segmentation_fast"][0].squeeze(-1).cpu().numpy().astype(np.int32)
        cache_box = _box_ids_top if cam_tag == "top" else _box_ids_wrist
        cache_cell = _cell_ids_top if cam_tag == "top" else _cell_ids_wrist
        find_ids(cam, cache_box, cache_cell)

        items: list[tuple[int, list[tuple[float, float]]]] = []
        # 박스 mask
        if cache_box:
            mask_box = np.isin(seg, cache_box)
            poly_box = mask_to_polygon_yolo(mask_box, args_cli.min_bbox_px)
            if poly_box is not None:
                items.append((YOLO_CLASS_ID_BOX, poly_box))
        # 셀 mask (4 wall 합집합 → morphological close 로 outline)
        if cache_cell:
            wall_mask = np.isin(seg, cache_cell)
            if wall_mask.any():
                kern = np.ones((9, 9), np.uint8)
                cell_mask = cv2.morphologyEx(
                    (wall_mask.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kern) > 0
                poly_cell = mask_to_polygon_yolo(cell_mask, args_cli.min_bbox_px)
                if poly_cell is not None:
                    items.append((YOLO_CLASS_ID_CELL, poly_cell))

        if not items:
            return False
        split = "val" if saved_count < val_per_cam else "train"
        idx = saved_count
        img_path = os.path.join(root, "images", split, f"{cam_tag}_{idx:06d}.jpg")
        lbl_path = os.path.join(root, "labels", split, f"{cam_tag}_{idx:06d}.txt")
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(img_path, rgb_bgr)
        write_yolo_seg_label(lbl_path, items)
        return True

    print(f"[collect] target={args_cli.target} (top={target_per_cam}+wrist={target_per_cam}), "
          f"val_ratio={args_cli.val_ratio}, output={root}")

    while saved_top < target_per_cam or saved_wrist < target_per_cam:
        iter_count += 1

        # Random scene
        bx = BOX_SPAWN[0] + random.uniform(-BOX_SPAWN_XY_NOISE, BOX_SPAWN_XY_NOISE)
        by = BOX_SPAWN[1] + random.uniform(-BOX_SPAWN_XY_NOISE, BOX_SPAWN_XY_NOISE)
        byaw = random.uniform(-BOX_SPAWN_YAW_MAX, BOX_SPAWN_YAW_MAX)
        set_box_pose(bx, by, byaw)

        cx = CELL_CENTER_X + random.uniform(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE)
        cy = CELL_CENTER_Y + random.uniform(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE)
        cyaw = random.uniform(-CELL_SPAWN_YAW_MAX, CELL_SPAWN_YAW_MAX)
        update_cell_walls(cx, cy, cyaw)

        # Random ee pose — 박스 위 OR 셀 위 (50/50). 손목 cam 이 둘 다 학습 데이터 확보.
        if random.random() < 0.5:
            anchor_xy = (bx, by)   # 박스 위
        else:
            anchor_xy = (cx, cy)   # 셀 위 (insert RL 시점 대비)
        ee_x = anchor_xy[0] + random.uniform(-EE_XY_NOISE, EE_XY_NOISE)
        ee_y = anchor_xy[1] + random.uniform(-EE_XY_NOISE, EE_XY_NOISE)
        ee_z = random.uniform(EE_Z_MIN, EE_Z_MAX)
        ee_yaw = random.uniform(-EE_YAW_MAX, EE_YAW_MAX)
        ee_target = torch.tensor([ee_x, ee_y, ee_z], device=device, dtype=torch.float)
        move_ee_to(ee_target, ee_yaw, settle_steps=args_cli.settle_steps)

        # Random dome light intensity (domain randomization)
        intensity = random.uniform(LIGHT_INTENSITY_MIN, LIGHT_INTENSITY_MAX)
        set_dome_light_intensity(None, intensity)

        # 손목 cam pose 갱신
        _update_wrist_cam_pose()

        # 캡처 시도 (양쪽 cam)
        if saved_top < target_per_cam:
            if save_capture(top_cam, "top", saved_top):
                saved_top += 1
            else:
                skipped += 1
        if saved_wrist < target_per_cam:
            if save_capture(wrist_cam, "wrist", saved_wrist):
                saved_wrist += 1
            else:
                skipped += 1

        if iter_count % 50 == 0:
            print(f"[collect] iter={iter_count} saved top={saved_top}/{target_per_cam} "
                  f"wrist={saved_wrist}/{target_per_cam} skipped={skipped} "
                  f"light={intensity:.0f}")

    # data.yaml 작성
    yaml_path = os.path.join(root, "data.yaml")
    abs_root = os.path.abspath(root)
    data_cfg = {
        "path": abs_root,
        "train": "images/train",
        "val": "images/val",
        "nc": len(YOLO_CLASS_NAMES),
        "names": YOLO_CLASS_NAMES,
    }
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_cfg, f, sort_keys=False)
    print(f"\n[collect] DONE. iter={iter_count} top={saved_top} wrist={saved_wrist} skipped={skipped}")
    print(f"[collect] data.yaml -> {yaml_path}")
    print(f"[collect] dataset root -> {abs_root}")


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.4, 0.4, 0.9], [0.3, -0.25, 0.10])
    scene_cfg = CollectSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[collect] Sim ready.")
    run_collection(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    os._exit(0)
