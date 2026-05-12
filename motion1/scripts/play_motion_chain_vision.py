"""motion1 — Motion chain (vision-only, 천장 + 손목 cam).

play_motion_chain_with_grasp_insert_camera.py 의 vision-only 버전:
 - **GT fallback 제거** — YOLO 검출 실패 시 그 run skip
 - `--use_vision` flag 제거 (항상 vision)
 - `EE_OFFSET = 0` — 카메라 추정 좌표로 직행, 추가 random noise 없음
 - 손목 cam 으로 cell yaw 도 재추정 (stage 4)
 - `--no_rl` flag: RL 정책 두 stage 다 skip → pure vision + motion

카메라:
 - **천장 cam** (top_cam): run 시작 시 1회 scan → 박스 xy/yaw + 셀 xy/yaw (YOLO class 0/1).
 - **손목 cam** (wrist_cam, link6 부착): grasp/insert stage 에서 매 step (RL) 또는 단발 (no_rl) refine.

흐름:
    1. 천장 cam scan → 박스/셀 pose 추정 (검출 실패 시 run skip)
    2. Stage 1 motion: home → 박스 위 (offset 0, vision 좌표 직행)
    3. Stage 2:
       - default: Grasp RL (손목 cam yaw 매 step refine → 정책 obs)
       - `--no_rl`: 손목 cam 단발 refine → motion 으로 ee yaw 정렬
    4. Stage 3a-c: descend → close → lift
    5. Stage 3d: transport → cell xy, ee_yaw → 0
    6. Stage 4:
       - default: Insert RL (yaw-only, 손목 cam 으로 cell_yaw 매 step refine)
       - `--no_rl`: 손목 cam 단발 refine → motion 으로 ee yaw 정렬
    7. Stage 5a-b: insert descend → release
    8. Stage 6a-b: retract up → home

실행:
    ./isaaclab.sh -p source/motion1/scripts/play_motion_chain_vision.py \\
        --enable_cameras --repeat 5 --hold_s 5
    ./isaaclab.sh -p source/motion1/scripts/play_motion_chain_vision.py \\
        --enable_cameras --no_rl --repeat 5 --hold_s 5
"""
from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

# -------------------- argparse / app launcher --------------------
parser = argparse.ArgumentParser(description="motion1 chain + grasp RL (wrist cam) + rule-based insert (top cam)")
parser.add_argument("--gripper_close", type=float, default=0.8)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--hold_s", type=float, default=5.0)
parser.add_argument("--grasp_checkpoint", type=str, default="checkpoints/motion1_grasp.zip")
parser.add_argument("--grasp_vecnorm",    type=str, default="checkpoints/motion1_grasp_vecnorm.pkl")
parser.add_argument("--insert_checkpoint", type=str, default="checkpoints/motion1_insert.zip")
parser.add_argument("--insert_vecnorm",    type=str, default="checkpoints/motion1_insert_vecnorm.pkl")
parser.add_argument("--rl_max_steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--no_rl", action="store_true",
                    help="RL 정책 두 stage 다 skip — pure vision + motion 만으로 진행")
parser.add_argument("--yolo_ckpt", type=str, default=None,
                    help="YOLO seg weights 경로 (없으면 default 모델). "
                         "예: runs/segment/source/motion1/yolo_runs/v1_seg/weights/best.pt")
parser.add_argument("--save_camera_debug", action="store_true",
                    help="scan 시 RGB/seg 이미지를 /tmp/motion1_cam_*.png 에 저장")
parser.add_argument("--csv_out", type=str, default=None,
                    help="벤치마크 CSV 저장 경로 (예: /tmp/bench_no_rl.csv)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------- imports (after app start) --------------------
import os
import pickle
import time
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
    quat_apply, quat_from_angle_axis, quat_mul, quat_slerp,
    subtract_frame_transforms, transform_points,
)

from stable_baselines3 import PPO

# YOLO (optional — yolo_ckpt 지정 시만 사용)
_yolo_model = None
if args_cli.yolo_ckpt is not None:
    from ultralytics import YOLO as _YOLO_cls
    _yolo_model = _YOLO_cls(args_cli.yolo_ckpt)
    print(f"[YOLO] loaded {args_cli.yolo_ckpt}")

from source.omy.omy_robot_cfg import OMY_OFF_SELF_COLLISION_CFG


def yolo_predict_mask(rgb_np: np.ndarray, class_id: int,
                      conf: float = 0.5) -> np.ndarray | None:
    """RGB (H,W,3) → 지정 class_id 의 best mask (H,W bool). 검출 실패 시 None.

    class_id: 0=box, 1=cell (v2_seg_2class 모델 기준).
    """
    if _yolo_model is None:
        return None
    H, W = rgb_np.shape[:2]
    results = _yolo_model.predict(rgb_np, conf=conf, verbose=False, imgsz=640)
    r = results[0]
    if r.masks is None or r.masks.xy is None or len(r.masks.xy) == 0:
        return None
    if r.boxes is None or r.boxes.cls is None:
        return None
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.ones_like(cls_ids, dtype=np.float32)
    candidates = [i for i, c in enumerate(cls_ids) if c == class_id]
    if not candidates:
        return None
    best_idx = max(candidates, key=lambda i: confs[i])
    poly = r.masks.xy[best_idx]
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
    return mask.astype(bool)


def yolo_predict_box_mask(rgb_np: np.ndarray, conf: float = 0.5) -> np.ndarray | None:
    """박스 (class_id=0) 호환 wrapper."""
    return yolo_predict_mask(rgb_np, class_id=0, conf=conf)

# -------------------- constants --------------------
BOX_SPAWN = (0.45, 0.10, 0.07)    # swap: cell 자리로 (오른쪽 위)
BOX_SIZE = (0.139, 0.044, 0.118)
BOX_MASS = 0.3
BOX_TOP_Z = BOX_SPAWN[2] + BOX_SIZE[2] / 2  # 0.129

CELL_CENTER_X = 0.35              # swap: box 자리로 (왼쪽 앞)
CELL_CENTER_Y = -0.20
CELL_INNER_X = 0.16
CELL_INNER_Y = 0.065
WALL_THICKNESS = 0.008
WALL_HEIGHT = 0.12
WALL_TOP_Z = WALL_HEIGHT  # 0.12

PRE_GRASP_Z = BOX_SPAWN[2] + 0.10        # 0.17
GRASP_Z     = BOX_SPAWN[2] + 0.058        # 0.128  (조금 위 잡기 — 박스 안정적 lift)
LIFT_Z      = 0.26
TRANSPORT_Z = 0.26
PLACE_Z     = BOX_SPAWN[2] + 0.095       # 0.165
RETRACT_Z   = 0.26

STAGE_DURATION_S: dict[str, float] = {
    "move_above_box": 2.5,
    "descend":        1.0,
    "close":          1.5,
    "lift":           2.0,
    "transport":      3.0,
    "insert":         1.5,
    "release":        0.7,
    "retract_up":     1.5,
    "retract_home":   3.0,
}
SETTLE_S = 0.8
GRIPPER_OPEN = 0.0

BOX_SPAWN_XY_NOISE = 0.00
BOX_SPAWN_YAW_MAX  = 1.396
CELL_SPAWN_XY_NOISE = 0.05       # ±5cm (insert dataset 분포 와 일치)
CELL_SPAWN_YAW_MAX  = 1.396      # ±80°
# Stage 1 끝 ee offset 0 — vision 좌표로 직행 (random noise 추가 X)
EE_OFFSET_MIN_M = 0.0
EE_OFFSET_MAX_M = 0.0
# Insert (transport 끝점) offset 0 — yaw-only 학습 환경과 일치 (cell xy 정확히)
INSERT_OFFSET_MIN_M = 0.0
INSERT_OFFSET_MAX_M = 0.0

RL_GRASP_ACTION_SCALE_XY = 0.01
RL_GRASP_ACTION_SCALE_YAW = 0.05
RL_INSERT_ACTION_SCALE_YAW = 0.05  # insert_env_cfg: ~2.86°/step (yaw-only)
RL_EE_YAW_MIN = -1.5708
RL_EE_YAW_MAX =  1.5708
RL_GRASP_ALIGN_XY = 0.005
RL_GRASP_ALIGN_YAW = 0.05
RL_GRASP_HOLD_STEPS = 30

# Insert RL success 판정 (yaw-only)
RL_INSERT_ALIGN_YAW = 0.052     # ~3°
RL_INSERT_HOLD_STEPS = 30       # 30 step (0.5초)

# Insert 시 ee target offset (cell frame 기준, cell_yaw 따라 회전)
# +X = cell long axis, +Y = cell short axis
INSERT_TARGET_X_OFFSET = -0.005     # -0.005: 아래로
INSERT_TARGET_Y_OFFSET = 0.03    # 0.02 m, cell short-axis (박스가 그리퍼에 빗나가게 잡힌 방향 보정)

_V_WALL_SIZE = (WALL_THICKNESS, CELL_INNER_Y + 2 * WALL_THICKNESS, WALL_HEIGHT)
_H_WALL_SIZE = (CELL_INNER_X + 2 * WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT)
_WALL_Z = WALL_HEIGHT / 2

# -------------------- camera 설정 --------------------
# 천장 cam: 워크스페이스 위 80cm, 정 아래 응시. ROS 180° about X.
TOP_CAM_POS_ENV = (0.40, -0.05, 0.80)   # box (0.45,+0.10) + cell (0.35,-0.20) 중간점
TOP_CAM_ROT_ROS = (0.0, 1.0, 0.0, 0.0)
TOP_CAM_W = 640
TOP_CAM_H = 480

# 손목 cam: link6 에 부착 (URDF + 검증된 OMY vision cfg 와 일치).
#   pos=(0,-0.1,0.084) link6-rel, rot=(0,0,0.7071,-0.7071) ROS — forward = link6 -Y (gripper).
WRIST_CAM_POS = (0.0, -0.1, 0.084)
WRIST_CAM_ROT = (0.0, 0.0, 0.7071068, -0.7071068)
WRIST_CAM_W = 320
WRIST_CAM_H = 240


# -------------------- Scene cfg --------------------
@configclass
class MotionSceneCfg(InteractiveSceneCfg):
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
        height=TOP_CAM_H,
        width=TOP_CAM_W,
        data_types=["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.05, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=TOP_CAM_POS_ENV, rot=TOP_CAM_ROT_ROS, convention="ros"),
        colorize_instance_id_segmentation=False,
        update_latest_camera_pose=True,  # pos_w/quat_w_world 매 update 시 갱신 (default=False 면 zero 로 남음)
    )

    # 손목 cam — articulation 외부 prim 으로 spawn (link6 child 면 hang).
    # 매 step manual 로 link6 world pose + WRIST offset 으로 set_world_poses 호출.
    # focal_length=11mm (FOV ≈ 87°, D405 spec 매칭). default 24mm 은 47° 좁음.
    wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/WristCam",
        update_period=0.0,
        height=WRIST_CAM_H,
        width=WRIST_CAM_W,
        data_types=["rgb", "distance_to_image_plane", "instance_id_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=11.0, focus_distance=0.25, horizontal_aperture=20.955,
            clipping_range=(0.01, 2.0),
        ),
        # offset 은 identity. 실제 pose 는 매 step set_world_poses 로.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        colorize_instance_id_segmentation=False,
        update_latest_camera_pose=True,
    )


# -------------------- helpers (motion + IK) --------------------
def grip_center_pos(robot, l_id, r_id):
    return 0.5 * (robot.data.body_pos_w[:, l_id] + robot.data.body_pos_w[:, r_id])

def grip_center_quat(robot, l_id):
    return robot.data.body_quat_w[:, l_id]

def grip_center_jacobian(robot, l_jac_idx, r_jac_idx, joint_ids):
    J = robot.root_physx_view.get_jacobians()
    j_l = J[:, l_jac_idx, :, :][:, :, joint_ids]
    j_r = J[:, r_jac_idx, :, :][:, :, joint_ids]
    return 0.5 * (j_l + j_r)

def grip_center_lin_vel(robot, l_id, r_id):
    return 0.5 * (robot.data.body_lin_vel_w[:, l_id] + robot.data.body_lin_vel_w[:, r_id])

def cartesian_lerp(start, end, num_steps):
    alphas = torch.linspace(0, 1, num_steps, device=start.device).unsqueeze(-1)
    return start.unsqueeze(0) * (1 - alphas) + end.unsqueeze(0) * alphas

def quat_z_yaw(q_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


# -------------------- vision helpers --------------------
def _id_to_path_mapping(info_entry: dict) -> dict[int, str]:
    mapping: dict[int, str] = {}
    if not info_entry:
        return mapping
    for k, v in info_entry.get("idToLabels", {}).items():
        try:
            mapping[int(k)] = str(v)
        except (ValueError, TypeError):
            continue
    return mapping


def _pixel_min_area_rect(mask_2d: np.ndarray) -> tuple[float, float, float, float]:
    """returns (cx_px, cy_px, angle_deg_image, area_px). 비어 있으면 area=0."""
    mask_u8 = mask_2d.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return -1.0, -1.0, 0.0, 0.0
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 5:
        return -1.0, -1.0, 0.0, 0.0
    (cx, cy), (w, h), angle = cv2.minAreaRect(cnt)
    if w < h:
        angle += 90.0
    return float(cx), float(cy), float(angle), float(cv2.contourArea(cnt))


def _unproject_pixel_to_world(
    u_px: float, v_px: float,
    depth_img: torch.Tensor,
    K: torch.Tensor,
    cam_pos_w: torch.Tensor,
    cam_quat_w_world: torch.Tensor,
    z_known: float | None = None,
    cam_z_w: float | None = None,
) -> tuple[float, float, float]:
    """단일 픽셀 (u,v) → world (x,y,z). depth 유효 윈도우 평균 → ROS pt → world axes swap."""
    H, W = depth_img.shape
    u_i, v_i = int(round(u_px)), int(round(v_px))
    u_i = max(0, min(W - 1, u_i))
    v_i = max(0, min(H - 1, v_i))
    win = 2
    u0, u1 = max(0, u_i - win), min(W, u_i + win + 1)
    v0, v1 = max(0, v_i - win), min(H, v_i + win + 1)
    patch = depth_img[v0:v1, u0:u1]
    valid = torch.isfinite(patch) & (patch > 0)
    if valid.any():
        d_use = float(patch[valid].mean().item())
    elif (z_known is not None) and (cam_z_w is not None):
        d_use = max(0.01, abs(cam_z_w - z_known))
    else:
        d_use = 0.5

    K_inv = torch.inverse(K)
    pix_h = torch.tensor(
        [u_px, v_px, 1.0], dtype=torch.float32, device=depth_img.device).unsqueeze(-1)
    ray_cam = (K_inv @ pix_h).squeeze(-1)
    ray_cam = ray_cam / ray_cam[-1]
    pt_cam = ray_cam * d_use  # ROS camera frame: x_right, y_down, z_forward
    # ROS → world-convention camera frame (forward=+X, left=+Y, up=+Z)
    pt_world_local = torch.stack([pt_cam[2], -pt_cam[0], -pt_cam[1]])
    pt_world = transform_points(
        pt_world_local.unsqueeze(0),
        cam_pos_w.unsqueeze(0).to(depth_img.device),
        cam_quat_w_world.unsqueeze(0).to(depth_img.device),
    ).squeeze(0)
    return float(pt_world[0]), float(pt_world[1]), float(pt_world[2])


def _world_yaw_from_image_angle_topdown(angle_deg_image: float) -> float:
    """천장 cam (180° about X). image x = world x, image y = world -y → world yaw = -image angle.

    minAreaRect 의 angle ∈ (0°, 180°] (w<h 보정 후) — long-edge 가 ±π 모호.
    Wrap to (-π/2, π/2] for canonical yaw.
    """
    yaw = -math.radians(angle_deg_image)
    if yaw > math.pi / 2:
        yaw -= math.pi
    elif yaw < -math.pi / 2:
        yaw += math.pi
    return yaw


# -------------------- pipeline --------------------
def run_pipeline(sim, scene):
    robot = scene["robot"]
    box   = scene["box"]
    top_cam = scene["top_cam"]
    wrist_cam = scene["wrist_cam"] if "wrist_cam" in scene.keys() else None
    if wrist_cam is None:
        print("[WARN] wrist_cam disabled — grasp RL 의 vision obs 비활성")
    device = sim.device
    dt = sim.get_physics_dt()
    DECIMATION = 1
    control_dt = dt * DECIMATION
    duration_to_steps = lambda s: max(1, int(s / control_dt))

    if args_cli.seed is not None:
        torch.manual_seed(args_cli.seed)

    arm_names = [f"joint{i}" for i in range(1, 7)]
    gripper_names = ["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"]
    arm_ids = [robot.find_joints(n)[0][0] for n in arm_names]
    gripper_ids = [robot.find_joints(n)[0][0] for n in gripper_names]
    all_joint_ids = arm_ids + gripper_ids
    left_id = robot.find_bodies("rh_p12_rn_l2")[0][0]
    link6_id = robot.find_bodies("link6")[0][0]  # 손목 cam manual pose 추적용
    right_id = robot.find_bodies("rh_p12_rn_r2")[0][0]
    if robot.is_fixed_base:
        l_jac, r_jac = left_id - 1, right_id - 1
    else:
        l_jac, r_jac = left_id, right_id

    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=device)

    def load_policy(ckpt_path: str, vec_path: str, tag: str):
        ckpt_abs = os.path.abspath(ckpt_path)
        vec_abs = os.path.abspath(vec_path)
        ckpt_size_kb = os.path.getsize(ckpt_path) / 1024 if os.path.exists(ckpt_path) else -1
        vec_size_kb = os.path.getsize(vec_path) / 1024 if os.path.exists(vec_path) else -1
        ckpt_mtime = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(ckpt_path)))
                      if os.path.exists(ckpt_path) else "N/A")
        vec_mtime = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(vec_path)))
                     if os.path.exists(vec_path) else "N/A")
        print(f"[{tag}] ckpt={ckpt_abs} ({ckpt_size_kb:.1f}KB, {ckpt_mtime})")
        print(f"[{tag}] vec ={vec_abs} ({vec_size_kb:.1f}KB, {vec_mtime})")
        m = PPO.load(ckpt_path, device=device)
        with open(vec_path, "rb") as f:
            v = pickle.load(f)
        v.training = False
        v.norm_reward = False
        def normalize(obs_np: np.ndarray) -> np.ndarray:
            return np.asarray(v.normalize_obs(obs_np[None, :])[0], dtype=np.float32)
        return m, normalize

    grasp_model, grasp_normalize_obs = load_policy(
        args_cli.grasp_checkpoint, args_cli.grasp_vecnorm, "chain+grasp")
    insert_model, insert_normalize_obs = load_policy(
        args_cli.insert_checkpoint, args_cli.insert_vecnorm, "chain+insert")

    HOME_JOINT_POS = {
        "joint1": 0.0, "joint2": -1.55, "joint3": 2.66,
        "joint4": -1.1, "joint5": 1.6, "joint6": 0.0,
        "rh_r1_joint": 0.0, "rh_r2": 0.0, "rh_l1": 0.0, "rh_l2": 0.0,
    }
    home_q = torch.zeros((scene.num_envs, robot.num_joints), device=device)
    for n, val in HOME_JOINT_POS.items():
        jid = robot.find_joints(n)[0][0]
        home_q[:, jid] = val
    joint_vel = torch.zeros_like(home_q)
    robot.write_joint_state_to_sim(home_q, joint_vel)
    robot.set_joint_position_target(home_q)
    robot.reset()
    box.reset()
    for _ in range(60):
        scene.write_data_to_sim(); sim.step(); scene.update(dt)

    env_origin = scene.env_origins[0]
    home_grip_w = grip_center_pos(robot, left_id, right_id)[0]
    home_grip_quat_w = grip_center_quat(robot, left_id)[0]
    home_grip_env = home_grip_w - env_origin
    print(f"[chain+rl+cam2] home grip env-rel: {home_grip_env.tolist()}")
    print(f"[chain+vision] vision=ON (always), no_rl={args_cli.no_rl}")

    base_ee_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
    gripper_close = float(args_cli.gripper_close)

    def control_step(target_pos_env, gripper_value, target_quat_w=None):
        if target_quat_w is None:
            target_quat_w = base_ee_quat
        target_pos_w = target_pos_env.unsqueeze(0) + env_origin.unsqueeze(0)
        tip_ratio = 2.3
        gripper_target = torch.tensor(
            [[gripper_value, gripper_value * tip_ratio, gripper_value, gripper_value * tip_ratio]],
            device=device).expand(scene.num_envs, -1)
        for _ in range(DECIMATION):
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

    def report(tag):
        ee = (grip_center_pos(robot, left_id, right_id)[0] - env_origin).tolist()
        bx_w = box.data.root_pos_w[0] - env_origin
        d = (grip_center_pos(robot, left_id, right_id)[0] - box.data.root_pos_w[0]).norm().item()
        print(f"  [status @ {tag:24s}] ee=({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f}) "
              f"box=({bx_w[0]:+.3f},{bx_w[1]:+.3f},{bx_w[2]:+.3f}) ee↔box={d*100:.1f}cm")

    def stage_move(start_pos_env, end_pos_env, dur_s, gripper_val, label,
                   start_quat_w=None, end_quat_w=None):
        n = duration_to_steps(dur_s); s = duration_to_steps(SETTLE_S)
        traj = cartesian_lerp(start_pos_env, end_pos_env, n)
        do_slerp = (start_quat_w is not None) and (end_quat_w is not None)
        slerp_tag = "+slerp" if do_slerp else ""
        print(f"[stage] {label:36s} | move={n} settle={s} grip={gripper_val:.2f}{slerp_tag}")
        for i in range(n):
            if do_slerp:
                tau = (i + 1) / n
                q_i = quat_slerp(start_quat_w[0], end_quat_w[0], tau).unsqueeze(0)
            else:
                q_i = end_quat_w
            control_step(traj[i], gripper_val, target_quat_w=q_i)
        end_q = end_quat_w if (end_quat_w is not None) else None
        for _ in range(s):
            control_step(end_pos_env, gripper_val, target_quat_w=end_q)
        report(label)

    def stage_hold(at_pos_env, dur_s, gripper_val, label, hold_quat_w=None):
        n = duration_to_steps(dur_s)
        print(f"[stage] {label:36s} | HOLD steps={n} grip={gripper_val:.2f}")
        for _ in range(n):
            control_step(at_pos_env, gripper_val, target_quat_w=hold_quat_w)
        report(label)

    # ============= 천장 cam scan (run 시작 시 1회) =============
    def top_cam_scan(rep_idx: int) -> dict:
        # 카메라 buffer 가 spawn 직후 갱신되도록 몇 frame 진행
        for _ in range(3):
            scene.write_data_to_sim(); sim.step(); scene.update(dt)
        top_cam.update(dt=control_dt, force_recompute=True)

        seg = top_cam.data.output["instance_id_segmentation_fast"][0].squeeze(-1).cpu().numpy().astype(np.int32)
        depth0 = top_cam.data.output["distance_to_image_plane"][0].squeeze(-1)
        rgb_t = top_cam.data.output.get("rgb")
        info_dict = top_cam.data.info[0].get("instance_id_segmentation_fast", {})
        K = top_cam.data.intrinsic_matrices[0]
        cam_pos_w = top_cam.data.pos_w[0]
        cam_quat = top_cam.data.quat_w_world[0]
        id_map = _id_to_path_mapping(info_dict)

        # ---- DEBUG ----
        print(f"  [DBG top_cam] cam_pos_w={cam_pos_w.tolist()} cam_quat={cam_quat.tolist()}")
        print(f"  [DBG top_cam] K={K.tolist()}")
        finite_mask = torch.isfinite(depth0) & (depth0 > 0)
        n_finite = int(finite_mask.sum().item())
        n_total = int(depth0.numel())
        if n_finite > 0:
            d_min = float(depth0[finite_mask].min().item())
            d_max = float(depth0[finite_mask].max().item())
            d_mean = float(depth0[finite_mask].mean().item())
        else:
            d_min = d_max = d_mean = float("nan")
        print(f"  [DBG top_cam] depth finite={n_finite}/{n_total} min={d_min:.3f} max={d_max:.3f} mean={d_mean:.3f}")
        print(f"  [DBG top_cam] seg uniq={np.unique(seg).tolist()[:10]} id_map_keys={list(id_map.keys())[:10]}")
        for k, v in list(id_map.items())[:8]:
            print(f"  [DBG top_cam]   id={k} path={v}")

        if args_cli.save_camera_debug and rgb_t is not None:
            os.makedirs("/tmp", exist_ok=True)
            rgb0 = rgb_t[0].cpu().numpy()
            cv2.imwrite(f"/tmp/motion1_topcam_rgb_run{rep_idx}.png",
                        cv2.cvtColor(rgb0[..., :3], cv2.COLOR_RGB2BGR))
            cv2.imwrite(f"/tmp/motion1_topcam_seg_run{rep_idx}.png",
                        (seg % 255).astype(np.uint8))
            print(f"  [top_cam] saved /tmp/motion1_topcam_{{rgb,seg}}_run{rep_idx}.png")

        box_xy_env = None
        box_yaw = None
        cell_xy_env = None
        cell_yaw = None

        # 박스 — YOLO 우선, 없으면 sim mask
        mask = None
        if _yolo_model is not None and rgb_t is not None:
            mask = yolo_predict_box_mask(rgb_t[0].cpu().numpy()[..., :3])
        if mask is None:
            box_ids = [i for i, p in id_map.items() if "/Box" in p and "CellWall" not in p]
            if box_ids:
                mask = np.isin(seg, box_ids)
        if mask is not None and mask.any():
            cx_px, cy_px, ang_img, area = _pixel_min_area_rect(mask)
            if area > 0:
                wx, wy, _ = _unproject_pixel_to_world(
                    cx_px, cy_px, depth0, K, cam_pos_w, cam_quat,
                    z_known=BOX_TOP_Z + env_origin[2].item(),
                    cam_z_w=float(cam_pos_w[2].item()))
                box_xy_env = (wx - env_origin[0].item(), wy - env_origin[1].item())
                box_yaw = _world_yaw_from_image_angle_topdown(ang_img)

        # 셀 — YOLO 우선 (class_id=1), 없으면 sim mask (4 wall 합집합 → morph close)
        cell_mask_bool = None
        if _yolo_model is not None and rgb_t is not None:
            cell_mask_bool = yolo_predict_mask(rgb_t[0].cpu().numpy()[..., :3], class_id=1)
        if cell_mask_bool is None:
            wall_ids = [i for i, p in id_map.items() if "CellWall" in p]
            if wall_ids:
                wall_mask = np.isin(seg, wall_ids)
                kern = np.ones((9, 9), np.uint8)
                cell_mask_bool = cv2.morphologyEx(
                    (wall_mask.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kern) > 0
        if cell_mask_bool is not None and cell_mask_bool.any():
            cx_px, cy_px, ang_img, area = _pixel_min_area_rect(cell_mask_bool)
            if area > 0:
                wx, wy, _ = _unproject_pixel_to_world(
                    cx_px, cy_px, depth0, K, cam_pos_w, cam_quat,
                    z_known=WALL_TOP_Z + env_origin[2].item(),
                    cam_z_w=float(cam_pos_w[2].item()))
                cell_xy_env = (wx - env_origin[0].item(), wy - env_origin[1].item())
                cell_yaw = _world_yaw_from_image_angle_topdown(ang_img)

        # gt 비교 print
        box_w = box.data.root_pos_w[0] - env_origin
        box_yaw_gt = float(quat_z_yaw(box.data.root_quat_w)[0].item())
        cell_xy_gt = torch.stack([
            scene["wall_v_left"].data.root_pos_w[0, :2],
            scene["wall_v_right"].data.root_pos_w[0, :2],
            scene["wall_h_front"].data.root_pos_w[0, :2],
            scene["wall_h_back"].data.root_pos_w[0, :2],
        ]).mean(dim=0) - env_origin[:2]
        cell_yaw_gt = float(quat_z_yaw(scene["wall_v_left"].data.root_quat_w)[0].item())

        def _fmt(xy, yaw):
            if xy is None:
                return "NONE"
            return f"({xy[0]:+.3f},{xy[1]:+.3f},yaw={math.degrees(yaw):+.1f}°)"
        print(f"  [top_cam] BOX  gt=({box_w[0]:+.3f},{box_w[1]:+.3f},yaw={math.degrees(box_yaw_gt):+.1f}°)  est={_fmt(box_xy_env, box_yaw if box_yaw else 0)}")
        print(f"  [top_cam] CELL gt=({cell_xy_gt[0]:+.3f},{cell_xy_gt[1]:+.3f},yaw={math.degrees(cell_yaw_gt):+.1f}°)  est={_fmt(cell_xy_env, cell_yaw if cell_yaw else 0)}")

        # 손목 cam home 자세 시점 RGB+overlay 저장
        _save_wrist_overlay(f"home_run{rep_idx}")

        return {
            "box_xy_env": box_xy_env, "box_yaw": box_yaw,
            "cell_xy_env": cell_xy_env, "cell_yaw": cell_yaw,
        }

    # ============= 손목 cam → 박스 pose 추정 (grasp 매 step) =============
    # 박스 instance ID 는 첫 호출 시 한 번만 알아내서 캐싱.
    _wrist_box_ids: list[int] = []

    def _save_wrist_overlay(label: str):
        """현재 손목 cam frame 을 RGB + box mask overlay (빨강) + 외곽선으로 /tmp 에 저장."""
        if wrist_cam is None or not args_cli.save_camera_debug:
            return
        _update_wrist_cam_pose()
        wrist_cam.update(dt=control_dt, force_recompute=True)
        if "rgb" not in wrist_cam.data.output:
            return
        rgb = wrist_cam.data.output["rgb"][0].cpu().numpy()
        if rgb.shape[-1] >= 3:
            rgb = rgb[..., :3]
        # YOLO 우선, 없으면 sim mask
        mask = None
        if _yolo_model is not None:
            mask = yolo_predict_box_mask(rgb)
        if mask is None:
            seg = wrist_cam.data.output["instance_id_segmentation_fast"][0].squeeze(-1).cpu().numpy().astype(np.int32)
            info = wrist_cam.data.info[0].get("instance_id_segmentation_fast", {})
            id_map = _id_to_path_mapping(info)
            box_ids_local = [i for i, p in id_map.items() if "/Box" in p and "CellWall" not in p]
            if box_ids_local:
                mask = np.isin(seg, box_ids_local)
        overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        if mask is not None and mask.any():
            red = np.zeros_like(overlay); red[..., 2] = 255
            overlay[mask] = (0.5 * overlay[mask] + 0.5 * red[mask]).astype(np.uint8)
            mask_u8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 1)
        path = f"/tmp/motion1_wristcam_{label}.png"
        cv2.imwrite(path, overlay)
        print(f"  [wrist_cam] saved {path}")

    # 손목 cam 의 link6-rel offset (URDF + omy_vision cfg 검증된 값)
    _wrist_offset_pos = torch.tensor([list(WRIST_CAM_POS)], device=device, dtype=torch.float32)
    _wrist_offset_quat = torch.tensor([list(WRIST_CAM_ROT)], device=device, dtype=torch.float32)

    def _update_wrist_cam_pose():
        """link6 world pose 를 읽어서 wrist_cam 을 link6 + WRIST offset 으로 set."""
        if wrist_cam is None:
            return
        link6_pos_w = robot.data.body_pos_w[:, link6_id]    # (N, 3)
        link6_quat_w = robot.data.body_quat_w[:, link6_id]  # (N, 4) wxyz
        # link6 frame 의 offset 위치를 world 로 회전
        offset_pos_world = quat_apply(link6_quat_w, _wrist_offset_pos.expand_as(link6_pos_w))
        cam_pos_w_t = link6_pos_w + offset_pos_world
        cam_quat_w_ros = quat_mul(link6_quat_w, _wrist_offset_quat.expand(link6_quat_w.shape))
        wrist_cam.set_world_poses(cam_pos_w_t, cam_quat_w_ros, convention="ros")

    def wrist_cam_estimate_box() -> tuple[float, float, float] | None:
        """현재 손목 cam 에서 박스 detection → world (box_x_env, box_y_env, box_yaw).

        검출 실패 또는 wrist_cam disable 시 None. 매 step 호출 가능.
        """
        if wrist_cam is None:
            return None
        _update_wrist_cam_pose()
        wrist_cam.update(dt=control_dt, force_recompute=True)
        if "distance_to_image_plane" not in wrist_cam.data.output:
            return None
        depth0 = wrist_cam.data.output["distance_to_image_plane"][0].squeeze(-1)
        K = wrist_cam.data.intrinsic_matrices[0]
        cam_pos_w = wrist_cam.data.pos_w[0]
        cam_quat = wrist_cam.data.quat_w_world[0]

        # YOLO 우선, 없으면 sim mask
        mask = None
        if _yolo_model is not None and "rgb" in wrist_cam.data.output:
            rgb = wrist_cam.data.output["rgb"][0].cpu().numpy()
            if rgb.shape[-1] >= 3:
                rgb = rgb[..., :3]
            mask = yolo_predict_box_mask(rgb)
        if mask is None:
            if "instance_id_segmentation_fast" not in wrist_cam.data.output:
                return None
            seg = wrist_cam.data.output["instance_id_segmentation_fast"][0].squeeze(-1).cpu().numpy().astype(np.int32)
            info_dict = wrist_cam.data.info[0].get("instance_id_segmentation_fast", {})
            id_map = _id_to_path_mapping(info_dict)
            nonlocal _wrist_box_ids
            if not _wrist_box_ids:
                _wrist_box_ids = [i for i, p in id_map.items() if "/Box" in p and "CellWall" not in p]
                if _wrist_box_ids:
                    print(f"  [wrist_cam] box ids cached: {_wrist_box_ids}")
            if not _wrist_box_ids:
                return None
            mask = np.isin(seg, _wrist_box_ids)
        # 4 corners + center 모두 image space 에서 추출 (cv2.minAreaRect)
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 5:
            return None
        rect = cv2.minAreaRect(cnt)
        (cx_px, cy_px), (w_rect, h_rect), _ang_img = rect
        box_pts = cv2.boxPoints(rect)  # (4, 2) in pixel coords

        # box top z 기준 — center pixel unproject 으로 xy
        z_known = BOX_TOP_Z + env_origin[2].item()
        cam_z_w_val = float(cam_pos_w[2].item())
        wx, wy, _ = _unproject_pixel_to_world(
            float(cx_px), float(cy_px), depth0, K, cam_pos_w, cam_quat,
            z_known=z_known, cam_z_w=cam_z_w_val)
        bx_env = wx - env_origin[0].item()
        by_env = wy - env_origin[1].item()

        # yaw — long edge 두 끝점 각각 world unproject → atan2 (cam quat 변환식 안 씀, robust)
        edge_01 = float(np.linalg.norm(box_pts[0] - box_pts[1]))
        edge_12 = float(np.linalg.norm(box_pts[1] - box_pts[2]))
        if edge_01 >= edge_12:
            p1, p2 = box_pts[0], box_pts[1]   # long edge endpoints
        else:
            p1, p2 = box_pts[1], box_pts[2]
        w1x, w1y, _ = _unproject_pixel_to_world(
            float(p1[0]), float(p1[1]), depth0, K, cam_pos_w, cam_quat,
            z_known=z_known, cam_z_w=cam_z_w_val)
        w2x, w2y, _ = _unproject_pixel_to_world(
            float(p2[0]), float(p2[1]), depth0, K, cam_pos_w, cam_quat,
            z_known=z_known, cam_z_w=cam_z_w_val)
        bx_yaw = math.atan2(w2y - w1y, w2x - w1x)
        # box long edge 방향성 모호 (±π 동일 박스). RL 학습 obs 의 yaw 정의 (±π/2 wrap) 와 맞추기 위해
        # ±π/2 안으로 wrap.
        bx_yaw = float(wrap_to_pi(torch.tensor([bx_yaw])).item())
        if bx_yaw > math.pi / 2:
            bx_yaw -= math.pi
        elif bx_yaw < -math.pi / 2:
            bx_yaw += math.pi

        return bx_env, by_env, bx_yaw

    # ============= 손목 cam → cell pose 추정 (insert 매 step) =============
    _wrist_cell_ids: list[int] = []

    def wrist_cam_estimate_cell() -> tuple[float, float, float] | None:
        """현재 손목 cam 에서 cell detection → (cell_x_env, cell_y_env, cell_yaw).

        검출 실패 시 None.
        """
        if wrist_cam is None:
            return None
        _update_wrist_cam_pose()
        wrist_cam.update(dt=control_dt, force_recompute=True)
        if "distance_to_image_plane" not in wrist_cam.data.output:
            return None
        depth0 = wrist_cam.data.output["distance_to_image_plane"][0].squeeze(-1)
        K = wrist_cam.data.intrinsic_matrices[0]
        cam_pos_w = wrist_cam.data.pos_w[0]
        cam_quat = wrist_cam.data.quat_w_world[0]

        # YOLO cell (class_id=1) 우선, 없으면 sim seg (4 wall → morph close)
        mask = None
        if _yolo_model is not None and "rgb" in wrist_cam.data.output:
            rgb = wrist_cam.data.output["rgb"][0].cpu().numpy()
            if rgb.shape[-1] >= 3:
                rgb = rgb[..., :3]
            mask = yolo_predict_mask(rgb, class_id=1)
        if mask is None:
            if "instance_id_segmentation_fast" not in wrist_cam.data.output:
                return None
            seg = wrist_cam.data.output["instance_id_segmentation_fast"][0].squeeze(-1).cpu().numpy().astype(np.int32)
            info_dict = wrist_cam.data.info[0].get("instance_id_segmentation_fast", {})
            id_map = _id_to_path_mapping(info_dict)
            nonlocal _wrist_cell_ids
            if not _wrist_cell_ids:
                _wrist_cell_ids = [i for i, p in id_map.items() if "CellWall" in p]
                if _wrist_cell_ids:
                    print(f"  [wrist_cam] cell wall ids cached: {_wrist_cell_ids}")
            if not _wrist_cell_ids:
                return None
            wall_mask = np.isin(seg, _wrist_cell_ids)
            kern = np.ones((9, 9), np.uint8)
            mask = cv2.morphologyEx((wall_mask.astype(np.uint8) * 255), cv2.MORPH_CLOSE, kern) > 0

        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 50:
            return None
        rect = cv2.minAreaRect(cnt)
        (cx_px, cy_px), (w_rect, h_rect), _ang_img = rect
        box_pts = cv2.boxPoints(rect)

        z_known = WALL_TOP_Z + env_origin[2].item()
        cam_z_w_val = float(cam_pos_w[2].item())
        wx, wy, _ = _unproject_pixel_to_world(
            float(cx_px), float(cy_px), depth0, K, cam_pos_w, cam_quat,
            z_known=z_known, cam_z_w=cam_z_w_val)
        cx_env_v = wx - env_origin[0].item()
        cy_env_v = wy - env_origin[1].item()

        edge_01 = float(np.linalg.norm(box_pts[0] - box_pts[1]))
        edge_12 = float(np.linalg.norm(box_pts[1] - box_pts[2]))
        if edge_01 >= edge_12:
            p1, p2 = box_pts[0], box_pts[1]
        else:
            p1, p2 = box_pts[1], box_pts[2]
        w1x, w1y, _ = _unproject_pixel_to_world(
            float(p1[0]), float(p1[1]), depth0, K, cam_pos_w, cam_quat,
            z_known=z_known, cam_z_w=cam_z_w_val)
        w2x, w2y, _ = _unproject_pixel_to_world(
            float(p2[0]), float(p2[1]), depth0, K, cam_pos_w, cam_quat,
            z_known=z_known, cam_z_w=cam_z_w_val)
        cy_yaw_v = math.atan2(w2y - w1y, w2x - w1x)
        cy_yaw_v = float(wrap_to_pi(torch.tensor([cy_yaw_v])).item())
        if cy_yaw_v > math.pi / 2:
            cy_yaw_v -= math.pi
        elif cy_yaw_v < -math.pi / 2:
            cy_yaw_v += math.pi

        return cx_env_v, cy_env_v, cy_yaw_v

    # ============= Grasp RL (천장 cam scan 결과 만 사용) =============
    def stage_rl_grasp(max_steps: int,
                       top_box_xy_env: tuple[float, float],
                       top_box_yaw: float,
                       rep_idx: int = 0) -> tuple[float, bool]:
        """학습된 PPO 정책으로 박스 정렬 (vision-only, 천장 cam 만).

        box xy + yaw 둘 다 천장 cam scan 결과 (run 시작 시 추정) 그대로 사용.
        매 step 정책에 같은 (top_box_xy_env, top_box_yaw) feed.
        """
        ee_target_yaw = 0.0
        prev_ee_target_yaw = 0.0
        aligned_count = 0
        success = False
        box_x_env_v, box_y_env_v = top_box_xy_env
        box_yaw_v = top_box_yaw

        print(f"[stage] 2. RL grasp align (PPO, top_cam only) | max_steps={max_steps}")
        sim_dt_ctrl = dt

        for step_i in range(max_steps):
            ee_pos_w = grip_center_pos(robot, left_id, right_id)
            ee_xy_env = ee_pos_w[:, :2] - scene.env_origins[:, :2]

            # ---- DBG: 매 60 step gt 비교 (top cam est 는 고정값이라 1회면 충분) ----
            if step_i == 0:
                gt_xy = box.data.root_pos_w[0, :2] - scene.env_origins[0, :2]
                gt_yaw = quat_z_yaw(box.data.root_quat_w)[0].item()
                diff_x = (box_x_env_v - gt_xy[0].item()) * 1000
                diff_y = (box_y_env_v - gt_xy[1].item()) * 1000
                diff_yaw = math.degrees(
                    wrap_to_pi(torch.tensor([box_yaw_v - gt_yaw])).item())
                print(f"  [DBG box] top_cam est=({box_x_env_v:+.3f},{box_y_env_v:+.3f},"
                      f"{math.degrees(box_yaw_v):+.1f}°) "
                      f"gt=({gt_xy[0]:+.3f},{gt_xy[1]:+.3f},{math.degrees(gt_yaw):+.1f}°) "
                      f"d=({diff_x:+5.1f},{diff_y:+5.1f})mm/{diff_yaw:+5.1f}°")

            obj_rel_x = box_x_env_v - ee_xy_env[0, 0].item()
            obj_rel_y = box_y_env_v - ee_xy_env[0, 1].item()
            obj_yaw_err = float(wrap_to_pi(torch.tensor([box_yaw_v - ee_target_yaw])).item())

            ee_vel = grip_center_lin_vel(robot, left_id, right_id)
            ee_vel_x = ee_vel[0, 0].item()
            ee_vel_y = ee_vel[0, 1].item()
            yaw_vel = (ee_target_yaw - prev_ee_target_yaw) / max(sim_dt_ctrl, 1e-6)

            obs_np = np.array(
                [obj_rel_x, obj_rel_y, obj_yaw_err, ee_vel_x, ee_vel_y, yaw_vel],
                dtype=np.float32)
            obs_norm = grasp_normalize_obs(obs_np)
            action, _ = grasp_model.predict(obs_norm, deterministic=True)
            action = np.clip(action, -1.0, 1.0)

            delta_xy = action[:2] * RL_GRASP_ACTION_SCALE_XY
            ee_target_xy_w = ee_pos_w[0, :2] + torch.tensor(delta_xy, device=device, dtype=torch.float)
            prev_ee_target_yaw = ee_target_yaw
            delta_yaw = float(action[2]) * RL_GRASP_ACTION_SCALE_YAW
            ee_target_yaw = max(RL_EE_YAW_MIN, min(RL_EE_YAW_MAX, ee_target_yaw + delta_yaw))

            target_pos_env = torch.tensor(
                [ee_target_xy_w[0].item() - env_origin[0].item(),
                 ee_target_xy_w[1].item() - env_origin[1].item(),
                 PRE_GRASP_Z], device=device, dtype=torch.float)
            yaw_q = quat_from_angle_axis(
                torch.tensor([ee_target_yaw], device=device, dtype=torch.float),
                torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float))
            ee_quat_now = quat_mul(yaw_q, base_ee_quat)
            control_step(target_pos_env, GRIPPER_OPEN, target_quat_w=ee_quat_now)

            aligned = (
                abs(obj_rel_x) < RL_GRASP_ALIGN_XY and
                abs(obj_rel_y) < RL_GRASP_ALIGN_XY and
                abs(obj_yaw_err) < RL_GRASP_ALIGN_YAW
            )
            if aligned:
                aligned_count += 1
                if aligned_count >= RL_GRASP_HOLD_STEPS:
                    success = True
                    print(f"  [stage 2] SUCCESS @ step {step_i+1} (aligned {aligned_count} steps)")
                    break
            else:
                aligned_count = 0
        if not success:
            print(f"  [stage 2] timeout {max_steps} steps — final aligned_count={aligned_count}")
        _save_wrist_overlay(f"grasp_end_run{rep_idx}")
        report("2. RL grasp align")
        return ee_target_yaw, success

    # ---- Stage 4: Insert RL inference (yaw-only, 천장 cam 만) ----
    def stage_rl_insert(max_steps: int,
                        top_cell_xy: tuple[float, float],
                        top_cell_yaw: float) -> tuple[bool, float]:
        """학습된 PPO 정책으로 cell yaw 정렬. xy 는 cell_xy + cell-rotated offset 고정.

        cell_yaw 는 천장 cam scan 결과 (top_cell_yaw) 고정 사용.

        Returns: (success, final_cell_yaw_used)
        """
        aligned_count = 0
        success = False
        cell_yaw_v = top_cell_yaw   # 천장 cam scan 결과 그대로
        print(f"[stage] 4. RL insert align (PPO, top_cam only) | max_steps={max_steps}")
        z_axis_t = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float)

        # DBG: stage 시작 시 1회 gt 비교
        gt_cell_yaw = float(quat_z_yaw(scene["wall_v_left"].data.root_quat_w)[0].item())
        d_yaw = math.degrees(wrap_to_pi(torch.tensor([cell_yaw_v - gt_cell_yaw])).item())
        print(f"  [DBG cell_yaw] top_cam est={math.degrees(cell_yaw_v):+7.1f}° "
              f"gt={math.degrees(gt_cell_yaw):+7.1f}° dyaw={d_yaw:+5.1f}°")

        for step_i in range(max_steps):
            ee_quat_w = grip_center_quat(robot, left_id)
            cur_ee_yaw = quat_z_yaw(ee_quat_w)[0].item()
            yaw_err = float(wrap_to_pi(torch.tensor([cell_yaw_v - cur_ee_yaw])).item())

            ang_l = robot.data.body_ang_vel_w[0, left_id, 2].item()
            ang_r = robot.data.body_ang_vel_w[0, right_id, 2].item()
            yaw_vel = 0.5 * (ang_l + ang_r)

            is_grasping = 1.0

            obs_np = np.array([yaw_err, yaw_vel, is_grasping], dtype=np.float32)
            obs_norm = insert_normalize_obs(obs_np)
            action, _ = insert_model.predict(obs_norm, deterministic=True)
            action = np.clip(action, -1.0, 1.0)

            delta_yaw = float(action[0]) * RL_INSERT_ACTION_SCALE_YAW
            new_yaw = max(RL_EE_YAW_MIN, min(RL_EE_YAW_MAX, cur_ee_yaw + delta_yaw))

            # xy 는 cell_xy + cell-rotated offset 에 고정
            cos_y = math.cos(cell_yaw_v)
            sin_y = math.sin(cell_yaw_v)
            dx_world = cos_y * INSERT_TARGET_X_OFFSET - sin_y * INSERT_TARGET_Y_OFFSET
            dy_world = sin_y * INSERT_TARGET_X_OFFSET + cos_y * INSERT_TARGET_Y_OFFSET
            target_pos_env = torch.tensor(
                [top_cell_xy[0] + dx_world,
                 top_cell_xy[1] + dy_world,
                 TRANSPORT_Z], device=device, dtype=torch.float)

            yaw_q = quat_from_angle_axis(
                torch.tensor([new_yaw], device=device, dtype=torch.float), z_axis_t)
            ee_quat_now = quat_mul(yaw_q, base_ee_quat)

            control_step(target_pos_env, gripper_close, target_quat_w=ee_quat_now)

            aligned = abs(yaw_err) < RL_INSERT_ALIGN_YAW
            if aligned:
                aligned_count += 1
                if aligned_count >= RL_INSERT_HOLD_STEPS:
                    success = True
                    print(f"  [stage 4] SUCCESS @ step {step_i+1} (aligned {aligned_count} steps)")
                    break
            else:
                aligned_count = 0

        if not success:
            print(f"  [stage 4] timeout {max_steps} steps — final aligned_count={aligned_count}")
        report("4. RL insert align")
        return success, cell_yaw_v

    # ============= spawn helpers =============
    def random_box_spawn():
        nx = float(torch.empty(1).uniform_(-BOX_SPAWN_XY_NOISE, BOX_SPAWN_XY_NOISE).item())
        ny = float(torch.empty(1).uniform_(-BOX_SPAWN_XY_NOISE, BOX_SPAWN_XY_NOISE).item())
        bx = BOX_SPAWN[0] + nx
        by = BOX_SPAWN[1] + ny
        yaw = float(torch.empty(1).uniform_(-BOX_SPAWN_YAW_MAX, BOX_SPAWN_YAW_MAX).item())
        box_pos_w = torch.tensor(
            [[bx + env_origin[0].item(),
              by + env_origin[1].item(),
              BOX_SPAWN[2] + env_origin[2].item()]], device=device, dtype=torch.float)
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float)
        box_quat = quat_from_angle_axis(
            torch.tensor([yaw], device=device, dtype=torch.float), z_axis)
        pose = torch.cat([box_pos_w, box_quat], dim=-1)
        vel = torch.zeros((1, 6), device=device, dtype=torch.float)
        box.write_root_pose_to_sim(pose)
        box.write_root_velocity_to_sim(vel)
        return bx, by, yaw

    def random_ee_offset():
        ang = float(torch.empty(1).uniform_(0, 2 * math.pi).item())
        dist = float(torch.empty(1).uniform_(EE_OFFSET_MIN_M, EE_OFFSET_MAX_M).item())
        return dist * math.cos(ang), dist * math.sin(ang)

    def random_cell_pose():
        nx = float(torch.empty(1).uniform_(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE).item())
        ny = float(torch.empty(1).uniform_(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE).item())
        cx_ = CELL_CENTER_X + nx
        cy_ = CELL_CENTER_Y + ny
        cyaw_ = float(torch.empty(1).uniform_(-CELL_SPAWN_YAW_MAX, CELL_SPAWN_YAW_MAX).item())
        return cx_, cy_, cyaw_

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
            [[math.cos(half), 0.0, 0.0, math.sin(half)]], device=device, dtype=torch.float)
        for name, dx_loc, dy_loc in _CELL_WALL_OFFSETS:
            wx_local = cos_y * dx_loc - sin_y * dy_loc
            wy_local = sin_y * dx_loc + cos_y * dy_loc
            wall_pos_w = torch.tensor(
                [[cx_ + wx_local + env_origin[0].item(),
                  cy_ + wy_local + env_origin[1].item(),
                  _WALL_Z + env_origin[2].item()]], device=device, dtype=torch.float)
            pose = torch.cat([wall_pos_w, cell_quat], dim=-1)
            scene[name].write_root_pose_to_sim(pose)
            scene[name].write_root_velocity_to_sim(
                torch.zeros((1, 6), device=device, dtype=torch.float))

    # ============= main repeat loop =============
    n_repeat = max(1, int(args_cli.repeat))
    grasp_success_count = 0
    insert_success_count = 0  # 박스 cell 안에 위치 + z 가 PLACE_Z 근처

    # ---- 벤치마크 데이터 (per-run) ----
    import time as _time
    benchmark_rows = []  # 각 run 의 metric dict
    run_t_start_all = _time.time()

    def _cell_world_yaw_from_walls(scene_obj):
        return float(quat_z_yaw(scene_obj["wall_v_left"].data.root_quat_w)[0].item())

    for rep in range(n_repeat):
        run_t_start = _time.time()
        print(f"\n========== run {rep+1}/{n_repeat} START ==========")
        robot.write_joint_state_to_sim(home_q, joint_vel)
        robot.set_joint_position_target(home_q)
        robot.reset()

        bx, by, byaw = random_box_spawn()
        cx, cy, cyaw = random_cell_pose()
        update_cell_walls(cx, cy, cyaw)
        print(f"[run {rep+1}] box gt: xy=({bx:+.3f},{by:+.3f}) yaw={math.degrees(byaw):+.1f}° "
              f"| cell gt: xy=({cx:+.3f},{cy:+.3f}) yaw={math.degrees(cyaw):+.1f}°")

        for _ in range(30):
            scene.write_data_to_sim(); sim.step(); scene.update(dt)

        # ---- 천장 cam scan ----
        top_est = top_cam_scan(rep_idx=rep + 1)

        # ---- waypoint / cell yaw — vision 만 사용 (GT fallback X). 검출 실패 시 run skip. ----
        if (top_est["box_xy_env"] is None or top_est["box_yaw"] is None
                or top_est["cell_xy_env"] is None or top_est["cell_yaw"] is None):
            print(f"  [run {rep+1}] ❌ YOLO 검출 실패 (box={top_est['box_xy_env']}, "
                  f"cell={top_est['cell_xy_env']}). run skip.")
            continue

        box_xy_used = top_est["box_xy_env"]
        box_yaw_used_init = top_est["box_yaw"]
        cell_xy_used = top_est["cell_xy_env"]
        cell_yaw_used = top_est["cell_yaw"]

        bx_wp, by_wp = box_xy_used
        cx_wp, cy_wp = cell_xy_used

        # ee offset 0 — vision 좌표로 직행
        off_grasp = (0.0, 0.0)

        pre_grasp_offset = torch.tensor(
            [bx_wp + off_grasp[0], by_wp + off_grasp[1], PRE_GRASP_Z],
            device=device, dtype=torch.float)
        grasp_pos = torch.tensor([bx_wp, by_wp, GRASP_Z], device=device, dtype=torch.float)
        lift_pos  = torch.tensor([bx_wp, by_wp, LIFT_Z], device=device, dtype=torch.float)
        # transport target = 셀 정확 위 (insert offset 0)
        transport_target = torch.tensor(
            [cx_wp, cy_wp, TRANSPORT_Z], device=device, dtype=torch.float)

        cell_yaw_q = quat_from_angle_axis(
            torch.tensor([cell_yaw_used], device=device, dtype=torch.float),
            torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float))
        cell_ee_quat = quat_mul(cell_yaw_q, base_ee_quat)

        # ---- 1. 룰베이스 motion: home → 박스 위 ----
        stage_move(home_grip_env, pre_grasp_offset,
                   STAGE_DURATION_S["move_above_box"], GRIPPER_OPEN,
                   "1. Move above box (3-5cm off)",
                   start_quat_w=home_grip_quat_w.unsqueeze(0),
                   end_quat_w=base_ee_quat)

        # ---- 2. Grasp 정렬 ----
        if args_cli.no_rl:
            # no_rl: 천장 cam scan 결과 그대로 사용 (손목 cam X)
            refined_box_x, refined_box_y = box_xy_used
            refined_box_yaw = box_yaw_used_init
            print(f"[stage] 2. no_rl: top_cam xy=({refined_box_x:+.3f},{refined_box_y:+.3f}) "
                  f"yaw={math.degrees(refined_box_yaw):+.1f}°")
            yaw_q_final = quat_from_angle_axis(
                torch.tensor([refined_box_yaw], device=device, dtype=torch.float),
                torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float))
            ee_quat_after_align = quat_mul(yaw_q_final, base_ee_quat)
            # 짧은 motion 으로 ee yaw 정렬 (xy/z 유지)
            stage_move(pre_grasp_offset, pre_grasp_offset, 0.8, GRIPPER_OPEN,
                       "2. (no_rl) yaw align",
                       start_quat_w=base_ee_quat, end_quat_w=ee_quat_after_align)
            final_yaw = refined_box_yaw
            grasp_success = True
        else:
            final_yaw, grasp_success = stage_rl_grasp(
                args_cli.rl_max_steps,
                top_box_xy_env=box_xy_used,
                top_box_yaw=box_yaw_used_init,
                rep_idx=rep + 1,
            )
            yaw_q_final = quat_from_angle_axis(
                torch.tensor([final_yaw], device=device, dtype=torch.float),
                torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float))
            ee_quat_after_align = quat_mul(yaw_q_final, base_ee_quat)
        if grasp_success:
            grasp_success_count += 1

        # ---- 3a. Descend — Stage 2 끝 ee xy 그대로 사용 (RL/wrist 정렬 결과 유지) ----
        cur_ee = (grip_center_pos(robot, left_id, right_id)[0] - env_origin)
        descend_start = torch.tensor(
            [cur_ee[0].item(), cur_ee[1].item(), PRE_GRASP_Z],
            device=device, dtype=torch.float)
        # grasp_pos / lift_pos 를 stage 2 끝 ee xy 로 재정의 (top cam xy 무시)
        grasp_pos = torch.tensor(
            [cur_ee[0].item(), cur_ee[1].item(), GRASP_Z],
            device=device, dtype=torch.float)
        lift_pos = torch.tensor(
            [cur_ee[0].item(), cur_ee[1].item(), LIFT_Z],
            device=device, dtype=torch.float)
        stage_move(descend_start, grasp_pos,
                   STAGE_DURATION_S["descend"], GRIPPER_OPEN,
                   "3a. Descend to grasp depth",
                   start_quat_w=ee_quat_after_align, end_quat_w=ee_quat_after_align)

        # ---- 3b. Close ----
        stage_hold(grasp_pos, STAGE_DURATION_S["close"], gripper_close,
                   "3b. Grasp close+hold", hold_quat_w=ee_quat_after_align)

        # ---- 3c. Lift ----
        stage_move(grasp_pos, lift_pos,
                   STAGE_DURATION_S["lift"], gripper_close,
                   "3c. Lift",
                   start_quat_w=ee_quat_after_align, end_quat_w=ee_quat_after_align)

        # ---- 3d. Transport → cell xy 정확히, ee yaw → 0 (cell_yaw 회전은 stage 4 RL 이 담당) ----
        stage_move(lift_pos, transport_target,
                   STAGE_DURATION_S["transport"], gripper_close,
                   "3d. Transport (yaw → 0)",
                   start_quat_w=ee_quat_after_align,
                   end_quat_w=base_ee_quat)

        # ---- 4. Insert 정렬 (yaw-only) ----
        if args_cli.no_rl:
            # no_rl: 천장 cam scan cell_yaw 그대로 사용 (손목 cam X)
            refined_cell_yaw = cell_yaw_used
            print(f"[stage] 4. no_rl: top_cam cell_yaw="
                  f"{math.degrees(refined_cell_yaw):+.1f}°")
            ref_cell_yaw_q = quat_from_angle_axis(
                torch.tensor([refined_cell_yaw], device=device, dtype=torch.float),
                torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float))
            cell_ee_quat = quat_mul(ref_cell_yaw_q, base_ee_quat)
            # cell-local INSERT_TARGET offset 적용 (RL 모드와 일관)
            cos_y = math.cos(refined_cell_yaw)
            sin_y = math.sin(refined_cell_yaw)
            dx_world = cos_y * INSERT_TARGET_X_OFFSET - sin_y * INSERT_TARGET_Y_OFFSET
            dy_world = sin_y * INSERT_TARGET_X_OFFSET + cos_y * INSERT_TARGET_Y_OFFSET
            # 시작 = 현재 ee xy (transport 끝, offset 없음), 끝 = cell xy + offset
            cur_ee_t = (grip_center_pos(robot, left_id, right_id)[0] - env_origin)
            start_pos = torch.tensor(
                [cur_ee_t[0].item(), cur_ee_t[1].item(), TRANSPORT_Z],
                device=device, dtype=torch.float)
            end_pos = torch.tensor(
                [cx_wp + dx_world, cy_wp + dy_world, TRANSPORT_Z],
                device=device, dtype=torch.float)
            stage_move(start_pos, end_pos, 0.8, gripper_close,
                       "4. (no_rl) xy_offset + yaw align",
                       start_quat_w=base_ee_quat, end_quat_w=cell_ee_quat)
            insert_success = True
        else:
            insert_success, refined_cell_yaw = stage_rl_insert(
                args_cli.rl_max_steps, (cx_wp, cy_wp), cell_yaw_used)
            # RL 정렬 끝의 cell_yaw 로 quat 재계산 (이후 stage 5a/b 에 사용)
            ref_cell_yaw_q = quat_from_angle_axis(
                torch.tensor([refined_cell_yaw], device=device, dtype=torch.float),
                torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float))
            cell_ee_quat = quat_mul(ref_cell_yaw_q, base_ee_quat)

        # ---- 5a. Insert descend (ee yaw 는 RL/motion 정렬 끝점 cell_ee_quat) ----
        cur_ee_2 = (grip_center_pos(robot, left_id, right_id)[0] - env_origin)
        descend_start_2 = torch.tensor(
            [cur_ee_2[0].item(), cur_ee_2[1].item(), TRANSPORT_Z],
            device=device, dtype=torch.float)
        descend_end_2 = torch.tensor(
            [cur_ee_2[0].item(), cur_ee_2[1].item(), PLACE_Z],
            device=device, dtype=torch.float)
        stage_move(descend_start_2, descend_end_2,
                   STAGE_DURATION_S["insert"], gripper_close,
                   "5a. Insert descend",
                   start_quat_w=cell_ee_quat, end_quat_w=cell_ee_quat)

        # ---- 5b. Release ----
        stage_hold(descend_end_2, STAGE_DURATION_S["release"], GRIPPER_OPEN,
                   "5b. Release", hold_quat_w=cell_ee_quat)

        # ---- 6a / 6b. Retract ----
        retract_pos_actual = torch.tensor(
            [cur_ee_2[0].item(), cur_ee_2[1].item(), RETRACT_Z],
            device=device, dtype=torch.float)
        stage_move(descend_end_2, retract_pos_actual,
                   STAGE_DURATION_S["retract_up"], GRIPPER_OPEN,
                   "6a. Retract up",
                   start_quat_w=cell_ee_quat, end_quat_w=base_ee_quat)
        stage_move(retract_pos_actual, home_grip_env,
                   STAGE_DURATION_S["retract_home"], GRIPPER_OPEN,
                   "6b. Retract home",
                   start_quat_w=base_ee_quat, end_quat_w=home_grip_quat_w.unsqueeze(0))

        obj_pos_w = box.data.root_pos_w[0] - env_origin
        cell_xy_dist = ((obj_pos_w[0] - cx) ** 2 + (obj_pos_w[1] - cy) ** 2).sqrt().item()
        # insert success 판정: 박스가 셀 안 (xy 거리 < 5cm) + z 가 ground 근처
        insert_ok = (cell_xy_dist < 0.05) and (obj_pos_w[2].item() < 0.15)
        if insert_ok:
            insert_success_count += 1

        # ---- 벤치마크 metric 수집 ----
        run_t_elapsed = _time.time() - run_t_start
        final_box_yaw = float(quat_z_yaw(box.data.root_quat_w)[0].item())
        cell_yaw_gt = _cell_world_yaw_from_walls(scene)
        yaw_err = float(wrap_to_pi(
            torch.tensor([final_box_yaw - cell_yaw_gt])).item())
        # vision 추정 오차 (top cam)
        box_xy_err = float('nan')
        box_yaw_err = float('nan')
        cell_xy_err = float('nan')
        cell_yaw_err = float('nan')
        if top_est["box_xy_env"] is not None:
            box_xy_err = math.sqrt((top_est["box_xy_env"][0] - bx) ** 2
                                    + (top_est["box_xy_env"][1] - by) ** 2)
        if top_est["box_yaw"] is not None:
            box_yaw_err = float(wrap_to_pi(
                torch.tensor([top_est["box_yaw"] - byaw])).item())
        if top_est["cell_xy_env"] is not None:
            cell_xy_err = math.sqrt((top_est["cell_xy_env"][0] - cx) ** 2
                                     + (top_est["cell_xy_env"][1] - cy) ** 2)
        if top_est["cell_yaw"] is not None:
            cell_yaw_err = float(wrap_to_pi(
                torch.tensor([top_est["cell_yaw"] - cyaw])).item())
        benchmark_rows.append({
            "run": rep + 1,
            "grasp_success": int(bool(grasp_success)),
            "insert_success": int(bool(insert_ok)),
            "final_box_x": float(obj_pos_w[0]),
            "final_box_y": float(obj_pos_w[1]),
            "final_box_z": float(obj_pos_w[2]),
            "final_xy_dist_cm": cell_xy_dist * 100.0,
            "final_yaw_err_deg": math.degrees(yaw_err),
            "box_gt_x": bx, "box_gt_y": by, "box_gt_yaw_deg": math.degrees(byaw),
            "cell_gt_x": cx, "cell_gt_y": cy, "cell_gt_yaw_deg": math.degrees(cyaw),
            "vision_box_xy_err_mm": box_xy_err * 1000.0 if not math.isnan(box_xy_err) else float('nan'),
            "vision_box_yaw_err_deg": math.degrees(box_yaw_err) if not math.isnan(box_yaw_err) else float('nan'),
            "vision_cell_xy_err_mm": cell_xy_err * 1000.0 if not math.isnan(cell_xy_err) else float('nan'),
            "vision_cell_yaw_err_deg": math.degrees(cell_yaw_err) if not math.isnan(cell_yaw_err) else float('nan'),
            "elapsed_s": run_t_elapsed,
        })

        print(f"\n[run {rep+1}] FINAL box (env-rel): "
              f"({obj_pos_w[0]:+.3f},{obj_pos_w[1]:+.3f},{obj_pos_w[2]:+.3f})")
        print(f"[run {rep+1}] FINAL box xy-dist to cell: {cell_xy_dist*100:.2f} cm "
              f"yaw_err: {math.degrees(yaw_err):+.1f}° elapsed: {run_t_elapsed:.1f}s")
        print(f"[run {rep+1}] grasp success: {grasp_success}, insert success: {insert_ok}")
        print(f"========== run {rep+1}/{n_repeat} DONE ==========\n")

    # ============= 종합 통계 =============
    total_elapsed = _time.time() - run_t_start_all
    n_total = len(benchmark_rows)
    n_grasp = sum(r["grasp_success"] for r in benchmark_rows)
    n_insert = sum(r["insert_success"] for r in benchmark_rows)
    print("\n" + "=" * 70)
    print(f"[BENCHMARK] mode={'no_rl' if args_cli.no_rl else 'rl'} total={n_total} "
          f"elapsed_total={total_elapsed/60:.1f}min")
    print(f"  grasp_success  : {n_grasp}/{n_total} ({n_grasp/n_total*100:.1f}%)")
    print(f"  insert_success : {n_insert}/{n_total} ({n_insert/n_total*100:.1f}%)")

    def _stats(vals, name):
        import statistics as _st
        vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        if not vals: return f"  {name}: (no data)"
        return (f"  {name}: mean={_st.mean(vals):+.3f} "
                f"median={_st.median(vals):+.3f} "
                f"std={_st.pstdev(vals):.3f} "
                f"min={min(vals):+.3f} max={max(vals):+.3f}")
    print(_stats([r["final_xy_dist_cm"] for r in benchmark_rows], "final_xy_dist_cm"))
    print(_stats([r["final_yaw_err_deg"] for r in benchmark_rows], "final_yaw_err_deg"))
    print(_stats([r["elapsed_s"] for r in benchmark_rows], "elapsed_s"))
    print(_stats([r["vision_box_xy_err_mm"] for r in benchmark_rows], "vision_box_xy_err_mm"))
    print(_stats([r["vision_box_yaw_err_deg"] for r in benchmark_rows], "vision_box_yaw_err_deg"))
    print(_stats([r["vision_cell_xy_err_mm"] for r in benchmark_rows], "vision_cell_xy_err_mm"))
    print(_stats([r["vision_cell_yaw_err_deg"] for r in benchmark_rows], "vision_cell_yaw_err_deg"))
    print("=" * 70)

    # CSV 저장
    if args_cli.csv_out:
        import csv as _csv
        os.makedirs(os.path.dirname(args_cli.csv_out) or ".", exist_ok=True)
        if benchmark_rows:
            with open(args_cli.csv_out, "w", newline="") as f:
                w = _csv.DictWriter(f, fieldnames=list(benchmark_rows[0].keys()))
                w.writeheader()
                w.writerows(benchmark_rows)
            print(f"[BENCHMARK] CSV saved → {args_cli.csv_out}")

    print(f"\n[chain+rl+cam2] grasp  success rate: {grasp_success_count}/{n_repeat}")
    print(f"[chain+rl+cam2] insert success rate: {insert_success_count}/{n_repeat}")

    # ---- hold ----
    is_headless = bool(getattr(args_cli, "headless", False))
    if args_cli.hold_s < 0:
        hold_s = 0.0 if is_headless else float("inf")
    else:
        hold_s = float(args_cli.hold_s)
    if hold_s == float("inf"):
        print("[chain+rl+cam2] holding final pose. Close window to exit.")
        while simulation_app.is_running():
            scene.write_data_to_sim(); sim.step(); scene.update(dt)
    elif hold_s > 0:
        print(f"[chain+rl+cam2] holding for {hold_s:.1f}s.")
        n_hold = duration_to_steps(hold_s)
        for _ in range(n_hold):
            if not simulation_app.is_running(): break
            scene.write_data_to_sim(); sim.step(); scene.update(dt)


# -------------------- main --------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.4, 0.4, 0.9], [0.3, -0.25, 0.10])
    scene_cfg = MotionSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[chain+rl+cam2] Sim ready.")
    run_pipeline(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    os._exit(0)
