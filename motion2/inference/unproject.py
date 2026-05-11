"""motion2 — Pixel → world coordinate unprojection.

sim/real 무관. depth + intrinsic + extrinsic 만 받으면 world 좌표 반환.
"""
from __future__ import annotations

import math
import numpy as np


def quat_mat_from_wxyz(q: np.ndarray) -> np.ndarray:
    """quat (w,x,y,z) → 3x3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),      1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),      2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def quat_z_yaw(q: np.ndarray) -> float:
    """quat (w,x,y,z) → world Z-axis yaw."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(angle: float) -> float:
    return ((angle + math.pi) % (2.0 * math.pi)) - math.pi


def unproject_pixel_to_world(
    u_px: float, v_px: float,
    depth_img: np.ndarray,         # (H, W) orthogonal/perspective depth (meters)
    K: np.ndarray,                 # (3, 3) intrinsic
    cam_pos_w: np.ndarray,         # (3,) world pos
    cam_quat_w_world: np.ndarray,  # (4,) wxyz, "world" convention (forward=+X, up=+Z)
    z_known: float | None = None,
    cam_z_w: float | None = None,
) -> tuple[float, float, float]:
    """Single pixel (u, v) → world (x, y, z).

    1. depth_img 의 5×5 윈도우 평균 사용 (NaN/inf 제외).
    2. 유효 depth 없으면 fallback = |cam_z_w - z_known| (또는 0.5 m).
    3. ROS camera frame point → world-convention camera frame 축 swap.
    4. cam_pos_w + R(cam_quat_w_world) @ point 로 world 변환.

    Args:
        depth_img: (H, W) depth array in meters
        K: 3x3 intrinsic matrix
        cam_pos_w: camera world position
        cam_quat_w_world: camera world orientation (wxyz, world convention)
        z_known: object's expected world z (fallback for invalid depth)
        cam_z_w: camera's world z (fallback)

    Returns:
        (world_x, world_y, world_z)
    """
    H, W = depth_img.shape[:2]
    u_i = int(round(u_px)); v_i = int(round(v_px))
    u_i = max(0, min(W - 1, u_i))
    v_i = max(0, min(H - 1, v_i))

    win = 2
    u0, u1 = max(0, u_i - win), min(W, u_i + win + 1)
    v0, v1 = max(0, v_i - win), min(H, v_i + win + 1)
    patch = depth_img[v0:v1, u0:u1]
    valid = np.isfinite(patch) & (patch > 0)
    if valid.any():
        d_use = float(patch[valid].mean())
    elif (z_known is not None) and (cam_z_w is not None):
        d_use = max(0.01, abs(cam_z_w - z_known))
    else:
        d_use = 0.5

    K_inv = np.linalg.inv(np.asarray(K, dtype=np.float32))
    pix_h = np.array([u_px, v_px, 1.0], dtype=np.float32)
    ray_cam = K_inv @ pix_h
    ray_cam = ray_cam / ray_cam[-1]
    pt_cam = ray_cam * d_use  # ROS camera frame: x_right, y_down, z_forward

    # ROS → world-convention camera frame (forward=+X, left=+Y, up=+Z)
    pt_world_local = np.array([pt_cam[2], -pt_cam[0], -pt_cam[1]], dtype=np.float32)

    R = quat_mat_from_wxyz(cam_quat_w_world)
    pt_world = R @ pt_world_local + np.asarray(cam_pos_w, dtype=np.float32)
    return float(pt_world[0]), float(pt_world[1]), float(pt_world[2])
