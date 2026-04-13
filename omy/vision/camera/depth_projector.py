from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


def intrinsics_from_fov(width: int, height: int, hfov_deg: float, vfov_deg: float) -> CameraIntrinsics:
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = (height / 2.0) / math.tan(math.radians(vfov_deg) / 2.0)
    return CameraIntrinsics(width=width, height=height, fx=fx, fy=fy, cx=width/2.0, cy=height/2.0)


def sample_depth_at_bbox_center(depth_hw: np.ndarray, bbox_xyxy: tuple[float, float, float, float], kernel: int = 5) -> Optional[float]:
    x1, y1, x2, y2 = bbox_xyxy
    u = int(round((x1 + x2) * 0.5))
    v = int(round((y1 + y2) * 0.5))
    h, w = depth_hw.shape[:2]
    u = max(0, min(w - 1, u))
    v = max(0, min(h - 1, v))
    r = max(1, kernel // 2)
    patch = depth_hw[max(0, v-r):min(h, v+r+1), max(0, u-r):min(w, u+r+1)]
    patch = patch.astype(np.float32)
    patch = patch[np.isfinite(patch)]
    patch = patch[patch > 1e-5]
    return None if patch.size == 0 else float(np.median(patch))


def pixel_to_camera_xyz(u: float, v: float, depth_m: float, K: CameraIntrinsics) -> np.ndarray:
    return np.array([(u - K.cx) * depth_m / K.fx, (v - K.cy) * depth_m / K.fy, depth_m], dtype=np.float32)


def quat_wxyz_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_wxyz / max(np.linalg.norm(quat_wxyz), 1e-8)
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)], [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)], [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]], dtype=np.float32)


def camera_to_world(point_cam: np.ndarray, cam_pos_w: np.ndarray, cam_quat_wxyz: np.ndarray) -> np.ndarray:
    R = quat_wxyz_to_rotmat(cam_quat_wxyz)
    return (R @ point_cam.reshape(3, 1)).reshape(3) + cam_pos_w.astype(np.float32)


def world_to_camera(point_w: np.ndarray, cam_pos_w: np.ndarray, cam_quat_wxyz: np.ndarray) -> np.ndarray:
    R = quat_wxyz_to_rotmat(cam_quat_wxyz)
    return R.T @ (point_w.astype(np.float32) - cam_pos_w.astype(np.float32))
