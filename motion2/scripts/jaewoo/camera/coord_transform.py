"""camera/coord_transform.py — 카메라 픽셀 좌표 → link0(robot base) 변환 유틸.

사용 방법:
  from camera.coord_transform import pixel_to_cam3d, CeilingTransform, lookup_T_frame

손목캠은 URDF 에 camera_joint 가 이미 정의돼 있으므로 hand-eye 캘리브 불필요.
TF2 로 link0 ← camera_depth_optical_frame 을 직접 조회하면 된다.
"""
from __future__ import annotations

import math
import pathlib
from typing import Tuple

import numpy as np
import yaml


# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────────

def pixel_to_cam3d(u: float, v: float, depth_m: float, K: np.ndarray) -> np.ndarray:
    """픽셀 좌표 + 깊이 → 카메라 optical 프레임 3D 좌표 [X, Y, Z] (m).

    optical 프레임 규약: x=오른쪽, y=아래, z=앞(depth 방향)

    Args:
        u, v     : 픽셀 좌표 (float)
        depth_m  : 해당 픽셀의 깊이 [m] (0 이하면 NaN 반환)
        K        : 3×3 카메라 내부 파라미터 (fx, fy, cx, cy)

    Returns:
        np.ndarray shape (3,) — [X, Y, Z] in camera optical frame [m]
    """
    if depth_m <= 0.0:
        return np.array([float("nan"), float("nan"), float("nan")])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * depth_m / fx
    Y = (v - cy) * depth_m / fy
    Z = depth_m
    return np.array([X, Y, Z], dtype=np.float64)


def yaw_from_angle_deg(angle_deg: float) -> float:
    """OpenCV minAreaRect 각도 [deg] → z축 yaw [rad], wrap_to_pi."""
    rad = math.radians(angle_deg)
    return (rad + math.pi) % (2.0 * math.pi) - math.pi


def quat_from_z_yaw(yaw: float) -> np.ndarray:
    """z축 회전 yaw [rad] → quaternion [w, x, y, z]."""
    half = yaw * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def _load_yaml(path: pathlib.Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# CeilingTransform — 천장캠 (static extrinsic yaml)
# ─────────────────────────────────────────────────────────────────────────────

class CeilingTransform:
    """천장캠 좌표 → link0(robot base) 변환.

    천장캠은 URDF 에 없으므로 체커보드 PnP 로 한 번 측정 후 yaml 저장.

    yaml 포맷:
      T_cam_to_base:
        - [r00, r01, r02, tx]
        - [r10, r11, r12, ty]
        - [r20, r21, r22, tz]
        - [0,   0,   0,   1 ]
    """

    def __init__(self, extrinsic_yaml: str | pathlib.Path):
        data = _load_yaml(pathlib.Path(extrinsic_yaml))
        rows = data["T_cam_to_base"]
        self._T = np.array(rows, dtype=np.float64).reshape(4, 4)

    @property
    def T(self) -> np.ndarray:
        return self._T.copy()

    def cam3d_to_base(self, pt_cam: np.ndarray) -> np.ndarray:
        """카메라 3D 좌표 [X,Y,Z] → link0 좌표 [x,y,z] [m]."""
        ph = np.array([pt_cam[0], pt_cam[1], pt_cam[2], 1.0])
        return (self._T @ ph)[:3]

    def pixel_to_base_xy(
        self, u: float, v: float, depth_m: float, K: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """픽셀 + 깊이 → link0 xy [m]. Returns (xy, valid)."""
        pt_cam = pixel_to_cam3d(u, v, depth_m, K)
        if np.any(np.isnan(pt_cam)):
            return np.zeros(2), False
        return self.cam3d_to_base(pt_cam)[:2], True


# ─────────────────────────────────────────────────────────────────────────────
# TF2 유틸 — 손목캠은 이걸로 충분 (URDF 가 camera_joint 이미 정의)
# ─────────────────────────────────────────────────────────────────────────────

def lookup_T_frame(
    node,
    rclpy,
    target_frame: str = "link0",
    source_frame: str = "camera_depth_optical_frame",
    timeout_sec: float = 5.0,
) -> np.ndarray:
    """TF2 에서 source_frame → target_frame 의 4×4 변환 행렬 조회.

    손목캠 사용 예:
        T = lookup_T_frame(node, rclpy, "link0", "camera_depth_optical_frame")

    URDF 에 camera_joint(link6→camera_link) 가 이미 정의돼 있으므로
    bringup 이 켜지면 TF2 가 자동으로 전체 체인을 브로드캐스트한다.
    따라서 hand-eye 캘리브레이션 없이 이 함수 하나로 변환이 가능하다.

    Returns:
        4×4 numpy 행렬 (source → target)

    Raises:
        RuntimeError : timeout 내 조회 실패 시
    """
    from tf2_ros import Buffer, TransformListener

    tf_buffer = Buffer()
    node._coord_tf_listener = TransformListener(tf_buffer, node)

    deadline = node.get_clock().now() + rclpy.duration.Duration(seconds=timeout_sec)
    while rclpy.ok():
        try:
            t = tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
            break
        except Exception:
            if node.get_clock().now() > deadline:
                raise RuntimeError(
                    f"TF2 lookup {target_frame}←{source_frame} timeout.\n"
                    f"  확인: ros2 run tf2_tools view_frames"
                )
            rclpy.spin_once(node, timeout_sec=0.05)

    tr = t.transform.translation
    ro = t.transform.rotation
    tx, ty, tz = tr.x, tr.y, tr.z
    qx, qy, qz, qw = ro.x, ro.y, ro.z, ro.w

    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


def pixel_to_base_xy_wrist(
    u: float, v: float, depth_m: float,
    K: np.ndarray,
    T_cam2base: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """픽셀 + 깊이 + T_cam2base → link0 xy [m].

    T_cam2base 는 lookup_T_frame(node, rclpy, "link0", "camera_depth_optical_frame") 로 얻는다.

    Returns:
        (xy_base [m], valid: bool)
    """
    pt_cam = pixel_to_cam3d(u, v, depth_m, K)
    if np.any(np.isnan(pt_cam)):
        return np.zeros(2), False
    ph = np.array([pt_cam[0], pt_cam[1], pt_cam[2], 1.0])
    pt_base = (T_cam2base @ ph)[:3]
    return pt_base[:2], True


# ─────────────────────────────────────────────────────────────────────────────
# 단독 실행 — 캘리브 yaml 검증 / TF2 프레임 확인
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="coord_transform 검증")
    parser.add_argument("--extrinsic", default=None, help="ceiling_extrinsic.yaml 경로")
    parser.add_argument("--check-tf",  action="store_true",
                        help="ROS2 TF2 로 손목캠 프레임 조회 테스트 (bringup 필요)")
    parser.add_argument("--cam-frame", default="camera_depth_optical_frame",
                        help="손목캠 TF2 프레임명")
    args = parser.parse_args()

    K = np.array([
        [428.5,   0.0, 212.3],
        [  0.0, 428.5, 117.8],
        [  0.0,   0.0,   1.0],
    ])

    if args.extrinsic:
        ct = CeilingTransform(args.extrinsic)
        xy, ok = ct.pixel_to_base_xy(320.0, 240.0, 0.8, K)
        print(f"[CeilingTransform] (320,240) depth=0.8m → base xy={xy} valid={ok}")
        print(f"  T_cam_to_base:\n{ct.T}")

    if args.check_tf:
        import rclpy as _rclpy
        from rclpy.node import Node as _Node
        _rclpy.init()
        _node = _Node("coord_transform_test")
        try:
            T = lookup_T_frame(_node, _rclpy, "link0", args.cam_frame)
            print(f"[TF2] link0 ← {args.cam_frame}:\n{T}")
            xy, ok = pixel_to_base_xy_wrist(320.0, 240.0, 0.3, K, T)
            print(f"  (320,240) depth=0.3m → base xy={xy} valid={ok}")
        finally:
            _node.destroy_node()
            _rclpy.shutdown()

    if not args.extrinsic and not args.check_tf:
        print("사용법:")
        print("  # 천장캠 yaml 검증")
        print("  python3 coord_transform.py --extrinsic calib/ceiling_extrinsic.yaml")
        print("  # 손목캠 TF2 확인 (bringup 실행 중이어야)")
        print("  python3 coord_transform.py --check-tf")
        print("  python3 coord_transform.py --check-tf --cam-frame camera_link")
