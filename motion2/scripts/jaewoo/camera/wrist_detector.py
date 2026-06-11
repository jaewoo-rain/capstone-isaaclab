"""camera/wrist_detector.py — 손목캠 depth ROI + PCA → link0 좌표.

역할:
  - approach 완료 후, 박스가 손목캠 시야 내에 들어온 상태에서 정밀 측정
  - depth ROI + PCA (또는 minAreaRect) 로 box center (xy) + yaw 추정
  - TF2 로 link0 ← camera_depth_optical_frame 변환 → link0 좌표 출력

hand-eye 캘리브레이션 불필요:
  URDF(omy_f3m.urdf line 238)에 camera_joint 가 이미 정의돼 있다.
    link6 → camera_bottom_screw_frame → camera_link
  bringup 이 켜지면 TF2 가 전체 체인을 자동 브로드캐스트하므로
  lookup_T_frame(node, rclpy, "link0", "camera_depth_optical_frame") 하나로 충분.

파이프라인:
  1. depth frame → threshold (0.05 ~ max_depth) → ROI crop
  2. ROI 내 픽셀 → 카메라 XY 좌표 (K 역투영)
  3. PCA → 주축 방향 → yaw_cam
     또는 YOLO seg mask → minAreaRect → yaw_cam
  4. T_cam2base (TF2) → link0 xy, yaw

단독 실행 (ROS2 없음 — PCA 시각화, xy 변환 불가):
  python3 wrist_detector.py \\
      --intrinsics calib/d405_intrinsics.yaml

ROS2 노드 모드 (link0 좌표 publish):
  python3 wrist_detector.py --ros \\
      --intrinsics calib/d405_intrinsics.yaml \\
      [--model yolov8n-seg.onnx]
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_JAEWOO_DIR = _THIS_DIR.parent
if str(_JAEWOO_DIR) not in sys.path:
    sys.path.insert(0, str(_JAEWOO_DIR))

from camera.coord_transform import (
    pixel_to_cam3d, pixel_to_base_xy_wrist,
    yaw_from_angle_deg, quat_from_z_yaw, lookup_T_frame,
)


# ─────────────────────────────────────────────────────────────────────────────
# 결과 자료형
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WristDetectionResult:
    center_px: Tuple[float, float]    # 마스크/클러스터 중심 픽셀 (u, v)
    yaw_cam: float                    # 카메라 프레임 기준 yaw [rad]
    xy_base: Optional[np.ndarray]     # link0 xy [m]  (None = TF2 없음)
    yaw_base: float                   # link0 기준 yaw [rad]
    depth_m: float                    # 중심 깊이 [m]
    method: str                       # "pca" | "minarearect" | "*_flipped"
    score: float                      # PCA: explained_ratio, seg: conf
    mask: Optional[np.ndarray] = field(default=None, repr=False)


# ─────────────────────────────────────────────────────────────────────────────
# Depth-only 검출 (PCA)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_depth_pca(
    depth_m: np.ndarray,
    K: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    min_depth: float = 0.05,
    max_depth: float = 0.60,
    min_points: int = 100,
) -> Optional[Tuple[float, float, float, float, float]]:
    """depth frame + PCA → (center_u, center_v, depth_center, yaw_cam, explained_ratio).

    Returns None if not enough valid points in ROI.
    """
    H, W = depth_m.shape
    if roi is not None:
        x1, y1, x2, y2 = max(0, roi[0]), max(0, roi[1]), min(W, roi[2]), min(H, roi[3])
        depth_roi = depth_m[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    else:
        depth_roi = depth_m
        offset_x, offset_y = 0, 0

    mask = (depth_roi > min_depth) & (depth_roi < max_depth)
    vs, us = np.where(mask)
    if len(us) < min_points:
        return None

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (us + offset_x - cx) / fx
    Y = (vs + offset_y - cy) / fy
    pts = np.column_stack([X, Y]).astype(np.float32)

    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = (centered.T @ centered) / len(pts)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]
    yaw_cam = math.atan2(float(major_axis[1]), float(major_axis[0]))
    explained = float(eigenvalues.max() / (eigenvalues.sum() + 1e-9))

    center_u = float(mean[0] * fx + cx)
    center_v = float(mean[1] * fy + cy)

    u_i, v_i = int(round(center_u)), int(round(center_v))
    d_patch = depth_m[max(0, v_i-2):min(H, v_i+3), max(0, u_i-2):min(W, u_i+3)]
    valid_d = d_patch[(d_patch > min_depth) & (d_patch < max_depth)]
    depth_center = float(np.median(valid_d)) if len(valid_d) > 0 else float(depth_m[v_i, u_i])

    return center_u, center_v, depth_center, yaw_cam, explained


# ─────────────────────────────────────────────────────────────────────────────
# WristDetector
# ─────────────────────────────────────────────────────────────────────────────

class WristDetector:
    """손목캠 정밀 박스 위치·yaw 추정.

    URDF 에 camera_joint 정의 → hand-eye yaml 불필요.
    detect() 에 T_cam2base (TF2 결과) 를 넘기면 link0 좌표 반환.
    T_cam2base 없이 호출하면 yaw_cam/center_px 만 반환 (xy_base=None).

    Args:
        intrinsics_yaml : d405_intrinsics.yaml (fx, fy, cx, cy)
        model_path      : YOLOv8n-seg ONNX (None 이면 depth PCA only)
        max_depth       : 유효 depth 상한 [m]
        min_depth       : 유효 depth 하한 [m]
        yaw_flip_thresh : 연속 프레임 yaw 차이가 이 이상이면 ±180° flip [rad]
    """

    def __init__(
        self,
        intrinsics_yaml: str | pathlib.Path,
        model_path: Optional[str | pathlib.Path] = None,
        max_depth: float = 0.60,
        min_depth: float = 0.05,
        yaw_flip_thresh: float = math.pi / 2,
    ):
        self._K = self._load_intrinsics(intrinsics_yaml)
        self._max_depth = max_depth
        self._min_depth = min_depth
        self._yaw_flip_thresh = yaw_flip_thresh
        self._prev_yaw: Optional[float] = None

        self._yolo = None
        if model_path is not None:
            from camera.ceiling_detector import _YoloSegONNX
            self._yolo = _YoloSegONNX(model_path, conf_thresh=0.35)
            self._yolo.set_class_names(["box"])

    @staticmethod
    def _load_intrinsics(yaml_path) -> np.ndarray:
        data = yaml.safe_load(open(yaml_path))
        return np.array([
            [data["fx"],       0.0, data["cx"]],
            [      0.0, data["fy"], data["cy"]],
            [      0.0,       0.0,       1.0],
        ], dtype=np.float64)

    def reset(self) -> None:
        """yaw temporal tracking 초기화 (새 물체 잡을 때 호출)."""
        self._prev_yaw = None

    def detect(
        self,
        color_img: np.ndarray,
        depth_img: np.ndarray,
        T_cam2base: Optional[np.ndarray] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        use_seg: bool = True,
    ) -> Optional[WristDetectionResult]:
        """단일 프레임 → 박스 위치·yaw 추정.

        Args:
            color_img  : BGR uint8 (H×W×3)
            depth_img  : uint16 [mm] or float32 [m] (H×W)
            T_cam2base : 4×4 (camera_depth_optical_frame → link0).
                         lookup_T_frame(node, rclpy, "link0", "camera_depth_optical_frame")
                         로 얻는다. None 이면 xy_base=None (yaw 만 반환).
            roi        : (x1, y1, x2, y2) 탐색 ROI. None 이면 전체 프레임.
            use_seg    : True + model 있으면 YOLO seg → minAreaRect. 아니면 PCA.

        Returns:
            WristDetectionResult or None
        """
        depth_m = self._to_depth_m(depth_img)

        result: Optional[WristDetectionResult] = None

        if use_seg and self._yolo is not None:
            result = self._detect_seg(color_img, depth_m, T_cam2base)

        if result is None:
            result = self._detect_pca(depth_m, T_cam2base, roi)

        if result is None:
            return None

        # yaw 시계열 consistency (180° flip 방지)
        if self._prev_yaw is not None:
            diff = (result.yaw_base - self._prev_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(diff) > self._yaw_flip_thresh:
                flipped = (result.yaw_base + math.pi + math.pi) % (2 * math.pi) - math.pi
                result = WristDetectionResult(
                    center_px=result.center_px,
                    yaw_cam=result.yaw_cam,
                    xy_base=result.xy_base,
                    yaw_base=flipped,
                    depth_m=result.depth_m,
                    method=result.method + "_flipped",
                    score=result.score,
                    mask=result.mask,
                )
        self._prev_yaw = result.yaw_base
        return result

    def _detect_seg(
        self, color_img, depth_m, T_cam2base
    ) -> Optional[WristDetectionResult]:
        H, W = depth_m.shape
        raw = self._yolo.infer(color_img)
        if not raw:
            return None

        label, score, box_xyxy, mask_bool = raw[0]
        contours, _ = cv2.findContours(
            mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        (cx, cy), _, angle_deg = cv2.minAreaRect(cnt)

        u_i, v_i = int(round(cx)), int(round(cy))
        d_patch = depth_m[max(0,v_i-2):min(H,v_i+3), max(0,u_i-2):min(W,u_i+3)]
        valid_d = d_patch[(d_patch > self._min_depth) & (d_patch < self._max_depth)]
        depth_val = float(np.median(valid_d)) if len(valid_d) > 0 else 0.0

        yaw_cam = yaw_from_angle_deg(angle_deg)
        xy_base, yaw_base = self._transform(cx, cy, depth_val, yaw_cam, T_cam2base)

        return WristDetectionResult(
            center_px=(float(cx), float(cy)),
            yaw_cam=yaw_cam,
            xy_base=xy_base,
            yaw_base=yaw_base,
            depth_m=depth_val,
            method="minarearect",
            score=float(score),
            mask=mask_bool,
        )

    def _detect_pca(
        self, depth_m, T_cam2base, roi
    ) -> Optional[WristDetectionResult]:
        res = _detect_depth_pca(
            depth_m, self._K, roi=roi,
            min_depth=self._min_depth, max_depth=self._max_depth,
        )
        if res is None:
            return None
        center_u, center_v, depth_val, yaw_cam, explained = res

        xy_base, yaw_base = self._transform(center_u, center_v, depth_val, yaw_cam, T_cam2base)

        return WristDetectionResult(
            center_px=(center_u, center_v),
            yaw_cam=yaw_cam,
            xy_base=xy_base,
            yaw_base=yaw_base,
            depth_m=depth_val,
            method="pca",
            score=float(explained),
        )

    def _transform(
        self,
        u: float, v: float, depth_val: float, yaw_cam: float,
        T_cam2base: Optional[np.ndarray],
    ) -> Tuple[Optional[np.ndarray], float]:
        """픽셀 + yaw_cam → (xy_base or None, yaw_base)."""
        if T_cam2base is None:
            return None, yaw_cam  # TF2 없으면 변환 불가

        xy_base, valid = pixel_to_base_xy_wrist(u, v, depth_val, self._K, T_cam2base)
        if not valid:
            return None, yaw_cam

        # T_cam2base 의 z축 회전 성분을 yaw에 더함
        r00, r10 = T_cam2base[0, 0], T_cam2base[1, 0]
        offset = math.atan2(r10, r00)
        yaw_base = (yaw_cam + offset + math.pi) % (2 * math.pi) - math.pi

        return xy_base, yaw_base

    @staticmethod
    def _to_depth_m(depth_img: np.ndarray) -> np.ndarray:
        if depth_img.dtype == np.uint16:
            return depth_img.astype(np.float32) / 1000.0
        return depth_img.astype(np.float32)

    def visualize(
        self,
        color_img: np.ndarray,
        result: Optional[WristDetectionResult],
    ) -> np.ndarray:
        vis = color_img.copy()
        if result is None:
            cv2.putText(vis, "NO DETECTION", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return vis

        if result.mask is not None:
            overlay = vis.copy()
            overlay[result.mask] = (0, 255, 0)
            vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        u, v = int(result.center_px[0]), int(result.center_px[1])
        cv2.circle(vis, (u, v), 5, (0, 0, 255), -1)
        ax = int(u + 40 * math.cos(result.yaw_cam))
        ay = int(v + 40 * math.sin(result.yaw_cam))
        cv2.arrowedLine(vis, (u, v), (ax, ay), (0, 255, 255), 2, tipLength=0.3)

        txt = (f"[{result.method}] score={result.score:.2f} "
               f"yaw_base={math.degrees(result.yaw_base):.1f}° "
               f"d={result.depth_m:.3f}m")
        if result.xy_base is not None:
            txt += f" xy=({result.xy_base[0]:.3f},{result.xy_base[1]:.3f})"
        cv2.putText(vis, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        return vis


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 노드 (--ros 플래그)
# ─────────────────────────────────────────────────────────────────────────────

def _run_ros_node(args) -> None:
    """ROS2 노드: RealSense D405 → /vision/box_fine publish."""
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    import pyrealsense2 as rs

    detector = WristDetector(
        intrinsics_yaml=args.intrinsics,
        model_path=args.model if not args.pca_only else None,
        max_depth=args.max_depth,
    )

    rclpy.init()
    node = Node("wrist_detector")
    pub = node.create_publisher(PoseStamped, "/vision/box_fine", 10)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 90)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  90)
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    node.get_logger().info(
        f"[wrist_detector] 시작 — TF2 프레임: link0 ← {args.cam_frame}")

    try:
        while rclpy.ok():
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            aligned = align.process(frames)
            color_f = np.asanyarray(aligned.get_color_frame().get_data())
            depth_f = np.asanyarray(aligned.get_depth_frame().get_data())

            try:
                T_cam2base = lookup_T_frame(node, rclpy, "link0", args.cam_frame)
            except RuntimeError as e:
                node.get_logger().warn(f"TF2 실패: {e}")
                rclpy.spin_once(node, timeout_sec=0.01)
                continue

            result = detector.detect(color_f, depth_f, T_cam2base)
            if result is not None and result.xy_base is not None:
                msg = PoseStamped()
                msg.header.stamp = node.get_clock().now().to_msg()
                msg.header.frame_id = "link0"
                msg.pose.position.x = float(result.xy_base[0])
                msg.pose.position.y = float(result.xy_base[1])
                msg.pose.position.z = float(result.depth_m)
                qw, qx, qy, qz = quat_from_z_yaw(result.yaw_base)
                msg.pose.orientation.w = float(qw)
                msg.pose.orientation.x = float(qx)
                msg.pose.orientation.y = float(qy)
                msg.pose.orientation.z = float(qz)
                pub.publish(msg)

            rclpy.spin_once(node, timeout_sec=0.0)

    finally:
        pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# 단독 실행 (ROS2 없음 — 시각화만, xy 변환 없음)
# ─────────────────────────────────────────────────────────────────────────────

def _run_standalone(args) -> None:
    """RealSense D405 스트림 — PCA/seg 시각화. xy 변환은 ROS2 모드만 가능."""
    import pyrealsense2 as rs

    detector = WristDetector(
        intrinsics_yaml=args.intrinsics,
        model_path=args.model if not args.pca_only else None,
        max_depth=args.max_depth,
    )

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 90)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  90)
    pipeline.start(cfg)
    align = rs.align(rs.stream.color)

    print("[wrist_detector] D405 스트림 시작 — q 로 종료")
    print("  (T_cam2base=None: yaw_cam, depth 만 출력. link0 xy 는 --ros 모드 사용)")

    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_f = np.asanyarray(aligned.get_color_frame().get_data())
        depth_f = np.asanyarray(aligned.get_depth_frame().get_data())

        # T_cam2base=None → xy 변환 없이 yaw 만
        result = detector.detect(color_f, depth_f, T_cam2base=None)
        if result:
            print(f"  [{result.method}] score={result.score:.2f}  "
                  f"d={result.depth_m:.3f}m  yaw_cam={math.degrees(result.yaw_cam):.1f}°")

        vis = detector.visualize(color_f, result)
        cv2.imshow("wrist_detector", vis)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    pipeline.stop()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="손목캠 depth PCA / inst.seg → link0 좌표 (URDF camera_joint 활용)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--intrinsics", required=True, help="d405_intrinsics.yaml (fx,fy,cx,cy)")
    parser.add_argument("--model",    default=None,  help="YOLOv8n-seg ONNX (없으면 PCA only)")
    parser.add_argument("--pca-only", action="store_true", help="YOLO seg 비활성화, PCA만 사용")
    parser.add_argument("--max-depth", type=float, default=0.60, help="유효 depth 상한 [m]")
    parser.add_argument("--cam-frame", default="camera_depth_optical_frame",
                        help="TF2 카메라 프레임명 (ros2 run tf2_tools view_frames 로 확인)")
    parser.add_argument("--ros", action="store_true", help="ROS2 노드 모드")
    args = parser.parse_args()

    if args.ros:
        _run_ros_node(args)
    else:
        _run_standalone(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
