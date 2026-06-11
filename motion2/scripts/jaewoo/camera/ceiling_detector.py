"""camera/ceiling_detector.py — 천장캠 instance segmentation + minAreaRect.

역할:
  - YOLOv8n-seg (ONNX) 로 박스·셀 인스턴스 마스크 추출
  - minAreaRect 로 center (u, v) + yaw 추정
  - CeilingTransform 으로 link0 좌표 변환
  - (선택) ROS2 노드로 실행 시 /vision/box_coarse, /vision/cell_coarse publish

단독 실행 (ROS2 없음, 이미지 파일 or 웹캠):
  python3 ceiling_detector.py \\
      --model yolov8n-seg.onnx \\
      --extrinsic calib/ceiling_extrinsic.yaml \\
      --intrinsics calib/d435i_intrinsics.yaml \\
      --source 0           # 웹캠 인덱스 or 이미지 경로

ROS2 노드 모드:
  python3 ceiling_detector.py --ros \\
      --model yolov8n-seg.onnx \\
      --extrinsic calib/ceiling_extrinsic.yaml \\
      --intrinsics calib/d435i_intrinsics.yaml
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

from camera.coord_transform import CeilingTransform, yaw_from_angle_deg, quat_from_z_yaw


# ─────────────────────────────────────────────────────────────────────────────
# 결과 자료형
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    label: str                         # 클래스 이름 (e.g. "box", "cell")
    score: float                       # 신뢰도 [0, 1]
    center_px: Tuple[float, float]     # 마스크 중심 픽셀 (u, v)
    angle_deg: float                   # minAreaRect 각도 [deg]  [-90, 0)
    rect_size: Tuple[float, float]     # minAreaRect (width, height) [px]
    xy_base: Optional[np.ndarray]      # link0 좌표 [x, y] [m]  (None = 깊이 없음)
    yaw_base: float                    # link0 기준 yaw [rad]
    depth_m: float                     # 검출 중심 깊이 [m]  (0 = 없음)
    mask: Optional[np.ndarray] = field(default=None, repr=False)  # HxW bool


# ─────────────────────────────────────────────────────────────────────────────
# ONNX 추론 래퍼
# ─────────────────────────────────────────────────────────────────────────────

class _YoloSegONNX:
    """YOLOv8n-seg ONNX 추론 (onnxruntime).

    YOLOv8 seg ONNX 출력:
      output0: (1, 116, 8400)  — [cx,cy,w,h,conf,cls×N,mask_coef×32]
      output1: (1, 32, 160, 160) — prototype masks
    """

    INPUT_SIZE = 640

    def __init__(self, model_path: str | pathlib.Path, conf_thresh: float = 0.4):
        import onnxruntime as ort
        self._sess = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._conf = conf_thresh
        self._in_name = self._sess.get_inputs()[0].name
        self._class_names: List[str] = []

    def set_class_names(self, names: List[str]) -> None:
        self._class_names = names

    def infer(
        self, img_bgr: np.ndarray
    ) -> List[Tuple[str, float, np.ndarray, np.ndarray]]:
        """추론 실행.

        Returns:
            list of (label, score, box_xyxy[4], mask_full[H×W bool])
        """
        H, W = img_bgr.shape[:2]
        blob, ratio, (dw, dh) = self._preprocess(img_bgr)
        out0, out1 = self._sess.run(None, {self._in_name: blob})
        return self._postprocess(out0[0], out1[0], ratio, dw, dh, H, W)

    def _preprocess(self, img):
        s = self.INPUT_SIZE
        h, w = img.shape[:2]
        ratio = min(s / h, s / w)
        nh, nw = int(h * ratio), int(w * ratio)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((s, s, 3), 114, dtype=np.uint8)
        dh, dw = (s - nh) // 2, (s - nw) // 2
        canvas[dh:dh+nh, dw:dw+nw] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[np.newaxis], ratio, (dw, dh)

    def _postprocess(self, out0, out1, ratio, dw, dh, orig_H, orig_W):
        # out0: (116, 8400), out1: (32, 160, 160)
        results = []
        preds = out0.T  # (8400, 116)
        num_cls = out0.shape[0] - 4 - 32
        for pred in preds:
            cx, cy, bw, bh = pred[:4]
            cls_scores = pred[4:4+num_cls]
            cls_id = int(np.argmax(cls_scores))
            conf = float(cls_scores[cls_id])
            if conf < self._conf:
                continue
            mask_coef = pred[4+num_cls:]  # (32,)

            # box → original coords
            x1 = int((cx - bw/2 - dw) / ratio)
            y1 = int((cy - bh/2 - dh) / ratio)
            x2 = int((cx + bw/2 - dw) / ratio)
            y2 = int((cy + bh/2 - dh) / ratio)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_W-1, x2), min(orig_H-1, y2)

            # prototype mask 합성 (160×160 → orig size)
            proto = out1  # (32, 160, 160)
            mask_160 = (mask_coef @ proto.reshape(32, -1)).reshape(160, 160)
            mask_160 = 1.0 / (1.0 + np.exp(-mask_160))  # sigmoid
            # 160×160 → 640×640 → crop → orig
            mask_640 = cv2.resize(mask_160, (640, 640), interpolation=cv2.INTER_LINEAR)
            # 패딩 제거
            mask_pad = mask_640[dh:dh+int(orig_H*ratio), dw:dw+int(orig_W*ratio)]
            mask_orig = cv2.resize(mask_pad, (orig_W, orig_H), interpolation=cv2.INTER_LINEAR)
            mask_bool = mask_orig > 0.5

            label = self._class_names[cls_id] if cls_id < len(self._class_names) else str(cls_id)
            results.append((label, conf, np.array([x1, y1, x2, y2]), mask_bool))

        return results


# ─────────────────────────────────────────────────────────────────────────────
# CeilingDetector
# ─────────────────────────────────────────────────────────────────────────────

class CeilingDetector:
    """천장캠 인스턴스 분할 → link0 좌표 추정.

    Args:
        model_path      : YOLOv8n-seg ONNX 경로
        extrinsic_yaml  : ceiling_extrinsic.yaml 경로
        intrinsics_yaml : 카메라 내부 파라미터 yaml (fx, fy, cx, cy)
        conf_thresh     : YOLO 신뢰도 임계값
        min_mask_ratio  : 마스크 면적 / bbox 면적 최소 비율 (잘린 마스크 필터)
    """

    def __init__(
        self,
        model_path: str | pathlib.Path,
        extrinsic_yaml: str | pathlib.Path,
        intrinsics_yaml: str | pathlib.Path,
        conf_thresh: float = 0.4,
        min_mask_ratio: float = 0.25,
    ):
        self._yolo = _YoloSegONNX(model_path, conf_thresh)
        self._tf = CeilingTransform(extrinsic_yaml)
        self._K = self._load_intrinsics(intrinsics_yaml)
        self._min_mask_ratio = min_mask_ratio

    @staticmethod
    def _load_intrinsics(yaml_path: str | pathlib.Path) -> np.ndarray:
        """intrinsics yaml → 3×3 K 행렬.

        yaml 포맷:
          fx: 615.0
          fy: 615.0
          cx: 320.0
          cy: 240.0
        """
        data = yaml.safe_load(open(yaml_path))
        return np.array([
            [data["fx"],       0.0, data["cx"]],
            [      0.0, data["fy"], data["cy"]],
            [      0.0,       0.0,       1.0],
        ], dtype=np.float64)

    def set_class_names(self, names: List[str]) -> None:
        self._yolo.set_class_names(names)

    def detect(
        self,
        color_img: np.ndarray,
        depth_img: np.ndarray,
        target_labels: Optional[List[str]] = None,
    ) -> List[DetectionResult]:
        """이미지 → 검출 결과 리스트.

        Args:
            color_img     : BGR uint8 (H×W×3)
            depth_img     : uint16 [mm] or float32 [m] (H×W)
            target_labels : None 이면 모든 클래스. ["box", "cell"] 처럼 지정 가능

        Returns:
            list[DetectionResult] — 신뢰도 높은 순 정렬
        """
        H, W = color_img.shape[:2]
        depth_m = self._to_depth_m(depth_img)

        raw = self._yolo.infer(color_img)
        results: List[DetectionResult] = []

        for label, score, box_xyxy, mask_bool in raw:
            if target_labels and label not in target_labels:
                continue

            # 잘린 마스크 필터 (bbox 가 이미지 경계에 닿으면 제외)
            x1, y1, x2, y2 = box_xyxy
            if x1 <= 1 or y1 <= 1 or x2 >= W-2 or y2 >= H-2:
                continue

            # 마스크 면적 비율 필터
            bbox_area = max(1, (x2 - x1) * (y2 - y1))
            mask_area = int(np.sum(mask_bool))
            if mask_area / bbox_area < self._min_mask_ratio:
                continue

            # minAreaRect
            contours, _ = cv2.findContours(
                mask_bool.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (rw, rh), angle_deg = rect

            # 깊이 조회 (마스크 중심 3×3 median)
            u_i, v_i = int(round(cx)), int(round(cy))
            d_patch = depth_m[
                max(0, v_i-1):min(H, v_i+2),
                max(0, u_i-1):min(W, u_i+2),
            ]
            valid_d = d_patch[d_patch > 0.05]
            depth_val = float(np.median(valid_d)) if len(valid_d) > 0 else 0.0

            # link0 변환
            xy_base, valid = self._tf.pixel_to_base_xy(cx, cy, depth_val, self._K)
            yaw = yaw_from_angle_deg(angle_deg)

            results.append(DetectionResult(
                label=label,
                score=score,
                center_px=(float(cx), float(cy)),
                angle_deg=float(angle_deg),
                rect_size=(float(rw), float(rh)),
                xy_base=xy_base if valid else None,
                yaw_base=yaw,
                depth_m=depth_val,
                mask=mask_bool,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _to_depth_m(depth_img: np.ndarray) -> np.ndarray:
        """depth 이미지를 [m] float32 로 정규화."""
        if depth_img.dtype == np.uint16:
            return depth_img.astype(np.float32) / 1000.0
        return depth_img.astype(np.float32)

    def visualize(self, color_img: np.ndarray, results: List[DetectionResult]) -> np.ndarray:
        """검출 결과를 이미지에 시각화."""
        vis = color_img.copy()
        for r in results:
            if r.mask is not None:
                overlay = vis.copy()
                overlay[r.mask] = (0, 255, 0)
                vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
            u, v = int(r.center_px[0]), int(r.center_px[1])
            cv2.circle(vis, (u, v), 4, (0, 0, 255), -1)
            txt = (f"{r.label} {r.score:.2f} "
                   f"yaw={math.degrees(r.yaw_base):.1f}deg")
            if r.xy_base is not None:
                txt += f" xy=({r.xy_base[0]:.3f},{r.xy_base[1]:.3f})"
            cv2.putText(vis, txt, (u+6, v-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        return vis


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 노드 (--ros 플래그 사용 시)
# ─────────────────────────────────────────────────────────────────────────────

def _run_ros_node(args) -> None:
    """ROS2 노드로 실행: RealSense 스트림 → /vision/box_coarse, /vision/cell_coarse publish."""
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    import pyrealsense2 as rs

    detector = CeilingDetector(
        model_path=args.model,
        extrinsic_yaml=args.extrinsic,
        intrinsics_yaml=args.intrinsics,
        conf_thresh=args.conf,
    )
    if args.classes:
        detector.set_class_names(args.classes.split(","))

    rclpy.init()
    node = Node("ceiling_detector")
    pub_box  = node.create_publisher(PoseStamped, "/vision/box_coarse",  10)
    pub_cell = node.create_publisher(PoseStamped, "/vision/cell_coarse", 10)

    # RealSense 파이프라인
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    node.get_logger().info("[ceiling_detector] 시작 — /vision/box_coarse, /vision/cell_coarse publish 중")

    try:
        while rclpy.ok():
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            aligned = align.process(frames)
            color_f = np.asanyarray(aligned.get_color_frame().get_data())
            depth_f = np.asanyarray(aligned.get_depth_frame().get_data())  # uint16 mm

            results = detector.detect(color_f, depth_f, target_labels=["box", "cell"])

            now = node.get_clock().now().to_msg()
            for r in results:
                if r.xy_base is None:
                    continue
                msg = PoseStamped()
                msg.header.stamp = now
                msg.header.frame_id = "link0"
                msg.pose.position.x = float(r.xy_base[0])
                msg.pose.position.y = float(r.xy_base[1])
                msg.pose.position.z = float(r.depth_m)
                qw, qx, qy, qz = quat_from_z_yaw(r.yaw_base)
                msg.pose.orientation.w = float(qw)
                msg.pose.orientation.x = float(qx)
                msg.pose.orientation.y = float(qy)
                msg.pose.orientation.z = float(qz)

                if r.label == "box":
                    pub_box.publish(msg)
                elif r.label == "cell":
                    pub_cell.publish(msg)

            rclpy.spin_once(node, timeout_sec=0.0)

    finally:
        pipeline.stop()
        node.destroy_node()
        rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# 단독 실행 — 웹캠 or 이미지 파일 테스트 (ROS2 없음)
# ─────────────────────────────────────────────────────────────────────────────

def _run_standalone(args) -> None:
    """웹캠 or 이미지로 detection 시각화 테스트."""
    import pyrealsense2 as rs

    detector = CeilingDetector(
        model_path=args.model,
        extrinsic_yaml=args.extrinsic,
        intrinsics_yaml=args.intrinsics,
        conf_thresh=args.conf,
    )
    if args.classes:
        detector.set_class_names(args.classes.split(","))

    if args.source.isdigit():
        # RealSense 스트림
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
        pipeline.start(config)
        align = rs.align(rs.stream.color)

        print("[ceiling_detector] RealSense 스트림 — q 로 종료")
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_f = np.asanyarray(aligned.get_color_frame().get_data())
            depth_f = np.asanyarray(aligned.get_depth_frame().get_data())
            results = detector.detect(color_f, depth_f)
            for r in results:
                print(f"  {r.label} score={r.score:.2f} xy={r.xy_base} "
                      f"yaw={math.degrees(r.yaw_base):.1f}°")
            vis = detector.visualize(color_f, results)
            cv2.imshow("ceiling_detector", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        pipeline.stop()

    else:
        # 이미지 파일 (depth 없음 → 좌표 변환 불가, 시각화만)
        img = cv2.imread(args.source)
        if img is None:
            raise FileNotFoundError(f"이미지 없음: {args.source}")
        depth = np.zeros(img.shape[:2], dtype=np.uint16)
        results = detector.detect(img, depth)
        for r in results:
            print(f"  {r.label} score={r.score:.2f} center={r.center_px} "
                  f"angle={r.angle_deg:.1f}°")
        vis = detector.visualize(img, results)
        cv2.imshow("ceiling_detector", vis)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="천장캠 instance seg + minAreaRect → link0 좌표",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      required=True, help="YOLOv8n-seg ONNX 경로")
    parser.add_argument("--extrinsic",  required=True, help="ceiling_extrinsic.yaml")
    parser.add_argument("--intrinsics", required=True, help="d435i_intrinsics.yaml (fx,fy,cx,cy)")
    parser.add_argument("--conf",  type=float, default=0.4, help="YOLO 신뢰도 임계값")
    parser.add_argument("--classes", default="box,cell", help="클래스 이름 (콤마 구분)")
    parser.add_argument("--source", default="0",
                        help="카메라 인덱스(숫자) or 이미지 파일 경로")
    parser.add_argument("--ros", action="store_true",
                        help="ROS2 노드 모드 (RealSense → topic publish)")
    args = parser.parse_args()

    if args.ros:
        _run_ros_node(args)
    else:
        _run_standalone(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
