"""motion2 — YOLO seg 박스/셀 detection + pose 추출.

YOLO 모델 + 카메라 데이터 (RGB, depth, intrinsic, extrinsic) 받아서:
 - 박스/셀의 world xy/yaw 추정
 - sim/real 무관

학습된 모델: v2_seg_2class (class 0=box, 1=cell)
"""
from __future__ import annotations

import math
import numpy as np
import cv2

from .unproject import unproject_pixel_to_world, wrap_to_pi


CLASS_BOX = 0
CLASS_CELL = 1


class YoloBoxDetector:
    """YOLO seg model wrapper. 박스/셀 mask + world pose 추출."""

    def __init__(self, ckpt_path: str, conf: float = 0.5):
        """Args:
            ckpt_path: YOLO seg .pt 모델 경로
            conf: detection confidence threshold
        """
        from ultralytics import YOLO
        self.model = YOLO(ckpt_path)
        self.conf = conf
        print(f"[YoloBoxDetector] loaded {ckpt_path}")

    def predict_mask(self, rgb: np.ndarray, class_id: int) -> np.ndarray | None:
        """RGB (H,W,3) → 지정 class_id 의 best (highest conf) mask (H,W bool)."""
        H, W = rgb.shape[:2]
        results = self.model.predict(rgb, conf=self.conf, verbose=False, imgsz=640)
        r = results[0]
        if r.masks is None or r.masks.xy is None or len(r.masks.xy) == 0:
            return None
        if r.boxes is None or r.boxes.cls is None:
            return None
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        confs = (r.boxes.conf.cpu().numpy()
                 if r.boxes.conf is not None
                 else np.ones_like(cls_ids, dtype=np.float32))
        cands = [i for i, c in enumerate(cls_ids) if c == class_id]
        if not cands:
            return None
        best_idx = max(cands, key=lambda i: confs[i])
        poly = r.masks.xy[best_idx]
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        return mask.astype(bool)

    @staticmethod
    def _pixel_min_area_rect(mask: np.ndarray):
        """mask (H,W bool) → cv2.minAreaRect 결과 + 4 corners + area."""
        mask_u8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 5:
            return None
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w, h), angle = rect
        box_pts = cv2.boxPoints(rect)  # (4, 2)
        return cx, cy, w, h, angle, box_pts, float(cv2.contourArea(cnt))

    def estimate_pose_from_topdown(
        self,
        mask: np.ndarray,
        depth: np.ndarray, K: np.ndarray,
        cam_pos_w: np.ndarray, cam_quat_w_world: np.ndarray,
        z_known: float,
    ) -> tuple[float, float, float] | None:
        """Top-down cam (정 아래 응시) 의 mask → world (x, y, yaw_rad).

        Yaw 변환: top-down 이라 image angle → world yaw = -image_angle (ROS 180° about X 회전 기준).
        cam orientation 이 다른 (rotated) cam 에서는 estimate_pose_general 사용.
        """
        res = self._pixel_min_area_rect(mask)
        if res is None:
            return None
        cx, cy, w, h, angle, _box_pts, _area = res
        # minAreaRect 의 angle 모호성 해소: long edge 기준
        if w < h:
            angle += 90.0
        cam_z_w = float(cam_pos_w[2])
        wx, wy, _ = unproject_pixel_to_world(
            cx, cy, depth, K, cam_pos_w, cam_quat_w_world,
            z_known=z_known, cam_z_w=cam_z_w)
        # top-down: image x = world x, image y = -world y → world yaw = -image angle
        yaw = -math.radians(angle)
        return wx, wy, yaw

    def estimate_pose_general(
        self,
        mask: np.ndarray,
        depth: np.ndarray, K: np.ndarray,
        cam_pos_w: np.ndarray, cam_quat_w_world: np.ndarray,
        z_known: float,
    ) -> tuple[float, float, float] | None:
        """일반 cam 자세 (rotated, tilted) 의 mask → world (x, y, yaw_rad).

        Long edge 의 두 끝점 각각 unproject → world delta atan2 으로 yaw.
        Cam orientation 무관 (robust). 결과 yaw 는 ±π/2 wrap (long edge ±π 모호성 해소).
        """
        res = self._pixel_min_area_rect(mask)
        if res is None:
            return None
        cx, cy, w, h, _angle, box_pts, _area = res
        cam_z_w = float(cam_pos_w[2])

        # center → world xy
        wx, wy, _ = unproject_pixel_to_world(
            cx, cy, depth, K, cam_pos_w, cam_quat_w_world,
            z_known=z_known, cam_z_w=cam_z_w)

        # long edge 두 끝점
        e01 = float(np.linalg.norm(box_pts[0] - box_pts[1]))
        e12 = float(np.linalg.norm(box_pts[1] - box_pts[2]))
        if e01 >= e12:
            p1, p2 = box_pts[0], box_pts[1]
        else:
            p1, p2 = box_pts[1], box_pts[2]
        w1x, w1y, _ = unproject_pixel_to_world(
            float(p1[0]), float(p1[1]), depth, K, cam_pos_w, cam_quat_w_world,
            z_known=z_known, cam_z_w=cam_z_w)
        w2x, w2y, _ = unproject_pixel_to_world(
            float(p2[0]), float(p2[1]), depth, K, cam_pos_w, cam_quat_w_world,
            z_known=z_known, cam_z_w=cam_z_w)
        yaw = math.atan2(w2y - w1y, w2x - w1x)
        yaw = wrap_to_pi(yaw)
        # long edge 방향성 ±π 모호 → ±π/2 wrap
        if yaw > math.pi / 2:
            yaw -= math.pi
        elif yaw < -math.pi / 2:
            yaw += math.pi
        return wx, wy, yaw
