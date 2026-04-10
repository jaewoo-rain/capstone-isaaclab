from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .vision_types import Detection2D


@dataclass
class SelectorOutput:
    target: Optional[Detection2D]
    candidate_objects: list[Detection2D]
    candidate_slots: list[Detection2D]


class TargetSelector:
    def __init__(self, object_class_name: str = 'object', slot_class_name: str = 'place_slot', slot_empty_iou_threshold: float = 0.10):
        self.object_class_name = object_class_name
        self.slot_class_name = slot_class_name
        self.slot_empty_iou_threshold = slot_empty_iou_threshold

    def select_for_pick(self, detections: list[Detection2D], ee_pos_robot: np.ndarray) -> SelectorOutput:
        objs = [d for d in detections if d.cls_name == self.object_class_name and d.pos_robot is not None]
        if not objs:
            return SelectorOutput(target=None, candidate_objects=[], candidate_slots=[])
        return SelectorOutput(target=min(objs, key=lambda d: np.linalg.norm(d.pos_robot - ee_pos_robot)), candidate_objects=objs, candidate_slots=[])

    def select_empty_slot(self, detections: list[Detection2D], reference_pos_robot: np.ndarray) -> SelectorOutput:
        slots = [d for d in detections if d.cls_name == self.slot_class_name]
        objs = [d for d in detections if d.cls_name == self.object_class_name]
        empty_slots = [s for s in slots if self._is_empty_slot(s, objs)]
        if not empty_slots:
            return SelectorOutput(target=None, candidate_objects=objs, candidate_slots=slots)
        def score(s: Detection2D):
            return np.linalg.norm(s.pos_robot - reference_pos_robot) if s.pos_robot is not None else np.linalg.norm(np.array(s.center_uv) - np.array([0.5, 0.5], dtype=np.float32))
        return SelectorOutput(target=min(empty_slots, key=score), candidate_objects=objs, candidate_slots=slots)

    def _is_empty_slot(self, slot: Detection2D, objects: list[Detection2D]) -> bool:
        sx1, sy1, sx2, sy2 = slot.bbox_xyxy
        s_area = max(1e-6, (sx2 - sx1) * (sy2 - sy1))
        for obj in objects:
            ox1, oy1, ox2, oy2 = obj.bbox_xyxy
            ix1, iy1, ix2, iy2 = max(sx1, ox1), max(sy1, oy1), min(sx2, ox2), min(sy2, oy2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter / s_area > self.slot_empty_iou_threshold:
                return False
        return True
