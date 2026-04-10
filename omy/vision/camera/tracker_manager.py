from __future__ import annotations

import cv2
import numpy as np

from .vision_types import Detection2D


class TrackerManager:
    def __init__(self, class_names: dict[int, str]):
        self.class_names = class_names
        self._next_fallback_track_id = 100000

    def parse_ultralytics_result(self, result) -> list[Detection2D]:
        detections: list[Detection2D] = []
        boxes = getattr(result, 'boxes', None)
        if boxes is None or len(boxes) == 0:
            return detections
        for i in range(len(boxes)):
            box = boxes[i]
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            conf = float(box.conf[0].item()) if box.conf is not None else 0.0
            cls_id = int(box.cls[0].item()) if box.cls is not None else -1
            track_id = int(box.id[0].item()) if box.id is not None else self._allocate_fallback_track_id()
            x1, y1, x2, y2 = xyxy
            detections.append(Detection2D(cls_id=cls_id, cls_name=self.class_names.get(cls_id, str(cls_id)), conf=conf, track_id=track_id, bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)), center_uv=(float((x1+x2)*0.5), float((y1+y2)*0.5))))
        return detections

    def _allocate_fallback_track_id(self) -> int:
        self._next_fallback_track_id += 1
        return self._next_fallback_track_id

    @staticmethod
    def draw(frame_bgr: np.ndarray, detections: list[Detection2D]) -> np.ndarray:
        vis = frame_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox_xyxy)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f'{det.cls_name}:{det.track_id} {det.conf:.2f}', (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis
