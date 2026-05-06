from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class YoloWrapper:
    def __init__(self, model_path: str, names_override: Optional[dict[int, str]] = None, conf: float = 0.35, iou: float = 0.45, tracker_cfg: str = 'bytetrack.yaml', device: str = 'cuda', imgsz: int = 640):
        self.model_path = str(model_path)
        self.names_override = names_override or {}
        self.conf = conf
        self.iou = iou
        self.tracker_cfg = tracker_cfg
        self.device = device
        self.imgsz = imgsz
        if YOLO is None:
            raise ImportError('ultralytics is not installed. pip install ultralytics')
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f'YOLO model not found: {self.model_path}')
        self.model = YOLO(self.model_path)

    def track(self, frame_bgr: np.ndarray):
        return self.model.track(source=frame_bgr, conf=self.conf, iou=self.iou, persist=True, tracker=self.tracker_cfg, device=self.device, imgsz=self.imgsz, verbose=False)

    def names(self) -> dict[int, str]:
        if self.names_override:
            return self.names_override
        return getattr(self.model, 'names', {}) or {}
