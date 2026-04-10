from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


@dataclass
class Detection2D:
    cls_id: int
    cls_name: str
    conf: float
    track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    center_uv: tuple[float, float]
    depth_m: Optional[float] = None
    pos_cam: Optional[np.ndarray] = None
    pos_robot: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)


@dataclass
class VisionState:
    detections: list[Detection2D]
    target_detection: Optional[Detection2D]
    miss_count: int
    is_stale: bool
    has_detection: bool
    debug_image: Optional[np.ndarray] = None


def torch_to_numpy_image(rgb: torch.Tensor) -> np.ndarray:
    if rgb is None:
        raise ValueError('rgb is None')
    if isinstance(rgb, torch.Tensor):
        img = rgb.detach().cpu()
        if img.ndim == 4:
            img = img[0]
        if img.shape[-1] == 4:
            img = img[..., :3]
        if img.dtype != torch.uint8:
            img = (img.float().clamp(0, 1) * 255.0).byte()
        return img.numpy()
    return rgb
