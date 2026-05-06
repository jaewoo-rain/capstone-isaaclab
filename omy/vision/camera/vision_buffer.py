from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .vision_types import Detection2D, VisionState


@dataclass
class LastValidState:
    target: Optional[Detection2D] = None
    miss_count: int = 0


class VisionBuffer:
    def __init__(self, max_stale_frames: int = 5):
        self.max_stale_frames = max_stale_frames
        self.state = LastValidState()

    def update(self, current_target: Optional[Detection2D], detections: list[Detection2D], debug_image=None) -> VisionState:
        if current_target is not None:
            self.state.target = current_target
            self.state.miss_count = 0
            return VisionState(detections=detections, target_detection=current_target, miss_count=0, is_stale=False, has_detection=True, debug_image=debug_image)
        self.state.miss_count += 1
        if self.state.target is not None and self.state.miss_count <= self.max_stale_frames:
            return VisionState(detections=detections, target_detection=self.state.target, miss_count=self.state.miss_count, is_stale=True, has_detection=False, debug_image=debug_image)
        return VisionState(detections=detections, target_detection=None, miss_count=self.state.miss_count, is_stale=True, has_detection=False, debug_image=debug_image)

    def reset(self) -> None:
        self.state = LastValidState()
