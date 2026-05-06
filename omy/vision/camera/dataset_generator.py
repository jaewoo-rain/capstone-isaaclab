from __future__ import annotations

from pathlib import Path

import cv2
import torch

from .depth_projector import intrinsics_from_fov
from .sim_annotation import compute_annotations_from_sim, xyxy_to_yolo_normalized
from .vision_types import torch_to_numpy_image


class IsaacYoloDatasetGenerator:
    def __init__(self, out_dir: str, image_width: int, image_height: int, hfov_deg: float, vfov_deg: float, object_size_xyz=(0.044, 0.118, 0.139), slot_size_xyz=(0.05, 0.13, 0.02)):
        self.out_dir = Path(out_dir)
        self.img_w = image_width
        self.img_h = image_height
        self.K = intrinsics_from_fov(image_width, image_height, hfov_deg, vfov_deg)
        self.object_size_xyz = object_size_xyz
        self.slot_size_xyz = slot_size_xyz
        for split in ['images/train', 'labels/train', 'images/val', 'labels/val']:
            (self.out_dir / split).mkdir(parents=True, exist_ok=True)

    def save_sample(self, split: str, stem: str, rgb_tensor: torch.Tensor, cam_pos_w, cam_quat_wxyz, object_positions_w, object_quats_wxyz, slot_positions_w, slot_quats_wxyz) -> None:
        img = torch_to_numpy_image(rgb_tensor)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        labels = []
        labels.extend(compute_annotations_from_sim(
            object_positions_w, object_quats_wxyz, 
            [self.object_size_xyz] * len(object_positions_w), 
            [0] * len(object_positions_w), 
            cam_pos_w, cam_quat_wxyz, 
            self.K, self.img_w, self.img_h
            ))
        
        labels.extend(compute_annotations_from_sim(
            slot_positions_w, slot_quats_wxyz, 
            [self.slot_size_xyz] * len(slot_positions_w), 
            [1] * len(slot_positions_w), cam_pos_w, cam_quat_wxyz, 
            self.K, self.img_w, self.img_h
            ))
        
        cv2.imwrite(str(self.out_dir / f'images/{split}/{stem}.png'), img_bgr)
        (self.out_dir / f'labels/{split}/{stem}.txt').write_text('\n'.join([xyxy_to_yolo_normalized(lb, self.img_w, self.img_h) for lb in labels]), encoding='utf-8')

    def write_yaml(self) -> None:
        (self.out_dir / 'dataset.yaml').write_text(f'''path: {self.out_dir.as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: object\n  1: place_slot\n''', encoding='utf-8')
