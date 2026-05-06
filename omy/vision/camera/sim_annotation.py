from __future__ import annotations

import numpy as np


def _as_np(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)

    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
    except Exception:
        pass

    if hasattr(x, "fx") and hasattr(x, "fy") and hasattr(x, "cx") and hasattr(x, "cy"):
        return np.array([
            [float(x.fx), 0.0, float(x.cx)],
            [0.0, float(x.fy), float(x.cy)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

    return np.asarray(x, dtype=np.float32)


def xyxy_to_yolo_normalized(label_xyxy: list[float], img_w: int, img_h: int) -> str:
    cls_id, x1, y1, x2, y2 = label_xyxy
    cx = ((x1 + x2) * 0.5) / img_w
    cy = ((y1 + y2) * 0.5) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return f"{int(cls_id)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def compute_annotations_from_sim(
    positions_w,
    quats_wxyz,
    sizes_xyz,
    class_ids,
    cam_pos_w,
    cam_quat_wxyz,
    K,
    img_w: int,
    img_h: int,
):
    """
    임시 안정화 버전:
    실제 투영 대신 화면 중앙 근처에 무조건 bbox 생성.
    먼저 라벨 파이프라인이 도는지 확인하는 용도.
    """
    labels = []

    # 일단 object는 중앙 기준, slot은 조금 아래쪽으로 배치
    for i, (pos_w, quat_wxyz, size_xyz, cls_id) in enumerate(zip(positions_w, quats_wxyz, sizes_xyz, class_ids)):
        if int(cls_id) == 0:   # object
            cx = 0.35 + 0.12 * i
            cy = 0.55
            bw = 0.10
            bh = 0.18
        else:                  # place_slot
            row = i // 3
            col = i % 3

            cx = 0.30 + 0.18 * col
            cy = 0.65 + 0.12 * row
            bw = 0.12
            bh = 0.08

        x1 = max(0.0, (cx - bw / 2.0) * img_w)
        y1 = max(0.0, (cy - bh / 2.0) * img_h)
        x2 = min(float(img_w - 1), (cx + bw / 2.0) * img_w)
        y2 = min(float(img_h - 1), (cy + bh / 2.0) * img_h)

        if x2 > x1 and y2 > y1:
            labels.append([float(cls_id), x1, y1, x2, y2])

    return labels