"""camera/dataset/auto_label_depth.py — depth 기반 박스 자동 라벨 + ultralytics seg 데이터셋 생성.

입력: dataset/raw/<cam>/NNNN_color.png + NNNN_depth.png (uint16) + K.yaml
출력: ultralytics YOLOv8-seg 데이터셋
  dataset/yolo_seg/
    images/{train,val}/<cam>_NNNN.png
    labels/{train,val}/<cam>_NNNN.txt   ← 각 줄: "cls x1 y1 x2 y2 ..." (정규화 폴리곤)
    data.yaml

──────────────────────────────────────────────────────────────────────────────
박스 자동 분할 원리 (--method depth, 기본):
  박스는 테이블 평면 위로 솟은 강체 → "테이블 평면" 을 제거하면 박스만 남는다.
  1) RANSAC 으로 depth 포인트클라우드의 지배 평면(테이블) 추정.
     (--table-z 를 주면 RANSAC 대신 '평면까지 깊이 = 그 값' 으로 고정 임계.)
  2) 평면보다 카메라에 --height-thresh[m] 이상 가까운 픽셀 = 전경(박스 윗면).
  3) connected components → 박스 인스턴스 분리.
  4) 각 인스턴스 외곽 컨투어 → approxPolyDP → 정규화 폴리곤(ultralytics seg).
  필터: 최소 면적, 화면 경계 닿은 컴포넌트 제외(ceiling_detector 컨벤션과 일관).

박스 분할 원리 (--method sam, 광택 금속 박스 대응):
  박스가 광택 금속이면 정반사로 박스 윗면 depth 에 구멍/노이즈가 생겨 depth 단독
  분할이 깨진다. → depth 를 'seed(대략 위치)' 로만 쓰고, color 이미지에 FastSAM
  (ultralytics 내장, 새 무거운 의존성 X)을 point-prompt 로 돌려 깨끗한 인스턴스
  마스크를 얻는다.
  1) depth 평면제거(위 1~2단계)로 전경 컴포넌트의 대표점(centroid/distance-transform
     peak)을 박스 seed 점으로 추출. (구멍 많아도 컴포넌트 1개당 점 1개만 있으면 됨.)
  2) FastSAM 을 각 seed 점에 point-prompt → seed 를 포함하는 깨끗한 마스크 선택.
  3) seed 가 전혀 없으면(박스 depth 완전 소실) FastSAM "everything" 모드 후
     크기/위치 휴리스틱으로 박스 후보 선택(fallback).
  4) 선택 마스크 → 외곽 컨투어 → approxPolyDP → 정규화 폴리곤. **depth 경로와
     동일한 출력 포맷/디렉터리/필터(최소면적·경계제외)를 재사용**한다.
  FastSAM 은 이 데스크 GPU 에서만 도는 라벨링 단계(Pi 추론과 무관). --device 지정.

셀(class 1) 라벨링 전략 — (a) 고정 ROI 템플릿 재사용 채택:
  천장캠은 '정적'이고 셀 그리드는 '고정 구조물'이라 모든 프레임에서 픽셀 위치가
  동일하다. 셀 슬롯은 오목/평면이라 '솟은 강체' depth 분할이 안 통한다(박스와 정반대).
  → 셀 폴리곤을 1회만 정의(cell_template.yaml)하고 전 프레임에 그대로 복사한다.
  근거: depth 로 셀 구조를 매 프레임 검출하는 것보다 (1) 안정적이고 (2) 노이즈 없고
  (3) 정적 카메라 가정과 정확히 맞는다. 셀 템플릿은 annotate_cells.py(stub, 하단 참고)
  로 1회 클릭 라벨하거나 손으로 cell_template.yaml 을 작성한다.
──────────────────────────────────────────────────────────────────────────────

사용:
  # 박스 자동 라벨 + 데이터셋 빌드(depth 경로, 기본; 셀 템플릿 있으면 함께 주입)
  python3 auto_label_depth.py --from-folder dataset/raw/d435i

  # 자동 라벨 검수(시각화, 저장 안 함)
  python3 auto_label_depth.py --from-folder dataset/raw/d435i --show --no-write

  # 테이블 평면 깊이 고정(RANSAC 대신), 셀 템플릿 지정
  python3 auto_label_depth.py --from-folder dataset/raw/d435i \\
      --table-z 0.80 --cell-template dataset/raw/d435i/cell_template.yaml

  # 광택 금속 박스: depth-seed + FastSAM 하이브리드(GPU 라벨링 단계)
  python3 auto_label_depth.py --from-folder dataset/raw/d435i --method sam --device cuda:0
  python3 auto_label_depth.py --from-folder dataset/raw/d435i --method sam --show --no-write
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_CAMERA_DIR = _THIS_DIR.parent
_JAEWOO_DIR = _CAMERA_DIR.parent
if str(_JAEWOO_DIR) not in sys.path:
    sys.path.insert(0, str(_JAEWOO_DIR))

CLASS_NAMES = ["box", "cell"]  # id 0 = box, id 1 = cell (변경 금지)


@dataclass
class LabelConfig:
    height_thresh: float = 0.012   # 평면 위 이 높이[m] 이상 = 박스 전경
    table_z: Optional[float] = None  # 고정 평면 깊이[m](None=RANSAC)
    ransac_iters: int = 200
    ransac_thresh: float = 0.006   # 평면 인라이어 허용 오차[m]
    min_area_px: int = 200         # 최소 컴포넌트 면적[px]
    max_area_frac: float = 0.5     # 화면 대비 최대 면적 비율(과대 전경 제외)
    border_margin: int = 2         # 경계 이 픽셀 안에 닿으면 제외
    poly_eps_frac: float = 0.01    # approxPolyDP epsilon = 둘레 * 이 값
    morph_kernel: int = 3          # 전경 마스크 open/close 커널
    depth_scale_mm: float = 1.0    # depth LSB → mm (K.yaml 에서 로드)
    # ── FastSAM(--method sam) 관련 ──
    sam_weights: str = "FastSAM-s.pt"  # ultralytics 자동 다운로드(FastSAM-s/x.pt)
    sam_device: str = "cuda:0"     # 라벨링 GPU(데스크 전용, Pi 무관)
    sam_imgsz: int = 1024          # FastSAM 추론 해상도
    sam_conf: float = 0.4          # FastSAM conf 임계
    sam_iou: float = 0.9           # FastSAM NMS IoU
    sam_seed_min_area: int = 60    # seed 컴포넌트 최소 면적[px](depth 구멍 관대)
    sam_max_seeds: int = 12        # 한 프레임 최대 seed 개수(과대 분할 방지)


# ─────────────────────────────────────────────────────────────────────────────
# depth → 평면 추정
# ─────────────────────────────────────────────────────────────────────────────

def _load_K(folder: pathlib.Path) -> Tuple[np.ndarray, float]:
    """K.yaml → (3x3 K, depth_scale_mm). 없으면 기본 가정."""
    kfile = folder / "K.yaml"
    if not kfile.exists():
        print(f"[auto_label] ⚠️ {kfile} 없음 → depth_scale 1mm/LSB 가정, K 추정 불가")
        return np.eye(3), 1.0
    d = yaml.safe_load(open(kfile))
    K = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]], dtype=np.float64)
    return K, float(d.get("depth_scale_mm", 1.0))


def _depth_to_m(depth_u16: np.ndarray, depth_scale_mm: float) -> np.ndarray:
    """uint16 depth(LSB) → m. (scale_mm = LSB당 mm)."""
    return depth_u16.astype(np.float32) * depth_scale_mm / 1000.0


def _ransac_plane_depth(depth_m: np.ndarray, cfg: LabelConfig) -> np.ndarray:
    """전 픽셀의 '추정 평면까지 깊이'[m] 맵 반환.

    카메라가 거의 top-down 이면 평면은 'depth ≈ 상수' 에 가깝다. 일반 기울기까지
    포괄하려면 z = a*u + b*v + c 평면을 RANSAC 으로 적합(픽셀좌표 선형). 충분히
    안정적이고 빠르다(포인트클라우드 풀 평면보다 단순).
    """
    H, W = depth_m.shape
    vs, us = np.nonzero(depth_m > 0.05)
    z = depth_m[vs, us]
    if len(z) < 50:
        # 유효 depth 거의 없음 → 평면 추정 불가, 전부 0 (전경 없음)
        return np.full((H, W), 1e9, dtype=np.float32)

    best_inliers = None
    best_coef = None
    n = len(z)
    rng = np.random.default_rng(0)
    A_all = np.stack([us, vs, np.ones_like(us)], axis=1).astype(np.float64)
    for _ in range(cfg.ransac_iters):
        idx = rng.choice(n, size=3, replace=False)
        A3 = A_all[idx]
        try:
            coef, *_ = np.linalg.lstsq(A3, z[idx], rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = A_all @ coef
        inliers = np.abs(pred - z) < cfg.ransac_thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_coef = coef
    # 인라이어로 최종 적합
    if best_inliers is not None and best_inliers.sum() >= 3:
        coef, *_ = np.linalg.lstsq(A_all[best_inliers], z[best_inliers], rcond=None)
    else:
        coef = best_coef if best_coef is not None else np.array([0, 0, float(np.median(z))])

    uu, vv = np.meshgrid(np.arange(W), np.arange(H))
    plane = (coef[0] * uu + coef[1] * vv + coef[2]).astype(np.float32)
    return plane


# ─────────────────────────────────────────────────────────────────────────────
# 박스 인스턴스 분할
# ─────────────────────────────────────────────────────────────────────────────

def segment_boxes(
    depth_m: np.ndarray, cfg: LabelConfig
) -> List[np.ndarray]:
    """depth(m) → 박스 인스턴스 폴리곤 리스트. 각 폴리곤 = (N,2) int 픽셀좌표."""
    H, W = depth_m.shape
    if cfg.table_z is not None:
        plane = np.full((H, W), float(cfg.table_z), dtype=np.float32)
    else:
        plane = _ransac_plane_depth(depth_m, cfg)

    valid = depth_m > 0.05
    # 카메라에 평면보다 height_thresh 이상 '가까운' 픽셀 = 솟은 박스 윗면.
    fg = valid & ((plane - depth_m) > cfg.height_thresh)
    fg = fg.astype(np.uint8) * 255

    k = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    polys: List[np.ndarray] = []
    img_area = H * W
    for i in range(1, num):  # 0 = 배경
        area = stats[i, cv2.CC_STAT_AREA]
        if area < cfg.min_area_px or area > cfg.max_area_frac * img_area:
            continue
        x, y, w, h = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                      stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        # 화면 경계 닿은 컴포넌트 제외(잘린 박스 → ceiling_detector 와 일관)
        m = cfg.border_margin
        if x <= m or y <= m or x + w >= W - m or y + h >= H - m:
            continue
        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        eps = cfg.poly_eps_frac * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        polys.append(approx.astype(np.int32))
    return polys


def _depth_foreground_mask(depth_m: np.ndarray, cfg: LabelConfig) -> np.ndarray:
    """depth(m) → 전경(솟은 박스) 이진 마스크(uint8 0/255). segment_boxes 와 동일 로직."""
    H, W = depth_m.shape
    if cfg.table_z is not None:
        plane = np.full((H, W), float(cfg.table_z), dtype=np.float32)
    else:
        plane = _ransac_plane_depth(depth_m, cfg)
    valid = depth_m > 0.05
    fg = valid & ((plane - depth_m) > cfg.height_thresh)
    fg = fg.astype(np.uint8) * 255
    k = np.ones((cfg.morph_kernel, cfg.morph_kernel), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    return fg


def _mask_to_poly(mask: np.ndarray, cfg: LabelConfig, W: int, H: int) -> Optional[np.ndarray]:
    """이진 마스크(0/255 또는 bool) → approxPolyDP 폴리곤(픽셀). 필터 통과 못하면 None.

    depth 경로(segment_boxes)와 동일한 필터: 최소면적·최대면적·경계제외·꼭짓점≥3.
    """
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    img_area = H * W
    if area < cfg.min_area_px or area > cfg.max_area_frac * img_area:
        return None
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    mg = cfg.border_margin
    # 화면 경계 닿은 컴포넌트 제외(잘린 박스 → ceiling_detector 와 일관)
    if x0 <= mg or y0 <= mg or x1 >= W - 1 - mg or y1 >= H - 1 - mg:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    eps = cfg.poly_eps_frac * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return None
    return approx.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# FastSAM 하이브리드 (--method sam): depth seed → FastSAM 인스턴스 마스크
# ─────────────────────────────────────────────────────────────────────────────

def depth_box_seeds(depth_m: np.ndarray, cfg: LabelConfig) -> List[Tuple[int, int]]:
    """depth 전경 컴포넌트 → 박스 seed 점(u,v) 리스트.

    광택 금속이라 컴포넌트가 구멍투성이여도, 컴포넌트당 대표점 1개만 있으면 FastSAM
    point-prompt 의 seed 로 충분하다. 대표점은 distance-transform peak(컴포넌트 내부
    가장 안쪽 점) → 마스크 안에 확실히 들어가는 점을 고른다.
    """
    fg = _depth_foreground_mask(depth_m, cfg)
    num, labels, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=8)
    seeds: List[Tuple[int, int]] = []
    order = sorted(range(1, num), key=lambda i: -stats[i, cv2.CC_STAT_AREA])
    for i in order:
        if stats[i, cv2.CC_STAT_AREA] < cfg.sam_seed_min_area:
            continue
        comp = (labels == i).astype(np.uint8)
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        _, _, _, maxloc = cv2.minMaxLoc(dist)  # maxloc=(u,v) 가장 안쪽 점
        seeds.append((int(maxloc[0]), int(maxloc[1])))
        if len(seeds) >= cfg.sam_max_seeds:
            break
    return seeds


# FastSAM 모델 1회 로드 캐시(프레임마다 재로드 방지)
_FASTSAM_CACHE: dict = {}


def _load_fastsam(cfg: LabelConfig):
    """ultralytics FastSAM 로드(캐시). 가중치는 최초 1회 자동 다운로드."""
    key = cfg.sam_weights
    if key in _FASTSAM_CACHE:
        return _FASTSAM_CACHE[key]
    from ultralytics import FastSAM  # ultralytics 내장(새 의존성 X)
    model = FastSAM(cfg.sam_weights)
    _FASTSAM_CACHE[key] = model
    return model


def _fastsam_masks(model, color: np.ndarray, cfg: LabelConfig,
                   points: Optional[List[Tuple[int, int]]]) -> List[np.ndarray]:
    """FastSAM 추론 → 마스크 리스트(각 (H,W) bool, 원해상도).

    points 가 주어지면 point-prompt(각 점당 마스크 1개), None 이면 everything 모드.
    """
    H, W = color.shape[:2]
    kw = dict(device=cfg.sam_device, retina_masks=True, imgsz=cfg.sam_imgsz,
              conf=cfg.sam_conf, iou=cfg.sam_iou, verbose=False)
    if points:
        kw["points"] = [[int(u), int(v)] for (u, v) in points]
        kw["labels"] = [1] * len(points)  # 1=전경 점
    res = model.predict(color, **kw)
    out: List[np.ndarray] = []
    if not res:
        return out
    r0 = res[0]
    if r0.masks is None:
        return out
    data = r0.masks.data.detach().cpu().numpy()  # (N,h,w) — retina_masks 면 원해상도
    for mk in data:
        m = mk.astype(np.uint8)
        if m.shape[:2] != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        out.append(m.astype(bool))
    return out


def segment_boxes_sam(
    color: np.ndarray, depth_m: np.ndarray, cfg: LabelConfig,
    return_debug: bool = False,
):
    """color+depth → FastSAM 박스 폴리곤 리스트.

    1) depth seed 추출 → seed 별 FastSAM point-prompt 마스크 선택.
    2) seed 가 0개면 everything 모드 + 크기/위치 휴리스틱 fallback.
    return_debug=True 면 (polys, seeds) 반환(시각 검수용).
    """
    H, W = color.shape[:2]
    model = _load_fastsam(cfg)
    seeds = depth_box_seeds(depth_m, cfg)
    polys: List[np.ndarray] = []
    used_masks: List[np.ndarray] = []  # 중복(IoU) 억제용

    def _too_similar(m: np.ndarray) -> bool:
        for um in used_masks:
            inter = np.logical_and(m, um).sum()
            uni = np.logical_or(m, um).sum()
            if uni > 0 and inter / uni > 0.6:
                return True
        return False

    if seeds:
        # seed 점들을 한 번에 prompt → seed 순서대로 마스크 1개씩 대응.
        masks = _fastsam_masks(model, color, cfg, points=seeds)
        for idx, (u, v) in enumerate(seeds):
            m = None
            # 우선 같은 인덱스 마스크가 seed 를 포함하면 사용.
            if idx < len(masks) and masks[idx][v, u]:
                m = masks[idx]
            else:
                # 인덱스 어긋나면 seed 점을 포함하는 마스크 중 최소면적 선택.
                cands = [mm for mm in masks if mm[v, u]]
                if cands:
                    m = min(cands, key=lambda mm: int(mm.sum()))
            if m is None or _too_similar(m):
                continue
            poly = _mask_to_poly(m, cfg, W, H)
            if poly is not None:
                polys.append(poly)
                used_masks.append(m)
    else:
        # fallback: 박스 depth 완전 소실 → everything 후 휴리스틱.
        masks = _fastsam_masks(model, color, cfg, points=None)
        img_area = H * W
        for m in masks:
            area = int(m.sum())
            # 박스 크기대(최소면적 ~ 화면의 일부)만 박스 후보로.
            if area < cfg.min_area_px or area > cfg.max_area_frac * img_area:
                continue
            if _too_similar(m):
                continue
            poly = _mask_to_poly(m, cfg, W, H)
            if poly is not None:
                polys.append(poly)
                used_masks.append(m)

    if return_debug:
        return polys, seeds
    return polys


# ─────────────────────────────────────────────────────────────────────────────
# 셀 템플릿 (고정 ROI 재사용)
# ─────────────────────────────────────────────────────────────────────────────

def load_cell_polys(template_path: Optional[pathlib.Path]) -> List[np.ndarray]:
    """cell_template.yaml → 셀 폴리곤 리스트(픽셀좌표). 없으면 빈 리스트.

    yaml 포맷:
      cells:
        - [[u0,v0],[u1,v1],[u2,v2],[u3,v3]]   # 셀 1 폴리곤
        - [[...]]                              # 셀 2 ...
    """
    if template_path is None or not template_path.exists():
        return []
    d = yaml.safe_load(open(template_path)) or {}
    polys = []
    for poly in d.get("cells", []):
        arr = np.array(poly, dtype=np.int32)
        if arr.ndim == 2 and arr.shape[0] >= 3:
            polys.append(arr)
    return polys


# ─────────────────────────────────────────────────────────────────────────────
# ultralytics seg 라벨 직렬화
# ─────────────────────────────────────────────────────────────────────────────

def poly_to_seg_line(cls_id: int, poly: np.ndarray, W: int, H: int) -> str:
    """폴리곤(픽셀) → ultralytics seg 한 줄: 'cls x1 y1 x2 y2 ...' (정규화)."""
    norm = poly.astype(np.float64).copy()
    norm[:, 0] = np.clip(norm[:, 0] / W, 0.0, 1.0)
    norm[:, 1] = np.clip(norm[:, 1] / H, 0.0, 1.0)
    coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in norm)
    return f"{cls_id} {coords}"


# ─────────────────────────────────────────────────────────────────────────────
# 데이터셋 빌드
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameLabel:
    color_path: pathlib.Path
    lines: List[str] = field(default_factory=list)
    box_count: int = 0
    cell_count: int = 0


def build_dataset(
    raw_folder: pathlib.Path,
    out_root: pathlib.Path,
    cam_tag: str,
    cfg: LabelConfig,
    cell_polys: List[np.ndarray],
    val_frac: float,
    write: bool,
    show: bool,
    method: str = "depth",
) -> None:
    colors = sorted(raw_folder.glob("*_color.png"))
    if not colors:
        raise FileNotFoundError(f"color 프레임 없음: {raw_folder}")

    frames: List[FrameLabel] = []
    for cpath in colors:
        dpath = cpath.with_name(cpath.name.replace("_color.png", "_depth.png"))
        if not dpath.exists():
            print(f"  ⚠️ depth 누락, skip: {cpath.name}")
            continue
        color = cv2.imread(str(cpath), cv2.IMREAD_COLOR)
        depth_u16 = cv2.imread(str(dpath), cv2.IMREAD_UNCHANGED)
        if color is None or depth_u16 is None:
            print(f"  ⚠️ 읽기 실패, skip: {cpath.name}")
            continue
        H, W = color.shape[:2]
        depth_m = _depth_to_m(depth_u16, cfg.depth_scale_mm)

        seeds: List[Tuple[int, int]] = []
        if method == "sam":
            box_polys, seeds = segment_boxes_sam(color, depth_m, cfg, return_debug=True)
        else:
            box_polys = segment_boxes(depth_m, cfg)
        fl = FrameLabel(color_path=cpath)
        for p in box_polys:
            fl.lines.append(poly_to_seg_line(0, p, W, H))
        fl.box_count = len(box_polys)
        for p in cell_polys:
            fl.lines.append(poly_to_seg_line(1, p, W, H))
        fl.cell_count = len(cell_polys)
        frames.append(fl)

        if show:
            vis = color.copy()
            cv2.polylines(vis, box_polys, True, (0, 255, 0), 2)
            cv2.polylines(vis, cell_polys, True, (0, 165, 255), 2)
            # --method sam: depth seed 점을 빨강 십자로 표시(seed→마스크 검수)
            for (u, v) in seeds:
                cv2.drawMarker(vis, (u, v), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
            tag = f"[{method}] {cpath.name} box={fl.box_count} cell={fl.cell_count}"
            if method == "sam":
                tag += f" seeds={len(seeds)}"
            cv2.putText(vis, tag, (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("auto_label (q=quit, any=next)", vis)
            if (cv2.waitKey(0) & 0xFF) == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()

    total_box = sum(f.box_count for f in frames)
    print(f"[auto_label] 프레임 {len(frames)}장, 박스 인스턴스 총 {total_box}개, "
          f"셀/프레임 {len(cell_polys)}개")

    if not write:
        print("[auto_label] --no-write → 데이터셋 미저장(검수 모드)")
        return

    # train/val 분할(셔플)
    random.seed(42)
    shuffled = frames[:]
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac)) if len(shuffled) > 1 else 0
    val_set = set(id(f) for f in shuffled[:n_val])

    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for f in frames:
        split = "val" if id(f) in val_set else "train"
        stem = f"{cam_tag}_{f.color_path.name.split('_')[0]}"
        img_dst = out_root / "images" / split / f"{stem}.png"
        lbl_dst = out_root / "labels" / split / f"{stem}.txt"
        cv2.imwrite(str(img_dst), cv2.imread(str(f.color_path), cv2.IMREAD_COLOR))
        with open(lbl_dst, "w") as fh:
            fh.write("\n".join(f.lines) + ("\n" if f.lines else ""))

    # data.yaml
    data_yaml = out_root / "data.yaml"
    with open(data_yaml, "w") as fh:
        yaml.safe_dump({
            "path": str(out_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {i: n for i, n in enumerate(CLASS_NAMES)},
        }, fh, default_flow_style=False, sort_keys=False)

    print(f"[auto_label] 저장 완료 → {out_root}")
    print(f"  train {len(frames) - len(val_set)}장 / val {len(val_set)}장")
    print(f"  data.yaml: {data_yaml}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="depth 자동 박스 라벨 → ultralytics YOLOv8-seg 데이터셋",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--from-folder", required=True,
                        help="dataset/raw/<cam> 경로(color+depth+K.yaml)")
    parser.add_argument("--method", choices=["depth", "sam"], default="depth",
                        help="박스 분할 방식. depth=평면제거(기본), "
                             "sam=depth seed + FastSAM(광택 금속 박스)")
    parser.add_argument("--out", default=None, help="출력 루트(기본 dataset/yolo_seg)")
    parser.add_argument("--cam-tag", default=None,
                        help="파일명 접두사(기본: from-folder 폴더명)")
    parser.add_argument("--cell-template", default=None,
                        help="cell_template.yaml(고정 ROI). 기본: raw 폴더 내 동명 파일")
    parser.add_argument("--val-frac", type=float, default=0.2, help="val 분할 비율")
    parser.add_argument("--table-z", type=float, default=None,
                        help="테이블 평면 깊이[m] 고정(RANSAC 대신)")
    parser.add_argument("--height-thresh", type=float, default=0.012,
                        help="평면 위 이 높이[m] 이상 = 박스 전경")
    parser.add_argument("--min-area", type=int, default=200, help="최소 컴포넌트 면적[px]")
    # ── FastSAM(--method sam) 전용 ──
    parser.add_argument("--device", default="cuda:0",
                        help="(method=sam) FastSAM 추론 device(데스크 GPU). 예: cuda:0, cpu")
    parser.add_argument("--sam-weights", default="FastSAM-s.pt",
                        help="(method=sam) FastSAM 가중치(자동 다운로드: FastSAM-s.pt/FastSAM-x.pt)")
    parser.add_argument("--sam-imgsz", type=int, default=1024,
                        help="(method=sam) FastSAM 추론 해상도")
    parser.add_argument("--sam-conf", type=float, default=0.4, help="(method=sam) conf 임계")
    parser.add_argument("--sam-iou", type=float, default=0.9, help="(method=sam) NMS IoU")
    parser.add_argument("--show", action="store_true", help="자동 라벨 시각화 검수")
    parser.add_argument("--no-write", action="store_true", help="저장 안 함(검수만)")
    args = parser.parse_args()

    raw_folder = pathlib.Path(args.from_folder)
    cam_tag = args.cam_tag or raw_folder.name
    out_root = pathlib.Path(args.out) if args.out else _THIS_DIR / "dataset" / "yolo_seg"

    _, depth_scale_mm = _load_K(raw_folder)
    cfg = LabelConfig(
        height_thresh=args.height_thresh,
        table_z=args.table_z,
        min_area_px=args.min_area,
        depth_scale_mm=depth_scale_mm,
        sam_weights=args.sam_weights,
        sam_device=args.device,
        sam_imgsz=args.sam_imgsz,
        sam_conf=args.sam_conf,
        sam_iou=args.sam_iou,
    )

    if args.method == "sam":
        print(f"[auto_label] method=sam → FastSAM({cfg.sam_weights}) on {cfg.sam_device}. "
              f"가중치는 최초 1회 자동 다운로드.")

    cell_tpl = (pathlib.Path(args.cell_template) if args.cell_template
                else raw_folder / "cell_template.yaml")
    cell_polys = load_cell_polys(cell_tpl)
    if cell_polys:
        print(f"[auto_label] 셀 템플릿 {len(cell_polys)}개 로드: {cell_tpl}")
    else:
        print(f"[auto_label] 셀 템플릿 없음({cell_tpl}) → box 만 라벨. "
              f"셀은 annotate_cells.py 로 1회 정의 권장.")

    build_dataset(
        raw_folder=raw_folder, out_root=out_root, cam_tag=cam_tag, cfg=cfg,
        cell_polys=cell_polys, val_frac=args.val_frac,
        write=not args.no_write, show=args.show, method=args.method,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
