"""camera/dataset/capture_dataset.py — RealSense color+depth(aligned) 쌍 캡처.

YOLOv8-seg 학습 데이터셋의 원본 프레임을 모은다. 박스/셀 라벨링은
auto_label_depth.py 가 이 raw 프레임을 읽어 자동 생성한다.

저장 구조 (cam = d435i | d405):
  dataset/raw/<cam>/
    0000_color.png   ← BGR uint8 (cv2.imwrite)
    0000_depth.png   ← uint16, 단위 mm (RealSense z16 그대로)
    0001_color.png
    0001_depth.png
    ...
    K.yaml           ← color 스트림 intrinsics (fx,fy,cx,cy,width,height,depth_scale_mm)

⚠️ color↔depth 정렬: rs.align(rs.stream.color) 로 depth 를 color 시점에 맞춘다.
   따라서 동일 픽셀 (u,v) 의 color 와 depth 가 같은 물체를 가리킨다(자동 라벨 전제).

사용:
  # 천장캠(D435i) 2초마다 자동 저장, 목표 80장
  python3 capture_dataset.py --cam d435i --interval 2.0 --n 80

  # 키보드 트리거(스페이스=저장, q=종료), 목표 60장
  python3 capture_dataset.py --cam d405 --trigger key --n 60

  # 카메라 없이: 이미 모아둔 raw 프레임 점검만 (머신 위치 독립)
  python3 capture_dataset.py --from-folder dataset/raw/d435i --list

코드 컨벤션: 한국어 주석, dataclass, argparse, _THIS_DIR sys.path 패턴.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_CAMERA_DIR = _THIS_DIR.parent
_JAEWOO_DIR = _CAMERA_DIR.parent
if str(_JAEWOO_DIR) not in sys.path:
    sys.path.insert(0, str(_JAEWOO_DIR))

# dump_intrinsics.py 의 프리셋을 단일 출처로 재사용(해상도 일치 보장).
from camera.dump_intrinsics import _PRESETS  # noqa: E402


@dataclass
class CaptureConfig:
    cam: str                       # "d435i" | "d405"
    out_root: pathlib.Path         # dataset/raw 루트
    n: int                         # 목표 저장 장수
    interval: float                # 자동 저장 주기 [s] (trigger=="time" 일 때)
    trigger: str                   # "time" | "key"
    width: int
    height: int
    fps: int


# ─────────────────────────────────────────────────────────────────────────────
# 저장 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _next_index(cam_dir: pathlib.Path) -> int:
    """이어쓰기: 기존 *_color.png 중 최대 인덱스 + 1 부터 시작."""
    existing = sorted(cam_dir.glob("*_color.png"))
    if not existing:
        return 0
    last = existing[-1].name.split("_")[0]
    try:
        return int(last) + 1
    except ValueError:
        return len(existing)


def _save_pair(
    cam_dir: pathlib.Path, idx: int, color: np.ndarray, depth: np.ndarray
) -> Tuple[pathlib.Path, pathlib.Path]:
    """color(BGR uint8) + depth(uint16 mm) 쌍 저장."""
    cpath = cam_dir / f"{idx:04d}_color.png"
    dpath = cam_dir / f"{idx:04d}_depth.png"
    cv2.imwrite(str(cpath), color)
    # depth 는 16-bit PNG. cv2.imwrite 는 uint16 단일 채널을 16-bit PNG 로 저장.
    cv2.imwrite(str(dpath), depth.astype(np.uint16))
    return cpath, dpath


def _save_intrinsics(cam_dir: pathlib.Path, intr, depth_scale_mm: float) -> None:
    """color 스트림 intrinsics + depth scale 을 K.yaml 로 저장."""
    data = {
        "fx": float(intr.fx), "fy": float(intr.fy),
        "cx": float(intr.ppx), "cy": float(intr.ppy),
        "width": int(intr.width), "height": int(intr.height),
        # depth.png 한 단위(LSB) 가 몇 mm 인지. RealSense z16 는 보통 1mm/LSB.
        "depth_scale_mm": float(depth_scale_mm),
        "model": str(intr.model), "coeffs": [float(c) for c in intr.coeffs],
    }
    with open(cam_dir / "K.yaml", "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)


# ─────────────────────────────────────────────────────────────────────────────
# RealSense 캡처
# ─────────────────────────────────────────────────────────────────────────────

def _run_capture(cfg: CaptureConfig) -> int:
    """RealSense 스트림에서 color+depth(aligned) 쌍을 cfg.n 장 저장."""
    import pyrealsense2 as rs

    cam_dir = cfg.out_root / cfg.cam
    cam_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps)
    config.enable_stream(rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    # depth 한 단위가 몇 m 인지 → mm 로 환산해 저장(보통 0.001m=1mm/LSB).
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale_mm = float(depth_sensor.get_depth_scale()) * 1000.0

    intr = (profile.get_stream(rs.stream.color)
            .as_video_stream_profile().get_intrinsics())
    _save_intrinsics(cam_dir, intr, depth_scale_mm)

    idx = _next_index(cam_dir)
    saved = 0
    print(f"[capture] {cfg.cam} @ {cfg.width}x{cfg.height}@{cfg.fps} "
          f"→ {cam_dir} (시작 인덱스 {idx:04d})")
    print(f"[capture] trigger={cfg.trigger} 목표 {cfg.n}장. "
          + ("스페이스=저장 q=종료" if cfg.trigger == "key" else f"{cfg.interval}s 주기"))

    last_t = 0.0
    try:
        # 노출 안정화
        for _ in range(30):
            pipeline.wait_for_frames()

        while saved < cfg.n:
            frames = pipeline.wait_for_frames(timeout_ms=2000)
            aligned = align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                continue
            color = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())  # uint16, scale 적용 전 raw LSB

            do_save = False
            if cfg.trigger == "time":
                now = time.time()
                if now - last_t >= cfg.interval:
                    do_save = True
                    last_t = now
                # 미리보기
                cv2.imshow("capture (q=quit)", color)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            else:  # key
                cv2.imshow("capture (space=save, q=quit)", color)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord(" "):
                    do_save = True

            if do_save:
                cpath, _ = _save_pair(cam_dir, idx, color, depth)
                idx += 1
                saved += 1
                print(f"  [{saved}/{cfg.n}] saved {cpath.name}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    print(f"[capture] 완료: {saved}장 저장 → {cam_dir}")
    print(f"[capture] 다음 단계: python3 auto_label_depth.py --from-folder {cam_dir}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# from-folder: 카메라 없이 기존 프레임 점검 (머신 위치 독립)
# ─────────────────────────────────────────────────────────────────────────────

def _run_from_folder(folder: pathlib.Path, do_list: bool) -> int:
    """기존 raw 폴더의 color/depth 쌍 무결성 점검(카메라 불필요)."""
    if not folder.is_dir():
        raise FileNotFoundError(f"폴더 없음: {folder}")
    colors = sorted(folder.glob("*_color.png"))
    print(f"[from-folder] {folder}: color {len(colors)}장")
    kfile = folder / "K.yaml"
    print(f"[from-folder] K.yaml: {'있음' if kfile.exists() else '없음(경고: depth 단위/intrinsics 불명)'}")

    missing = []
    for c in colors:
        d = c.with_name(c.name.replace("_color.png", "_depth.png"))
        if not d.exists():
            missing.append(c.name)
    if missing:
        print(f"[from-folder] ⚠️ depth 누락 {len(missing)}장: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    else:
        print("[from-folder] 모든 color 에 대응 depth 존재 ✓")

    if do_list:
        for c in colors:
            print(f"  {c.name}")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="RealSense color+depth(aligned) 캡처 → dataset/raw/<cam>/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cam", choices=sorted(_PRESETS), default="d435i",
                        help="카메라(해상도 프리셋 결정)")
    parser.add_argument("--out", default=None,
                        help="raw 루트(기본 dataset/raw)")
    parser.add_argument("--n", type=int, default=80, help="목표 저장 장수")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="자동 저장 주기 [s] (--trigger time)")
    parser.add_argument("--trigger", choices=["time", "key"], default="time",
                        help="time=N초마다 자동 / key=스페이스 수동")
    parser.add_argument("--width",  type=int, default=None, help="color 가로 오버라이드")
    parser.add_argument("--height", type=int, default=None, help="color 세로 오버라이드")
    parser.add_argument("--fps",    type=int, default=None, help="color fps 오버라이드")
    parser.add_argument("--from-folder", default=None,
                        help="카메라 없이 기존 raw 폴더 점검(머신 위치 독립)")
    parser.add_argument("--list", action="store_true", help="--from-folder 시 파일 나열")
    args = parser.parse_args()

    if args.from_folder:
        return _run_from_folder(pathlib.Path(args.from_folder), args.list)

    pw, ph, pfps = _PRESETS[args.cam]
    out_root = pathlib.Path(args.out) if args.out else _THIS_DIR / "dataset" / "raw"
    cfg = CaptureConfig(
        cam=args.cam,
        out_root=out_root,
        n=args.n,
        interval=args.interval,
        trigger=args.trigger,
        width=args.width or pw,
        height=args.height or ph,
        fps=args.fps or pfps,
    )
    return _run_capture(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
