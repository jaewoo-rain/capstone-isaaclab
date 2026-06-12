"""camera/dump_intrinsics.py — RealSense 카메라 내부 파라미터(intrinsics) 저장.

천장캠(D435i)·손목캠(D405)의 fx, fy, cx, cy 를 RealSense SDK 에서 직접 읽어
calib/<name>_intrinsics.yaml 로 저장한다. bringup·로봇 불필요(카메라 USB만).

사용:
  # 천장캠 (D435i, 기본 640x480x30)
  python3 dump_intrinsics.py --name d435i

  # 손목캠 (D405, 기본 640x480x90)
  python3 dump_intrinsics.py --name d405

  # 해상도 직접 지정
  python3 dump_intrinsics.py --name d435i --width 1280 --height 720 --fps 30

⚠️ intrinsics 는 '해상도에 종속'이다. detector 가 실제로 쓰는 스트림 해상도와
   같은 해상도로 측정해야 한다. ceiling_detector/wrist_detector 는 현재 640x480 사용.
"""
from __future__ import annotations

import argparse
import pathlib

import yaml

# 카메라별 기본 스트림 프리셋 (detector 가 쓰는 해상도와 일치시킬 것)
_PRESETS = {
    "d435i": (640, 480, 30),
    "d405":  (640, 480, 90),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RealSense intrinsics → calib/<name>_intrinsics.yaml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--name", required=True, choices=sorted(_PRESETS),
                        help="카메라 이름 (저장 파일명 결정)")
    parser.add_argument("--width",  type=int, default=None, help="color 스트림 가로")
    parser.add_argument("--height", type=int, default=None, help="color 스트림 세로")
    parser.add_argument("--fps",    type=int, default=None, help="color 스트림 fps")
    parser.add_argument("--out", default=None, help="출력 yaml 경로(기본 calib/<name>_intrinsics.yaml)")
    args = parser.parse_args()

    import pyrealsense2 as rs

    pw, ph, pfps = _PRESETS[args.name]
    w   = args.width  or pw
    h   = args.height or ph
    fps = args.fps    or pfps

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    profile = pipeline.start(cfg)
    try:
        intr = (profile.get_stream(rs.stream.color)
                .as_video_stream_profile().get_intrinsics())
    finally:
        pipeline.stop()

    data = {
        "fx": float(intr.fx), "fy": float(intr.fy),
        "cx": float(intr.ppx), "cy": float(intr.ppy),
        # 참고용 메타(코드가 읽진 않음)
        "width": int(intr.width), "height": int(intr.height),
        "model": str(intr.model), "coeffs": [float(c) for c in intr.coeffs],
    }

    out_path = pathlib.Path(args.out) if args.out else \
        pathlib.Path(__file__).resolve().parent / "calib" / f"{args.name}_intrinsics.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=None, sort_keys=False)

    print(f"[dump_intrinsics] {args.name} @ {w}x{h}@{fps}")
    print(f"  fx={data['fx']:.2f} fy={data['fy']:.2f} "
          f"cx={data['cx']:.2f} cy={data['cy']:.2f}")
    print(f"  저장: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
