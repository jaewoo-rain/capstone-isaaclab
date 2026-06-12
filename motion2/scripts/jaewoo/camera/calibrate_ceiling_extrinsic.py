"""camera/calibrate_ceiling_extrinsic.py — 천장캠 외부 파라미터(extrinsic) 측정.

체커보드 PnP 로 "천장캠 → link0(robot base)" 4×4 변환행렬을 구해
calib/ceiling_extrinsic.yaml 로 저장한다. 천장캠이 link0 어디에 있는지 알아야
픽셀 좌표를 로봇이 움직일 link0 좌표/yaw 로 바꿀 수 있다.

bringup·로봇 불필요(천장캠 USB + 체커보드만). 1회성.

──────────────────────────────────────────────────────────────────────────────
★★★ 정밀도 주의 — 이 행렬의 회전 오차가 그대로 '박스 yaw' 오차가 된다 ★★★

  yaw_cam_to_base() 가 이 extrinsic 의 회전 성분으로 박스 yaw 를 계산한다.
  따라서 체커보드를 link0 와 '축까지' 맞춰 놓아야 한다:
    - 체커보드 (0,0) 코너  = link0 원점(로봇 발 밑)
    - 체커보드 X축(긴 줄)   = link0 +x (로봇 앞쪽)
    - 체커보드 Y축          = link0 +y
    - 수평(테이블 위 평평)  유지
  비뚤게 놓으면 그 각도가 박스 yaw 에 상수 오차로 더해진다.
──────────────────────────────────────────────────────────────────────────────

사용:
  # 기본 (7×5 내부코너, 30mm, 30프레임 평균)
  python3 calibrate_ceiling_extrinsic.py

  # 체커보드 사양/해상도 지정 + 검출 시각화
  python3 calibrate_ceiling_extrinsic.py --cols 7 --rows 5 --square 0.030 \\
      --width 1280 --height 720 --fps 30 --show

검증:
  python3 coord_transform.py --extrinsic calib/ceiling_extrinsic.yaml
"""
from __future__ import annotations

import argparse
import pathlib

import cv2
import numpy as np
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(
        description="천장캠 extrinsic (체커보드 PnP) → calib/ceiling_extrinsic.yaml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cols", type=int, default=7, help="체커보드 내부코너 가로 개수")
    parser.add_argument("--rows", type=int, default=5, help="체커보드 내부코너 세로 개수")
    parser.add_argument("--square", type=float, default=0.030, help="정사각형 한 변 [m]")
    parser.add_argument("--width",  type=int, default=1280, help="color 스트림 가로")
    parser.add_argument("--height", type=int, default=720,  help="color 스트림 세로")
    parser.add_argument("--fps",    type=int, default=30,   help="color 스트림 fps")
    parser.add_argument("--samples", type=int, default=30,
                        help="평균낼 검출 프레임 수(정적 보드 → 코너 평균으로 노이즈↓)")
    parser.add_argument("--show", action="store_true", help="검출 코너 시각화")
    parser.add_argument("--out", default=None,
                        help="출력 경로(기본 calib/ceiling_extrinsic.yaml)")
    args = parser.parse_args()

    import pyrealsense2 as rs

    board = (args.cols, args.rows)

    # 체커보드 3D 좌표 (link0 평면 z=0, (0,0) 코너가 원점) — 단위 [m]
    objp = np.zeros((args.cols * args.rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:args.cols, 0:args.rows].T.reshape(-1, 2)
    objp *= args.square

    # ── RealSense 시작 + device intrinsics ──
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    profile = pipeline.start(cfg)
    intr = (profile.get_stream(rs.stream.color)
            .as_video_stream_profile().get_intrinsics())
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)

    print(f"[calib] device intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}")
    print(f"[calib] 체커보드 {args.cols}x{args.rows}, square={args.square*1000:.0f}mm")
    print(f"[calib] {args.samples} 프레임 검출 시도 중 ... (보드가 화면에 또렷이 보여야 함)")

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corner_stack = []
    tried = 0
    try:
        # 노출 안정화
        for _ in range(30):
            pipeline.wait_for_frames()

        while len(corner_stack) < args.samples and tried < args.samples * 10:
            tried += 1
            frames = pipeline.wait_for_frames()
            img = np.asanyarray(frames.get_color_frame().get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ok, corners = cv2.findChessboardCorners(
                gray, board,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not ok:
                continue
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            corner_stack.append(corners)
            if args.show:
                vis = img.copy()
                cv2.drawChessboardCorners(vis, board, corners, ok)
                cv2.imshow("ceiling calib (q to abort)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        pipeline.stop()
        if args.show:
            cv2.destroyAllWindows()

    if not corner_stack:
        raise RuntimeError(
            "체커보드를 한 번도 찾지 못했습니다.\n"
            "  - 조명(반사/그림자) 확인, 보드 전체가 화면 안에 또렷이\n"
            "  - --cols/--rows 가 '내부 코너' 개수인지 확인 (사각형 개수-1)")

    # 정적 보드 → 코너 픽셀 평균으로 노이즈 감소 후 1회 solvePnP
    corners_mean = np.mean(np.stack(corner_stack, axis=0), axis=0).astype(np.float32)
    print(f"[calib] 검출 성공 {len(corner_stack)}/{tried} 프레임 → 코너 평균으로 PnP")

    ok, rvec, tvec = cv2.solvePnP(objp, corners_mean, K, dist)
    if not ok:
        raise RuntimeError("solvePnP 실패")

    # 재투영 오차(품질 지표)
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    reproj_err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners_mean.reshape(-1, 2)) ** 2, axis=1))))

    R, _ = cv2.Rodrigues(rvec)
    T_base2cam = np.eye(4)          # link0(=board) → camera
    T_base2cam[:3, :3] = R
    T_base2cam[:3, 3] = tvec.flatten()
    T_cam2base = np.linalg.inv(T_base2cam)   # camera → link0  (detector 가 쓰는 방향)

    out = {"T_cam_to_base": T_cam2base.tolist()}
    out_path = pathlib.Path(args.out) if args.out else \
        pathlib.Path(__file__).resolve().parent / "calib" / "ceiling_extrinsic.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(out, f, default_flow_style=None, sort_keys=False)

    cam_pos = T_cam2base[:3, 3]
    print(f"[calib] 재투영 오차 RMS = {reproj_err:.2f} px  (1px 이하 권장, 3px 넘으면 재측정)")
    print(f"[calib] 카메라 위치(link0 기준) ≈ "
          f"x={cam_pos[0]:.3f} y={cam_pos[1]:.3f} z={cam_pos[2]:.3f} m")
    print(f"[calib] T_cam_to_base:\n{np.round(T_cam2base, 4)}")
    print(f"[calib] 저장: {out_path}")
    print("[calib] 검증: python3 coord_transform.py --extrinsic "
          f"{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
