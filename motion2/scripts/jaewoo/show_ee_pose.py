"""현재 OMY-F3M end-effector 위치를 읽어 run_ee_pose_move.py 인자 형식으로 출력한다.

로봇을 원하는 위치로 먼저 이동시킨 뒤 이 스크립트를 실행하면,
run_ee_pose_move.py 에 바로 붙여 쓸 수 있는 --x --y --z 값을 얻을 수 있다.

사용 흐름:
    1. 로봇을 원하는 위치로 이동 (teach 또는 수동 조작)
    2. python3 motion2/scripts/show_ee_pose.py
    3. 출력된 명령어를 복사해 run_ee_pose_move.py 에 사용

사용 예시:
    python3 motion2/scripts/show_ee_pose.py
    python3 motion2/scripts/show_ee_pose.py --samples 3   # 3회 평균
    python3 motion2/scripts/show_ee_pose.py --ee-frame link6
"""
from __future__ import annotations

import argparse
import time

import numpy as np


DEFAULT_BASE_FRAME = "link0"
DEFAULT_EE_FRAME = "link6"


def _lookup_tf(node, rclpy, base_frame: str, ee_frame: str, timeout_s: float):
    """TF2로 base_frame → ee_frame 변환을 읽어 (pos, quat_wxyz) 반환."""
    from tf2_ros import Buffer, TransformListener

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    deadline = time.monotonic() + timeout_s
    last_err = None
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            tf = tf_buffer.lookup_transform(base_frame, ee_frame, rclpy.time.Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            pos = np.array([t.x, t.y, t.z], dtype=np.float64)
            quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float64)
            return pos, quat
        except Exception as exc:
            last_err = exc
    raise RuntimeError(
        f"TF timeout: {base_frame} → {ee_frame}. Last error: {last_err}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="현재 EE 위치를 읽어 run_ee_pose_move.py 인자 형식으로 출력.")
    parser.add_argument("--base-frame", default=DEFAULT_BASE_FRAME)
    parser.add_argument("--ee-frame", default=DEFAULT_EE_FRAME)
    parser.add_argument(
        "--samples", type=int, default=1,
        help="여러 번 읽어 평균을 낼 횟수 (로봇이 진동 중일 때 유용)")
    parser.add_argument("--interval", type=float, default=0.3,
                        help="샘플 간 대기 시간 [s]")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be > 0")

    import rclpy
    from rclpy.node import Node

    rclpy.init(args=None)
    node = Node("motion2_show_ee_pose")
    try:
        positions = []
        quats = []
        for i in range(args.samples):
            pos, quat = _lookup_tf(node, rclpy, args.base_frame, args.ee_frame, args.timeout)
            positions.append(pos)
            quats.append(quat)
            if i < args.samples - 1:
                time.sleep(args.interval)

        mean_pos = np.mean(positions, axis=0)
        mean_quat = np.mean(quats, axis=0)
        mean_quat /= np.linalg.norm(mean_quat)  # 평균 쿼터니언 정규화

        x, y, z = float(mean_pos[0]), float(mean_pos[1]), float(mean_pos[2])
        w, qx, qy, qz = (float(mean_quat[0]), float(mean_quat[1]),
                          float(mean_quat[2]), float(mean_quat[3]))

        print(f"\n{'='*60}")
        print(f"[show-ee-pose] frame: {args.base_frame} → {args.ee_frame}")
        if args.samples > 1:
            print(f"[show-ee-pose] samples: {args.samples}회 평균")
        print(f"[show-ee-pose] position  : x={x:.6f}  y={y:.6f}  z={z:.6f}  [m]")
        print(f"[show-ee-pose] quat_wxyz : w={w:.6f}  x={qx:.6f}  y={qy:.6f}  z={qz:.6f}")
        print(f"{'='*60}")
        print()
        print("# run_ee_pose_move.py 에 사용할 인자 (복사해서 사용):")
        print(f"  --x {x:.6f} --y {y:.6f} --z {z:.6f}")
        print()
        print("# 바로 실행할 명령어 (dry-run):")
        print(
            f"  python3 motion2/scripts/run_ee_pose_move.py "
            f"--x {x:.6f} --y {y:.6f} --z {z:.6f}")
        print()
        print("# 실제 실행:")
        print(
            f"  python3 motion2/scripts/run_ee_pose_move.py "
            f"--x {x:.6f} --y {y:.6f} --z {z:.6f} "
            f"--execute --confirm EXECUTE_EE_POSE_MOVE")
        print(f"{'='*60}\n")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
