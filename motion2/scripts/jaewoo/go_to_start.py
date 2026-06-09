"""YAML waypoints['start'] 관절값으로 로봇을 홈 자세로 복귀시킨다.

start 자세:
  - joint2 ≈ -π/2, joint3 ≈ 2.65: 팔이 위쪽으로 접힌 안전 대기 자세
  - EE 위치: x≈-0.05, y≈-0.11, z≈0.39 (베이스 근처, 낮은 높이)
  - 그리퍼: 열림 (0.0) 상태로 기록됨

이 스크립트는 MoveIt 없이 FollowJointTrajectory 를 직접 전송한다.
이상 상황(로봇이 예상치 못한 자세에 있을 때) 홈 복귀용으로 사용.

기본 동작은 dry-run. 실제 실행:
    --execute --confirm GO_TO_START

사용 예시:
    python3 motion2/scripts/jaewoo/go_to_start.py
    python3 motion2/scripts/jaewoo/go_to_start.py --execute --confirm GO_TO_START
    python3 motion2/scripts/jaewoo/go_to_start.py --duration 10.0 --execute --confirm GO_TO_START
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import numpy as np
import yaml


CONFIRM_TEXT = "GO_TO_START"
WAYPOINT_KEY = "start"
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
DEFAULT_CONFIG = "motion2/config/teach_pick_place_waypoints.yaml"
DEFAULT_ARM_ACTION = "/arm_controller/follow_joint_trajectory"


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[3]


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML 최상위가 mapping이 아님")
    return data


def _read_arm_joints(data: dict[str, Any], key: str) -> list[float]:
    wp = data.get("waypoints", {}).get(key)
    if not isinstance(wp, dict):
        raise KeyError(f"YAML에 waypoints['{key}'] 없음")
    arm = wp.get("arm")
    if not isinstance(arm, dict):
        raise KeyError(f"waypoints['{key}']['arm'] 없음")
    missing = [j for j in ARM_JOINTS if j not in arm]
    if missing:
        raise KeyError(f"waypoints['{key}'] 관절 누락: {missing}")
    return [float(arm[j]) for j in ARM_JOINTS]


def _joint_state_once(node, rclpy, timeout: float) -> dict[str, float]:
    from sensor_msgs.msg import JointState

    result: dict[str, float] = {}
    done = False

    def _cb(msg: JointState) -> None:
        nonlocal done
        for name, pos in zip(msg.name, msg.position):
            result[name] = float(pos)
        done = True

    sub = node.create_subscription(JointState, "/joint_states", _cb, 10)
    deadline = node.get_clock().now().nanoseconds + int(timeout * 1e9)
    while not done:
        rclpy.spin_once(node, timeout_sec=0.05)
        if node.get_clock().now().nanoseconds > deadline:
            node.destroy_subscription(sub)
            raise RuntimeError(
                f"Timed out waiting for /joint_states ({timeout:.1f}s)")
    node.destroy_subscription(sub)
    return result


def _send_arm_traj(node, rclpy, client, positions: list[float], duration_s: float) -> bool:
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_JOINTS)
    pt = JointTrajectoryPoint()
    pt.positions = [float(v) for v in positions]
    pt.velocities = [0.0] * len(positions)
    pt.time_from_start.sec = int(duration_s)
    pt.time_from_start.nanosec = int((duration_s % 1) * 1_000_000_000)
    goal.trajectory.points.append(pt)

    fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut)
    handle = fut.result()
    if handle is None or not handle.accepted:
        print("[go-to-start] arm goal rejected")
        return False
    res_fut = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res_fut)
    code = int(res_fut.result().result.error_code)
    if code == 0:
        print("[go-to-start] SUCCESS command_sent=true")
        return True
    print(f"[go-to-start] FAILED error_code={code}")
    return False


def main() -> int:
    scripts_dir = str(pathlib.Path(__file__).resolve().parent.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory

    parser = argparse.ArgumentParser(
        description=f"YAML waypoints['{WAYPOINT_KEY}'] 자세로 홈 복귀.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--arm-action", default=DEFAULT_ARM_ACTION)
    parser.add_argument("--duration", type=float, default=9.0,
                        help="이동 시간 [s] (최소 4.0, 이상 상황이므로 여유 있게 설정 권장)")
    parser.add_argument("--max-joint-delta", type=float, default=6.30,
                        help="현재→목표 최대 허용 관절 변위 [rad]. "
                             "이상 상황에서는 큰 값이 필요할 수 있음 (기본 6.30 ≈ 2π)")
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"Refusing. --confirm {CONFIRM_TEXT} 필요")
    if args.duration < 4.0:
        raise ValueError("--duration >= 4.0 s 필요")

    config_path = pathlib.Path(args.config)
    if not config_path.is_absolute():
        config_path = _repo_root() / config_path
    data = _load_yaml(config_path)
    target_joints = _read_arm_joints(data, WAYPOINT_KEY)

    # EE 위치 참고값 출력
    wp = data["waypoints"][WAYPOINT_KEY]
    if "ee_pose" in wp:
        pos = wp["ee_pose"]["position"]
        print(f"[go-to-start] target EE (참고): "
              f"x={pos['x']:.4f}  y={pos['y']:.4f}  z={pos['z']:.4f} [m]")
    gripper_val = wp.get("gripper", {}).get("rh_r1_joint", "?")
    print(f"[go-to-start] 기록된 그리퍼 상태: {gripper_val}  (이 스크립트는 그리퍼 명령 없음)")

    rclpy.init(args=None)
    node = Node("motion2_go_to_start")
    arm_client = ActionClient(node, FollowJointTrajectory, args.arm_action)

    try:
        current_joints = _joint_state_once(node, rclpy, args.joint_state_timeout)
        missing = [j for j in ARM_JOINTS if j not in current_joints]
        if missing:
            raise RuntimeError(f"/joint_states 에 관절 없음: {missing}")
        current_arm = np.array([current_joints[j] for j in ARM_JOINTS], dtype=np.float64)
        target_arm = np.array(target_joints, dtype=np.float64)

        deltas = np.abs(target_arm - current_arm)
        max_delta = float(np.max(deltas))
        max_joint = ARM_JOINTS[int(np.argmax(deltas))]

        print(f"\n[go-to-start] ── 계획 ───────────────────────────────────────")
        print(f"[go-to-start] execute      : {args.execute}")
        print(f"[go-to-start] waypoint     : '{WAYPOINT_KEY}'")
        print(f"[go-to-start] duration     : {args.duration:.1f} s")
        for j, tgt, cur, d in zip(ARM_JOINTS, target_arm, current_arm, deltas):
            marker = " ← MAX" if j == max_joint else ""
            print(f"  {j}: cur={cur:.6f}  target={tgt:.6f}  delta={d:.6f} rad{marker}")
        print(f"[go-to-start] max_delta    : {max_delta:.6f} rad at {max_joint} "
              f"(limit={args.max_joint_delta:.6f} rad)")

        if max_delta > args.max_joint_delta:
            print(f"[go-to-start] GUARD VIOLATED: {max_delta:.4f} > {args.max_joint_delta:.4f} rad")
            print("[go-to-start] --max-joint-delta 를 높이거나 로봇 자세를 먼저 확인하세요.")
            return 2
        print(f"[go-to-start] ──────────────────────────────────────────────────\n")

        if not args.execute:
            print("[go-to-start] dry-run complete; command_sent=false")
            return 0

        print(f"[go-to-start] arm action 대기: {args.arm_action}")
        if not arm_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"arm action 없음: {args.arm_action}")

        ok = _send_arm_traj(node, rclpy, arm_client, target_joints, args.duration)
        return 0 if ok else 2

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
