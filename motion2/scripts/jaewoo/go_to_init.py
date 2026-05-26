"""YAML waypoints['init'] 관절값으로 로봇을 초기화 자세로 이동시킨다.

init 자세:
  - joint1~6 ≈ 0: 모든 관절이 원점 (팔이 정면 수직으로 완전히 펴진 상태)
  - EE 위치: x≈0.00, y≈-0.11, z≈0.75 (가장 높은 위치, 주변 충돌 주의)
  - 주의: YAML 기록 당시 그리퍼 = 1.12 (닫힘) — 이 스크립트는 그리퍼 명령 없음

이 스크립트는 MoveIt 없이 FollowJointTrajectory 를 직접 전송한다.
관절 원점 복귀가 필요한 이상 상황에서 사용.

주의사항:
  - 팔을 완전히 수직으로 세우므로 위쪽 공간에 장애물이 없는지 반드시 확인
  - 어느 자세에서든 큰 관절 이동이 발생할 수 있으므로 --duration 을 충분히 설정
  - 기본 delta 한계가 2π(≈6.28)로 설정되어 있어 대부분의 자세에서 실행 가능

기본 동작은 dry-run. 실제 실행:
    --execute --confirm GO_TO_INIT

사용 예시:
    python3 motion2/scripts/jaewoo/go_to_init.py
    python3 motion2/scripts/jaewoo/go_to_init.py --execute --confirm GO_TO_INIT
    python3 motion2/scripts/jaewoo/go_to_init.py --duration 12.0 --execute --confirm GO_TO_INIT
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import numpy as np
import yaml


CONFIRM_TEXT = "GO_TO_INIT"
WAYPOINT_KEY = "init"
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
        print("[go-to-init] arm goal rejected")
        return False
    res_fut = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res_fut)
    code = int(res_fut.result().result.error_code)
    if code == 0:
        print("[go-to-init] SUCCESS command_sent=true")
        return True
    print(f"[go-to-init] FAILED error_code={code}")
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
        description=f"YAML waypoints['{WAYPOINT_KEY}'] 자세로 관절 원점 복귀.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--arm-action", default=DEFAULT_ARM_ACTION)
    parser.add_argument("--duration", type=float, default=12.0,
                        help="이동 시간 [s] (최소 4.0). init은 큰 이동이 예상되므로 기본 12s)")
    parser.add_argument("--max-joint-delta", type=float, default=6.30,
                        help="현재→목표 최대 허용 관절 변위 [rad]. "
                             "init 복귀는 대각이 클 수 있어 기본 6.30 ≈ 2π")
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

    # EE 위치 참고값 및 경고 출력
    wp = data["waypoints"][WAYPOINT_KEY]
    if "ee_pose" in wp:
        pos = wp["ee_pose"]["position"]
        print(f"[go-to-init] target EE (참고): "
              f"x={pos['x']:.4f}  y={pos['y']:.4f}  z={pos['z']:.4f} [m]")
    gripper_val = wp.get("gripper", {}).get("rh_r1_joint", "?")
    print(f"[go-to-init] 기록된 그리퍼 상태: {gripper_val}  (이 스크립트는 그리퍼 명령 없음)")
    print("[go-to-init] !!! 주의: 팔이 수직으로 세워짐 — 위쪽 공간 장애물 확인 필수 !!!")

    rclpy.init(args=None)
    node = Node("motion2_go_to_init")
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

        print(f"\n[go-to-init] ── 계획 ────────────────────────────────────────")
        print(f"[go-to-init] execute      : {args.execute}")
        print(f"[go-to-init] waypoint     : '{WAYPOINT_KEY}'")
        print(f"[go-to-init] duration     : {args.duration:.1f} s")
        for j, tgt, cur, d in zip(ARM_JOINTS, target_arm, current_arm, deltas):
            marker = " ← MAX" if j == max_joint else ""
            print(f"  {j}: cur={cur:.6f}  target={tgt:.6f}  delta={d:.6f} rad{marker}")
        print(f"[go-to-init] max_delta    : {max_delta:.6f} rad at {max_joint} "
              f"(limit={args.max_joint_delta:.6f} rad)")

        if max_delta > args.max_joint_delta:
            print(f"[go-to-init] GUARD VIOLATED: {max_delta:.4f} > {args.max_joint_delta:.4f} rad")
            return 2
        print(f"[go-to-init] ──────────────────────────────────────────────────\n")

        if not args.execute:
            print("[go-to-init] dry-run complete; command_sent=false")
            return 0

        print(f"[go-to-init] arm action 대기: {args.arm_action}")
        if not arm_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"arm action 없음: {args.arm_action}")

        ok = _send_arm_traj(node, rclpy, arm_client, target_joints, args.duration)
        return 0 if ok else 2

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
