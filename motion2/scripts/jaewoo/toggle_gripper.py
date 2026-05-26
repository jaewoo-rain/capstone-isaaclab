"""그리퍼 토글: 현재 상태를 읽어 열려있으면 닫고, 닫혀있으면 연다.

판단 기준 (--threshold, 기본 0.5):
    현재 위치 < threshold  → 열림 상태로 판단 → 닫기 (1.05)
    현재 위치 >= threshold → 닫힘 상태로 판단 → 열기 (0.0)

기본 동작은 dry-run. 실제 실행:
    --execute --confirm TOGGLE_GRIPPER

사용 예시:
    python3 motion2/scripts/jaewoo/toggle_gripper.py
    python3 motion2/scripts/jaewoo/toggle_gripper.py --execute --confirm TOGGLE_GRIPPER
    python3 motion2/scripts/jaewoo/toggle_gripper.py --close-pos 1.05 --open-pos 0.0
"""
from __future__ import annotations

import argparse

import numpy as np


CONFIRM_TEXT = "TOGGLE_GRIPPER"
GRIPPER_JOINT = "rh_r1_joint"
DEFAULT_GRIPPER_ACTION = "/gripper_controller/gripper_cmd"
DEFAULT_CLOSE_POS = 1.05
DEFAULT_OPEN_POS = 0.0
DEFAULT_THRESHOLD = 0.5   # 이 값보다 작으면 열림, 크거나 같으면 닫힘


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
            raise RuntimeError(f"Timed out waiting for /joint_states ({timeout:.1f}s)")
    node.destroy_subscription(sub)
    return result


def _send_gripper(node, rclpy, client, position: float, max_effort: float) -> bool:
    from control_msgs.action import GripperCommand

    goal = GripperCommand.Goal()
    goal.command.position = float(position)
    goal.command.max_effort = float(max_effort)

    fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut)
    handle = fut.result()
    if handle is None or not handle.accepted:
        print("[toggle-gripper] goal rejected")
        return False

    res_fut = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res_fut)
    r = res_fut.result().result
    ok = bool(r.reached_goal or r.stalled)
    print(
        f"[toggle-gripper] pos={r.position:.4f}  effort={r.effort:.4f}  "
        f"stalled={r.stalled}  reached={r.reached_goal}  → {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import GripperCommand

    parser = argparse.ArgumentParser(
        description="그리퍼 토글: 열려있으면 닫고, 닫혀있으면 연다.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--close-pos", type=float, default=DEFAULT_CLOSE_POS,
                        help="닫을 때 목표 위치 [m]")
    parser.add_argument("--open-pos", type=float, default=DEFAULT_OPEN_POS,
                        help="열 때 목표 위치 [m]")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="열림/닫힘 판단 기준 [m]. 현재값 < threshold → 열림으로 판단")
    parser.add_argument("--max-effort", type=float, default=0.0)
    parser.add_argument("--gripper-action", default=DEFAULT_GRIPPER_ACTION)
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"Refusing. --confirm {CONFIRM_TEXT} 필요")

    rclpy.init(args=None)
    node = Node("motion2_toggle_gripper")
    gripper_client = ActionClient(node, GripperCommand, args.gripper_action)

    try:
        # 현재 그리퍼 위치 읽기
        joints = _joint_state_once(node, rclpy, args.joint_state_timeout)
        if GRIPPER_JOINT not in joints:
            raise RuntimeError(f"/joint_states 에 '{GRIPPER_JOINT}' 없음")

        current = joints[GRIPPER_JOINT]

        # 토글 방향 결정
        if current < args.threshold:
            action = "닫기"
            target = args.close_pos
        else:
            action = "열기"
            target = args.open_pos

        print(f"\n[toggle-gripper] 현재 위치 : {current:.4f} m")
        print(f"[toggle-gripper] 판단      : {'열림' if current < args.threshold else '닫힘'} "
              f"(threshold={args.threshold})")
        print(f"[toggle-gripper] 동작      : {action} → {target:.4f} m")
        print(f"[toggle-gripper] execute   : {args.execute}\n")

        if not args.execute:
            print("[toggle-gripper] dry-run complete; command_sent=false")
            return 0

        print(f"[toggle-gripper] gripper action 대기: {args.gripper_action}")
        if not gripper_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"gripper action 없음: {args.gripper_action}")

        ok = _send_gripper(node, rclpy, gripper_client, target, args.max_effort)
        return 0 if ok else 2

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
