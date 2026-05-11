"""Guarded gripper smoke test for OMY-F3M.

Default behavior is dry-run only. The script reads `rh_r1_joint`, computes a
small target, and sends `control_msgs/action/GripperCommand` only when both
flags are provided:

    --execute --confirm EXECUTE_GRIPPER_SMOKE_TEST
"""
from __future__ import annotations

import argparse


CONFIRM_TEXT = "EXECUTE_GRIPPER_SMOKE_TEST"


def _joint_state_once(node, rclpy, timeout_s: float = 5.0) -> dict[str, float]:
    from sensor_msgs.msg import JointState
    import time

    latest = {"msg": None}

    def _cb(msg):
        latest["msg"] = msg

    sub = node.create_subscription(JointState, "/joint_states", _cb, 10)
    deadline = time.monotonic() + timeout_s
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if latest["msg"] is not None:
                msg = latest["msg"]
                return dict(zip(msg.name, msg.position))
    finally:
        node.destroy_subscription(sub)
    raise RuntimeError("Timed out waiting for /joint_states")


def _make_goal(position: float, max_effort: float):
    from control_msgs.action import GripperCommand

    goal = GripperCommand.Goal()
    goal.command.position = float(position)
    goal.command.max_effort = float(max_effort)
    return goal


def main() -> int:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import GripperCommand

    parser = argparse.ArgumentParser(description="Guarded tiny gripper smoke test.")
    parser.add_argument("--action-name", default="/gripper_controller/gripper_cmd")
    parser.add_argument("--joint", default="rh_r1_joint")
    parser.add_argument("--delta", type=float, default=0.03)
    parser.add_argument("--max-delta", type=float, default=0.04)
    parser.add_argument("--min-position", type=float, default=0.0)
    parser.add_argument("--max-position", type=float, default=1.12)
    parser.add_argument("--max-effort", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if abs(args.delta) <= 0.0:
        raise ValueError("--delta must be non-zero")
    if abs(args.delta) > args.max_delta:
        raise ValueError(
            f"delta {args.delta:.4f} exceeds max {args.max_delta:.4f}")
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"Refusing to execute. Re-run with --confirm {CONFIRM_TEXT}")

    rclpy.init(args=None)
    node = Node("motion2_gripper_smoke_test")
    client = ActionClient(node, GripperCommand, args.action_name)
    try:
        joints = _joint_state_once(node, rclpy)
        if args.joint not in joints:
            raise RuntimeError(f"Missing {args.joint} in /joint_states")

        start = float(joints[args.joint])
        target = start + float(args.delta)
        if not (args.min_position <= target <= args.max_position):
            raise ValueError(
                f"target {target:.4f} outside "
                f"[{args.min_position:.4f}, {args.max_position:.4f}]")

        print("[gripper-smoke] action:", args.action_name)
        print("[gripper-smoke] joint:", args.joint)
        print("[gripper-smoke] start_value:", f"{start:.6f}")
        print("[gripper-smoke] target_value:", f"{target:.6f}")
        print("[gripper-smoke] delta:", f"{args.delta:.6f}")
        print("[gripper-smoke] max_effort:", f"{args.max_effort:.6f}")
        print("[gripper-smoke] execute:", bool(args.execute))

        if not args.execute:
            print("[gripper-smoke] dry-run complete; command_sent=false")
            return 0

        node.get_logger().info(f"Waiting for action {args.action_name}")
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Action server not available: {args.action_name}")

        goal = _make_goal(target, args.max_effort)
        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            print("[gripper-smoke] goal rejected")
            return 2

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        result = result_future.result().result
        print(
            f"[gripper-smoke] result position={result.position:.6f} "
            f"effort={result.effort:.6f} stalled={result.stalled} "
            f"reached_goal={result.reached_goal} command_sent=true")
        return 0 if result.reached_goal else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
