"""Guarded joint-space smoke test for OMY-F3M.

Default behavior is dry-run only. The script reads `/joint_states`, builds a
tiny one-joint trajectory, and prints the command. It sends a
FollowJointTrajectory goal only when both flags are provided:

    --execute --confirm EXECUTE_JOINT_SPACE_SMOKE_TEST

Use this before any Cartesian/MoveIt execution because joint-space commands are
easier to bound and inspect.
"""
from __future__ import annotations

import argparse


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
CONFIRM_TEXT = "EXECUTE_JOINT_SPACE_SMOKE_TEST"


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


def _make_goal(joint_names: list[str], positions: list[float], duration_s: float):
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = joint_names
    point = JointTrajectoryPoint()
    point.positions = [float(v) for v in positions]
    point.velocities = [0.0 for _ in positions]
    point.time_from_start.sec = int(duration_s)
    point.time_from_start.nanosec = int((duration_s - int(duration_s)) * 1_000_000_000)
    goal.trajectory.points.append(point)
    return goal


def main() -> int:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory

    parser = argparse.ArgumentParser(description="Guarded tiny joint-space smoke test.")
    parser.add_argument("--action-name", default="/arm_controller/follow_joint_trajectory")
    parser.add_argument("--joint", default="joint6", choices=ARM_JOINTS)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--max-delta", type=float, default=0.02)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if abs(args.delta) <= 0.0:
        raise ValueError("--delta must be non-zero")
    if abs(args.delta) > args.max_delta:
        raise ValueError(
            f"delta {args.delta:.4f} rad exceeds max {args.max_delta:.4f} rad")
    if args.duration < 1.0:
        raise ValueError("--duration must be >= 1.0 s")
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"Refusing to execute. Re-run with --confirm {CONFIRM_TEXT}")

    rclpy.init(args=None)
    node = Node("motion2_joint_space_smoke_test")
    client = ActionClient(node, FollowJointTrajectory, args.action_name)
    try:
        joints = _joint_state_once(node, rclpy)
        missing = [name for name in ARM_JOINTS if name not in joints]
        if missing:
            raise RuntimeError(f"Missing arm joints in /joint_states: {missing}")

        target = [float(joints[name]) for name in ARM_JOINTS]
        idx = ARM_JOINTS.index(args.joint)
        start_value = target[idx]
        target[idx] = start_value + float(args.delta)

        print("[joint-smoke] action:", args.action_name)
        print("[joint-smoke] joint_names:", ARM_JOINTS)
        print("[joint-smoke] selected_joint:", args.joint)
        print("[joint-smoke] start_value:", f"{start_value:.6f}")
        print("[joint-smoke] target_value:", f"{target[idx]:.6f}")
        print("[joint-smoke] delta:", f"{args.delta:.6f}")
        print("[joint-smoke] duration:", f"{args.duration:.2f}s")
        print("[joint-smoke] execute:", bool(args.execute))

        if not args.execute:
            print("[joint-smoke] dry-run complete; command_sent=false")
            return 0

        node.get_logger().info(f"Waiting for action {args.action_name}")
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Action server not available: {args.action_name}")

        goal = _make_goal(ARM_JOINTS, target, args.duration)
        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            print("[joint-smoke] goal rejected")
            return 2

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        result = result_future.result().result
        code = int(result.error_code)
        if code == 0:
            print("[joint-smoke] SUCCESS command_sent=true")
            return 0
        print(f"[joint-smoke] FAILED error_code={code} command_sent=true")
        return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
