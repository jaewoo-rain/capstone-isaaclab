"""Guarded real-robot relative motion smoke test.

Default behavior is plan-only. The robot can move only when both flags are
provided:

    --execute --confirm EXECUTE_RELATIVE_SMOKE_TEST

The script allows only a very small relative EE pose target and does not command
the gripper. Use after bringup and move_group are already running.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


def _add_repo_root_to_path() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _norm_xyz(x: float, y: float, z: float) -> float:
    return float(np.linalg.norm(np.array([x, y, z], dtype=np.float64)))


def _joint_state_once(node, rclpy, timeout_s: float = 5.0) -> dict[str, float]:
    from sensor_msgs.msg import JointState

    latest = {"msg": None}

    def _cb(msg):
        latest["msg"] = msg

    sub = node.create_subscription(JointState, "/joint_states", _cb, 10)
    import time

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


def _max_joint_delta(trajectory, current_joints: dict[str, float]) -> tuple[float, str]:
    max_delta = 0.0
    max_joint = ""
    names = list(trajectory.joint_names)
    for point in trajectory.points:
        for idx, joint_name in enumerate(names):
            if joint_name not in current_joints:
                continue
            delta = abs(float(point.positions[idx]) - float(current_joints[joint_name]))
            if delta > max_delta:
                max_delta = delta
                max_joint = joint_name
    return max_delta, max_joint


def main() -> int:
    _add_repo_root_to_path()

    from plan_manual_targets_moveit import (
        _lookup_current_pose,
        _make_move_group_goal,
    )

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from moveit_msgs.action import MoveGroup

    parser = argparse.ArgumentParser(
        description="Guarded + tiny MoveIt relative smoke test.")
    parser.add_argument("--frame-id", default="link0")
    parser.add_argument("--group-name", default="arm")
    parser.add_argument("--link-name", default="end_effector_link")
    parser.add_argument("--action-name", default="/move_action")
    parser.add_argument("--relative-x", type=float, default=0.0)
    parser.add_argument("--relative-y", type=float, default=0.0)
    parser.add_argument("--relative-z", type=float, default=0.01)
    parser.add_argument("--max-translation", type=float, default=0.011)
    parser.add_argument("--planning-time", type=float, default=10.0)
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-tolerance", type=float, default=0.20)
    parser.add_argument("--velocity-scale", type=float, default=0.03)
    parser.add_argument("--acceleration-scale", type=float, default=0.03)
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.20,
        help="Reject planned trajectories whose largest joint delta exceeds this radian limit.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    delta_norm = _norm_xyz(args.relative_x, args.relative_y, args.relative_z)
    if delta_norm <= 0.0:
        raise ValueError("relative motion must be non-zero")
    if delta_norm > args.max_translation:
        raise ValueError(
            f"relative motion norm {delta_norm:.4f} m exceeds limit "
            f"{args.max_translation:.4f} m")
    if args.execute and args.confirm != "EXECUTE_RELATIVE_SMOKE_TEST":
        raise RuntimeError(
            "Refusing to execute. Re-run with "
            "--confirm EXECUTE_RELATIVE_SMOKE_TEST")

    workspace = {
        "x_min": -0.10,
        "x_max": 0.45,
        "y_min": -0.45,
        "y_max": 0.20,
        "z_min": 0.10,
        "z_max": 0.50,
    }

    rclpy.init(args=None)
    node = Node("motion2_execute_relative_smoke_test")
    client = ActionClient(node, MoveGroup, args.action_name)
    try:
        current_pos, current_quat = _lookup_current_pose(
            node, rclpy, args.frame_id, args.link_name)
        current_joints = _joint_state_once(node, rclpy)
        target_pos = current_pos + np.array(
            [args.relative_x, args.relative_y, args.relative_z], dtype=np.float32)

        print("[smoke-test] current_pos:", current_pos.tolist())
        print("[smoke-test] target_pos:", target_pos.tolist())
        print("[smoke-test] quat_wxyz:", current_quat.tolist())
        print("[smoke-test] execute:", bool(args.execute))
        print("[smoke-test] gripper command: none")

        node.get_logger().info(f"Waiting for MoveGroup action {args.action_name}")
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"MoveGroup action server not available: {args.action_name}")

        goal = _make_move_group_goal(
            frame_id=args.frame_id,
            group_name=args.group_name,
            link_name=args.link_name,
            pos=target_pos,
            quat_wxyz=current_quat,
            workspace=workspace,
            planning_time=args.planning_time,
            attempts=args.attempts,
            pos_tol=args.position_tolerance,
            ori_tol=args.orientation_tolerance,
            velocity_scale=args.velocity_scale,
            acceleration_scale=args.acceleration_scale,
        )
        goal.planning_options.plan_only = not args.execute

        send_future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, send_future)
        handle = send_future.result()
        if handle is None or not handle.accepted:
            print("[smoke-test] goal rejected")
            return 2

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(node, result_future)
        result = result_future.result().result
        code = int(result.error_code.val)
        trajectory = result.planned_trajectory.joint_trajectory
        points = len(trajectory.points)
        max_delta, max_joint = _max_joint_delta(trajectory, current_joints)
        print(
            f"[smoke-test] max_joint_delta={max_delta:.4f} "
            f"joint={max_joint or 'n/a'} limit={args.max_joint_delta:.4f}")
        if code == 1:
            if max_delta > args.max_joint_delta:
                print(
                    "[smoke-test] REFUSING trajectory: max joint delta exceeds "
                    "safety limit. execute=false")
                return 3
            print(
                f"[smoke-test] SUCCESS points={points} "
                f"execute={bool(args.execute)}")
            return 0
        print(
            f"[smoke-test] FAILED error_code={code} points={points} "
            f"execute={bool(args.execute)}")
        return 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
