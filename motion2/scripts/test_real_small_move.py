"""Plan or explicitly execute a tiny current-pose MoveIt motion for OMY-F3M."""
from __future__ import annotations

import argparse

import numpy as np

from real_moveit_common import (
    assert_in_workspace,
    current_to_planned_start_deltas,
    default_workspace,
    joint_state_once,
    lookup_current_pose,
    plan_or_execute_pose,
)


CONFIRM_TEXT = "EXECUTE_SMALL_MOVE"
DEFAULT_ACTION_NAME = "/move_action"
DEFAULT_GROUP_NAME = "arm"
DEFAULT_BASE_FRAME = "link0"
DEFAULT_EE_FRAME = "link6"
CHECK_COMMANDS = (
    "ros2 action list",
    "ros2 node list",
    "ros2 topic echo /joint_states",
    "ros2 run tf2_ros tf2_echo link0 link6",
)


def _offset_from_axis(axis: str, distance: float) -> np.ndarray:
    offset = np.zeros(3, dtype=np.float32)
    offset["xyz".index(axis)] = float(distance)
    return offset


def _check_hint() -> str:
    return "Check inside the Docker container: " + "; ".join(CHECK_COMMANDS)


def main() -> int:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from moveit_msgs.action import MoveGroup

    parser = argparse.ArgumentParser(description="Tiny current-pose move for real OMY-F3M.")
    parser.add_argument("--action-name", default=DEFAULT_ACTION_NAME)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--base-frame", default=DEFAULT_BASE_FRAME)
    parser.add_argument("--ee-frame", default=DEFAULT_EE_FRAME)
    parser.add_argument("--eef-link", default=None, help="Alias for --ee-frame.")
    parser.add_argument("--planner", default="", help="Optional MoveIt planner_id, e.g. RRTConnect.")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--distance", type=float, default=0.005)
    parser.add_argument("--dx", type=float, default=None, help="Override --axis/--distance with explicit x offset.")
    parser.add_argument("--dy", type=float, default=None, help="Override --axis/--distance with explicit y offset.")
    parser.add_argument("--dz", type=float, default=None, help="Override --axis/--distance with explicit z offset.")
    parser.add_argument("--position-tolerance", type=float, default=0.005)
    parser.add_argument("--orientation-tolerance", type=float, default=0.20)
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--velocity-scale", type=float, default=0.03)
    parser.add_argument("--acceleration-scale", type=float, default=0.03)
    parser.add_argument("--max-start-to-goal-joint-delta", type=float, default=0.20)
    parser.add_argument("--max-segment-joint-delta", type=float, default=0.08)
    parser.add_argument("--position-only", action="store_true", help="Use only position constraint.")
    parser.add_argument("--keep-orientation", action="store_true", help="Keep current EE orientation constraint.")
    parser.add_argument("--joint-state-timeout", type=float, default=3.0)
    parser.add_argument("--plan-only", action="store_true", help="Default behavior; accepted for explicit CLI use.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"Refusing to execute. Re-run with --confirm {CONFIRM_TEXT}")
    if args.position_only and args.keep_orientation:
        raise ValueError("Use only one of --position-only or --keep-orientation")

    ee_frame = args.eef_link or args.ee_frame
    keep_orientation = args.keep_orientation or not args.position_only

    if any(value is not None for value in (args.dx, args.dy, args.dz)):
        offset = np.array([
            0.0 if args.dx is None else args.dx,
            0.0 if args.dy is None else args.dy,
            0.0 if args.dz is None else args.dz,
        ], dtype=np.float32)
    else:
        offset = _offset_from_axis(args.axis, args.distance)

    workspace = default_workspace()
    rclpy.init(args=None)
    node = Node("motion2_test_real_small_move")
    client = ActionClient(node, MoveGroup, args.action_name)
    try:
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(
                f"MoveGroup action server not available: {args.action_name}. {_check_hint()}")

        current_pos, current_quat = lookup_current_pose(
            node, rclpy, args.base_frame, ee_frame)
        current_joints = joint_state_once(node, rclpy, args.joint_state_timeout)
        target_pos = current_pos + offset
        assert_in_workspace("small_move_target", target_pos, workspace)

        plan_only = not args.execute
        print(f"[small-move] plan_only={plan_only} execute={args.execute}")
        print(f"[small-move] planning_frame={args.base_frame}")
        print(f"[small-move] group_name={args.group_name}")
        print(f"[small-move] eef_link={ee_frame}")
        print(f"[small-move] planner_id={args.planner!r}")
        print(f"[small-move] keep_orientation={keep_orientation}")
        print(f"[small-move] offset={np.round(offset, 6).tolist()}")
        print(f"[small-move] current_pos={np.round(current_pos, 6).tolist()}")
        print(f"[small-move] target_pos={np.round(target_pos, 6).tolist()}")
        print(f"[small-move] current_quat_wxyz={np.round(current_quat, 6).tolist()}")
        print(f"[small-move] target_quat_wxyz={np.round(current_quat, 6).tolist()}")
        print("[small-move] current_joint_states:")
        for joint_name in sorted(current_joints):
            if joint_name.startswith("joint") or "gripper" in joint_name or "rh_" in joint_name:
                print(f"  {joint_name}={current_joints[joint_name]:.6f}")

        ok, report = plan_or_execute_pose(
            node=node,
            rclpy=rclpy,
            client=client,
            frame_id=args.base_frame,
            group_name=args.group_name,
            link_name=ee_frame,
            name="small_move",
            pos=target_pos,
            quat_wxyz=current_quat,
            workspace=workspace,
            planning_time=args.planning_time,
            attempts=args.attempts,
            pos_tol=args.position_tolerance,
            ori_tol=args.orientation_tolerance,
            velocity_scale=args.velocity_scale,
            acceleration_scale=args.acceleration_scale,
            plan_only=True,
            keep_orientation=keep_orientation,
            planner_id=args.planner,
            max_start_to_goal_delta=args.max_start_to_goal_joint_delta,
            max_segment_delta=args.max_segment_joint_delta,
        )
        print(
            "[small-move] result "
            f"success={ok} error_code={report.get('error_code')} "
            f"points={report.get('point_count')} "
            f"max_start_to_goal_delta={float(report.get('max_start_to_goal_delta', 0.0)):.6f} "
            f"max_segment_delta={float(report.get('max_segment_delta', 0.0)):.6f} "
            f"max_joint={report.get('max_joint', '')}")
        joint_names = list(report.get("joint_names", []))
        planned_start = list(report.get("start_positions", []))
        planned_goal = list(report.get("goal_positions", []))
        planned_deltas = list(report.get("start_to_goal_deltas", []))
        if joint_names and planned_start:
            print("[small-move] planned_start_vs_current_joint_delta:")
            for joint_name, delta in current_to_planned_start_deltas(
                current_joints, joint_names, planned_start):
                print(f"  {joint_name}: planned_start-current={delta:.6f}")
        if joint_names and planned_goal:
            print("[small-move] planned_start_to_goal_joint_delta:")
            for joint_name, start, goal, delta in zip(
                joint_names, planned_start, planned_goal, planned_deltas):
                print(
                    f"  {joint_name}: start={start:.6f} goal={goal:.6f} "
                    f"delta={delta:.6f}")
        if not report.get("guard_ok", False):
            print("[small-move] blocked_by_joint_delta_guard=true")
        if not ok:
            return 2

        if not args.execute:
            print("[small-move] plan-only complete; command_sent=false")
            return 0

        print("[small-move] executing confirmed move after guarded plan-only success")
        exec_ok, exec_report = plan_or_execute_pose(
            node=node,
            rclpy=rclpy,
            client=client,
            frame_id=args.base_frame,
            group_name=args.group_name,
            link_name=ee_frame,
            name="small_move_execute",
            pos=target_pos,
            quat_wxyz=current_quat,
            workspace=workspace,
            planning_time=args.planning_time,
            attempts=args.attempts,
            pos_tol=args.position_tolerance,
            ori_tol=args.orientation_tolerance,
            velocity_scale=args.velocity_scale,
            acceleration_scale=args.acceleration_scale,
            plan_only=False,
            keep_orientation=keep_orientation,
            planner_id=args.planner,
            max_start_to_goal_delta=args.max_start_to_goal_joint_delta,
            max_segment_delta=args.max_segment_joint_delta,
        )
        print(
            "[small-move] execute_result "
            f"success={exec_ok} error_code={exec_report.get('error_code')} "
            f"points={exec_report.get('point_count')} "
            f"command_sent=true")
        return 0 if exec_ok else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
