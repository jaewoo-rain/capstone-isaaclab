"""MoveIt waypoint pick-and-place for real OMY-F3M manual coordinates.

Default is plan-only. Execution is allowed only with:

    --execute --confirm EXECUTE_MANUAL_PICK_PLACE
"""
from __future__ import annotations

import argparse

import numpy as np

from real_moveit_common import (
    assert_in_workspace,
    default_workspace,
    lookup_current_pose,
    plan_or_execute_pose,
    quat_from_z_yaw,
    quat_mul,
)


CONFIRM_TEXT = "EXECUTE_MANUAL_PICK_PLACE"
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


def _check_hint() -> str:
    return "Check inside the Docker container: " + "; ".join(CHECK_COMMANDS)


def _build_stages(args) -> list[tuple[str, np.ndarray, np.ndarray]]:
    base_ee_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    object_quat = quat_mul(quat_from_z_yaw(args.object_yaw), base_ee_quat)
    target_quat = quat_mul(quat_from_z_yaw(args.target_yaw), base_ee_quat)
    pre_grasp_z = args.object_z + args.approach_z
    lift_object_z = args.object_z + args.lift_z
    transport_z = args.target_z + args.lift_z
    retract_z = args.target_z + args.approach_z
    return [
        ("pre_grasp", np.array([args.object_x, args.object_y, pre_grasp_z], dtype=np.float32), object_quat),
        ("grasp", np.array([args.object_x, args.object_y, args.object_z], dtype=np.float32), object_quat),
        ("lift", np.array([args.object_x, args.object_y, lift_object_z], dtype=np.float32), object_quat),
        ("transport", np.array([args.target_x, args.target_y, transport_z], dtype=np.float32), target_quat),
        ("place", np.array([args.target_x, args.target_y, args.target_z], dtype=np.float32), target_quat),
        ("retract", np.array([args.target_x, args.target_y, retract_z], dtype=np.float32), base_ee_quat),
    ]


def main() -> int:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from moveit_msgs.action import MoveGroup

    parser = argparse.ArgumentParser(description="Manual-coordinate real OMY-F3M pick/place MoveIt runner.")
    parser.add_argument("--action-name", default=DEFAULT_ACTION_NAME)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--base-frame", default=DEFAULT_BASE_FRAME)
    parser.add_argument("--ee-frame", default=DEFAULT_EE_FRAME)
    parser.add_argument("--object-x", type=float, required=True)
    parser.add_argument("--object-y", type=float, required=True)
    parser.add_argument("--object-z", type=float, required=True)
    parser.add_argument("--object-yaw", type=float, default=0.0)
    parser.add_argument("--target-x", type=float, required=True)
    parser.add_argument("--target-y", type=float, required=True)
    parser.add_argument("--target-z", type=float, required=True)
    parser.add_argument("--target-yaw", type=float, default=0.0)
    parser.add_argument("--approach-z", type=float, default=0.10)
    parser.add_argument("--lift-z", type=float, default=0.12)
    parser.add_argument("--position-tolerance", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance", type=float, default=0.25)
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--velocity-scale", type=float, default=0.03)
    parser.add_argument("--acceleration-scale", type=float, default=0.03)
    parser.add_argument("--max-start-to-goal-joint-delta", type=float, default=0.45)
    parser.add_argument("--max-segment-joint-delta", type=float, default=0.12)
    parser.add_argument("--use-current-ee-orientation", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Optional stage names to plan/execute.")
    parser.add_argument("--plan-only", action="store_true", help="Default behavior; accepted for explicit CLI use.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"Refusing to execute. Re-run with --confirm {CONFIRM_TEXT}")

    workspace = default_workspace()
    stages = _build_stages(args)
    if args.only:
        wanted = set(args.only)
        stages = [stage for stage in stages if stage[0] in wanted]
    if not stages:
        raise ValueError("No stages selected")
    for name, pos, _ in stages:
        assert_in_workspace(name, pos, workspace)

    rclpy.init(args=None)
    node = Node("motion2_manual_pick_place_real")
    client = ActionClient(node, MoveGroup, args.action_name)
    try:
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(
                f"MoveGroup action server not available: {args.action_name}. {_check_hint()}")

        current_pos, current_quat = lookup_current_pose(
            node, rclpy, args.base_frame, args.ee_frame)
        if args.use_current_ee_orientation:
            stages = [(name, pos, current_quat.copy()) for name, pos, _ in stages]

        print(f"[manual-real] execute={args.execute} plan_only={not args.execute}")
        print(f"[manual-real] current_pos={np.round(current_pos, 6).tolist()}")
        print(f"[manual-real] current_quat_wxyz={np.round(current_quat, 6).tolist()}")
        print(
            "[manual-real] joint_delta_guard "
            f"max_start_to_goal={args.max_start_to_goal_joint_delta:.6f} "
            f"max_segment={args.max_segment_joint_delta:.6f}")

        ok_count = 0
        for idx, (name, pos, quat) in enumerate(stages, start=1):
            print(
                f"[manual-real] stage {idx:02d}/{len(stages)} {name}: "
                f"pos={np.round(pos, 6).tolist()} quat_wxyz={np.round(quat, 6).tolist()}")
            ok, report = plan_or_execute_pose(
                node=node,
                rclpy=rclpy,
                client=client,
                frame_id=args.base_frame,
                group_name=args.group_name,
                link_name=args.ee_frame,
                name=name,
                pos=pos,
                quat_wxyz=quat,
                workspace=workspace,
                planning_time=args.planning_time,
                attempts=args.attempts,
                pos_tol=args.position_tolerance,
                ori_tol=args.orientation_tolerance,
                velocity_scale=args.velocity_scale,
                acceleration_scale=args.acceleration_scale,
                plan_only=True,
                keep_orientation=True,
                planner_id="",
                max_start_to_goal_delta=args.max_start_to_goal_joint_delta,
                max_segment_delta=args.max_segment_joint_delta,
            )
            print(
                f"[manual-real] {name}: success={ok} error_code={report.get('error_code')} "
                f"points={report.get('point_count')} "
                f"max_start_to_goal_delta={float(report.get('max_start_to_goal_delta', 0.0)):.6f} "
                f"max_segment_delta={float(report.get('max_segment_delta', 0.0)):.6f} "
                f"max_joint={report.get('max_joint', '')} "
                f"guard_ok={report.get('guard_ok', False)}")
            if not ok:
                print(f"[manual-real] stopping at stage {name}; command_sent=false")
                return 2

            if args.execute:
                print(f"[manual-real] executing confirmed stage after guarded plan-only success: {name}")
                exec_ok, exec_report = plan_or_execute_pose(
                    node=node,
                    rclpy=rclpy,
                    client=client,
                    frame_id=args.base_frame,
                    group_name=args.group_name,
                    link_name=args.ee_frame,
                    name=f"{name}_execute",
                    pos=pos,
                    quat_wxyz=quat,
                    workspace=workspace,
                    planning_time=args.planning_time,
                    attempts=args.attempts,
                    pos_tol=args.position_tolerance,
                    ori_tol=args.orientation_tolerance,
                    velocity_scale=args.velocity_scale,
                    acceleration_scale=args.acceleration_scale,
                    plan_only=False,
                    keep_orientation=True,
                    planner_id="",
                    max_start_to_goal_delta=args.max_start_to_goal_joint_delta,
                    max_segment_delta=args.max_segment_joint_delta,
                )
                print(
                    f"[manual-real] {name} execute_result: success={exec_ok} "
                    f"error_code={exec_report.get('error_code')} "
                    f"points={exec_report.get('point_count')} command_sent=true")
                if not exec_ok:
                    print(f"[manual-real] stopping after execute failure: {name}")
                    return 2
            ok_count += 1

        print(
            f"[manual-real] summary: {ok_count}/{len(stages)} successful "
            f"command_sent={args.execute}")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
