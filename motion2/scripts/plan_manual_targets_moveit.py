"""Plan manual-target waypoints with MoveIt without executing trajectories.

This script sends MoveGroup action goals with ``plan_only=True``. It is intended
for checking whether the manual waypoints are reachable/collision-free before
any real robot execution code is written.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

import numpy as np
import yaml


def _add_repo_root_to_path() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _quat_from_z_yaw(yaw: float) -> np.ndarray:
    half = yaw / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def _load_config(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _section(data: dict, name: str) -> dict:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing mapping section: {name}")
    return value


def _xy_yaw(data: dict, name: str) -> tuple[float, float, float]:
    section = _section(data, name)
    return float(section["x"]), float(section["y"]), float(section["yaw"])


def _override_if_set(section: dict, key: str, value: float | None) -> None:
    if value is not None:
        section[key] = float(value)


def _apply_overrides(data: dict, args) -> None:
    box = _section(data, "box")
    cell = _section(data, "cell")
    heights = _section(data, "heights")

    _override_if_set(box, "x", args.box_x)
    _override_if_set(box, "y", args.box_y)
    _override_if_set(box, "yaw", args.box_yaw)
    _override_if_set(cell, "x", args.cell_x)
    _override_if_set(cell, "y", args.cell_y)
    _override_if_set(cell, "yaw", args.cell_yaw)

    for key in (
        "pre_grasp_z",
        "grasp_z",
        "lift_z",
        "transport_z",
        "place_z",
        "retract_z",
    ):
        _override_if_set(heights, key, getattr(args, key))


def _ros_quat_from_wxyz(q_wxyz):
    from geometry_msgs.msg import Quaternion

    q = Quaternion()
    q.w = float(q_wxyz[0])
    q.x = float(q_wxyz[1])
    q.y = float(q_wxyz[2])
    q.z = float(q_wxyz[3])
    return q


def _make_pose_constraints(
    *,
    frame_id: str,
    link_name: str,
    pos,
    quat_wxyz,
    pos_tol: float,
    ori_tol: float,
):
    from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
    from shape_msgs.msg import SolidPrimitive

    constraints = Constraints()
    constraints.name = f"{link_name}_pose_goal"

    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [float(pos_tol)]

    pc = PositionConstraint()
    pc.header.frame_id = frame_id
    pc.link_name = link_name
    pc.constraint_region.primitives.append(sphere)

    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x = float(pos[0])
    pose.position.y = float(pos[1])
    pose.position.z = float(pos[2])
    pose.orientation.w = 1.0
    pc.constraint_region.primitive_poses.append(pose)
    pc.weight = 1.0

    oc = OrientationConstraint()
    oc.header.frame_id = frame_id
    oc.link_name = link_name
    oc.orientation = _ros_quat_from_wxyz(quat_wxyz)
    oc.absolute_x_axis_tolerance = float(ori_tol)
    oc.absolute_y_axis_tolerance = float(ori_tol)
    oc.absolute_z_axis_tolerance = float(ori_tol)
    oc.weight = 1.0

    constraints.position_constraints.append(pc)
    constraints.orientation_constraints.append(oc)
    return constraints


def _make_move_group_goal(
    *,
    frame_id: str,
    group_name: str,
    link_name: str,
    pos,
    quat_wxyz,
    workspace: dict,
    planning_time: float,
    attempts: int,
    pos_tol: float,
    ori_tol: float,
    velocity_scale: float,
    acceleration_scale: float,
):
    from moveit_msgs.action import MoveGroup

    goal = MoveGroup.Goal()
    req = goal.request
    req.group_name = group_name
    req.num_planning_attempts = int(attempts)
    req.allowed_planning_time = float(planning_time)
    req.max_velocity_scaling_factor = float(velocity_scale)
    req.max_acceleration_scaling_factor = float(acceleration_scale)

    req.workspace_parameters.header.frame_id = frame_id
    req.workspace_parameters.min_corner.x = float(workspace["x_min"])
    req.workspace_parameters.min_corner.y = float(workspace["y_min"])
    req.workspace_parameters.min_corner.z = float(workspace["z_min"])
    req.workspace_parameters.max_corner.x = float(workspace["x_max"])
    req.workspace_parameters.max_corner.y = float(workspace["y_max"])
    req.workspace_parameters.max_corner.z = float(workspace["z_max"])

    # Empty diff means "use current monitored robot state".
    req.start_state.is_diff = True
    req.goal_constraints.append(_make_pose_constraints(
        frame_id=frame_id,
        link_name=link_name,
        pos=pos,
        quat_wxyz=quat_wxyz,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
    ))

    goal.planning_options.plan_only = True
    goal.planning_options.look_around = False
    goal.planning_options.replan = False
    return goal


def _stage_targets(data: dict) -> list[tuple[str, np.ndarray, np.ndarray]]:
    box_x, box_y, box_yaw = _xy_yaw(data, "box")
    cell_x, cell_y, cell_yaw = _xy_yaw(data, "cell")
    heights = _section(data, "heights")
    base_ee_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    box_quat = _quat_mul(_quat_from_z_yaw(box_yaw), base_ee_quat)
    cell_quat = _quat_mul(_quat_from_z_yaw(cell_yaw), base_ee_quat)

    return [
        ("pre_grasp", np.array([box_x, box_y, float(heights["pre_grasp_z"])], dtype=np.float32), box_quat),
        ("grasp", np.array([box_x, box_y, float(heights["grasp_z"])], dtype=np.float32), box_quat),
        ("lift", np.array([box_x, box_y, float(heights["lift_z"])], dtype=np.float32), box_quat),
        ("transport", np.array([cell_x, cell_y, float(heights["transport_z"])], dtype=np.float32), cell_quat),
        ("insert", np.array([cell_x, cell_y, float(heights["place_z"])], dtype=np.float32), cell_quat),
        ("retract", np.array([cell_x, cell_y, float(heights["retract_z"])], dtype=np.float32), base_ee_quat),
    ]


def _lookup_current_quat_wxyz(node, rclpy, frame_id: str, link_name: str) -> np.ndarray:
    from tf2_ros import Buffer, TransformListener

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    deadline = time.monotonic() + 5.0
    last_error = None
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            tf = tf_buffer.lookup_transform(frame_id, link_name, rclpy.time.Time())
            q = tf.transform.rotation
            return np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
        except Exception as exc:  # tf2_ros exception classes vary by distro.
            last_error = exc
    raise RuntimeError(
        f"Timed out waiting for TF {frame_id} -> {link_name}. "
        f"Last error: {last_error}") from last_error


def _lookup_current_pose(node, rclpy, frame_id: str, link_name: str) -> tuple[np.ndarray, np.ndarray]:
    from tf2_ros import Buffer, TransformListener

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    deadline = time.monotonic() + 5.0
    last_error = None
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            tf = tf_buffer.lookup_transform(frame_id, link_name, rclpy.time.Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            pos = np.array([t.x, t.y, t.z], dtype=np.float32)
            quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
            return pos, quat
        except Exception as exc:  # tf2_ros exception classes vary by distro.
            last_error = exc
    raise RuntimeError(
        f"Timed out waiting for TF {frame_id} -> {link_name}. "
        f"Last error: {last_error}") from last_error


def main() -> int:
    parser = argparse.ArgumentParser(description="MoveIt plan-only check for manual motion2 targets.")
    parser.add_argument("--config", default="motion2/config/manual_targets.yaml")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--action-name", default="/move_action")
    parser.add_argument("--group-name", default="arm")
    parser.add_argument("--link-name", default="end_effector_link")
    parser.add_argument("--position-tolerance", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance", type=float, default=0.20)
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--velocity-scale", type=float, default=0.05)
    parser.add_argument("--acceleration-scale", type=float, default=0.05)
    parser.add_argument("--only", nargs="*", default=None, help="Optional stage names to plan.")
    parser.add_argument("--box-x", type=float, default=None)
    parser.add_argument("--box-y", type=float, default=None)
    parser.add_argument("--box-yaw", type=float, default=None)
    parser.add_argument("--cell-x", type=float, default=None)
    parser.add_argument("--cell-y", type=float, default=None)
    parser.add_argument("--cell-yaw", type=float, default=None)
    parser.add_argument("--pre-grasp-z", dest="pre_grasp_z", type=float, default=None)
    parser.add_argument("--grasp-z", type=float, default=None)
    parser.add_argument("--lift-z", type=float, default=None)
    parser.add_argument("--transport-z", type=float, default=None)
    parser.add_argument("--place-z", type=float, default=None)
    parser.add_argument("--retract-z", type=float, default=None)
    parser.add_argument(
        "--use-current-ee-orientation",
        action="store_true",
        help="Use current TF orientation for all target pose constraints.")
    parser.add_argument(
        "--relative-z",
        type=float,
        default=None,
        help="Plan one target at current EE pose plus this z offset in meters.")
    parser.add_argument(
        "--relative-x",
        type=float,
        default=0.0,
        help="X offset used with --relative-z/relative planning.")
    parser.add_argument(
        "--relative-y",
        type=float,
        default=0.0,
        help="Y offset used with --relative-z/relative planning.")
    args = parser.parse_args()

    _add_repo_root_to_path()

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from moveit_msgs.action import MoveGroup

    data = _load_config(pathlib.Path(args.config))
    if data.get("frame_id") != "link0":
        raise ValueError(f"expected frame_id=link0, got {data.get('frame_id')!r}")
    if not data.get("verified", False) and not args.allow_unverified:
        raise RuntimeError(
            f"{args.config} has verified=false. Use --allow-unverified for plan-only checks.")
    _apply_overrides(data, args)

    stages = _stage_targets(data)
    if args.only:
        wanted = set(args.only)
        stages = [stage for stage in stages if stage[0] in wanted]
    workspace = _section(_section(data, "safety"), "workspace")

    rclpy.init(args=None)
    node = Node("motion2_manual_targets_plan_only")
    client = ActionClient(node, MoveGroup, args.action_name)
    try:
        if args.relative_z is not None:
            current_pos, current_quat = _lookup_current_pose(
                node, rclpy, data["frame_id"], args.link_name)
            target_pos = current_pos + np.array(
                [args.relative_x, args.relative_y, args.relative_z], dtype=np.float32)
            stages = [("relative_current_pose", target_pos, current_quat)]
            print(
                "[plan-only] relative target from current EE pose: "
                f"current_pos={current_pos.tolist()} "
                f"target_pos={target_pos.tolist()} "
                f"quat_wxyz={current_quat.tolist()}")

        if args.use_current_ee_orientation:
            current_quat = _lookup_current_quat_wxyz(
                node, rclpy, data["frame_id"], args.link_name)
            stages = [(name, pos, current_quat.copy()) for name, pos, _ in stages]
            print(
                "[plan-only] using current EE orientation for all targets: "
                f"quat_wxyz={current_quat.tolist()}")

        node.get_logger().info(f"Waiting for MoveGroup action {args.action_name}")
        if not client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"MoveGroup action server not available: {args.action_name}")

        ok_count = 0
        for name, pos, quat in stages:
            print(f"[plan-only] {name}: pos={pos.tolist()} quat_wxyz={quat.tolist()}")
            goal = _make_move_group_goal(
                frame_id=data["frame_id"],
                group_name=args.group_name,
                link_name=args.link_name,
                pos=pos,
                quat_wxyz=quat,
                workspace=workspace,
                planning_time=args.planning_time,
                attempts=args.attempts,
                pos_tol=args.position_tolerance,
                ori_tol=args.orientation_tolerance,
                velocity_scale=args.velocity_scale,
                acceleration_scale=args.acceleration_scale,
            )
            send_future = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(node, send_future)
            handle = send_future.result()
            if handle is None or not handle.accepted:
                print(f"[plan-only] {name}: REJECTED")
                continue

            result_future = handle.get_result_async()
            rclpy.spin_until_future_complete(node, result_future)
            result = result_future.result().result
            code = int(result.error_code.val)
            points = len(result.planned_trajectory.joint_trajectory.points)
            if code == 1:
                ok_count += 1
                print(f"[plan-only] {name}: SUCCESS points={points} execute=false")
            else:
                print(f"[plan-only] {name}: FAILED error_code={code} points={points} execute=false")

        print(f"[plan-only] summary: {ok_count}/{len(stages)} successful, execute=false")
        return 0 if ok_count == len(stages) else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
