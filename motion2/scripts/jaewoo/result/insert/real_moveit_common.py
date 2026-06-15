"""Small ROS2/MoveIt helpers for OMY-F3M real-robot MVP scripts."""
from __future__ import annotations

import math
import time

import numpy as np


SUCCESS = 1


def quat_from_z_yaw(yaw: float) -> np.ndarray:
    half = yaw / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


def ros_quat_from_wxyz(q_wxyz):
    from geometry_msgs.msg import Quaternion

    q = Quaternion()
    q.w = float(q_wxyz[0])
    q.x = float(q_wxyz[1])
    q.y = float(q_wxyz[2])
    q.z = float(q_wxyz[3])
    return q


def make_pose_constraints(
    *,
    frame_id: str,
    link_name: str,
    pos,
    quat_wxyz,
    pos_tol: float,
    ori_tol: float,
    keep_orientation: bool,
):
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
    from shape_msgs.msg import SolidPrimitive

    constraints = Constraints()
    constraints.name = f"{link_name}_pose_goal"

    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [float(pos_tol)]

    pose = Pose()
    pose.position.x = float(pos[0])
    pose.position.y = float(pos[1])
    pose.position.z = float(pos[2])
    pose.orientation.w = 1.0

    pc = PositionConstraint()
    pc.header.frame_id = frame_id
    pc.link_name = link_name
    pc.constraint_region.primitives.append(sphere)
    pc.constraint_region.primitive_poses.append(pose)
    pc.weight = 1.0

    constraints.position_constraints.append(pc)
    if keep_orientation:
        oc = OrientationConstraint()
        oc.header.frame_id = frame_id
        oc.link_name = link_name
        oc.orientation = ros_quat_from_wxyz(quat_wxyz)
        oc.absolute_x_axis_tolerance = float(ori_tol)
        oc.absolute_y_axis_tolerance = float(ori_tol)
        oc.absolute_z_axis_tolerance = float(ori_tol)
        oc.weight = 1.0
        constraints.orientation_constraints.append(oc)
    return constraints


def make_move_group_goal(
    *,
    frame_id: str,
    group_name: str,
    link_name: str,
    pos,
    quat_wxyz,
    workspace: dict[str, float],
    planning_time: float,
    attempts: int,
    pos_tol: float,
    ori_tol: float,
    velocity_scale: float,
    acceleration_scale: float,
    plan_only: bool,
    keep_orientation: bool,
    planner_id: str,
):
    from moveit_msgs.action import MoveGroup

    goal = MoveGroup.Goal()
    req = goal.request
    req.group_name = group_name
    req.num_planning_attempts = int(attempts)
    req.allowed_planning_time = float(planning_time)
    req.max_velocity_scaling_factor = float(velocity_scale)
    req.max_acceleration_scaling_factor = float(acceleration_scale)
    req.planner_id = str(planner_id)

    req.workspace_parameters.header.frame_id = frame_id
    req.workspace_parameters.min_corner.x = float(workspace["x_min"])
    req.workspace_parameters.min_corner.y = float(workspace["y_min"])
    req.workspace_parameters.min_corner.z = float(workspace["z_min"])
    req.workspace_parameters.max_corner.x = float(workspace["x_max"])
    req.workspace_parameters.max_corner.y = float(workspace["y_max"])
    req.workspace_parameters.max_corner.z = float(workspace["z_max"])
    req.start_state.is_diff = True
    req.goal_constraints.append(make_pose_constraints(
        frame_id=frame_id,
        link_name=link_name,
        pos=pos,
        quat_wxyz=quat_wxyz,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
        keep_orientation=keep_orientation,
    ))

    goal.planning_options.plan_only = bool(plan_only)
    goal.planning_options.look_around = False
    goal.planning_options.replan = False
    return goal


def lookup_current_pose(node, rclpy, frame_id: str, link_name: str, timeout_s: float = 5.0):
    from tf2_ros import Buffer, TransformListener

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            tf = tf_buffer.lookup_transform(frame_id, link_name, rclpy.time.Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            return (
                np.array([t.x, t.y, t.z], dtype=np.float32),
                np.array([q.w, q.x, q.y, q.z], dtype=np.float32),
            )
        except Exception as exc:  # tf2_ros exception classes vary by distro.
            last_error = exc
    raise RuntimeError(
        f"Timed out waiting for TF {frame_id} -> {link_name}. "
        f"Last error: {last_error}") from last_error


def trajectory_delta_report(trajectory) -> dict[str, object]:
    joint_traj = trajectory.joint_trajectory
    names = list(joint_traj.joint_names)
    points = list(joint_traj.points)
    if len(points) == 0:
        return {
            "joint_names": names,
            "point_count": 0,
            "max_start_to_goal_delta": 0.0,
            "max_segment_delta": 0.0,
            "max_joint": "",
            "start_positions": [],
            "goal_positions": [],
            "start_to_goal_deltas": [],
        }

    first = np.asarray(points[0].positions, dtype=np.float64)
    last = np.asarray(points[-1].positions, dtype=np.float64)
    start_to_goal = np.abs(last - first)
    max_start_idx = int(np.argmax(start_to_goal)) if len(start_to_goal) else 0
    max_start_delta = float(start_to_goal[max_start_idx]) if len(start_to_goal) else 0.0

    max_segment_delta = 0.0
    if len(points) > 1:
        previous = first
        for point in points[1:]:
            current = np.asarray(point.positions, dtype=np.float64)
            if len(current) == len(previous):
                max_segment_delta = max(max_segment_delta, float(np.max(np.abs(current - previous))))
            previous = current

    return {
        "joint_names": names,
        "point_count": len(points),
        "max_start_to_goal_delta": max_start_delta,
        "max_segment_delta": max_segment_delta,
        "max_joint": names[max_start_idx] if max_start_idx < len(names) else "",
        "start_positions": first.tolist(),
        "goal_positions": last.tolist(),
        "start_to_goal_deltas": (last - first).tolist(),
    }


def plan_or_execute_pose(
    *,
    node,
    rclpy,
    client,
    frame_id: str,
    group_name: str,
    link_name: str,
    name: str,
    pos,
    quat_wxyz,
    workspace: dict[str, float],
    planning_time: float,
    attempts: int,
    pos_tol: float,
    ori_tol: float,
    velocity_scale: float,
    acceleration_scale: float,
    plan_only: bool,
    keep_orientation: bool,
    planner_id: str,
    max_start_to_goal_delta: float,
    max_segment_delta: float,
) -> tuple[bool, dict[str, object]]:
    goal = make_move_group_goal(
        frame_id=frame_id,
        group_name=group_name,
        link_name=link_name,
        pos=pos,
        quat_wxyz=quat_wxyz,
        workspace=workspace,
        planning_time=planning_time,
        attempts=attempts,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
        velocity_scale=velocity_scale,
        acceleration_scale=acceleration_scale,
        plan_only=plan_only,
        keep_orientation=keep_orientation,
        planner_id=planner_id,
    )
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    handle = send_future.result()
    if handle is None or not handle.accepted:
        return False, {"name": name, "error": "rejected"}

    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    code = int(result.error_code.val)
    report = trajectory_delta_report(result.planned_trajectory)
    report["name"] = name
    report["error_code"] = code
    report["guard_ok"] = (
        float(report["max_start_to_goal_delta"]) <= max_start_to_goal_delta
        and float(report["max_segment_delta"]) <= max_segment_delta
    )
    ok = code == SUCCESS and bool(report["guard_ok"])
    return ok, report


def joint_state_once(node, rclpy, timeout_s: float = 3.0) -> dict[str, float]:
    from sensor_msgs.msg import JointState

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


def current_to_planned_start_deltas(
    current_joints: dict[str, float],
    joint_names: list[str],
    planned_start: list[float],
) -> list[tuple[str, float]]:
    deltas = []
    for name, planned_value in zip(joint_names, planned_start):
        if name in current_joints:
            deltas.append((name, float(planned_value) - float(current_joints[name])))
    return deltas


def default_workspace() -> dict[str, float]:
    return {
        "x_min": -0.10,
        "x_max": 0.45,
        "y_min": -0.45,
        "y_max": 0.20,
        "z_min": 0.10,
        "z_max": 0.50,
    }


def assert_in_workspace(name: str, pos, workspace: dict[str, float]) -> None:
    x, y, z = [float(v) for v in pos]
    checks = (
        ("x", x, workspace["x_min"], workspace["x_max"]),
        ("y", y, workspace["y_min"], workspace["y_max"]),
        ("z", z, workspace["z_min"], workspace["z_max"]),
    )
    for axis, value, lower, upper in checks:
        if not (lower <= value <= upper):
            raise ValueError(
                f"{name}: {axis}={value:.6f} outside [{lower:.6f}, {upper:.6f}]")
