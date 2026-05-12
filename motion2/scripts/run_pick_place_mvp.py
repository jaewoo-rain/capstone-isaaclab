"""Guarded MVP pick/place runner for OMY-F3M.

Default mode is dry-run: compute and print stage targets only.

Execution mode is intentionally explicit:
  --execute --confirm EXECUTE_PICK_PLACE_MVP

For arm motion this script asks MoveIt for plan-only trajectories, validates the
joint deltas, then sends the validated trajectory to /execute_trajectory. It
does not use MoveGroup plan-and-execute because that would skip our local
trajectory guard.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import yaml


CONFIRM_TEXT = "EXECUTE_PICK_PLACE_MVP"
DEMO_SAFE_STAGES = ["pre_grasp", "lift", "transport", "insert", "retract"]


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _add_repo_root_to_path() -> None:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_yaml(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _resolve_repo_path(path_text: str) -> pathlib.Path:
    path = pathlib.Path(path_text)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _section(data: dict, name: str) -> dict:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing mapping section: {name}")
    return value


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


def _override(section: dict, key: str, value: float | None) -> None:
    if value is not None:
        section[key] = float(value)


@dataclass(frozen=True)
class Stage:
    name: str
    kind: str
    pos: np.ndarray | None = None
    quat: np.ndarray | None = None
    gripper: float | None = None


def _build_stages(data: dict, args, current_ee_quat: np.ndarray | None = None) -> list[Stage]:
    box = _section(data, "box").copy()
    cell = _section(data, "cell").copy()
    heights = _section(data, "heights").copy()

    _override(box, "x", args.object_x)
    _override(box, "y", args.object_y)
    _override(box, "yaw", args.object_yaw)
    _override(cell, "x", args.slot_x)
    _override(cell, "y", args.slot_y)
    _override(cell, "yaw", args.slot_yaw)

    # These are real-gripper absolute command positions, not sim normalized
    # values. Keep them conservative by default.
    open_value = float(args.gripper_open)
    close_value = float(args.gripper_close)

    base_ee_quat = (
        np.asarray(current_ee_quat, dtype=np.float32)
        if current_ee_quat is not None
        else np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    )
    box_quat = _quat_mul(_quat_from_z_yaw(float(box["yaw"])), base_ee_quat)
    cell_quat = _quat_mul(_quat_from_z_yaw(float(cell["yaw"])), base_ee_quat)
    box_xy = (float(box["x"]), float(box["y"]))
    cell_xy = (float(cell["x"]), float(cell["y"]))

    return [
        Stage("pre_grasp", "arm", np.array([box_xy[0], box_xy[1], float(heights["pre_grasp_z"])], dtype=np.float32), box_quat),
        Stage("grasp", "arm", np.array([box_xy[0], box_xy[1], float(heights["grasp_z"])], dtype=np.float32), box_quat),
        Stage("close_gripper", "gripper", gripper=close_value),
        Stage("lift", "arm", np.array([box_xy[0], box_xy[1], float(heights["lift_z"])], dtype=np.float32), box_quat),
        Stage("transport", "arm", np.array([cell_xy[0], cell_xy[1], float(heights["transport_z"])], dtype=np.float32), cell_quat),
        Stage("insert", "arm", np.array([cell_xy[0], cell_xy[1], float(heights["place_z"])], dtype=np.float32), cell_quat),
        Stage("open_gripper", "gripper", gripper=open_value),
        Stage("retract", "arm", np.array([cell_xy[0], cell_xy[1], float(heights["retract_z"])], dtype=np.float32), base_ee_quat),
    ]


def _select_stages(stages: list[Stage], only: list[str] | None) -> list[Stage]:
    if not only:
        return stages
    aliases = {"place": "insert"}
    wanted = {aliases.get(name, name) for name in only}
    return [stage for stage in stages if stage.name in wanted]


def _ros_quat_from_wxyz(q_wxyz):
    from geometry_msgs.msg import Quaternion

    q = Quaternion()
    q.w = float(q_wxyz[0])
    q.x = float(q_wxyz[1])
    q.y = float(q_wxyz[2])
    q.z = float(q_wxyz[3])
    return q


def _make_pose_constraints(frame_id: str, link_name: str, pos, quat_wxyz, pos_tol: float, ori_tol: float):
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
    from shape_msgs.msg import SolidPrimitive

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

    oc = OrientationConstraint()
    oc.header.frame_id = frame_id
    oc.link_name = link_name
    oc.orientation = _ros_quat_from_wxyz(quat_wxyz)
    oc.absolute_x_axis_tolerance = float(ori_tol)
    oc.absolute_y_axis_tolerance = float(ori_tol)
    oc.absolute_z_axis_tolerance = float(ori_tol)
    oc.weight = 1.0

    constraints = Constraints()
    constraints.name = f"{link_name}_{frame_id}_goal"
    constraints.position_constraints.append(pc)
    constraints.orientation_constraints.append(oc)
    return constraints


def _make_move_group_goal(args, workspace: dict, stage: Stage):
    from moveit_msgs.action import MoveGroup

    goal = MoveGroup.Goal()
    req = goal.request
    req.group_name = args.group_name
    req.num_planning_attempts = int(args.attempts)
    req.allowed_planning_time = float(args.planning_time)
    req.max_velocity_scaling_factor = float(args.velocity_scale)
    req.max_acceleration_scaling_factor = float(args.acceleration_scale)
    req.workspace_parameters.header.frame_id = args.frame_id
    req.workspace_parameters.min_corner.x = float(workspace["x_min"])
    req.workspace_parameters.min_corner.y = float(workspace["y_min"])
    req.workspace_parameters.min_corner.z = float(workspace["z_min"])
    req.workspace_parameters.max_corner.x = float(workspace["x_max"])
    req.workspace_parameters.max_corner.y = float(workspace["y_max"])
    req.workspace_parameters.max_corner.z = float(workspace["z_max"])
    req.start_state.is_diff = True
    req.goal_constraints.append(_make_pose_constraints(
        args.frame_id,
        args.link_name,
        stage.pos,
        stage.quat,
        args.position_tolerance,
        args.orientation_tolerance,
    ))
    goal.planning_options.plan_only = True
    goal.planning_options.look_around = False
    goal.planning_options.replan = False
    return goal


def _latest_joint_state(node, rclpy, timeout_sec: float):
    from sensor_msgs.msg import JointState

    holder = {"msg": None}
    sub = node.create_subscription(JointState, "/joint_states", lambda msg: holder.update(msg=msg), 10)
    deadline = node.get_clock().now().nanoseconds / 1e9 + float(timeout_sec)
    while node.get_clock().now().nanoseconds / 1e9 < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if holder["msg"] is not None:
            node.destroy_subscription(sub)
            return holder["msg"]
    node.destroy_subscription(sub)
    raise RuntimeError("Timed out waiting for /joint_states")


def _lookup_current_quat_wxyz(node, rclpy, frame_id: str, link_name: str, timeout_sec: float) -> np.ndarray:
    from tf2_ros import Buffer, TransformListener

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    deadline = node.get_clock().now().nanoseconds / 1e9 + float(timeout_sec)
    last_error = None
    while node.get_clock().now().nanoseconds / 1e9 < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            tf = tf_buffer.lookup_transform(frame_id, link_name, rclpy.time.Time())
            q = tf.transform.rotation
            return np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Timed out waiting for TF {frame_id} -> {link_name}. "
        f"Last error: {last_error}") from last_error


def _lookup_current_pose(node, rclpy, frame_id: str, link_name: str, timeout_sec: float) -> tuple[np.ndarray, np.ndarray]:
    from tf2_ros import Buffer, TransformListener

    tf_buffer = Buffer()
    TransformListener(tf_buffer, node)
    deadline = node.get_clock().now().nanoseconds / 1e9 + float(timeout_sec)
    last_error = None
    while node.get_clock().now().nanoseconds / 1e9 < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        try:
            tf = tf_buffer.lookup_transform(frame_id, link_name, rclpy.time.Time())
            t = tf.transform.translation
            q = tf.transform.rotation
            pos = np.array([t.x, t.y, t.z], dtype=np.float32)
            quat = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
            return pos, quat
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        f"Timed out waiting for TF {frame_id} -> {link_name}. "
        f"Last error: {last_error}") from last_error


def _validate_trajectory(traj, current_joint_state, args) -> tuple[bool, str]:
    jt = traj.joint_trajectory
    points = list(jt.points)
    names = list(jt.joint_names)
    if not names or not points:
        return False, "empty joint trajectory"

    current = dict(zip(current_joint_state.name, current_joint_state.position))
    missing = [name for name in names if name not in current]
    if missing:
        return False, f"trajectory joints missing from /joint_states: {missing}"

    first = np.array(points[0].positions, dtype=float)
    current_vec = np.array([current[name] for name in names], dtype=float)
    start_delta = np.abs(first - current_vec)
    if float(start_delta.max()) > args.max_start_delta:
        return False, (
            f"start delta too large: max={start_delta.max():.4f} "
            f"limit={args.max_start_delta:.4f}")

    prev = first
    total_min = first.copy()
    total_max = first.copy()
    for point in points[1:]:
        pos = np.array(point.positions, dtype=float)
        step_delta = np.abs(pos - prev)
        if float(step_delta.max()) > args.max_point_delta:
            return False, (
                f"point-to-point joint delta too large: max={step_delta.max():.4f} "
                f"limit={args.max_point_delta:.4f}")
        total_min = np.minimum(total_min, pos)
        total_max = np.maximum(total_max, pos)
        prev = pos

    total_delta = total_max - total_min
    if float(total_delta.max()) > args.max_total_delta:
        return False, (
            f"total joint span too large: max={total_delta.max():.4f} "
            f"limit={args.max_total_delta:.4f}")
    return True, (
        f"valid points={len(points)} max_start_delta={start_delta.max():.4f} "
        f"max_total_delta={total_delta.max():.4f}")


def _send_gripper(node, rclpy, client, target: float, args) -> None:
    from control_msgs.action import GripperCommand

    if not (args.gripper_min <= target <= args.gripper_max):
        raise RuntimeError(
            f"Refusing gripper target {target:.4f}; outside "
            f"[{args.gripper_min:.4f}, {args.gripper_max:.4f}]")
    goal = GripperCommand.Goal()
    goal.command.position = float(target)
    goal.command.max_effort = float(args.gripper_max_effort)
    future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, future)
    handle = future.result()
    if handle is None or not handle.accepted:
        raise RuntimeError("gripper goal rejected")
    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    print(
        f"[mvp] gripper result position={result.position:.6f} "
        f"effort={result.effort:.6f} stalled={result.stalled} "
        f"reached_goal={result.reached_goal}")


def _make_runtime_stages(node, rclpy, data: dict, args) -> list[Stage]:
    if args.relative_z is not None:
        current_pos, current_quat = _lookup_current_pose(
            node, rclpy, args.frame_id, args.link_name, args.joint_state_timeout)
        target_pos = current_pos + np.array(
            [args.relative_x, args.relative_y, args.relative_z], dtype=np.float32)
        print(
            "[mvp] relative target from current EE pose: "
            f"current_pos={current_pos.tolist()} "
            f"target_pos={target_pos.tolist()} "
            f"quat_wxyz={current_quat.tolist()}")
        return [Stage("relative_current_pose", "arm", target_pos, current_quat)]

    current_quat = None
    if args.use_current_ee_orientation:
        current_quat = _lookup_current_quat_wxyz(
            node, rclpy, args.frame_id, args.link_name, args.joint_state_timeout)
        print(
            "[mvp] using current EE orientation for all targets: "
            f"quat_wxyz={current_quat.tolist()}")
    return _select_stages(_build_stages(data, args, current_quat), args.only)


def _run_plan_guard(data: dict, args) -> int:
    import rclpy
    from moveit_msgs.action import MoveGroup
    from rclpy.action import ActionClient
    from rclpy.node import Node

    workspace = _section(_section(data, "safety"), "workspace")
    rclpy.init(args=None)
    node = Node("motion2_pick_place_mvp_plan_guard")
    move_group = ActionClient(node, MoveGroup, args.move_action)
    try:
        stages = _make_runtime_stages(node, rclpy, data, args)
        print(f"[mvp] config={args.config} execute=false plan_guard=true stages={len(stages)}")
        for idx, stage in enumerate(stages, start=1):
            if stage.kind == "arm":
                print(
                    f"[mvp] step {idx}/{len(stages)} {stage.name}: "
                    f"pos={stage.pos.tolist()} quat_wxyz={stage.quat.tolist()}")
            else:
                print(
                    f"[mvp] step {idx}/{len(stages)} {stage.name}: "
                    f"gripper_target={stage.gripper:.6f}")

        node.get_logger().info(f"Waiting for action {args.move_action}")
        if not move_group.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"action server not available: {args.move_action}")

        ok_count = 0
        for idx, stage in enumerate(stages, start=1):
            print(f"[mvp] plan-guard step {idx}/{len(stages)} {stage.name} kind={stage.kind}")
            if stage.kind == "gripper":
                target = float(stage.gripper)
                if not (args.gripper_min <= target <= args.gripper_max):
                    print(f"[mvp] gripper target INVALID target={target:.6f}")
                    continue
                print(f"[mvp] gripper target valid target={target:.6f} execute=false")
                ok_count += 1
                continue

            joint_state = _latest_joint_state(node, rclpy, args.joint_state_timeout)
            plan_goal = _make_move_group_goal(args, workspace, stage)
            plan_future = move_group.send_goal_async(plan_goal)
            rclpy.spin_until_future_complete(node, plan_future)
            plan_handle = plan_future.result()
            if plan_handle is None or not plan_handle.accepted:
                print(f"[mvp] {stage.name}: MoveGroup rejected execute=false")
                continue
            result_future = plan_handle.get_result_async()
            rclpy.spin_until_future_complete(node, result_future)
            result = result_future.result().result
            code = int(result.error_code.val)
            if code != 1:
                print(f"[mvp] {stage.name}: PLAN_FAILED error_code={code} execute=false")
                continue

            valid, reason = _validate_trajectory(result.planned_trajectory, joint_state, args)
            status = "PASS" if valid else "REJECT"
            print(f"[mvp] {stage.name}: plan_guard={status} {reason} execute=false")
            if valid:
                ok_count += 1

        print(f"[mvp] plan-guard summary: {ok_count}/{len(stages)} passed execute=false")
        return 0 if ok_count == len(stages) else 2
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _run_execute(data: dict, args) -> int:
    import rclpy
    from control_msgs.action import GripperCommand
    from moveit_msgs.action import ExecuteTrajectory, MoveGroup
    from rclpy.action import ActionClient
    from rclpy.node import Node

    if args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"Refusing to execute. Re-run with --confirm {CONFIRM_TEXT}")

    workspace = _section(_section(data, "safety"), "workspace")
    rclpy.init(args=None)
    node = Node("motion2_pick_place_mvp")
    move_group = ActionClient(node, MoveGroup, args.move_action)
    execute_traj = ActionClient(node, ExecuteTrajectory, args.execute_action)
    gripper = ActionClient(node, GripperCommand, args.gripper_action)
    try:
        stages = _make_runtime_stages(node, rclpy, data, args)
        print(f"[mvp] config={args.config} execute=true stages={len(stages)}")
        for idx, stage in enumerate(stages, start=1):
            if stage.kind == "arm":
                print(
                    f"[mvp] step {idx}/{len(stages)} {stage.name}: "
                    f"pos={stage.pos.tolist()} quat_wxyz={stage.quat.tolist()}")
            else:
                print(
                    f"[mvp] step {idx}/{len(stages)} {stage.name}: "
                    f"gripper_target={stage.gripper:.6f}")

        for client, name in (
            (move_group, args.move_action),
            (execute_traj, args.execute_action),
            (gripper, args.gripper_action),
        ):
            node.get_logger().info(f"Waiting for action {name}")
            if not client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"action server not available: {name}")

        ok_count = 0
        for idx, stage in enumerate(stages, start=1):
            print(f"[mvp] execute step {idx}/{len(stages)} {stage.name} kind={stage.kind}")
            if stage.kind == "gripper":
                _send_gripper(node, rclpy, gripper, float(stage.gripper), args)
                ok_count += 1
                continue

            joint_state = _latest_joint_state(node, rclpy, args.joint_state_timeout)
            plan_goal = _make_move_group_goal(args, workspace, stage)
            plan_future = move_group.send_goal_async(plan_goal)
            rclpy.spin_until_future_complete(node, plan_future)
            plan_handle = plan_future.result()
            if plan_handle is None or not plan_handle.accepted:
                raise RuntimeError(f"MoveGroup rejected stage {stage.name}")
            result_future = plan_handle.get_result_async()
            rclpy.spin_until_future_complete(node, result_future)
            result = result_future.result().result
            code = int(result.error_code.val)
            if code != 1:
                raise RuntimeError(f"MoveIt plan failed for {stage.name}: error_code={code}")

            valid, reason = _validate_trajectory(result.planned_trajectory, joint_state, args)
            print(f"[mvp] trajectory guard {stage.name}: {reason}")
            if not valid:
                raise RuntimeError(f"Refusing to execute {stage.name}: {reason}")

            exec_goal = ExecuteTrajectory.Goal()
            exec_goal.trajectory = result.planned_trajectory
            exec_future = execute_traj.send_goal_async(exec_goal)
            rclpy.spin_until_future_complete(node, exec_future)
            exec_handle = exec_future.result()
            if exec_handle is None or not exec_handle.accepted:
                raise RuntimeError(f"ExecuteTrajectory rejected stage {stage.name}")
            exec_result_future = exec_handle.get_result_async()
            rclpy.spin_until_future_complete(node, exec_result_future)
            exec_result = exec_result_future.result().result
            exec_code = int(exec_result.error_code.val)
            if exec_code != 1:
                raise RuntimeError(f"ExecuteTrajectory failed for {stage.name}: error_code={exec_code}")
            print(f"[mvp] {stage.name}: EXECUTE_SUCCESS")
            ok_count += 1

        print(f"[mvp] summary: {ok_count}/{len(stages)} successful execute=true")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _current_orientation_for_dry_run(args) -> np.ndarray | None:
    if not args.use_current_ee_orientation:
        return None

    import rclpy
    from rclpy.node import Node

    rclpy.init(args=None)
    node = Node("motion2_pick_place_mvp_dry_run")
    try:
        current_quat = _lookup_current_quat_wxyz(
            node, rclpy, args.frame_id, args.link_name, args.joint_state_timeout)
        print(
            "[mvp] using current EE orientation for all targets: "
            f"quat_wxyz={current_quat.tolist()}")
        return current_quat
    finally:
        node.destroy_node()
        rclpy.shutdown()


def _run_demo_safe(data: dict, args) -> int:
    if args.execute or args.plan_guard:
        raise RuntimeError("--demo-safe cannot be combined with --execute or --plan-guard")

    stages = _select_stages(_build_stages(data, args), DEMO_SAFE_STAGES)
    print("[mvp-demo] safe presentation mode; command_sent=false")
    print(f"[mvp-demo] config={args.config}")
    print("[mvp-demo] object-to-slot chain shape:")
    for idx, stage in enumerate(stages, start=1):
        print(
            f"[mvp-demo] step {idx}/{len(stages)} {stage.name}: "
            f"pos={stage.pos.tolist()} quat_wxyz={stage.quat.tolist()}")

    print("[mvp-demo] gripper stages available but not sent:")
    print(f"[mvp-demo] close_gripper target={float(args.gripper_close):.6f}")
    print(f"[mvp-demo] open_gripper target={float(args.gripper_open):.6f}")
    print("[mvp-demo] next safe commands:")
    print("  python3 motion2/scripts/run_pick_place_mvp.py --allow-unverified --use-current-ee-orientation")
    print("  python3 motion2/scripts/run_pick_place_mvp.py --allow-unverified --plan-guard --relative-z 0.002 --planning-time 10.0 --attempts 10")
    print("  python3 motion2/scripts/run_smoke_sequence.py --name full_basic_smoke")
    print("[mvp-demo] do not execute full pick-place on the real robot yet")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Guarded dry-run/execute MVP pick-place chain.")
    parser.add_argument("--config", default="motion2/config/manual_targets_safe_dryrun.yaml")
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--demo-safe", action="store_true", help="Print the current safe demo flow without ROS commands.")
    parser.add_argument("--plan-guard", action="store_true", help="Plan and validate trajectories without executing.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--only", nargs="*", default=None, help="Optional stage names to include.")
    parser.add_argument(
        "--use-current-ee-orientation",
        action="store_true",
        help="Use current TF orientation for all arm target pose constraints.")
    parser.add_argument("--object-x", type=float, default=None)
    parser.add_argument("--object-y", type=float, default=None)
    parser.add_argument("--object-yaw", type=float, default=None)
    parser.add_argument("--slot-x", type=float, default=None)
    parser.add_argument("--slot-y", type=float, default=None)
    parser.add_argument("--slot-yaw", type=float, default=None)
    parser.add_argument("--relative-z", type=float, default=None)
    parser.add_argument("--relative-x", type=float, default=0.0)
    parser.add_argument("--relative-y", type=float, default=0.0)
    parser.add_argument("--gripper-open", type=float, default=0.0)
    parser.add_argument("--gripper-close", type=float, default=0.06)
    parser.add_argument("--gripper-min", type=float, default=0.0)
    parser.add_argument("--gripper-max", type=float, default=1.12)
    parser.add_argument("--gripper-max-effort", type=float, default=0.0)
    parser.add_argument("--frame-id", default="link0")
    parser.add_argument("--link-name", default="end_effector_link")
    parser.add_argument("--group-name", default="arm")
    parser.add_argument("--move-action", default="/move_action")
    parser.add_argument("--execute-action", default="/execute_trajectory")
    parser.add_argument("--gripper-action", default="/gripper_controller/gripper_cmd")
    parser.add_argument("--position-tolerance", type=float, default=0.015)
    parser.add_argument("--orientation-tolerance", type=float, default=0.30)
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--velocity-scale", type=float, default=0.03)
    parser.add_argument("--acceleration-scale", type=float, default=0.03)
    parser.add_argument("--max-start-delta", type=float, default=0.08)
    parser.add_argument("--max-point-delta", type=float, default=0.12)
    parser.add_argument("--max-total-delta", type=float, default=1.20)
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    args = parser.parse_args()

    _add_repo_root_to_path()
    cfg_path = _resolve_repo_path(args.config)
    data = _load_yaml(cfg_path)
    if data.get("frame_id") != args.frame_id:
        raise ValueError(f"expected frame_id={args.frame_id}, got {data.get('frame_id')!r}")
    if not data.get("verified", False) and not args.allow_unverified:
        raise RuntimeError(
            f"{cfg_path} has verified=false. Use --allow-unverified for MVP dry-run/guarded execute.")

    if args.demo_safe:
        return _run_demo_safe(data, args)
    if args.plan_guard:
        return _run_plan_guard(data, args)
    if args.execute:
        return _run_execute(data, args)

    current_quat = None
    current_quat = _current_orientation_for_dry_run(args)
    stages = _select_stages(_build_stages(data, args, current_quat), args.only)
    if not stages:
        raise ValueError("no stages selected")

    print(f"[mvp] config={cfg_path} execute={bool(args.execute)} stages={len(stages)}")
    for idx, stage in enumerate(stages, start=1):
        if stage.kind == "arm":
            print(
                f"[mvp] step {idx}/{len(stages)} {stage.name}: "
                f"pos={stage.pos.tolist()} quat_wxyz={stage.quat.tolist()}")
        else:
            print(
                f"[mvp] step {idx}/{len(stages)} {stage.name}: "
                f"gripper_target={stage.gripper:.6f}")

    print("[mvp] dry-run complete; command_sent=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
