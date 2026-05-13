"""Run a taught joint-waypoint pick/place sequence for OMY-F3M.

Default behavior is dry-run only. The script sends robot commands only when
both flags are provided:

    --execute --confirm EXECUTE_TEACH_PICK_PLACE

This runner does not use MoveIt pose goals or Cartesian execution. Arm motion is
sent directly to `/arm_controller/follow_joint_trajectory`; gripper motion is
sent to `/gripper_controller/gripper_cmd`.
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Any

import yaml


CONFIRM_TEXT = "EXECUTE_TEACH_PICK_PLACE"
DEFAULT_CONFIG = "motion2/config/teach_pick_place_waypoints.yaml"
DEFAULT_SEQUENCE = [
    "pre_grasp",
    "grasp",
    "close_gripper",
    "lift",
    "place",
    "open_gripper",
    "retract",
]
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT = "rh_r1_joint"


@dataclass(frozen=True)
class ArmStep:
    name: str
    positions: list[float]
    max_delta_from_previous: float


@dataclass(frozen=True)
class GripperStep:
    name: str
    position: float


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _resolve_repo_path(root: pathlib.Path, path_text: str) -> pathlib.Path:
    path = pathlib.Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _joint_state_once(node, rclpy, timeout_s: float) -> dict[str, float]:
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


def _make_arm_goal(joint_names: list[str], positions: list[float], duration_s: float):
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(joint_names)
    point = JointTrajectoryPoint()
    point.positions = [float(v) for v in positions]
    point.velocities = [0.0 for _ in positions]
    point.time_from_start.sec = int(duration_s)
    point.time_from_start.nanosec = int((duration_s - int(duration_s)) * 1_000_000_000)
    goal.trajectory.points.append(point)
    return goal


def _make_gripper_goal(position: float, max_effort: float):
    from control_msgs.action import GripperCommand

    goal = GripperCommand.Goal()
    goal.command.position = float(position)
    goal.command.max_effort = float(max_effort)
    return goal


def _as_float_map(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return {str(k): float(v) for k, v in value.items()}


def _parse_sequence(sequence_text: str | None, config_sequence: Any) -> list[str]:
    if sequence_text:
        names = [part.strip() for part in sequence_text.split(",") if part.strip()]
    elif isinstance(config_sequence, list) and config_sequence:
        names = [str(name) for name in config_sequence]
    else:
        names = list(DEFAULT_SEQUENCE)
    if not names:
        raise ValueError("sequence is empty")
    return names


def _validate_gripper_position(name: str, position: float, min_position: float, max_position: float) -> None:
    if not (min_position <= position <= max_position):
        raise ValueError(
            f"{name}: gripper target {position:.6f} outside "
            f"[{min_position:.6f}, {max_position:.6f}]")


def _build_steps(
    data: dict[str, Any],
    sequence: list[str],
    arm_joints: list[str],
    current_arm: list[float],
    max_joint_delta: float,
    min_gripper_position: float,
    max_gripper_position: float,
    close_gripper: float | None,
    open_gripper: float | None,
) -> list[ArmStep | GripperStep]:
    waypoints = data.get("waypoints", {})
    if not isinstance(waypoints, dict):
        raise ValueError("waypoints must be a mapping")
    gripper_targets = data.get("gripper_targets", {})
    if not isinstance(gripper_targets, dict):
        gripper_targets = {}

    close_value = float(close_gripper) if close_gripper is not None else float(gripper_targets.get("close_gripper", 0.60))
    open_value = float(open_gripper) if open_gripper is not None else float(gripper_targets.get("open_gripper", 0.00))
    _validate_gripper_position("close_gripper", close_value, min_gripper_position, max_gripper_position)
    _validate_gripper_position("open_gripper", open_value, min_gripper_position, max_gripper_position)

    previous_arm = list(current_arm)
    steps: list[ArmStep | GripperStep] = []
    for name in sequence:
        if name in ("close_gripper", "open_gripper"):
            steps.append(GripperStep(name=name, position=close_value if name == "close_gripper" else open_value))
            continue

        waypoint = waypoints.get(name)
        if not isinstance(waypoint, dict):
            available = sorted(str(key) for key in waypoints)
            raise ValueError(f"missing arm waypoint {name!r}; available={available}")
        arm = _as_float_map(waypoint.get("arm"), f"waypoints.{name}.arm")
        missing = [joint for joint in arm_joints if joint not in arm]
        if missing:
            raise ValueError(f"waypoints.{name}.arm missing joints: {missing}")
        positions = [float(arm[joint]) for joint in arm_joints]
        deltas = [abs(target - start) for target, start in zip(positions, previous_arm)]
        max_delta = max(deltas) if deltas else 0.0
        if max_delta > max_joint_delta:
            max_joint = arm_joints[deltas.index(max_delta)]
            raise ValueError(
                f"{name}: max joint delta {max_delta:.6f} rad at {max_joint} "
                f"exceeds limit {max_joint_delta:.6f} rad")
        steps.append(ArmStep(name=name, positions=positions, max_delta_from_previous=max_delta))
        previous_arm = positions
    return steps


def _print_steps(
    steps: list[ArmStep | GripperStep],
    arm_joints: list[str],
    arm_action: str,
    gripper_action: str,
    arm_duration: float,
    gripper_max_effort: float,
    execute: bool,
) -> None:
    print(f"[teach-replay] execute: {bool(execute)}")
    print(f"[teach-replay] arm action: {arm_action}")
    print(f"[teach-replay] gripper action: {gripper_action}")
    print(f"[teach-replay] arm duration: {arm_duration:.2f}s")
    print(f"[teach-replay] gripper max_effort: {gripper_max_effort:.6f}")
    for idx, step in enumerate(steps, start=1):
        if isinstance(step, ArmStep):
            values = ", ".join(
                f"{joint}={value:.6f}" for joint, value in zip(arm_joints, step.positions))
            print(
                f"[teach-replay] step {idx:02d} arm {step.name}: "
                f"max_delta={step.max_delta_from_previous:.6f} {values}")
        else:
            print(
                f"[teach-replay] step {idx:02d} gripper {step.name}: "
                f"position={step.position:.6f}")


def _send_arm_goal(node, rclpy, client, arm_joints: list[str], step: ArmStep, duration: float) -> int:
    goal = _make_arm_goal(arm_joints, step.positions, duration)
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    handle = send_future.result()
    if handle is None or not handle.accepted:
        print(f"[teach-replay] arm goal rejected: {step.name}")
        return 2

    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    code = int(result.error_code)
    if code == 0:
        print(f"[teach-replay] arm success: {step.name}")
        return 0
    print(f"[teach-replay] arm failed: {step.name} error_code={code}")
    return 2


def _send_gripper_goal(node, rclpy, client, step: GripperStep, max_effort: float) -> int:
    goal = _make_gripper_goal(step.position, max_effort)
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    handle = send_future.result()
    if handle is None or not handle.accepted:
        print(f"[teach-replay] gripper goal rejected: {step.name}")
        return 2

    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    print(
        f"[teach-replay] gripper result: {step.name} "
        f"position={result.position:.6f} effort={result.effort:.6f} "
        f"stalled={result.stalled} reached_goal={result.reached_goal}")
    return 0 if result.reached_goal else 2


def main() -> int:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory, GripperCommand

    parser = argparse.ArgumentParser(description="Dry-run or execute taught OMY-F3M pick/place waypoints.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--sequence", default=None, help="Comma-separated step names. Defaults to YAML sequence.")
    parser.add_argument("--arm-action", default=None)
    parser.add_argument("--gripper-action", default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--max-joint-delta", type=float, default=None)
    parser.add_argument("--close-gripper", type=float, default=None)
    parser.add_argument("--open-gripper", type=float, default=None)
    parser.add_argument("--min-gripper-position", type=float, default=None)
    parser.add_argument("--max-gripper-position", type=float, default=None)
    parser.add_argument("--gripper-max-effort", type=float, default=None)
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"Refusing to execute. Re-run with --confirm {CONFIRM_TEXT}")
    if args.joint_state_timeout <= 0.0:
        raise ValueError("--joint-state-timeout must be > 0")

    root = _repo_root()
    data = _load_yaml(_resolve_repo_path(root, args.config))
    controller = data.get("controller", {})
    safety = data.get("safety", {})
    arm_joints = [str(name) for name in data.get("arm_joints", ARM_JOINTS)]
    if arm_joints != ARM_JOINTS:
        raise ValueError(f"Unexpected arm_joints for OMY-F3M: {arm_joints}")

    arm_action = args.arm_action or str(controller.get("arm_action", "/arm_controller/follow_joint_trajectory"))
    gripper_action = args.gripper_action or str(controller.get("gripper_action", "/gripper_controller/gripper_cmd"))
    arm_duration = float(args.duration if args.duration is not None else safety.get("default_arm_duration", 5.0))
    max_joint_delta = float(args.max_joint_delta if args.max_joint_delta is not None else safety.get("max_joint_delta_per_step", 0.35))
    min_gripper_position = float(args.min_gripper_position if args.min_gripper_position is not None else safety.get("min_gripper_position", 0.0))
    max_gripper_position = float(args.max_gripper_position if args.max_gripper_position is not None else safety.get("max_gripper_position", 1.12))
    gripper_max_effort = float(args.gripper_max_effort if args.gripper_max_effort is not None else safety.get("default_gripper_max_effort", 0.0))

    if arm_duration < 4.0:
        raise ValueError("--duration must be >= 4.0 s for teach replay")
    if max_joint_delta <= 0.0:
        raise ValueError("--max-joint-delta must be > 0")
    if min_gripper_position < 0.0 or max_gripper_position > 1.12 or min_gripper_position >= max_gripper_position:
        raise ValueError("gripper guard must stay within [0.0, 1.12] with min < max")

    sequence = _parse_sequence(args.sequence, data.get("sequence"))

    rclpy.init(args=None)
    node = Node("motion2_run_teach_pick_place")
    arm_client = ActionClient(node, FollowJointTrajectory, arm_action)
    gripper_client = ActionClient(node, GripperCommand, gripper_action)
    try:
        joints = _joint_state_once(node, rclpy, args.joint_state_timeout)
        missing = [name for name in arm_joints if name not in joints]
        if missing:
            raise RuntimeError(f"Missing arm joints in /joint_states: {missing}")
        current_arm = [float(joints[name]) for name in arm_joints]

        steps = _build_steps(
            data=data,
            sequence=sequence,
            arm_joints=arm_joints,
            current_arm=current_arm,
            max_joint_delta=max_joint_delta,
            min_gripper_position=min_gripper_position,
            max_gripper_position=max_gripper_position,
            close_gripper=args.close_gripper,
            open_gripper=args.open_gripper,
        )
        _print_steps(
            steps,
            arm_joints,
            arm_action,
            gripper_action,
            arm_duration,
            gripper_max_effort,
            args.execute,
        )

        if not args.execute:
            print("[teach-replay] dry-run complete; command_sent=false")
            return 0

        node.get_logger().info(f"Waiting for arm action {arm_action}")
        if not arm_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Action server not available: {arm_action}")
        node.get_logger().info(f"Waiting for gripper action {gripper_action}")
        if not gripper_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Action server not available: {gripper_action}")

        for idx, step in enumerate(steps, start=1):
            print(f"[teach-replay] executing step {idx}/{len(steps)}: {step.name}")
            if isinstance(step, ArmStep):
                rc = _send_arm_goal(node, rclpy, arm_client, arm_joints, step, arm_duration)
            else:
                rc = _send_gripper_goal(node, rclpy, gripper_client, step, gripper_max_effort)
            if rc != 0:
                print(f"[teach-replay] stopping after failed step: {step.name}")
                return rc

        print("[teach-replay] SUCCESS command_sent=true")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
