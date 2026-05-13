"""Record the current OMY-F3M joint state as a teach waypoint.

This script only reads `/joint_states` and writes a YAML waypoint file. It does
not send any robot commands.
"""
from __future__ import annotations

import argparse
import datetime as _datetime
import pathlib
import re
from typing import Any

import yaml


ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT = "rh_r1_joint"
DEFAULT_CONFIG = "motion2/config/teach_pick_place_waypoints.yaml"


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _resolve_repo_path(root: pathlib.Path, path_text: str) -> pathlib.Path:
    path = pathlib.Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def _load_config(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return _default_config()
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return _default_config()
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _default_config() -> dict[str, Any]:
    return {
        "verified": False,
        "bringup": "omy_f3m.launch.py",
        "mode": "teach_waypoint_replay",
        "controller": {
            "arm_action": "/arm_controller/follow_joint_trajectory",
            "gripper_action": "/gripper_controller/gripper_cmd",
        },
        "safety": {
            "max_joint_delta_per_step": 0.35,
            "min_gripper_position": 0.0,
            "max_gripper_position": 1.12,
            "default_arm_duration": 5.0,
            "default_gripper_max_effort": 0.0,
            "execute_requires_confirm": True,
            "forbidden_execution": ["MoveIt pose execute", "Cartesian execute"],
        },
        "arm_joints": list(ARM_JOINTS),
        "gripper_joint": GRIPPER_JOINT,
        "gripper_targets": {
            "close_gripper": 0.60,
            "open_gripper": 0.00,
        },
        "sequence": [
            "pre_grasp",
            "grasp",
            "close_gripper",
            "lift",
            "place",
            "open_gripper",
            "retract",
        ],
        "waypoints": {},
    }


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


def _validate_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]*", name):
        raise ValueError(
            "--name must contain only letters, numbers, underscore, or hyphen, "
            "and must not start with hyphen")


def _record_from_joint_state(joints: dict[str, float], arm_joints: list[str], gripper_joint: str) -> dict[str, Any]:
    required = list(arm_joints) + [gripper_joint]
    missing = [name for name in required if name not in joints]
    if missing:
        raise RuntimeError(f"Missing required joints in /joint_states: {missing}")

    return {
        "arm": {name: float(joints[name]) for name in arm_joints},
        "gripper": {gripper_joint: float(joints[gripper_joint])},
        "recorded_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
    }


def _write_config(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> int:
    import rclpy
    from rclpy.node import Node

    parser = argparse.ArgumentParser(
        description="Record the current /joint_states as a teach waypoint.")
    parser.add_argument("--name", required=True, help="Waypoint name, e.g. pre_grasp")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _validate_name(args.name)
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be > 0")

    root = _repo_root()
    config_path = _resolve_repo_path(root, args.config)
    data = _load_config(config_path)
    arm_joints = list(data.get("arm_joints", ARM_JOINTS))
    gripper_joint = str(data.get("gripper_joint", GRIPPER_JOINT))
    waypoints = data.setdefault("waypoints", {})
    if not isinstance(waypoints, dict):
        raise ValueError(f"{config_path}: waypoints must be a mapping")
    if args.name in waypoints and not args.overwrite:
        raise RuntimeError(
            f"Waypoint {args.name!r} already exists. Re-run with --overwrite to replace it.")

    rclpy.init(args=None)
    node = Node("motion2_record_teach_waypoint")
    try:
        joints = _joint_state_once(node, rclpy, args.timeout)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    waypoints[args.name] = _record_from_joint_state(joints, arm_joints, gripper_joint)
    _write_config(config_path, data)

    arm_text = ", ".join(f"{name}={waypoints[args.name]['arm'][name]:.6f}" for name in arm_joints)
    gripper_value = waypoints[args.name]["gripper"][gripper_joint]
    print(f"[teach-record] saved: {args.name}")
    print(f"[teach-record] config: {config_path}")
    print(f"[teach-record] arm: {arm_text}")
    print(f"[teach-record] gripper: {gripper_joint}={gripper_value:.6f}")
    print("[teach-record] command_sent=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
