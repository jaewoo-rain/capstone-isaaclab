"""Capstone-level MVP pipeline summary for motion2.

This script is presentation-oriented: it shows the complete object-to-slot
pipeline with rule-based inputs and reports which parts are real-robot verified.
It never sends robot commands.
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass

import yaml


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _resolve_repo_path(path_text: str) -> pathlib.Path:
    path = pathlib.Path(path_text)
    if path.is_absolute():
        return path
    return _repo_root() / path


def _load_yaml(path: pathlib.Path) -> dict:
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


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class Stage:
    name: str
    source: str
    command: str
    status: str


def _pose_from_section(data: dict, name: str) -> Pose2D:
    section = _section(data, name)
    return Pose2D(float(section["x"]), float(section["y"]), float(section["yaw"]))


def _apply_overrides(pose: Pose2D, *, x: float | None, y: float | None, yaw: float | None) -> Pose2D:
    return Pose2D(
        pose.x if x is None else float(x),
        pose.y if y is None else float(y),
        pose.yaw if yaw is None else float(yaw),
    )


def _build_stages(config: dict, obj: Pose2D, slot: Pose2D) -> list[Stage]:
    heights = _section(config, "heights")
    return [
        Stage(
            "vision_object_pose",
            "test/manual input",
            f"object PoseStamped frame=link0 x={obj.x:.3f} y={obj.y:.3f} yaw={obj.yaw:.3f}",
            "MVP fallback ready; D435 not calibrated",
        ),
        Stage(
            "slot_selection",
            "rule based",
            f"selected slot frame=link0 x={slot.x:.3f} y={slot.y:.3f} yaw={slot.yaw:.3f}",
            "ready",
        ),
        Stage(
            "pre_grasp",
            "MoveIt plan-only",
            f"pose=({obj.x:.3f}, {obj.y:.3f}, {float(heights['pre_grasp_z']):.3f}) yaw={obj.yaw:.3f}",
            "plan-only candidate",
        ),
        Stage(
            "grasp",
            "MoveIt plan-only",
            f"pose=({obj.x:.3f}, {obj.y:.3f}, {float(heights['grasp_z']):.3f}) yaw={obj.yaw:.3f}",
            "blocked on real robot: current grasp_z failed planning",
        ),
        Stage(
            "close_gripper",
            "gripper action",
            "run_pick_place_mvp.py --only close_gripper",
            "real action path verified",
        ),
        Stage(
            "lift",
            "MoveIt plan-only",
            f"pose=({obj.x:.3f}, {obj.y:.3f}, {float(heights['lift_z']):.3f}) yaw={obj.yaw:.3f}",
            "plan-only candidate",
        ),
        Stage(
            "transport",
            "MoveIt plan-only + guard",
            f"pose=({slot.x:.3f}, {slot.y:.3f}, {float(heights['transport_z']):.3f}) yaw={slot.yaw:.3f}",
            "blocked for execute: guard caught large joint span",
        ),
        Stage(
            "insert",
            "MoveIt plan-only + guard",
            f"pose=({slot.x:.3f}, {slot.y:.3f}, {float(heights['place_z']):.3f}) yaw={slot.yaw:.3f}",
            "blocked for execute until absolute pose guard is stable",
        ),
        Stage(
            "open_gripper",
            "gripper action",
            "run_pick_place_mvp.py --only open_gripper",
            "real action path verified",
        ),
        Stage(
            "retract",
            "MoveIt plan-only",
            f"pose=({slot.x:.3f}, {slot.y:.3f}, {float(heights['retract_z']):.3f}) yaw={slot.yaw:.3f}",
            "plan-only candidate",
        ),
    ]


def _print_stage_table(stages: list[Stage]) -> None:
    print("[capstone-mvp] pipeline stages:")
    for idx, stage in enumerate(stages, start=1):
        print(f"[capstone-mvp] {idx:02d}. {stage.name}")
        print(f"[capstone-mvp]     source: {stage.source}")
        print(f"[capstone-mvp]     command: {stage.command}")
        print(f"[capstone-mvp]     status: {stage.status}")


def _print_real_robot_status() -> None:
    print("[capstone-mvp] real robot verified:")
    print("[capstone-mvp]   - regular bringup: omy_f3m.launch.py")
    print("[capstone-mvp]   - controllers: joint_state_broadcaster, arm_controller, gripper_controller active")
    print("[capstone-mvp]   - full_basic_smoke sequence: executed successfully")
    print("[capstone-mvp]   - MVP gripper close/open action path: executed successfully")
    print("[capstone-mvp]   - guarded relative z +/-0.002 m pose smoke: executed successfully")
    print("[capstone-mvp] real robot blocked:")
    print("[capstone-mvp]   - full pick-place execute")
    print("[capstone-mvp]   - grasp_z=0.115 execute")
    print("[capstone-mvp]   - absolute transport/insert execute")
    print("[capstone-mvp]   - camera/RL-driven real commands")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the capstone MVP pipeline without moving the robot.")
    parser.add_argument("--config", default="motion2/config/manual_targets_safe_dryrun.yaml")
    parser.add_argument("--object-x", type=float, default=None)
    parser.add_argument("--object-y", type=float, default=None)
    parser.add_argument("--object-yaw", type=float, default=None)
    parser.add_argument("--slot-x", type=float, default=None)
    parser.add_argument("--slot-y", type=float, default=None)
    parser.add_argument("--slot-yaw", type=float, default=None)
    args = parser.parse_args()

    config_path = _resolve_repo_path(args.config)
    config = _load_yaml(config_path)
    if config.get("frame_id") != "link0":
        raise ValueError(f"expected frame_id=link0, got {config.get('frame_id')!r}")

    obj = _apply_overrides(
        _pose_from_section(config, "box"),
        x=args.object_x,
        y=args.object_y,
        yaw=args.object_yaw,
    )
    slot = _apply_overrides(
        _pose_from_section(config, "cell"),
        x=args.slot_x,
        y=args.slot_y,
        yaw=args.slot_yaw,
    )

    print("[capstone-mvp] command_sent=false")
    print(f"[capstone-mvp] config={args.config}")
    print("[capstone-mvp] mode=rule_based_mvp")
    _print_stage_table(_build_stages(config, obj, slot))
    _print_real_robot_status()
    print("[capstone-mvp] next demo command:")
    print("  python3 motion2/scripts/run_pick_place_mvp.py --allow-unverified --demo-safe")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
