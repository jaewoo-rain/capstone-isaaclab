"""Run a named smoke-test sequence.

Default behavior is dry-run. With --execute, each step is executed through
run_smoke_waypoint.py and still requires the appropriate confirm token.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import yaml


JOINT_CONFIRM = "EXECUTE_JOINT_SPACE_SMOKE_TEST"
GRIPPER_CONFIRM = "EXECUTE_GRIPPER_SMOKE_TEST"


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _resolve_repo_path(root: pathlib.Path, path_text: str) -> pathlib.Path:
    path = pathlib.Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def _load_yaml(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _waypoint_type(waypoint_config: dict, name: str) -> str:
    for command in waypoint_config.get("tested_commands", []):
        if command.get("name") == name:
            return str(command.get("type"))
    raise ValueError(f"unknown waypoint in waypoint config: {name}")


def _confirm_for_type(command_type: str) -> str:
    if command_type == "arm_delta":
        return JOINT_CONFIRM
    if command_type == "gripper_delta":
        return GRIPPER_CONFIRM
    raise ValueError(f"unsupported waypoint type: {command_type}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a named smoke-test sequence.")
    parser.add_argument("--sequence-config", default="motion2/config/smoke_sequences.yaml")
    parser.add_argument("--waypoint-config", default="motion2/config/joint_space_smoke_waypoints.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        default=True,
        help="Stop the sequence at the first failed step.")
    args = parser.parse_args()

    root = _repo_root()
    sequence_config = _load_yaml(_resolve_repo_path(root, args.sequence_config))
    waypoint_config = _load_yaml(_resolve_repo_path(root, args.waypoint_config))
    sequences = sequence_config.get("sequences", {})
    if args.name not in sequences:
        raise ValueError(f"unknown sequence {args.name!r}; available={list(sequences)}")

    steps = list(sequences[args.name].get("steps", []))
    if not steps:
        raise ValueError(f"sequence {args.name!r} has no steps")

    print(f"[sequence] name={args.name} steps={len(steps)} execute={bool(args.execute)}")
    ok_count = 0
    for idx, step in enumerate(steps, start=1):
        cmd = [
            sys.executable,
            str(root / "motion2/scripts/run_smoke_waypoint.py"),
            "--name",
            str(step),
            "--config",
            args.waypoint_config,
        ]
        if args.execute:
            command_type = _waypoint_type(waypoint_config, str(step))
            cmd += ["--execute", "--confirm", _confirm_for_type(command_type)]

        print(f"[sequence] step {idx}/{len(steps)}: {step}")
        rc = subprocess.call(cmd)
        if rc == 0:
            ok_count += 1
            continue
        print(f"[sequence] step failed: {step} rc={rc}")
        if args.stop_on_failure:
            break

    print(f"[sequence] summary: {ok_count}/{len(steps)} successful execute={bool(args.execute)}")
    return 0 if ok_count == len(steps) else 2


if __name__ == "__main__":
    raise SystemExit(main())
