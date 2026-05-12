"""Run a named smoke waypoint from joint_space_smoke_waypoints.yaml.

Default behavior is dry-run. Real execution still requires the underlying
script's explicit `--execute` and `--confirm` flags.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

import yaml


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _resolve_repo_path(root: pathlib.Path, path_text: str) -> pathlib.Path:
    path = pathlib.Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def _load_config(path: pathlib.Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _find_command(data: dict, name: str) -> dict:
    for command in data.get("tested_commands", []):
        if command.get("name") == name:
            return command
    names = [cmd.get("name") for cmd in data.get("tested_commands", [])]
    raise ValueError(f"unknown smoke waypoint {name!r}; available={names}")


def _run(cmd: list[str]) -> int:
    print("[smoke-runner]", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a named smoke waypoint.")
    parser.add_argument("--config", default="motion2/config/joint_space_smoke_waypoints.yaml")
    parser.add_argument("--name", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    root = _repo_root()
    data = _load_config(_resolve_repo_path(root, args.config))
    command = _find_command(data, args.name)
    safety = data.get("safety", {})

    if command["type"] == "arm_delta":
        delta_map = command.get("delta", {})
        if len(delta_map) != 1:
            raise ValueError(f"{args.name}: arm_delta must contain exactly one joint")
        joint, delta = next(iter(delta_map.items()))
        cmd = [
            sys.executable,
            str(root / "motion2/scripts/joint_space_smoke_test.py"),
            "--joint", str(joint),
            "--delta", str(delta),
            "--max-delta", str(safety.get("max_joint_delta_per_step", 0.02)),
            "--duration", str(safety.get("default_duration", 3.0)),
        ]
        if args.execute:
            cmd += ["--execute", "--confirm", args.confirm]
        return _run(cmd)

    if command["type"] == "gripper_delta":
        cmd = [
            sys.executable,
            str(root / "motion2/scripts/gripper_smoke_test.py"),
            "--joint", str(command.get("joint", "rh_r1_joint")),
            "--delta", str(command["delta"]),
            "--max-delta", str(safety.get("max_gripper_delta_per_step", 0.04)),
        ]
        if args.execute:
            cmd += ["--execute", "--confirm", args.confirm]
        return _run(cmd)

    raise ValueError(f"unsupported command type: {command.get('type')!r}")


if __name__ == "__main__":
    raise SystemExit(main())
