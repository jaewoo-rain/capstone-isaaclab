"""Dry-run the manual-target motion2 chain without moving the robot.

This script is for the current real-lab setup where the D405 wrist camera may
be mounted, but the D435 top-view camera is not mounted/calibrated. It skips
top_cam_scan(), reads box/cell poses from YAML, and sends stage targets only to
DryRunAdapter, which logs commands without publishing robot goals.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

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


def _require_section(data: dict, name: str) -> dict:
    section = data.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"missing mapping section: {name}")
    return section


def _pose_xy_yaw(section: dict, name: str) -> tuple[float, float, float]:
    try:
        return float(section["x"]), float(section["y"]), float(section["yaw"])
    except KeyError as exc:
        raise ValueError(f"{name} missing required key: {exc.args[0]}") from exc


def _stage(adapter, name: str, pos, quat, gripper: float, steps: int) -> None:
    pos = np.asarray(pos, dtype=np.float32)
    quat = np.asarray(quat, dtype=np.float32)
    print(f"[manual dry-run] {name}: pos={pos.tolist()} quat_wxyz={quat.tolist()} gripper={gripper}")
    adapter.set_ee_target(pos, quat, gripper)
    adapter.step(steps)


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run manual-target motion2 chain.")
    parser.add_argument(
        "--config",
        default="motion2/config/manual_targets.yaml",
        help="manual target YAML path")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="allow configs with verified=false. Still sends no robot commands.")
    parser.add_argument(
        "--steps-per-stage",
        type=int,
        default=3,
        help="DryRunAdapter step() calls after each logged target.")
    args = parser.parse_args()

    _add_repo_root_to_path()
    from motion2.adapters.dry_run_adapter import DryRunAdapter, WorkspaceLimits

    cfg_path = pathlib.Path(args.config)
    data = _load_config(cfg_path)
    if data.get("frame_id") != "link0":
        raise ValueError(f"expected frame_id=link0, got {data.get('frame_id')!r}")
    if not data.get("verified", False) and not args.allow_unverified:
        raise RuntimeError(
            f"{cfg_path} has verified=false. Re-run with --allow-unverified for "
            "logging-only dry-run, or verify the target poses first.")

    box_x, box_y, box_yaw = _pose_xy_yaw(_require_section(data, "box"), "box")
    cell_x, cell_y, cell_yaw = _pose_xy_yaw(_require_section(data, "cell"), "cell")
    heights = _require_section(data, "heights")
    gripper = _require_section(data, "gripper")
    safety = _require_section(_require_section(data, "safety"), "workspace")

    workspace = WorkspaceLimits(
        x_min=float(safety["x_min"]),
        x_max=float(safety["x_max"]),
        y_min=float(safety["y_min"]),
        y_max=float(safety["y_max"]),
        z_min=float(safety["z_min"]),
        z_max=float(safety["z_max"]),
    )
    adapter = DryRunAdapter(workspace=workspace)
    try:
        adapter.step(30)
        adapter.reset_to_home()
        home_pos, home_quat = adapter.get_home_ee_pose()
        base_ee_quat = adapter.get_base_ee_quat()
        box_quat = _quat_mul(_quat_from_z_yaw(box_yaw), base_ee_quat)
        cell_quat = _quat_mul(_quat_from_z_yaw(cell_yaw), base_ee_quat)

        open_val = float(gripper["open"])
        close_val = float(gripper["close"])
        steps = max(1, int(args.steps_per_stage))

        print(f"[manual dry-run] config={cfg_path} verified={data.get('verified')}")
        print("[manual dry-run] no robot command/action will be sent")
        print(f"[manual dry-run] current home pos={home_pos.tolist()} quat_wxyz={home_quat.tolist()}")

        _stage(adapter, "home", home_pos, home_quat, open_val, steps)
        _stage(adapter, "pre_grasp",
               [box_x, box_y, float(heights["pre_grasp_z"])], box_quat, open_val, steps)
        _stage(adapter, "grasp",
               [box_x, box_y, float(heights["grasp_z"])], box_quat, open_val, steps)
        _stage(adapter, "close",
               [box_x, box_y, float(heights["grasp_z"])], box_quat, close_val, steps)
        _stage(adapter, "lift",
               [box_x, box_y, float(heights["lift_z"])], box_quat, close_val, steps)
        _stage(adapter, "transport",
               [cell_x, cell_y, float(heights["transport_z"])], cell_quat, close_val, steps)
        _stage(adapter, "insert",
               [cell_x, cell_y, float(heights["place_z"])], cell_quat, close_val, steps)
        _stage(adapter, "release",
               [cell_x, cell_y, float(heights["place_z"])], cell_quat, open_val, steps)
        _stage(adapter, "retract",
               [cell_x, cell_y, float(heights["retract_z"])], base_ee_quat, open_val, steps)
        _stage(adapter, "return_home", home_pos, home_quat, open_val, steps)

        print("[manual dry-run] complete; command_sent=false for all targets")
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
