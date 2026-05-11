"""Smoke-test DryRunAdapter without moving the robot.

Run this inside the robot ROS2 Docker after bringup is active. It reads TF and
/joint_states, then calls set_ee_target() with the current EE pose so the adapter
prints what it would command. No ROS command/action is published.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np


def _add_repo_root_to_path() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> int:
    _add_repo_root_to_path()

    from motion2.adapters.dry_run_adapter import DryRunAdapter

    adapter = DryRunAdapter()
    try:
        adapter.step(30)
        ee = adapter.get_ee_pose()
        print("[check] current EE pose")
        print(f"  pos: {np.round(ee.pos_w, 6).tolist()}")
        print(f"  quat_wxyz: {np.round(ee.quat_w, 6).tolist()}")
        print(f"  lin_vel: {np.round(ee.lin_vel, 6).tolist()}")

        home_pos, home_quat = adapter.get_home_ee_pose()
        print("[check] cached home EE pose")
        print(f"  pos: {np.round(home_pos, 6).tolist()}")
        print(f"  quat_wxyz: {np.round(home_quat, 6).tolist()}")

        adapter.set_ee_target(ee.pos_w, ee.quat_w, gripper_value=0.0)
        adapter.step(5)
        print("[check] dry-run complete; command_sent=false")
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
