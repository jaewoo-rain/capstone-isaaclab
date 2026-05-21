"""Print the current OMY-F3M end-effector pose from TF."""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


DEFAULT_BASE_FRAME = "link0"
DEFAULT_EE_FRAME = "end_effector_link"
CHECK_COMMANDS = (
    "ros2 topic list",
    "ros2 node list",
    "ros2 topic echo /joint_states",
    "ros2 run tf2_ros tf2_echo link0 end_effector_link",
)


def _add_repo_root_to_path() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Print current real OMY-F3M EE pose.")
    parser.add_argument("--base-frame", default=DEFAULT_BASE_FRAME)
    parser.add_argument("--ee-frame", default=DEFAULT_EE_FRAME)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--spin-steps", type=int, default=10)
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be > 0")

    _add_repo_root_to_path()
    from motion2.adapters.real_adapter import RealAdapter

    adapter = RealAdapter(base_frame=args.base_frame, ee_frame=args.ee_frame)
    try:
        try:
            adapter.step(args.spin_steps)
            for idx in range(args.samples):
                ee = adapter.get_ee_pose()
                print(f"[ee-pose] sample={idx + 1}")
                print(f"  frame: {args.base_frame} -> {args.ee_frame}")
                print(f"  pos: {np.round(ee.pos_w, 6).tolist()}")
                print(f"  quat_wxyz: {np.round(ee.quat_w, 6).tolist()}")
                print(f"  lin_vel: {np.round(ee.lin_vel, 6).tolist()}")
                if idx + 1 < args.samples:
                    adapter.step(max(1, args.spin_steps))
        except RuntimeError as exc:
            print(f"[ee-pose] failed: {exc}")
            print("[ee-pose] check inside the Docker container:")
            for command in CHECK_COMMANDS:
                print(f"  {command}")
            return 2
        return 0
    finally:
        adapter.close()


if __name__ == "__main__":
    raise SystemExit(main())
