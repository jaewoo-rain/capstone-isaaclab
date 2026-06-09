"""Inspect a motion2 sim-exported trajectory before real replay.

This is an offline checker for files produced by
export_sim_pick_place_trajectory.py. It never talks to ROS and never commands
the robot.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np


DEFAULT_NPZ = "motion2/config/sim_exported_pick_place_trajectory.npz"
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _resolve_path(path_text: str) -> pathlib.Path:
    p = pathlib.Path(path_text).expanduser()
    return p if p.is_absolute() else _repo_root() / p


def _load_npz(path: pathlib.Path):
    data = np.load(path, allow_pickle=False)
    required = ["time_s", "arm_joint_pos", "gripper_command", "target_pos", "ee_pos"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{path}: missing required arrays: {missing}")
    return data


def _filter_run(data, run_index: int):
    time_s = np.asarray(data["time_s"], dtype=np.float64)
    arm = np.asarray(data["arm_joint_pos"], dtype=np.float64)
    gripper = np.asarray(data["gripper_command"], dtype=np.float64)
    if "run_index" not in data.files:
        if run_index != 0:
            raise ValueError("NPZ has no run_index array; only --run-index 0 is valid")
        return time_s, arm, gripper

    runs = np.asarray(data["run_index"], dtype=np.int32)
    available = sorted(set(int(v) for v in runs.tolist()))
    if run_index not in available:
        raise ValueError(f"--run-index {run_index} not found; available={available}")
    mask = runs == int(run_index)
    return time_s[mask], arm[mask], gripper[mask]


def _transition_indices(values: np.ndarray, tol: float) -> list[int]:
    changes = np.flatnonzero(np.abs(np.diff(values)) > tol) + 1
    return [int(i) for i in changes]


def _print_joint_vector(label: str, values: np.ndarray) -> None:
    text = ", ".join(f"{j}={v:+.6f}" for j, v in zip(ARM_JOINTS, values))
    print(f"{label}: {text}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect sim-exported NPZ trajectory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz", default=DEFAULT_NPZ)
    parser.add_argument("--run-index", type=int, default=0)
    parser.add_argument("--downsample-stride", type=int, default=5)
    parser.add_argument("--max-segment-delta", type=float, default=0.25)
    parser.add_argument("--max-abs-joint", type=float, default=3.2)
    parser.add_argument("--gripper-change-tol", type=float, default=1e-4)
    args = parser.parse_args()

    path = _resolve_path(args.npz)
    data = _load_npz(path)
    time_s, arm, gripper = _filter_run(data, args.run_index)

    if arm.ndim != 2 or arm.shape[1] != 6:
        raise ValueError(f"arm_joint_pos must have shape (N, 6), got {arm.shape}")
    if len(time_s) != len(arm) or len(gripper) != len(arm):
        raise ValueError("time_s, arm_joint_pos, gripper_command lengths differ")

    duration = float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0
    raw_step_delta = np.abs(np.diff(arm, axis=0)) if len(arm) > 1 else np.zeros((0, 6))
    raw_max_per_joint = raw_step_delta.max(axis=0) if len(raw_step_delta) else np.zeros(6)
    raw_worst_idx = np.unravel_index(int(np.argmax(raw_step_delta)), raw_step_delta.shape) if len(raw_step_delta) else (0, 0)

    stride = max(1, int(args.downsample_stride))
    keep = list(range(0, len(arm), stride))
    if keep[-1] != len(arm) - 1:
        keep.append(len(arm) - 1)
    for idx in _transition_indices(gripper, args.gripper_change_tol):
        keep.extend([max(0, idx - 1), idx])
    keep = sorted(set(keep))
    ds_arm = arm[keep]
    ds_delta = np.abs(np.diff(ds_arm, axis=0)) if len(ds_arm) > 1 else np.zeros((0, 6))
    ds_max_delta = float(ds_delta.max()) if len(ds_delta) else 0.0
    ds_worst_joint = ARM_JOINTS[int(np.argmax(ds_delta) % 6)] if len(ds_delta) else ""

    abs_max = np.max(np.abs(arm), axis=0)
    abs_worst_joint = ARM_JOINTS[int(np.argmax(abs_max))]
    transitions = _transition_indices(gripper, args.gripper_change_tol)

    print(f"[inspect-sim-export] file: {path}")
    print(f"[inspect-sim-export] run_index: {args.run_index}")
    print(f"[inspect-sim-export] samples: {len(arm)}")
    print(f"[inspect-sim-export] time: start={time_s[0]:.6f}s end={time_s[-1]:.6f}s duration={duration:.6f}s")
    print(f"[inspect-sim-export] gripper command range: {gripper.min():.6f} .. {gripper.max():.6f}")
    print(f"[inspect-sim-export] gripper transitions: {len(transitions)} at indices={transitions[:20]}")
    print(f"[inspect-sim-export] raw max per-step delta: {float(raw_step_delta.max()) if len(raw_step_delta) else 0.0:.6f} rad")
    if len(raw_step_delta):
        print(
            f"[inspect-sim-export] raw worst step: {raw_worst_idx[0]}->{raw_worst_idx[0]+1} "
            f"joint={ARM_JOINTS[raw_worst_idx[1]]}"
        )
    _print_joint_vector("[inspect-sim-export] raw max per joint", raw_max_per_joint)
    _print_joint_vector("[inspect-sim-export] abs max per joint", abs_max)
    _print_joint_vector("[inspect-sim-export] first arm", arm[0])
    _print_joint_vector("[inspect-sim-export] last arm", arm[-1])

    print(f"[inspect-sim-export] downsample stride: {stride}")
    print(f"[inspect-sim-export] downsampled points: {len(ds_arm)}")
    print(f"[inspect-sim-export] downsampled max segment delta: {ds_max_delta:.6f} rad at {ds_worst_joint}")

    guard_ok = True
    if ds_max_delta > args.max_segment_delta:
        print(
            f"[inspect-sim-export] GUARD: downsampled segment delta {ds_max_delta:.6f} "
            f"> limit {args.max_segment_delta:.6f}"
        )
        guard_ok = False
    if float(abs_max.max()) > args.max_abs_joint:
        print(
            f"[inspect-sim-export] GUARD: abs joint {float(abs_max.max()):.6f} at {abs_worst_joint} "
            f"> limit {args.max_abs_joint:.6f}"
        )
        guard_ok = False

    print(f"[inspect-sim-export] guard_ok={guard_ok}")
    return 0 if guard_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
