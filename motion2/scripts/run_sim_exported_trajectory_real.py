"""Replay a sim-exported joint trajectory on the real OMY-F3M.

Default mode is dry-run. Real execution requires:

    --execute --confirm EXECUTE_SIM_EXPORTED_TRAJECTORY

The runner splits the arm trajectory at gripper command transitions. Between
arm segments it can send GripperCommand goals, mapping sim gripper command
0.8 to real gripper position 1.05 by default.
"""
from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np


CONFIRM_TEXT = "EXECUTE_SIM_EXPORTED_TRAJECTORY"
DEFAULT_NPZ = "motion2/config/sim_exported_pick_place_trajectory.npz"
DEFAULT_ARM_ACTION = "/arm_controller/follow_joint_trajectory"
DEFAULT_GRIPPER_ACTION = "/gripper_controller/gripper_cmd"
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _resolve_path(path_text: str) -> pathlib.Path:
    p = pathlib.Path(path_text).expanduser()
    return p if p.is_absolute() else _repo_root() / p


def _load_npz(path: pathlib.Path):
    data = np.load(path, allow_pickle=False)
    required = ["time_s", "arm_joint_pos", "gripper_command"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{path}: missing required arrays: {missing}")
    time_s = np.asarray(data["time_s"], dtype=np.float64)
    arm = np.asarray(data["arm_joint_pos"], dtype=np.float64)
    gripper = np.asarray(data["gripper_command"], dtype=np.float64)
    if arm.ndim != 2 or arm.shape[1] != 6:
        raise ValueError(f"arm_joint_pos must have shape (N, 6), got {arm.shape}")
    if len(time_s) != len(arm) or len(gripper) != len(arm):
        raise ValueError("time_s, arm_joint_pos, gripper_command lengths differ")
    return time_s, arm, gripper


def _joint_state_once(node, rclpy, timeout_s: float) -> dict[str, float]:
    from sensor_msgs.msg import JointState

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
                return {name: float(pos) for name, pos in zip(msg.name, msg.position)}
    finally:
        node.destroy_subscription(sub)
    raise RuntimeError("Timed out waiting for /joint_states")


def _transition_indices(values: np.ndarray, tol: float) -> list[int]:
    return [int(i) for i in np.flatnonzero(np.abs(np.diff(values)) > tol) + 1]


def _downsample_indices(n: int, stride: int, gripper: np.ndarray, tol: float) -> list[int]:
    keep = list(range(0, n, max(1, stride)))
    if keep[-1] != n - 1:
        keep.append(n - 1)
    for idx in _transition_indices(gripper, tol):
        keep.extend([max(0, idx - 1), idx])
    return sorted(set(keep))


def _split_segments(indices: list[int], gripper: np.ndarray, tol: float) -> list[list[int]]:
    transition_set = set(_transition_indices(gripper, tol))
    segments: list[list[int]] = []
    current: list[int] = []
    for idx in indices:
        if idx in transition_set and current:
            if current[-1] != idx - 1 and idx - 1 >= 0:
                current.append(idx - 1)
            segments.append(sorted(set(current)))
            current = [idx]
        else:
            current.append(idx)
    if current:
        segments.append(sorted(set(current)))
    return [seg for seg in segments if len(seg) >= 1]


def _max_delta(values: np.ndarray) -> tuple[float, str]:
    if len(values) < 2:
        return 0.0, ""
    delta = np.abs(np.diff(values, axis=0))
    flat = int(np.argmax(delta))
    return float(delta.reshape(-1)[flat]), ARM_JOINTS[flat % 6]


def _make_arm_goal(indices: list[int], time_s: np.ndarray, arm: np.ndarray, duration_scale: float):
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(ARM_JOINTS)
    t0 = float(time_s[indices[0]])
    for n, idx in enumerate(indices):
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in arm[idx]]
        point.velocities = [0.0] * 6
        t = max(0.2, (float(time_s[idx]) - t0) * duration_scale)
        if n == 0:
            t = 0.2
        point.time_from_start.sec = int(t)
        point.time_from_start.nanosec = int((t - int(t)) * 1_000_000_000)
        goal.trajectory.points.append(point)
    return goal


def _make_gripper_goal(position: float, max_effort: float):
    from control_msgs.action import GripperCommand

    goal = GripperCommand.Goal()
    goal.command.position = float(position)
    goal.command.max_effort = float(max_effort)
    return goal


def _map_gripper(sim_value: float, sim_close: float, real_close: float, real_open: float) -> float:
    if abs(sim_close) < 1e-9:
        return real_open
    alpha = max(0.0, min(1.0, float(sim_value) / float(sim_close)))
    return real_open + alpha * (real_close - real_open)


def _send_arm_segment(node, rclpy, client, goal, name: str) -> bool:
    fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut)
    handle = fut.result()
    if handle is None or not handle.accepted:
        print(f"[sim-replay] {name}: arm goal rejected")
        return False
    res_fut = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res_fut)
    code = int(res_fut.result().result.error_code)
    if code == 0:
        print(f"[sim-replay] {name}: arm success")
        return True
    print(f"[sim-replay] {name}: arm failed error_code={code}")
    return False


def _send_gripper(node, rclpy, client, position: float, max_effort: float, timeout_s: float) -> bool:
    goal = _make_gripper_goal(position, max_effort)
    fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout_s)
    handle = fut.result()
    if handle is None or not handle.accepted:
        print("[sim-replay] gripper goal rejected")
        return False
    res_fut = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res_fut, timeout_sec=timeout_s)
    if not res_fut.done():
        print(f"[sim-replay] gripper timeout after {timeout_s:.2f}s; continuing")
        return True
    result = res_fut.result().result
    print(
        f"[sim-replay] gripper result position={result.position:.6f} "
        f"stalled={result.stalled} reached={result.reached_goal}"
    )
    return bool(result.reached_goal or result.stalled)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay sim-exported NPZ trajectory on real OMY-F3M.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--npz", default=DEFAULT_NPZ)
    parser.add_argument("--arm-action", default=DEFAULT_ARM_ACTION)
    parser.add_argument("--gripper-action", default=DEFAULT_GRIPPER_ACTION)
    parser.add_argument("--downsample-stride", type=int, default=5)
    parser.add_argument("--duration-scale", type=float, default=2.0)
    parser.add_argument("--max-start-delta", type=float, default=3.0)
    parser.add_argument("--max-segment-delta", type=float, default=0.35)
    parser.add_argument("--gripper-change-tol", type=float, default=1e-4)
    parser.add_argument("--sim-close", type=float, default=0.8)
    parser.add_argument("--real-close", type=float, default=1.05)
    parser.add_argument("--real-open", type=float, default=0.0)
    parser.add_argument("--gripper-max-effort", type=float, default=5.0)
    parser.add_argument("--gripper-timeout", type=float, default=2.0)
    parser.add_argument("--arm-only", action="store_true")
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"Refusing to execute. --confirm {CONFIRM_TEXT} required.")
    if args.duration_scale < 1.0:
        raise ValueError("--duration-scale must be >= 1.0")

    path = _resolve_path(args.npz)
    time_s, arm, gripper = _load_npz(path)
    keep = _downsample_indices(len(arm), args.downsample_stride, gripper, args.gripper_change_tol)
    segments = _split_segments(keep, gripper, args.gripper_change_tol)

    ds_arm = arm[keep]
    max_seg_delta, max_seg_joint = _max_delta(ds_arm)
    sim_duration = float(time_s[-1] - time_s[0])
    replay_duration = sim_duration * args.duration_scale

    print(f"[sim-replay] file: {path}")
    print(f"[sim-replay] execute={args.execute} arm_only={args.arm_only}")
    print(f"[sim-replay] raw_samples={len(arm)} downsampled_points={len(keep)} segments={len(segments)}")
    print(f"[sim-replay] sim_duration={sim_duration:.3f}s replay_duration≈{replay_duration:.3f}s")
    print(f"[sim-replay] max_downsampled_segment_delta={max_seg_delta:.6f} rad joint={max_seg_joint}")
    print(f"[sim-replay] gripper sim range={float(gripper.min()):.6f}..{float(gripper.max()):.6f}")
    print(
        f"[sim-replay] gripper map: sim {args.sim_close:.3f} -> real {args.real_close:.3f}, "
        f"open={args.real_open:.3f}"
    )

    if max_seg_delta > args.max_segment_delta:
        print(
            f"[sim-replay] GUARD VIOLATED: segment delta {max_seg_delta:.6f} "
            f"> limit {args.max_segment_delta:.6f}"
        )
        return 2

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory, GripperCommand

    rclpy.init(args=None)
    node = Node("motion2_run_sim_exported_trajectory")
    arm_client = ActionClient(node, FollowJointTrajectory, args.arm_action)
    gripper_client = ActionClient(node, GripperCommand, args.gripper_action)

    try:
        current = _joint_state_once(node, rclpy, args.joint_state_timeout)
        missing = [j for j in ARM_JOINTS if j not in current]
        if missing:
            raise RuntimeError(f"/joint_states missing arm joints: {missing}")
        current_arm = np.array([current[j] for j in ARM_JOINTS], dtype=np.float64)
        start_delta = np.abs(arm[0] - current_arm)
        max_start_delta = float(start_delta.max())
        max_start_joint = ARM_JOINTS[int(np.argmax(start_delta))]
        print(f"[sim-replay] current_to_first_delta={max_start_delta:.6f} rad joint={max_start_joint}")
        if max_start_delta > args.max_start_delta:
            print(
                f"[sim-replay] GUARD VIOLATED: current-to-first delta {max_start_delta:.6f} "
                f"> limit {args.max_start_delta:.6f}"
            )
            return 2

        for i, seg in enumerate(segments, start=1):
            seg_delta, seg_joint = _max_delta(arm[seg])
            g0 = float(gripper[seg[0]])
            g1 = float(gripper[seg[-1]])
            print(
                f"[sim-replay] segment {i:02d}/{len(segments)} "
                f"points={len(seg)} idx={seg[0]}..{seg[-1]} "
                f"grip={g0:.3f}->{g1:.3f} max_delta={seg_delta:.6f} {seg_joint}"
            )

        if not args.execute:
            print("[sim-replay] dry-run complete; command_sent=false")
            return 0

        print(f"[sim-replay] waiting for arm action: {args.arm_action}")
        if not arm_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"arm action server not available: {args.arm_action}")
        if not args.arm_only:
            print(f"[sim-replay] waiting for gripper action: {args.gripper_action}")
            if not gripper_client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"gripper action server not available: {args.gripper_action}")

        for i, seg in enumerate(segments, start=1):
            label = f"segment {i:02d}/{len(segments)}"
            goal = _make_arm_goal(seg, time_s, arm, args.duration_scale)
            if not _send_arm_segment(node, rclpy, arm_client, goal, label):
                print(f"[sim-replay] stopping after failed {label}")
                return 2

            if args.arm_only or i == len(segments):
                continue
            next_gripper = float(gripper[segments[i][0]])
            mapped = _map_gripper(next_gripper, args.sim_close, args.real_close, args.real_open)
            print(f"[sim-replay] gripper event after {label}: sim={next_gripper:.6f} real={mapped:.6f}")
            if not _send_gripper(node, rclpy, gripper_client, mapped, args.gripper_max_effort, args.gripper_timeout):
                print("[sim-replay] stopping after failed gripper event")
                return 2

        print("[sim-replay] SUCCESS command_sent=true")
        return 0

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
