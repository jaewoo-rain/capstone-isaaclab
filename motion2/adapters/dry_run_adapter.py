"""motion2 — ROS2 dry-run adapter for OMY-F3M.

This adapter is intentionally read-only from the robot's point of view:
it may read TF and joint state, but it never publishes commands and never
sends FollowJointTrajectory goals. Use it before implementing RealAdapter.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
import numpy as np

from .base_adapter import BaseAdapter, CamData, EePose, BoxGtPose


@dataclass(frozen=True)
class WorkspaceLimits:
    """Conservative link0-frame target limits for dry-run validation."""

    x_min: float = -0.10
    x_max: float = 0.45
    y_min: float = -0.45
    y_max: float = 0.20
    z_min: float = 0.10
    z_max: float = 0.50


class DryRunAdapter(BaseAdapter):
    """ROS2 adapter that validates targets without moving the robot.

    Frame convention:
      - base_frame: real robot base frame, observed as ``link0`` on OMY-F3M
      - ee_frame: MoveIt end-effector frame, observed as ``end_effector_link``
      - wrist_cam_frame: observed as ``camera_link``
      - returned quaternions are motion2 convention ``wxyz``
    """

    def __init__(
        self,
        *,
        base_frame: str = "link0",
        ee_frame: str = "end_effector_link",
        wrist_cam_frame: str = "camera_link",
        control_dt: float = 1.0 / 60.0,
        workspace: WorkspaceLimits | None = None,
        gripper_min: float = 0.0,
        gripper_max: float = 1.12,
        node_name: str = "motion2_dry_run_adapter",
    ):
        # Lazy imports keep this module importable on machines without ROS2.
        import rclpy
        from rclpy.node import Node
        from tf2_ros import Buffer, TransformListener
        from sensor_msgs.msg import JointState

        if not rclpy.ok():
            rclpy.init(args=None)

        class _Node(Node):
            pass

        self._rclpy = rclpy
        self._joint_state_msg_type = JointState
        self.node = _Node(node_name)
        self.base_frame = base_frame
        self.ee_frame = ee_frame
        self.wrist_cam_frame = wrist_cam_frame
        self._control_dt = float(control_dt)
        self.workspace = workspace or WorkspaceLimits()
        self.gripper_min = float(gripper_min)
        self.gripper_max = float(gripper_max)
        self.command_count = 0
        self.last_target: tuple[np.ndarray, np.ndarray, float] | None = None
        self._last_ee_pose: EePose | None = None
        self._last_ee_stamp_ns: int | None = None
        self._home_pose: tuple[np.ndarray, np.ndarray] | None = None
        self._joint_state = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self.node)
        self.node.create_subscription(
            JointState, "/joint_states", self._on_joint_state, 10)

        self.node.get_logger().info(
            "DryRunAdapter ready: commands will be logged, not sent.")

    # ===== ROS helpers =====
    @staticmethod
    def _xyzw_to_wxyz(q_xyzw) -> np.ndarray:
        return np.array(
            [q_xyzw.w, q_xyzw.x, q_xyzw.y, q_xyzw.z], dtype=np.float32)

    @staticmethod
    def _vec3_to_np(v) -> np.ndarray:
        return np.array([v.x, v.y, v.z], dtype=np.float32)

    def _on_joint_state(self, msg) -> None:
        self._joint_state = msg

    def _spin_some(self) -> None:
        self._rclpy.spin_once(self.node, timeout_sec=0.0)

    def _lookup_transform(self, source_frame: str):
        deadline = time.monotonic() + 5.0
        last_error = None
        while time.monotonic() < deadline:
            self._spin_some()
            try:
                return self.tf_buffer.lookup_transform(
                    self.base_frame, source_frame, self._rclpy.time.Time())
            except Exception as exc:  # tf2_ros exception classes vary by distro.
                last_error = exc
                self._rclpy.spin_once(self.node, timeout_sec=0.05)
        raise RuntimeError(
            f"Timed out waiting for TF {self.base_frame} -> {source_frame}. "
            f"Last error: {last_error}") from last_error

    def _validate_target(self, target_pos: np.ndarray, gripper_value: float) -> list[str]:
        x, y, z = [float(v) for v in target_pos]
        w = self.workspace
        warnings = []
        if not (w.x_min <= x <= w.x_max):
            warnings.append(f"x={x:.3f} outside [{w.x_min:.3f}, {w.x_max:.3f}]")
        if not (w.y_min <= y <= w.y_max):
            warnings.append(f"y={y:.3f} outside [{w.y_min:.3f}, {w.y_max:.3f}]")
        if not (w.z_min <= z <= w.z_max):
            warnings.append(f"z={z:.3f} outside [{w.z_min:.3f}, {w.z_max:.3f}]")
        if not (self.gripper_min <= gripper_value <= self.gripper_max):
            warnings.append(
                f"gripper={gripper_value:.3f} outside "
                f"[{self.gripper_min:.3f}, {self.gripper_max:.3f}]")
        return warnings

    # ===== BaseAdapter API =====
    def get_top_cam(self) -> CamData:
        raise NotImplementedError(
            "DryRunAdapter does not capture the top camera yet. "
            "Verify RealSense calibration before enabling camera-based chain runs.")

    def get_wrist_cam(self) -> CamData:
        raise NotImplementedError(
            "DryRunAdapter does not capture the wrist camera yet. "
            "Use TF-only dry-runs before enabling camera-based RL grasp.")

    def get_ee_pose(self) -> EePose:
        tf = self._lookup_transform(self.ee_frame)
        pos = self._vec3_to_np(tf.transform.translation)
        quat = self._xyzw_to_wxyz(tf.transform.rotation)

        stamp_ns = tf.header.stamp.sec * 1_000_000_000 + tf.header.stamp.nanosec
        lin_vel = np.zeros(3, dtype=np.float32)
        ang_vel_z = 0.0
        if self._last_ee_pose is not None and self._last_ee_stamp_ns is not None:
            dt = (stamp_ns - self._last_ee_stamp_ns) / 1_000_000_000.0
            if dt > 1e-6:
                lin_vel = ((pos - self._last_ee_pose.pos_w) / dt).astype(np.float32)

        pose = EePose(pos_w=pos, quat_w=quat, lin_vel=lin_vel, ang_vel_z=ang_vel_z)
        self._last_ee_pose = pose
        self._last_ee_stamp_ns = stamp_ns
        return pose

    def set_ee_target(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        gripper_value: float,
    ) -> None:
        target_pos = np.asarray(target_pos, dtype=np.float32)
        target_quat = np.asarray(target_quat, dtype=np.float32)
        gripper_value = float(gripper_value)
        self.command_count += 1
        self.last_target = (target_pos.copy(), target_quat.copy(), gripper_value)

        warnings = self._validate_target(target_pos, gripper_value)
        msg = (
            f"[DRY RUN #{self.command_count}] set_ee_target "
            f"pos={target_pos.tolist()} quat_wxyz={target_quat.tolist()} "
            f"gripper={gripper_value:.3f} command_sent=false")
        if warnings:
            msg += " warnings=" + "; ".join(warnings)
            self.node.get_logger().warning(msg)
        else:
            self.node.get_logger().info(msg)

    def step(self, n: int = 1) -> None:
        for _ in range(max(0, int(n))):
            self._rclpy.spin_once(self.node, timeout_sec=self._control_dt)
            time.sleep(0.0)

    def reset_to_home(self) -> None:
        self._spin_some()
        if self._joint_state is not None:
            joints = dict(zip(self._joint_state.name, self._joint_state.position))
            self.node.get_logger().info(
                "[DRY RUN] reset_to_home skipped; current joints="
                + ", ".join(f"{k}={v:.4f}" for k, v in joints.items()))
        else:
            self.node.get_logger().warning(
                "[DRY RUN] reset_to_home skipped; /joint_states not received yet.")

    def get_base_ee_quat(self) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    def get_home_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self._home_pose is None:
            ee = self.get_ee_pose()
            self._home_pose = (ee.pos_w.copy(), ee.quat_w.copy())
            self.node.get_logger().info(
                "[DRY RUN] cached current TF as home EE pose "
                f"pos={self._home_pose[0].tolist()} "
                f"quat_wxyz={self._home_pose[1].tolist()}")
        return self._home_pose

    def spawn_random_box(self) -> tuple[float, float, float]:
        self.node.get_logger().info("[DRY RUN] spawn_random_box skipped.")
        return (0.0, 0.0, 0.0)

    def spawn_random_cell(self) -> tuple[float, float, float]:
        self.node.get_logger().info("[DRY RUN] spawn_random_cell skipped.")
        return (0.0, 0.0, 0.0)

    def get_box_gt(self) -> BoxGtPose | None:
        return None

    def get_cell_gt(self) -> BoxGtPose | None:
        return None

    @property
    def control_dt(self) -> float:
        return self._control_dt

    def close(self) -> None:
        self.node.destroy_node()
        if self._rclpy.ok():
            self._rclpy.shutdown()
