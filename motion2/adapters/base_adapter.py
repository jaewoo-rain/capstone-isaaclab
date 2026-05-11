"""motion2 — Abstract adapter interface (sim/real 공통).

ChainStateMachine 이 호출하는 모든 sensor/robot interface 를 추상화.
sim_adapter (IsaacLab) 와 real_adapter (RealSense + OMY ROS) 가 같은 메서드 구현.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class CamData:
    """카메라 1 frame 캡처 결과."""
    rgb: np.ndarray              # (H, W, 3) uint8
    depth: np.ndarray            # (H, W) float32, meters (NaN/inf 가능)
    K: np.ndarray                # (3, 3) intrinsic
    pos_w: np.ndarray            # (3,) camera world position
    quat_w_world: np.ndarray     # (4,) wxyz, world convention (forward=+X, up=+Z)


@dataclass
class EePose:
    """End-effector (gripper center) 의 world pose + velocity."""
    pos_w: np.ndarray            # (3,)
    quat_w: np.ndarray           # (4,) wxyz
    lin_vel: np.ndarray          # (3,) world linear velocity m/s
    ang_vel_z: float             # world Z-axis angular velocity rad/s


@dataclass
class BoxGtPose:
    """Ground truth box pose — sim 만 제공. real 에선 None."""
    xy: tuple[float, float]      # env-rel xy (real 에선 world xy)
    yaw: float                   # rad


class BaseAdapter(ABC):
    """sim/real 추상 interface.

    Convention:
      - 모든 world 좌표는 robot base frame 기준 (sim 의 env_origin 적용된 좌표).
        sim 에선 env-rel xy = world xy - env_origin. real 에선 robot base 좌표계 그대로.
      - 모든 메서드는 sync (blocking). adapter 가 내부적으로 sim step 또는 ROS spin 처리.
    """

    @abstractmethod
    def get_top_cam(self) -> CamData:
        """천장 cam (RealSense D435) 1 frame."""
        ...

    @abstractmethod
    def get_wrist_cam(self) -> CamData:
        """손목 cam (RealSense D405, link6 부착) 1 frame. real 에선 ee 자세 따라 자동 변화."""
        ...

    @abstractmethod
    def get_ee_pose(self) -> EePose:
        """현재 ee (gripper center) world pose + velocity."""
        ...

    @abstractmethod
    def set_ee_target(self, target_pos: np.ndarray, target_quat: np.ndarray,
                      gripper_value: float) -> None:
        """IK + actuator 명령 한 번. adapter 가 내부적으로 step() 호출 안 함.

        Args:
            target_pos: (3,) world target xyz
            target_quat: (4,) wxyz world target orientation
            gripper_value: 0.0 (open) ~ 1.0 (close). sim 의 tip_ratio + real 의 gripper API
                          내부에서 처리. real 에선 normalized 0~1.
        """
        ...

    @abstractmethod
    def step(self, n: int = 1) -> None:
        """sim 의 sim.step() 또는 real 의 sleep(1/control_rate)."""
        ...

    @abstractmethod
    def reset_to_home(self) -> None:
        """robot home joint pose 로 reset. sim 은 write_joint_state, real 은 home pose 이동."""
        ...

    # ===== sim 전용 (real 에선 no-op 또는 None) =====

    def spawn_random_box(self) -> tuple[float, float, float]:
        """sim: random xy/yaw 으로 박스 spawn. real: no-op + 측정된 박스 pose 반환.

        Returns: (x_env, y_env, yaw_rad)
        """
        return (0.0, 0.0, 0.0)

    def spawn_random_cell(self) -> tuple[float, float, float]:
        """sim: random xy/yaw 으로 셀 wall 4개 spawn. real: no-op + 측정 셀 pose."""
        return (0.0, 0.0, 0.0)

    def get_box_gt(self) -> BoxGtPose | None:
        """sim 만 ground truth 반환. real 은 None."""
        return None

    def get_cell_gt(self) -> BoxGtPose | None:
        return None

    @abstractmethod
    def get_base_ee_quat(self) -> np.ndarray:
        """ee 가 정 아래 향하는 base orientation (4,) wxyz. yaw 0 일 때 사용."""
        ...

    @abstractmethod
    def get_home_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """home 자세에서의 ee pos/quat. (3,), (4,)."""
        ...

    @property
    @abstractmethod
    def control_dt(self) -> float:
        """제어 주기 (s). sim 1/60, real 자체 control loop rate."""
        ...
