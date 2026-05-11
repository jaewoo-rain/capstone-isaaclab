"""motion2 — Real adapter (RealSense + OMY ROS).

== 친구 (real robot 담당자) 가 작성하는 파일 ==

BaseAdapter 의 메서드들을 RealSense D435/D405 + OMY ROS 토픽 으로 구현.
ChainStateMachine 는 변경 X — 이 파일만 채우면 sim 검증된 흐름 그대로 real 에서 동작.

== 필수 구현 메서드 ==
1. get_top_cam() / get_wrist_cam()
   - RealSense pyrealsense2 SDK 로 RGB + depth + intrinsic + extrinsic 캡처
   - extrinsic: 카메라 mount 의 robot base 좌표계 기준 pose
2. get_ee_pose()
   - OMY proprioception (joint state → forward kinematics) 또는 tf2 lookup
3. set_ee_target(pos, quat, gripper_value)
   - OMY IK solver (MoveIt / OMY SDK) 로 joint target 계산 → publish
   - gripper_value: 0.0~1.0 normalized
4. step(n)
   - real 의 control loop rate 따라 sleep (예: 1/60 s)
5. reset_to_home()
   - 정의된 home joint pose 로 robot 이동 (blocking)
6. get_base_ee_quat() / get_home_ee_pose()
   - hardcoded constants (sim 과 동일)
7. control_dt
   - 1/60 권장 (sim 학습 정책의 dt 와 일치)

== 선택 구현 (sim 만 의미) ==
- spawn_random_box() / spawn_random_cell() — real 에선 user 가 박스/셀 수동 배치. 호출 시 no-op 또는 측정값 반환
- get_box_gt() / get_cell_gt() — real 에선 None 그대로 (실제 측정 = vision pipeline 결과)

== RealSense 카메라 mount ==
- 천장 cam (D435): workspace 위 ~80cm. robot base 기준 fixed pose.
- 손목 cam (D405): OMY link6 끝에 부착. link6 의 -Y 방향 (gripper 향) 으로 camera 의 forward.
  link6 frame 기준 offset: pos=(0.0, -0.1, 0.084), rot=(0,0,0.7071,-0.7071) [wxyz, ROS convention]
- 매 frame 손목 cam 의 world pose 는 robot proprioception 에서 link6 pose 읽고 offset 적용해서 계산

== Coordinate convention ==
- 모든 pose 는 robot base frame 기준 (sim env_origin 적용된 world 와 동일)
- quat = wxyz
- ee = gripper finger center (양 finger 의 world 좌표 평균)

== 주의 ==
- ChainStateMachine 의 RL 정책은 sim 의 control_dt = 1/60 (60 Hz) 으로 학습됨.
  real 에서 같은 rate 로 호출해야 정책 분포 일치.
- gripper_value 의 의미는 sim 의 tip_ratio=2.3 같은 mechanism. real OMY 의 gripper API
  가 0~1 받는다면 그대로 사용. 다른 형식이면 adapter 안에서 변환.

== 학습된 모델 (그대로 사용) ==
- YOLO seg: motion2/models/yolo11n_seg_v2_box_cell.pt (class 0=box, 1=cell)
- Grasp policy: motion2/models/motion1_grasp.zip + _vecnorm.pkl
"""
from __future__ import annotations

import math
import numpy as np

from .base_adapter import BaseAdapter, CamData, EePose, BoxGtPose


class RealAdapter(BaseAdapter):
    """RealSense D435/D405 + OMY ROS 기반 adapter — 친구가 채워 넣음.

    아래 메서드들을 자신의 robot/cam SDK 호출로 채우면 됨. 다른 파일 (chain_state_machine.py)
    은 건드릴 필요 없음.
    """

    def __init__(self, control_dt: float = 1.0 / 60.0,
                 home_joint_pos: dict | None = None):
        self._control_dt = control_dt
        # === 친구가 추가할 init ===
        #   - rs.pipeline 시작 (천장 + 손목 cam)
        #   - rospy.init_node() / ros2 init
        #   - OMY joint state subscriber
        #   - OMY IK / motion service client
        #   - gripper publisher
        raise NotImplementedError(
            "RealAdapter 구현 필요. base_adapter.py 의 메서드 7개 구현 후 NotImplementedError 제거.")

    # ===== 필수 구현 =====
    def get_top_cam(self) -> CamData:
        """RealSense D435 (천장) RGB + depth + intrinsic + extrinsic.

        Steps:
          1. pipeline.wait_for_frames() 으로 RGB + depth frame 가져오기
          2. depth → numpy float32 (meters, NaN/inf 처리)
          3. intrinsic (rs.intrinsics) → K (3,3)
          4. extrinsic: robot base 기준 카메라 pose
             - 사전 calibration (eye-to-base) 결과 사용
             - pos_w: (3,) world position
             - quat_w_world: (4,) wxyz, world convention (forward=+X, up=+Z)
        """
        raise NotImplementedError("get_top_cam() 구현 필요")

    def get_wrist_cam(self) -> CamData:
        """RealSense D405 (손목, link6 부착) RGB + depth + intrinsic + extrinsic.

        extrinsic 계산:
          link6_pose_w = get_link6_world_pose()  # tf2 또는 OMY FK
          cam_pos_w   = link6_pose_w.translation + link6_rotation @ (0.0, -0.1, 0.084)
          cam_quat_w  = link6_quat * (0, 0, 0.7071, -0.7071)  # ROS convention
        """
        raise NotImplementedError("get_wrist_cam() 구현 필요")

    def get_ee_pose(self) -> EePose:
        """현재 ee (gripper finger center) world pose + velocity.

        - pos_w: forward kinematics 결과 (양 finger 위치 평균) 또는 tf2 lookup
        - quat_w: ee orientation
        - lin_vel: ee 의 world linear velocity. joint vel + jacobian 으로 계산
        - ang_vel_z: ee 의 world z-axis angular velocity
        """
        raise NotImplementedError("get_ee_pose() 구현 필요")

    def set_ee_target(self, target_pos: np.ndarray, target_quat: np.ndarray,
                      gripper_value: float) -> None:
        """OMY IK 로 joint target 계산 + publish.

        Args:
            target_pos: (3,) world target xyz
            target_quat: (4,) wxyz world target orientation
            gripper_value: 0.0 (open) ~ 1.0 (close). OMY gripper API 형식으로 변환.

        Steps:
          1. IK solve (MoveIt / OMY SDK)
          2. joint target publish (control_dt 안에서 도달 가능한 increment)
          3. gripper publish
        주의: blocking 아님. step() 가 실제로 control loop 진행.
        """
        raise NotImplementedError("set_ee_target() 구현 필요")

    def step(self, n: int = 1) -> None:
        """real 의 control loop n step 진행 (각 step = control_dt).

        구현:
          - ROS 면 rospy.sleep(control_dt) 또는 rate.sleep()
          - rate = 1 / control_dt (보통 60 Hz)
        """
        raise NotImplementedError("step() 구현 필요")

    def reset_to_home(self) -> None:
        """robot 을 home joint pose 로 이동 (blocking until complete).

        home joint pose (sim 과 동일, OMY OpenManipulator 기준):
            joint1=0.0, joint2=-1.55, joint3=2.66, joint4=-1.1, joint5=1.6, joint6=0.0
            (gripper 4 joints 는 0 = open)
        """
        raise NotImplementedError("reset_to_home() 구현 필요")

    def get_base_ee_quat(self) -> np.ndarray:
        """ee 가 정 아래 향하는 base orientation (sim 과 동일)."""
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    def get_home_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """home 자세에서의 ee pos/quat.

        reset_to_home() 호출 후 get_ee_pose() 결과를 캐싱해서 반환해도 됨.
        또는 sim 에서 측정된 값 hardcode: pos=(0.136, -0.108, 0.388), quat=(0.7071, 0, 0.7071, 0)
        """
        raise NotImplementedError("get_home_ee_pose() 구현 필요")

    @property
    def control_dt(self) -> float:
        return self._control_dt

    # ===== real 에선 no-op (사용자 박스/셀 수동 배치) =====
    def spawn_random_box(self) -> tuple[float, float, float]:
        """real 에선 박스를 수동 배치. 호출 시 (0, 0, 0) 반환 — chain_state_machine 가
        그 다음 천장 cam scan 으로 실제 위치 측정."""
        return (0.0, 0.0, 0.0)

    def spawn_random_cell(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def get_box_gt(self) -> BoxGtPose | None:
        return None  # real 에선 GT 없음

    def get_cell_gt(self) -> BoxGtPose | None:
        return None
