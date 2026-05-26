"""Run a taught joint-waypoint pick/place sequence for OMY-F3M.

Default behavior is dry-run only. The script sends robot commands only when
both flags are provided:

    --execute --confirm EXECUTE_TEACH_PICK_PLACE

This runner does not use MoveIt pose goals or Cartesian execution. Arm motion is
sent directly to `/arm_controller/follow_joint_trajectory`; gripper motion is
sent to `/gripper_controller/gripper_cmd`.

전체 흐름:
    1. YAML에서 waypoint 목록을 읽는다 (record_teach_waypoint.py 로 기록한 값들)
    2. /joint_states 토픽에서 현재 로봇 관절 값을 한 번 읽는다
    3. 각 waypoint 간 최대 joint 변위가 safety limit을 넘지 않는지 검증한다
    4. --execute 없이 실행하면 dry-run으로 계획만 출력하고 종료한다
    5. --execute --confirm EXECUTE_TEACH_PICK_PLACE 로 실행하면 순서대로 전송한다
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Any

import yaml


# 실제 실행 시 사용자가 CLI에 직접 입력해야 하는 확인 문자열 (의도치 않은 실행 방지)
CONFIRM_TEXT = "EXECUTE_TEACH_PICK_PLACE"
# 기본 waypoint 설정 파일 경로 (repo 루트 기준 상대경로)
DEFAULT_CONFIG = "motion2/config/teach_pick_place_waypoints.yaml"
# 기본 실행 시퀀스: 파지(pre_grasp→grasp→close) → 이동(lift) → 내려놓기(place→open) → 복귀(retract)
DEFAULT_SEQUENCE = [
    "pre_grasp",
    "grasp",
    "close_gripper",
    "lift",
    "place",
    "open_gripper",
    "retract",
]
# OMY-F3M 6축 암 관절 이름 (YAML과 반드시 일치해야 함)
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
# 그리퍼 관절 이름 (control_msgs/GripperCommand 에서 사용)
GRIPPER_JOINT = "rh_r1_joint"


@dataclass(frozen=True)
class ArmStep:
    """암 관절 이동 단계 하나를 표현한다.

    positions: ARM_JOINTS 순서에 맞춘 목표 관절 각도(rad) 리스트
    max_delta_from_previous: 이전 step과의 최대 관절 변위(rad) — safety check 결과값
    ee_pose: 기록 시점의 end-effector 자세 (선택, 참고용 메타데이터)
    """
    name: str
    positions: list[float]
    max_delta_from_previous: float
    ee_pose: dict[str, Any] | None = None


@dataclass(frozen=True)
class GripperStep:
    """그리퍼 열기/닫기 단계 하나를 표현한다.

    position: 목표 그리퍼 위치 (0.0 = 완전히 열림, ~1.12 = 완전히 닫힘)
    """
    name: str
    position: float


def _repo_root() -> pathlib.Path:
    """이 파일 위치로부터 두 단계 위(repo 루트)를 반환한다.

    scripts/ → motion2/ → repo_root/ 구조에 의존한다.
    """
    return pathlib.Path(__file__).resolve().parents[2]


def _resolve_repo_path(root: pathlib.Path, path_text: str) -> pathlib.Path:
    """절대 경로면 그대로, 상대 경로면 repo 루트 기준으로 해석한다."""
    path = pathlib.Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    """YAML 파일을 읽어 dict 로 반환한다. 최상위가 mapping이 아니면 예외."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _joint_state_once(node, rclpy, timeout_s: float) -> dict[str, float]:
    """/joint_states 토픽에서 첫 번째 메시지를 받아 {joint_name: position} 딕셔너리로 반환한다.

    dry-run/실행 모두 사용: 현재 로봇 자세 기준으로 첫 번째 waypoint까지의
    joint delta를 검증하기 위해 반드시 필요하다.
    timeout_s 초 안에 메시지가 오지 않으면 RuntimeError를 발생시킨다.
    """
    from sensor_msgs.msg import JointState
    import time

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
                return dict(zip(msg.name, msg.position))
    finally:
        node.destroy_subscription(sub)
    raise RuntimeError("Timed out waiting for /joint_states")


def _make_arm_goal(joint_names: list[str], positions: list[float], duration_s: float):
    """FollowJointTrajectory.Goal 객체를 생성한다.

    단일 trajectory point를 사용한다: 로봇이 duration_s 초 안에 positions에 도달하도록
    요청하고, 목표 속도는 0으로 설정하여 종점에서 완전히 정지하게 한다.
    """
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(joint_names)
    point = JointTrajectoryPoint()
    point.positions = [float(v) for v in positions]
    point.velocities = [0.0 for _ in positions]  # 종점에서 정지
    # ROS2 Duration은 sec + nanosec 두 필드로 분리하여 설정한다
    point.time_from_start.sec = int(duration_s)
    point.time_from_start.nanosec = int((duration_s - int(duration_s)) * 1_000_000_000)
    goal.trajectory.points.append(point)
    return goal


def _make_gripper_goal(position: float, max_effort: float):
    """GripperCommand.Goal 객체를 생성한다.

    position: 목표 그리퍼 위치 (m 단위, OMY-F3M 기준 0.0~1.12)
    max_effort: 최대 토크 제한 (0.0 = 컨트롤러 기본값 사용)
    """
    from control_msgs.action import GripperCommand

    goal = GripperCommand.Goal()
    goal.command.position = float(position)
    goal.command.max_effort = float(max_effort)
    return goal


def _as_float_map(value: Any, label: str) -> dict[str, float]:
    """YAML에서 읽은 값을 {str: float} 딕셔너리로 변환한다. mapping이 아니면 예외."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return {str(k): float(v) for k, v in value.items()}


def _parse_sequence(sequence_text: str | None, config_sequence: Any) -> list[str]:
    """실행할 step 이름 목록을 결정한다.

    우선순위:
        1. CLI --sequence 인자 (쉼표 구분)
        2. YAML sequence 필드
        3. DEFAULT_SEQUENCE 상수
    """
    if sequence_text:
        names = [part.strip() for part in sequence_text.split(",") if part.strip()]
    elif isinstance(config_sequence, list) and config_sequence:
        names = [str(name) for name in config_sequence]
    else:
        names = list(DEFAULT_SEQUENCE)
    if not names:
        raise ValueError("sequence is empty")
    return names


def _validate_gripper_position(name: str, position: float, min_position: float, max_position: float) -> None:
    """그리퍼 목표 위치가 safety guard 범위 [min, max] 안에 있는지 검증한다."""
    if not (min_position <= position <= max_position):
        raise ValueError(
            f"{name}: gripper target {position:.6f} outside "
            f"[{min_position:.6f}, {max_position:.6f}]")


def _build_steps(
    data: dict[str, Any],
    sequence: list[str],
    arm_joints: list[str],
    current_arm: list[float],
    max_joint_delta: float,
    min_gripper_position: float,
    max_gripper_position: float,
    close_gripper: float | None,
    open_gripper: float | None,
) -> list[ArmStep | GripperStep]:
    """YAML 데이터와 현재 로봇 자세를 기반으로 실행 step 리스트를 생성한다.

    각 arm waypoint마다:
      - YAML에서 joint 값을 읽어 ARM_JOINTS 순서로 정렬한다
      - 이전 step(또는 현재 자세)과의 최대 joint 변위를 계산한다
      - max_joint_delta를 초과하면 즉시 ValueError를 발생시킨다 (실행 전 검증)

    close_gripper / open_gripper 이름은 특별 처리: waypoint lookup 없이
    gripper_targets 설정값(또는 CLI 오버라이드)으로 GripperStep을 생성한다.
    """
    waypoints = data.get("waypoints", {})
    if not isinstance(waypoints, dict):
        raise ValueError("waypoints must be a mapping")
    gripper_targets = data.get("gripper_targets", {})
    if not isinstance(gripper_targets, dict):
        gripper_targets = {}

    # CLI 오버라이드 → YAML gripper_targets → 하드코딩 기본값 순으로 fallback
    close_value = float(close_gripper) if close_gripper is not None else float(gripper_targets.get("close_gripper", 0.60))
    open_value = float(open_gripper) if open_gripper is not None else float(gripper_targets.get("open_gripper", 0.00))
    _validate_gripper_position("close_gripper", close_value, min_gripper_position, max_gripper_position)
    _validate_gripper_position("open_gripper", open_value, min_gripper_position, max_gripper_position)

    previous_arm = list(current_arm)  # 첫 번째 step의 delta 계산 기준은 현재 로봇 자세
    steps: list[ArmStep | GripperStep] = []
    for name in sequence:
        # 그리퍼 특수 step: waypoint가 아닌 gripper_targets 값을 사용
        if name in ("close_gripper", "open_gripper"):
            steps.append(GripperStep(name=name, position=close_value if name == "close_gripper" else open_value))
            continue

        waypoint = waypoints.get(name)
        if not isinstance(waypoint, dict):
            available = sorted(str(key) for key in waypoints)
            raise ValueError(f"missing arm waypoint {name!r}; available={available}")
        arm = _as_float_map(waypoint.get("arm"), f"waypoints.{name}.arm")
        missing = [joint for joint in arm_joints if joint not in arm]
        if missing:
            raise ValueError(f"waypoints.{name}.arm missing joints: {missing}")
        positions = [float(arm[joint]) for joint in arm_joints]

        # 각 관절의 변위를 계산하여 최대값이 safety limit을 초과하는지 확인
        deltas = [abs(target - start) for target, start in zip(positions, previous_arm)]
        max_delta = max(deltas) if deltas else 0.0
        if max_delta > max_joint_delta:
            max_joint = arm_joints[deltas.index(max_delta)]
            raise ValueError(
                f"{name}: max joint delta {max_delta:.6f} rad at {max_joint} "
                f"exceeds limit {max_joint_delta:.6f} rad")
        # ee_pose는 기록 당시의 참고 자세 — 실행에는 영향 없고 로그 출력에만 사용
        ee_pose = waypoint.get("ee_pose")
        if ee_pose is not None and not isinstance(ee_pose, dict):
            raise ValueError(f"waypoints.{name}.ee_pose must be a mapping when present")
        steps.append(ArmStep(
            name=name,
            positions=positions,
            max_delta_from_previous=max_delta,
            ee_pose=ee_pose,
        ))
        previous_arm = positions  # 다음 step의 delta 계산 기준을 현재 목표값으로 갱신
    return steps


def _print_steps(
    steps: list[ArmStep | GripperStep],
    arm_joints: list[str],
    arm_action: str,
    gripper_action: str,
    arm_duration: float,
    gripper_max_effort: float,
    execute: bool,
) -> None:
    """전체 실행 계획을 터미널에 출력한다.

    dry-run에서는 이 출력만 하고 종료한다.
    실행 모드에서는 이 출력 후 실제 action goal을 전송한다.
    """
    print(f"[teach-replay] execute: {bool(execute)}")
    print(f"[teach-replay] arm action: {arm_action}")
    print(f"[teach-replay] gripper action: {gripper_action}")
    print(f"[teach-replay] arm duration: {arm_duration:.2f}s")
    print(f"[teach-replay] gripper max_effort: {gripper_max_effort:.6f}")
    for idx, step in enumerate(steps, start=1):
        if isinstance(step, ArmStep):
            values = ", ".join(
                f"{joint}={value:.6f}" for joint, value in zip(arm_joints, step.positions))
            print(
                f"[teach-replay] step {idx:02d} arm {step.name}: "
                f"max_delta={step.max_delta_from_previous:.6f} {values}")
            if step.ee_pose:
                pos = step.ee_pose.get("position", {})
                quat = step.ee_pose.get("quat_wxyz", {})
                print(
                    f"[teach-replay] step {idx:02d} ee_pose {step.name}: "
                    f"{step.ee_pose.get('frame_id')} -> {step.ee_pose.get('link_name')} "
                    f"pos=[{float(pos.get('x', 0.0)):.6f}, "
                    f"{float(pos.get('y', 0.0)):.6f}, "
                    f"{float(pos.get('z', 0.0)):.6f}] "
                    f"quat_wxyz=[{float(quat.get('w', 0.0)):.6f}, "
                    f"{float(quat.get('x', 0.0)):.6f}, "
                    f"{float(quat.get('y', 0.0)):.6f}, "
                    f"{float(quat.get('z', 0.0)):.6f}]")
        else:
            print(
                f"[teach-replay] step {idx:02d} gripper {step.name}: "
                f"position={step.position:.6f}")


def _send_arm_goal(node, rclpy, client, arm_joints: list[str], step: ArmStep, duration: float) -> int:
    """arm_action 서버로 FollowJointTrajectory goal을 전송하고 결과를 기다린다.

    반환값: 0 = 성공, 2 = 실패(goal rejected 또는 error_code != 0)
    error_code 0이 아닌 경우 FollowJointTrajectory 표준 에러 코드를 나타낸다.
    """
    goal = _make_arm_goal(arm_joints, step.positions, duration)
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    handle = send_future.result()
    if handle is None or not handle.accepted:
        print(f"[teach-replay] arm goal rejected: {step.name}")
        return 2

    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    code = int(result.error_code)
    if code == 0:
        print(f"[teach-replay] arm success: {step.name}")
        return 0
    print(f"[teach-replay] arm failed: {step.name} error_code={code}")
    return 2


def _send_gripper_goal(node, rclpy, client, step: GripperStep, max_effort: float) -> int:
    """gripper_action 서버로 GripperCommand goal을 전송하고 결과를 기다린다.

    reached_goal=True 여야 성공으로 간주한다.
    stalled=True 이면 장애물에 걸려 정지한 것으로, 파지 중 발생할 수 있다.
    반환값: 0 = reached_goal, 2 = not reached_goal
    """
    goal = _make_gripper_goal(step.position, max_effort)
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    handle = send_future.result()
    if handle is None or not handle.accepted:
        print(f"[teach-replay] gripper goal rejected: {step.name}")
        return 2

    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    print(
        f"[teach-replay] gripper result: {step.name} "
        f"position={result.position:.6f} effort={result.effort:.6f} "
        f"stalled={result.stalled} reached_goal={result.reached_goal}")
    return 0 if result.reached_goal else 2


def _confirm_step(step_name: str) -> None:
    """각 step 실행 전 사용자가 step 이름을 직접 타이핑하도록 요구한다.

    --no-step-prompts 가 없는 한 기본적으로 활성화된다.
    의도치 않은 연속 실행을 방지하기 위한 이중 안전장치다.
    """
    typed = input(f"[teach-replay] type {step_name} to execute this step: ").strip()
    if typed != step_name:
        raise RuntimeError(f"Refusing to execute step {step_name!r}; typed {typed!r}")


def main() -> int:
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory, GripperCommand

    parser = argparse.ArgumentParser(description="Dry-run or execute taught OMY-F3M pick/place waypoints.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--sequence", default=None, help="Comma-separated step names. Defaults to YAML sequence.")
    parser.add_argument("--arm-action", default=None)
    parser.add_argument("--gripper-action", default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--max-joint-delta", type=float, default=None)
    parser.add_argument("--close-gripper", type=float, default=None)
    parser.add_argument("--open-gripper", type=float, default=None)
    parser.add_argument("--min-gripper-position", type=float, default=None)
    parser.add_argument("--max-gripper-position", type=float, default=None)
    parser.add_argument("--gripper-max-effort", type=float, default=None)
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--plan-only", action="store_true", help="Alias for default dry-run validation.")
    parser.add_argument(
        "--no-step-prompts",
        action="store_true",
        help="Do not ask for per-step typed confirmation after --confirm. Not recommended.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    # 실행 의도를 명확히 요구: --execute 단독으로는 실행 불가
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(f"Refusing to execute. Re-run with --confirm {CONFIRM_TEXT}")
    if args.joint_state_timeout <= 0.0:
        raise ValueError("--joint-state-timeout must be > 0")

    root = _repo_root()
    data = _load_yaml(_resolve_repo_path(root, args.config))
    controller = data.get("controller", {})
    safety = data.get("safety", {})
    arm_joints = [str(name) for name in data.get("arm_joints", ARM_JOINTS)]
    # OMY-F3M 전용: 다른 로봇 모델과 혼용 방지를 위해 관절 이름을 엄격히 검증
    if arm_joints != ARM_JOINTS:
        raise ValueError(f"Unexpected arm_joints for OMY-F3M: {arm_joints}")

    # CLI 인자 > YAML safety 섹션 > 하드코딩 기본값 순으로 파라미터를 결정
    arm_action = args.arm_action or str(controller.get("arm_action", "/arm_controller/follow_joint_trajectory"))
    gripper_action = args.gripper_action or str(controller.get("gripper_action", "/gripper_controller/gripper_cmd"))
    arm_duration = float(args.duration if args.duration is not None else safety.get("default_arm_duration", 5.0))
    max_joint_delta = float(args.max_joint_delta if args.max_joint_delta is not None else safety.get("max_joint_delta_per_step", 0.35))
    min_gripper_position = float(args.min_gripper_position if args.min_gripper_position is not None else safety.get("min_gripper_position", 0.0))
    max_gripper_position = float(args.max_gripper_position if args.max_gripper_position is not None else safety.get("max_gripper_position", 1.12))
    gripper_max_effort = float(args.gripper_max_effort if args.gripper_max_effort is not None else safety.get("default_gripper_max_effort", 0.0))

    # teach replay는 관절 이동이 크므로 최소 4초 이상 여유를 둬야 안전하다
    if arm_duration < 4.0:
        raise ValueError("--duration must be >= 4.0 s for teach replay")
    if max_joint_delta <= 0.0:
        raise ValueError("--max-joint-delta must be > 0")
    # OMY-F3M 그리퍼 하드웨어 한계: 0.0 ~ 1.12 m
    if min_gripper_position < 0.0 or max_gripper_position > 1.12 or min_gripper_position >= max_gripper_position:
        raise ValueError("gripper guard must stay within [0.0, 1.12] with min < max")

    sequence = _parse_sequence(args.sequence, data.get("sequence"))

    rclpy.init(args=None)
    node = Node("motion2_run_teach_pick_place")
    arm_client = ActionClient(node, FollowJointTrajectory, arm_action)
    gripper_client = ActionClient(node, GripperCommand, gripper_action)
    try:
        # 현재 로봇 자세 취득 — dry-run에서도 반드시 필요 (첫 waypoint delta 검증용)
        joints = _joint_state_once(node, rclpy, args.joint_state_timeout)
        missing = [name for name in arm_joints if name not in joints]
        if missing:
            raise RuntimeError(f"Missing arm joints in /joint_states: {missing}")
        current_arm = [float(joints[name]) for name in arm_joints]

        # step 리스트 생성 & 모든 safety guard 검증 (실행 전 전체 경로 검사)
        steps = _build_steps(
            data=data,
            sequence=sequence,
            arm_joints=arm_joints,
            current_arm=current_arm,
            max_joint_delta=max_joint_delta,
            min_gripper_position=min_gripper_position,
            max_gripper_position=max_gripper_position,
            close_gripper=args.close_gripper,
            open_gripper=args.open_gripper,
        )
        _print_steps(
            steps,
            arm_joints,
            arm_action,
            gripper_action,
            arm_duration,
            gripper_max_effort,
            args.execute,
        )

        # dry-run: 계획 출력 후 여기서 종료 (실제 명령 전송 없음)
        if not args.execute:
            print("[teach-replay] dry-run complete; command_sent=false")
            return 0

        # action server 준비 대기 (타임아웃 10초)
        node.get_logger().info(f"Waiting for arm action {arm_action}")
        if not arm_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Action server not available: {arm_action}")
        node.get_logger().info(f"Waiting for gripper action {gripper_action}")
        if not gripper_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"Action server not available: {gripper_action}")

        # 각 step 순서대로 실행; 하나라도 실패하면 즉시 중단 (부분 실행 방지)
        for idx, step in enumerate(steps, start=1):
            print(f"[teach-replay] executing step {idx}/{len(steps)}: {step.name}")
            if not args.no_step_prompts:
                _confirm_step(step.name)  # 사용자 타이핑 확인 (이중 안전장치)
            if isinstance(step, ArmStep):
                rc = _send_arm_goal(node, rclpy, arm_client, arm_joints, step, arm_duration)
            else:
                rc = _send_gripper_goal(node, rclpy, gripper_client, step, gripper_max_effort)
            if rc != 0:
                print(f"[teach-replay] stopping after failed step: {step.name}")
                return rc

        print("[teach-replay] SUCCESS command_sent=true")
        return 0
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
