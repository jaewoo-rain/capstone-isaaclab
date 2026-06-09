"""Move OMY-F3M end-effector to a single Cartesian target pose.

OMY-F3M 파지 특성:
  - 그리퍼가 수직 방향으로 물체를 측면 파지 (top-down이 아님)
  - joint5 ≈ π/2 를 유지해야 수직 파지 orientation이 유지됨
  - 기본 EE 방향: quat_wxyz ≈ [0.506, 0.494, 0.494, 0.506]
    (EE z-axis → global +x: 앞쪽에서 수평으로 접근)
  - YAML waypoints '1','2' 가 이 orientation에서의 최대 도달 거리 기준

안전 설계 (3단계):
  1. 목표 위치가 workspace 박스 안인지 확인
  2. MoveIt plan 내부 delta 검증 (plan start→goal, segment 간)
  3. 현재 실제 관절값 → plan 목표 관절값 delta 검증

기본 동작은 dry-run. 실제 실행:
    --execute --confirm EXECUTE_EE_POSE_MOVE

Orientation 지정 방법 (기본값: VERTICAL_GRIP_QUAT):
    --rpy ROLL PITCH YAW    ZYX 오일러각 [rad]
    --quat W X Y Z          쿼터니언 wxyz (자동 정규화)
    --use-current-ori       현재 EE orientation 사용 (수직 파지 아님)

사용 예시:
    # dry-run (기본값: 수직 파지 orientation + joint5 constraint)
    python3 motion2/scripts/run_ee_pose_move.py --x 0.40 --y -0.11 --z 0.37

    # joint5 constraint 없이 (orientation만 지정)
    python3 motion2/scripts/run_ee_pose_move.py --x 0.40 --y -0.11 --z 0.37 \\
        --no-constrain-joint5

    # 실제 실행
    python3 motion2/scripts/run_ee_pose_move.py --x 0.40 --y -0.11 --z 0.37 \\
        --execute --confirm EXECUTE_EE_POSE_MOVE
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Any

import numpy as np


# ── 실행 안전 장치 ─────────────────────────────────────────────────────────
CONFIRM_TEXT = "EXECUTE_EE_POSE_MOVE"

# ── OMY-F3M 관절 정의 ──────────────────────────────────────────────────────
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# ── 수직 파지 상수 ──────────────────────────────────────────────────────────
# YAML waypoints '1','2' 에서 측정한 수직 파지 EE orientation
# 물리적 의미: EE z-axis → global +x (앞쪽에서 수평 접근, 그리퍼 finger는 수직)
_VERTICAL_GRIP_QUAT_RAW = np.array([0.5063, 0.4936, 0.4936, 0.5063], dtype=np.float64)
VERTICAL_GRIP_QUAT = _VERTICAL_GRIP_QUAT_RAW / np.linalg.norm(_VERTICAL_GRIP_QUAT_RAW)

# joint5 = π/2: 이 값이 유지되어야 수직 파지 orientation 고정
JOINT5_VERTICAL_POSITION = math.pi / 2   # ≈ 1.5708 rad
# path constraint 허용 오차: YAML 측정값이 모두 1.5707~1.5708 범위이므로 ±0.08 rad 여유
JOINT5_TOLERANCE = 0.08                  # ≈ ±4.6°

# ── ROS2 action 기본값 ─────────────────────────────────────────────────────
DEFAULT_ARM_ACTION = "/arm_controller/follow_joint_trajectory"
DEFAULT_MOVE_GROUP_ACTION = "/move_action"
DEFAULT_GROUP_NAME = "arm"
DEFAULT_BASE_FRAME = "link0"
DEFAULT_EE_FRAME = "link6"

# ── 작업 공간 기본값 ───────────────────────────────────────────────────────
# x_max=0.55: YAML '1','2' 위치(x≈0.496)를 커버하도록 여유 확보
DEFAULT_WORKSPACE = {
    "x_min": -0.10, "x_max": 0.55,
    "y_min": -0.45, "y_max": 0.20,
    "z_min": 0.10,  "z_max": 0.50,
}

# MoveIt 성공 코드 (moveit_msgs/MoveItErrorCodes)
_MOVEIT_SUCCESS = 1


def _scripts_dir() -> str:
    return str(pathlib.Path(__file__).resolve().parent.parent)


def _rpy_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX 오일러각(rad) → 쿼터니언 [w, x, y, z] 변환 (ROS 표준)."""
    cr, cp, cy = math.cos(roll / 2), math.cos(pitch / 2), math.cos(yaw / 2)
    sr, sp, sy = math.sin(roll / 2), math.sin(pitch / 2), math.sin(yaw / 2)
    return np.array([
        cr * cp * cy + sr * sp * sy,  # w
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,  # z
    ], dtype=np.float64)


def _make_arm_goal(joint_names: list[str], positions: list[float], duration_s: float):
    """단일 trajectory point를 담은 FollowJointTrajectory.Goal 생성."""
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(joint_names)
    point = JointTrajectoryPoint()
    point.positions = [float(v) for v in positions]
    point.velocities = [0.0 for _ in positions]  # 목표에서 완전 정지
    point.time_from_start.sec = int(duration_s)
    point.time_from_start.nanosec = int((duration_s - int(duration_s)) * 1_000_000_000)
    goal.trajectory.points.append(point)
    return goal


def _send_arm_goal(
    node, rclpy, client,
    joint_names: list[str], positions: list[float], duration: float,
) -> int:
    """FollowJointTrajectory goal 전송 및 결과 대기. 0=성공, 2=실패."""
    goal = _make_arm_goal(joint_names, positions, duration)
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    handle = send_future.result()
    if handle is None or not handle.accepted:
        print("[ee-pose-move] arm goal rejected by controller")
        return 2

    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    code = int(result.error_code)
    if code == 0:
        print("[ee-pose-move] arm success")
        return 0
    print(f"[ee-pose-move] arm failed error_code={code}")
    return 2


def _plan_with_constraints(
    *,
    node,
    rclpy,
    client,
    frame_id: str,
    group_name: str,
    link_name: str,
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    workspace: dict[str, float],
    planning_time: float,
    attempts: int,
    pos_tol: float,
    ori_tol: float,
    velocity_scale: float,
    acceleration_scale: float,
    keep_orientation: bool,
    planner_id: str,
    max_joint_delta: float,
    max_segment_delta: float,
    path_joint_constraints: list[dict[str, Any]],
    start_joint_names: list[str] | None = None,
    start_joint_positions: list[float] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """MoveIt plan_only=True + path joint constraints 지원 버전.

    real_moveit_common.plan_or_execute_pose는 path constraint를 지원하지 않으므로
    make_move_group_goal → constraint 추가 → 직접 전송 순서로 처리한다.

    path_joint_constraints 형식:
        [{"name": "joint5", "position": 1.5708, "tol": 0.08, "weight": 1.0}]
    """
    from real_moveit_common import make_move_group_goal, trajectory_delta_report
    from moveit_msgs.msg import Constraints, JointConstraint

    goal = make_move_group_goal(
        frame_id=frame_id,
        group_name=group_name,
        link_name=link_name,
        pos=pos,
        quat_wxyz=quat_wxyz,
        workspace=workspace,
        planning_time=planning_time,
        attempts=attempts,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
        velocity_scale=velocity_scale,
        acceleration_scale=acceleration_scale,
        plan_only=True,           # MoveIt은 계획만 — 절대 실행하지 않는다
        keep_orientation=keep_orientation,
        planner_id=planner_id,
    )

    # move_group robot state monitor가 비활성일 수 있으므로 is_diff 대신 현재 joint 값을 직접 주입
    if start_joint_names and start_joint_positions:
        from sensor_msgs.msg import JointState
        goal.request.start_state.is_diff = False
        js = JointState()
        js.header.stamp = node.get_clock().now().to_msg()
        js.name = list(start_joint_names)
        js.position = [float(p) for p in start_joint_positions]
        goal.request.start_state.joint_state = js

    # path constraint 추가: 경로 전체에서 해당 관절 값을 유지하도록 요청
    if path_joint_constraints:
        path = Constraints()
        for jc in path_joint_constraints:
            c = JointConstraint()
            c.joint_name = str(jc["name"])
            c.position = float(jc["position"])
            c.tolerance_above = float(jc.get("tol", 0.08))
            c.tolerance_below = float(jc.get("tol", 0.08))
            c.weight = float(jc.get("weight", 1.0))
            path.joint_constraints.append(c)
        goal.request.path_constraints = path

    # goal 전송 및 결과 대기
    send_future = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_future)
    handle = send_future.result()
    if handle is None or not handle.accepted:
        empty: dict[str, Any] = {
            "error": "goal rejected", "error_code": -1, "guard_ok": False,
            "joint_names": [], "goal_positions": [], "start_positions": [],
            "point_count": 0, "max_start_to_goal_delta": 0.0,
            "max_segment_delta": 0.0, "max_joint": "",
        }
        return False, empty

    result_future = handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result
    code = int(result.error_code.val)

    report = trajectory_delta_report(result.planned_trajectory)
    report["error_code"] = code
    report["guard_ok"] = (
        float(report["max_start_to_goal_delta"]) <= max_joint_delta
        and float(report["max_segment_delta"]) <= max_segment_delta
    )
    ok = (code == _MOVEIT_SUCCESS) and bool(report["guard_ok"])
    return ok, report


def _extract_goal_joints(
    report: dict[str, Any],
    arm_joints: list[str],
) -> np.ndarray:
    """MoveIt plan report에서 ARM_JOINTS 순서로 목표 관절값을 추출한다.

    MoveIt이 반환하는 joint 순서가 ARM_JOINTS와 다를 수 있으므로 이름 기준으로 정렬.
    """
    plan_names: list[str] = report["joint_names"]
    plan_positions: list[float] = report["goal_positions"]
    name_to_pos = dict(zip(plan_names, plan_positions))
    missing = [j for j in arm_joints if j not in name_to_pos]
    if missing:
        raise RuntimeError(
            f"MoveIt plan missing arm joints: {missing}. "
            f"Plan contained: {plan_names}")
    return np.array([name_to_pos[j] for j in arm_joints], dtype=np.float64)


def main() -> int:
    scripts = _scripts_dir()
    if scripts not in sys.path:
        sys.path.insert(0, scripts)

    from real_moveit_common import assert_in_workspace, joint_state_once, lookup_current_pose

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory
    from moveit_msgs.action import MoveGroup

    parser = argparse.ArgumentParser(
        description=(
            "OMY-F3M EE를 목표 Cartesian 위치로 이동. "
            "기본값으로 수직 파지 orientation + joint5=π/2 path constraint 적용. "
            "MoveIt plan_only=True → FollowJointTrajectory 실행."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 목표 위치 (필수) ─────────────────────────────────────────────────────
    parser.add_argument("--x", type=float, required=True,
                        help="목표 EE x [m] (link0 기준)")
    parser.add_argument("--y", type=float, required=True,
                        help="목표 EE y [m] (link0 기준)")
    parser.add_argument("--z", type=float, required=True,
                        help="목표 EE z [m] (link0 기준)")

    # ── 목표 방향 (기본값: VERTICAL_GRIP_QUAT) ──────────────────────────────
    ori_group = parser.add_mutually_exclusive_group()
    ori_group.add_argument(
        "--rpy", nargs=3, type=float, metavar=("R", "P", "Y"),
        help="EE 방향: ZYX 오일러각 [rad]. 미지정 시 수직 파지 방향 사용.")
    ori_group.add_argument(
        "--quat", nargs=4, type=float, metavar=("W", "X", "Y", "Z"),
        help="EE 방향: 쿼터니언 wxyz. 미지정 시 수직 파지 방향 사용.")
    ori_group.add_argument(
        "--use-current-ori", action="store_true",
        help="TF2로 읽은 현재 EE orientation 사용 (수직 파지 기본값 무시).")

    # ── joint5 path constraint ───────────────────────────────────────────────
    parser.add_argument(
        "--constrain-joint5", dest="constrain_joint5",
        action="store_true", default=True,
        help="MoveIt planning 중 joint5=π/2 path constraint 적용 (기본 활성화).")
    parser.add_argument(
        "--no-constrain-joint5", dest="constrain_joint5", action="store_false",
        help="joint5 constraint 비활성화 (orientation만으로 파지 자세를 제어).")
    parser.add_argument(
        "--joint5-tolerance", type=float, default=JOINT5_TOLERANCE,
        help=f"joint5 constraint 허용 오차 [rad] (기본 {JOINT5_TOLERANCE:.2f})")
    parser.add_argument(
        "--joint5-position", type=float, default=JOINT5_VERTICAL_POSITION,
        help=f"joint5 constraint 목표값 [rad] (기본 π/2 ≈ {JOINT5_VERTICAL_POSITION:.4f})")

    # ── ROS2 / action 설정 ───────────────────────────────────────────────────
    parser.add_argument("--move-group-action", default=DEFAULT_MOVE_GROUP_ACTION)
    parser.add_argument("--arm-action", default=DEFAULT_ARM_ACTION)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--base-frame", default=DEFAULT_BASE_FRAME)
    parser.add_argument("--ee-frame", default=DEFAULT_EE_FRAME)

    # ── 이동 파라미터 ────────────────────────────────────────────────────────
    parser.add_argument("--duration", type=float, default=6.0,
                        help="FollowJointTrajectory 실행 시간 [s] (최소 4.0)")
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--velocity-scale", type=float, default=0.03)
    parser.add_argument("--acceleration-scale", type=float, default=0.03)

    # ── 허용 오차 ────────────────────────────────────────────────────────────
    parser.add_argument("--position-tolerance", type=float, default=0.01,
                        help="위치 허용 오차 [m]")
    parser.add_argument("--orientation-tolerance", type=float, default=0.15,
                        help="방향 허용 오차 [rad] (수직 파지 시 엄격하게 0.15 기본)")
    parser.add_argument(
        "--no-keep-orientation", dest="keep_orientation", action="store_false", default=True,
        help="방향 constraint 없이 위치만 지정 (권장하지 않음: 수직 파지 자세 유지 불가)")

    # ── Safety guard ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--max-joint-delta", type=float, default=0.35,
        help="현재→목표 최대 허용 관절 변위 [rad] (기본 0.35)")
    parser.add_argument(
        "--max-segment-delta", type=float, default=0.12,
        help="plan 내 연속 포인트 간 최대 관절 변위 [rad] (기본 0.12)")

    # ── 작업 공간 오버라이드 ─────────────────────────────────────────────────
    parser.add_argument("--x-min", type=float, default=DEFAULT_WORKSPACE["x_min"])
    parser.add_argument("--x-max", type=float, default=DEFAULT_WORKSPACE["x_max"])
    parser.add_argument("--y-min", type=float, default=DEFAULT_WORKSPACE["y_min"])
    parser.add_argument("--y-max", type=float, default=DEFAULT_WORKSPACE["y_max"])
    parser.add_argument("--z-min", type=float, default=DEFAULT_WORKSPACE["z_min"])
    parser.add_argument("--z-max", type=float, default=DEFAULT_WORKSPACE["z_max"])

    # ── 실행 제어 ────────────────────────────────────────────────────────────
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="",
                        help=f"실행 확인 문자열: '{CONFIRM_TEXT}'")

    args = parser.parse_args()

    # ── 전제 조건 검사 ────────────────────────────────────────────────────────
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(
            f"Refusing to execute. Re-run with --confirm {CONFIRM_TEXT}")
    if args.duration < 4.0:
        raise ValueError("--duration must be >= 4.0 s")
    if args.joint_state_timeout <= 0.0:
        raise ValueError("--joint-state-timeout must be > 0")
    if args.max_joint_delta <= 0.0:
        raise ValueError("--max-joint-delta must be > 0")

    workspace = {
        "x_min": args.x_min, "x_max": args.x_max,
        "y_min": args.y_min, "y_max": args.y_max,
        "z_min": args.z_min, "z_max": args.z_max,
    }
    target_pos = np.array([args.x, args.y, args.z], dtype=np.float64)
    assert_in_workspace("target", target_pos, workspace)

    # joint5 path constraint 목록 구성
    path_joint_constraints: list[dict[str, Any]] = []
    if args.constrain_joint5:
        path_joint_constraints.append({
            "name": "joint5",
            "position": args.joint5_position,
            "tol": args.joint5_tolerance,
            "weight": 1.0,
        })

    rclpy.init(args=None)
    node = Node("motion2_run_ee_pose_move")
    move_group_client = ActionClient(node, MoveGroup, args.move_group_action)
    arm_client = ActionClient(node, FollowJointTrajectory, args.arm_action)

    try:
        # ── 현재 상태 취득 ────────────────────────────────────────────────────
        print("[ee-pose-move] reading current EE pose (TF2) and joint states ...")
        current_pos, current_quat = lookup_current_pose(
            node, rclpy, args.base_frame, args.ee_frame, args.joint_state_timeout)
        current_joints = joint_state_once(node, rclpy, args.joint_state_timeout)
        missing = [j for j in ARM_JOINTS if j not in current_joints]
        if missing:
            raise RuntimeError(f"Missing arm joints in /joint_states: {missing}")
        current_arm = np.array([current_joints[j] for j in ARM_JOINTS], dtype=np.float64)

        # ── 목표 방향 결정 ────────────────────────────────────────────────────
        if args.rpy is not None:
            target_quat = _rpy_to_quat(*args.rpy)
            ori_source = f"--rpy {args.rpy}"
        elif args.quat is not None:
            q = np.array(args.quat, dtype=np.float64)
            norm = float(np.linalg.norm(q))
            if norm < 1e-6:
                raise ValueError("--quat: 영벡터는 사용할 수 없습니다")
            target_quat = q / norm
            ori_source = "--quat (정규화 적용)"
        elif args.use_current_ori:
            target_quat = current_quat.copy()
            ori_source = "현재 EE orientation (TF2)"
        else:
            # 기본값: 수직 파지 orientation (YAML waypoints '1','2' 에서 측정)
            target_quat = VERTICAL_GRIP_QUAT.copy()
            ori_source = f"수직 파지 기본값 {np.round(VERTICAL_GRIP_QUAT, 4).tolist()}"

        # ── 계획 정보 출력 ────────────────────────────────────────────────────
        print(f"[ee-pose-move] execute        : {bool(args.execute)}")
        print(f"[ee-pose-move] target pos     : [{args.x:.6f}, {args.y:.6f}, {args.z:.6f}] m")
        print(f"[ee-pose-move] target quat    : {np.round(target_quat, 6).tolist()}")
        print(f"[ee-pose-move] ori source     : {ori_source}")
        print(f"[ee-pose-move] current pos    : {np.round(current_pos, 6).tolist()} m")
        print(f"[ee-pose-move] current joints : "
              + ", ".join(f"{j}={v:.4f}" for j, v in zip(ARM_JOINTS, current_arm)))
        if args.constrain_joint5:
            j5_actual = float(current_joints.get("joint5", float("nan")))
            print(
                f"[ee-pose-move] joint5 constraint: {args.joint5_position:.4f} "
                f"± {args.joint5_tolerance:.4f} rad  "
                f"(현재 joint5={j5_actual:.4f} rad)")
        else:
            print("[ee-pose-move] joint5 constraint: 비활성화")
        print(f"[ee-pose-move] duration       : {args.duration:.2f} s")
        print(f"[ee-pose-move] max_joint_delta: {args.max_joint_delta:.4f} rad")
        print(
            f"[ee-pose-move] workspace      : "
            f"x=[{args.x_min},{args.x_max}] "
            f"y=[{args.y_min},{args.y_max}] "
            f"z=[{args.z_min},{args.z_max}]")

        # ── MoveGroup action server 대기 ──────────────────────────────────────
        print(f"[ee-pose-move] waiting for MoveGroup: {args.move_group_action}")
        if not move_group_client.wait_for_server(timeout_sec=3.0):
            raise RuntimeError(
                f"MoveGroup action server not available: {args.move_group_action}")

        # ── MoveIt planning (plan_only=True + path joint constraints) ─────────
        constraint_desc = (
            f"joint5={args.joint5_position:.4f}±{args.joint5_tolerance:.4f}"
            if args.constrain_joint5 else "없음"
        )
        print(f"[ee-pose-move] planning ... (path_constraint: {constraint_desc})")

        ok, report = _plan_with_constraints(
            node=node,
            rclpy=rclpy,
            client=move_group_client,
            frame_id=args.base_frame,
            group_name=args.group_name,
            link_name=args.ee_frame,
            pos=target_pos,
            quat_wxyz=target_quat,
            workspace=workspace,
            planning_time=args.planning_time,
            attempts=args.attempts,
            pos_tol=args.position_tolerance,
            ori_tol=args.orientation_tolerance,
            velocity_scale=args.velocity_scale,
            acceleration_scale=args.acceleration_scale,
            keep_orientation=args.keep_orientation,
            planner_id="",
            max_joint_delta=args.max_joint_delta,
            max_segment_delta=args.max_segment_delta,
            path_joint_constraints=path_joint_constraints,
            start_joint_names=ARM_JOINTS,
            start_joint_positions=current_arm.tolist(),
        )

        print(
            f"[ee-pose-move] plan result    : ok={ok} "
            f"error_code={report.get('error_code')} "
            f"points={report.get('point_count')} "
            f"plan_max_delta={float(report.get('max_start_to_goal_delta', 0.0)):.6f} rad "
            f"segment_max={float(report.get('max_segment_delta', 0.0)):.6f} rad "
            f"worst_joint={report.get('max_joint', '')} "
            f"guard_ok={report.get('guard_ok', False)}")

        if not ok:
            print("[ee-pose-move] plan failed or plan-level guard violated; command_sent=false")
            if args.constrain_joint5:
                print(
                    "[ee-pose-move] hint: joint5 constraint로 인해 계획 실패 시 "
                    "--no-constrain-joint5 로 재시도하거나 목표 위치를 수직 파지 가능 범위로 조정")
            return 2

        # ── Plan에서 목표 관절값 추출 ─────────────────────────────────────────
        goal_arm = _extract_goal_joints(report, ARM_JOINTS)

        # ── joint5 결과 검증: plan이 constraint를 실제로 만족하는지 확인 ────────
        if args.constrain_joint5:
            j5_planned = goal_arm[ARM_JOINTS.index("joint5")]
            j5_err = abs(j5_planned - args.joint5_position)
            j5_ok = j5_err <= args.joint5_tolerance
            print(
                f"[ee-pose-move] joint5 result  : "
                f"planned={j5_planned:.6f} rad  "
                f"target={args.joint5_position:.6f} rad  "
                f"error={j5_err:.6f} rad  "
                f"within_tol={j5_ok}")
            if not j5_ok:
                print(
                    f"[ee-pose-move] JOINT5 CONSTRAINT NOT MET: "
                    f"error {j5_err:.4f} rad > tol {args.joint5_tolerance:.4f} rad; "
                    f"command_sent=false")
                return 2

        # ── 현재 실제 자세 기준 delta 재검증 ─────────────────────────────────
        deltas = np.abs(goal_arm - current_arm)
        max_delta = float(np.max(deltas))
        max_joint_name = ARM_JOINTS[int(np.argmax(deltas))]

        print("[ee-pose-move] goal joints (from plan):")
        for jname, tgt, cur, dlt in zip(ARM_JOINTS, goal_arm, current_arm, deltas):
            marker = " ← MAX" if jname == max_joint_name else ""
            print(f"  {jname}: cur={cur:.6f}  target={tgt:.6f}  delta={dlt:.6f} rad{marker}")
        print(
            f"[ee-pose-move] actual max_delta: {max_delta:.6f} rad "
            f"at {max_joint_name} (limit: {args.max_joint_delta:.6f} rad)")

        if max_delta > args.max_joint_delta:
            print(
                f"[ee-pose-move] GUARD VIOLATED: actual delta {max_delta:.6f} > "
                f"limit {args.max_joint_delta:.6f} at {max_joint_name}; "
                f"command_sent=false")
            return 2

        # ── Dry-run 완료 ──────────────────────────────────────────────────────
        if not args.execute:
            print("[ee-pose-move] dry-run complete; command_sent=false")
            return 0

        # ── 실제 실행: FollowJointTrajectory (MoveIt execute 아님) ────────────
        print(f"[ee-pose-move] waiting for arm action: {args.arm_action}")
        if not arm_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(
                f"Arm action server not available: {args.arm_action}")

        print(
            f"[ee-pose-move] sending FollowJointTrajectory "
            f"(duration={args.duration:.2f}s)")
        rc = _send_arm_goal(
            node, rclpy, arm_client,
            ARM_JOINTS, goal_arm.tolist(), args.duration)

        if rc == 0:
            print("[ee-pose-move] SUCCESS command_sent=true")
        else:
            print("[ee-pose-move] FAILED command_sent=true")
        return rc

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
