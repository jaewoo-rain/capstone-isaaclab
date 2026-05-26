"""OMY-F3M pick-and-place 통합 파이프라인: 좌표를 받아 9단계를 순서대로 실행한다.

시퀀스 (9단계):
    1. pre_grasp     : 물체 위 이동      (x1, y1, approach_z)
    2. grasp         : 물체 높이 하강    (x1, y1, grasp_z)
    3. close_gripper : 그리퍼 닫기       ← 파지
    4. lift          : 들어올리기        (x1, y1, lift_z)
    5. transport     : 목표 위치 이동    (x2, y2, lift_z)
    6. place_descend : 놓을 높이 하강    (x2, y2, place_z = grasp_z)
    7. open_gripper  : 그리퍼 열기       ← 해제
    8. retract       : 들어올리기        (x2, y2, lift_z)
    9. home          : 홈 복귀          (YAML 'start' 관절값 직접 재생)

좌표 결정 우선순위:
    CLI --pick-x/y, --place-x/y 지정 시 → 해당 값 사용
    미지정 시 → YAML waypoints['2'] (pick), waypoints['4'] (place) 의 ee_pose x,y 자동 사용

Z 좌표 (YAML 에서 자동 읽기, CLI 로 오버라이드 가능):
    approach_z = waypoints['1']['ee_pose']['position']['z']  ≈ 0.372 m
    grasp_z    = waypoints['2']['ee_pose']['position']['z']  ≈ 0.345 m
    place_z    = grasp_z  (동일)
    lift_z     = approach_z + --lift-offset                  기본 +0.06 m

홈 복귀는 YAML 'start' 관절값을 FollowJointTrajectory 로 직접 전송.

기본 동작은 dry-run. 실제 실행:
    --execute --confirm EXECUTE_PICK_PLACE

사용 예시:
    # 좌표 미지정 → YAML '2','4' 위치로 자동 실행 (dry-run)
    python3 motion2/scripts/run_pick_place.py

    # 좌표 직접 지정 (dry-run)
    python3 motion2/scripts/run_pick_place.py \\
        --pick-x 0.496 --pick-y -0.113 \\
        --place-x 0.321 --place-y -0.394

    # 실제 실행 (YAML 기본 좌표)
    python3 motion2/scripts/run_pick_place.py \\
        --execute --confirm EXECUTE_PICK_PLACE

    # 실제 실행 (좌표 직접 지정)
    python3 motion2/scripts/run_pick_place.py \\
        --pick-x 0.496 --pick-y -0.113 \\
        --place-x 0.321 --place-y -0.394 \\
        --execute --confirm EXECUTE_PICK_PLACE
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys
from typing import Any

import numpy as np
import yaml


# ── 실행 안전 장치 ─────────────────────────────────────────────────────────
CONFIRM_TEXT = "EXECUTE_PICK_PLACE"

# ── 관절 / 프레임 정의 ─────────────────────────────────────────────────────
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
DEFAULT_CONFIG = "motion2/config/teach_pick_place_waypoints.yaml"
DEFAULT_ARM_ACTION = "/arm_controller/follow_joint_trajectory"
DEFAULT_MOVE_GROUP_ACTION = "/move_action"
DEFAULT_GRIPPER_ACTION = "/gripper_controller/gripper_cmd"
DEFAULT_GROUP_NAME = "arm"
DEFAULT_BASE_FRAME = "link0"
DEFAULT_EE_FRAME = "link6"

# ── 수직 파지 orientation (YAML waypoints '1','2' 측정값) ──────────────────
_VGQ_RAW = np.array([0.5063, 0.4936, 0.4936, 0.5063], dtype=np.float64)
VERTICAL_GRIP_QUAT = _VGQ_RAW / np.linalg.norm(_VGQ_RAW)

# joint5 path constraint: 수직 파지 자세 유지
JOINT5_VERTICAL = math.pi / 2   # ≈ 1.5708 rad
JOINT5_TOLERANCE = 0.08

# MoveIt 성공 코드
MOVEIT_SUCCESS = 1

# 작업 공간 (x_max=0.55: YAML '1','2' 위치 x≈0.496 커버)
DEFAULT_WORKSPACE = {
    "x_min": -0.10, "x_max": 0.55,
    "y_min": -0.45, "y_max": 0.20,
    "z_min": 0.10,  "z_max": 0.55,
}


# ─────────────────────────────────────────────────────────────────────────────
# YAML 로딩 및 Z/관절값 읽기
# ─────────────────────────────────────────────────────────────────────────────

def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _resolve_path(path_text: str) -> pathlib.Path:
    p = pathlib.Path(path_text)
    return p if p.is_absolute() else _repo_root() / p


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: YAML 최상위가 mapping이 아님")
    return data


def _read_z_from_waypoint(data: dict[str, Any], key: str) -> float:
    """YAML waypoints[key]['ee_pose']['position']['z'] 를 읽는다."""
    wp = data.get("waypoints", {}).get(key)
    if not isinstance(wp, dict):
        raise KeyError(f"YAML에 waypoints['{key}'] 가 없음")
    ee = wp.get("ee_pose")
    if not isinstance(ee, dict):
        raise KeyError(f"waypoints['{key}']['ee_pose'] 가 없음")
    pos = ee.get("position")
    if not isinstance(pos, dict):
        raise KeyError(f"waypoints['{key}']['ee_pose']['position'] 가 없음")
    return float(pos["z"])


def _read_xy_from_waypoint(data: dict[str, Any], key: str) -> tuple[float, float]:
    """YAML waypoints[key]['ee_pose']['position'] 에서 (x, y) 반환."""
    wp = data.get("waypoints", {}).get(key)
    if not isinstance(wp, dict):
        raise KeyError(f"YAML에 waypoints['{key}'] 가 없음")
    ee = wp.get("ee_pose")
    if not isinstance(ee, dict):
        raise KeyError(f"waypoints['{key}']['ee_pose'] 가 없음")
    pos = ee.get("position")
    if not isinstance(pos, dict):
        raise KeyError(f"waypoints['{key}']['ee_pose']['position'] 가 없음")
    return float(pos["x"]), float(pos["y"])


def _read_home_joints(data: dict[str, Any], key: str = "start") -> list[float]:
    """YAML waypoints[key]['arm'] 에서 홈 관절값을 읽는다."""
    wp = data.get("waypoints", {}).get(key)
    if not isinstance(wp, dict):
        raise KeyError(f"YAML에 홈 waypoint '{key}' 가 없음")
    arm = wp.get("arm")
    if not isinstance(arm, dict):
        raise KeyError(f"waypoints['{key}']['arm'] 가 없음")
    missing = [j for j in ARM_JOINTS if j not in arm]
    if missing:
        raise KeyError(f"홈 waypoint '{key}' 에 관절 누락: {missing}")
    return [float(arm[j]) for j in ARM_JOINTS]


def _read_gripper_targets(data: dict[str, Any]) -> tuple[float, float]:
    """(close_value, open_value) 반환."""
    gt = data.get("gripper_targets", {})
    return float(gt.get("close_gripper", 0.60)), float(gt.get("open_gripper", 0.00))


# ─────────────────────────────────────────────────────────────────────────────
# 저수준 ROS2 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _make_arm_traj_goal(joint_names: list[str], positions: list[float], duration_s: float):
    """단일 point FollowJointTrajectory.Goal 생성."""
    from control_msgs.action import FollowJointTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = list(joint_names)
    pt = JointTrajectoryPoint()
    pt.positions = [float(v) for v in positions]
    pt.velocities = [0.0] * len(positions)
    pt.time_from_start.sec = int(duration_s)
    pt.time_from_start.nanosec = int((duration_s % 1) * 1_000_000_000)
    goal.trajectory.points.append(pt)
    return goal


def _send_arm_traj(node, rclpy, client, joint_names, positions, duration_s, label) -> bool:
    """FollowJointTrajectory 전송 → True=성공, False=실패."""
    goal = _make_arm_traj_goal(joint_names, positions, duration_s)
    fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut)
    handle = fut.result()
    if handle is None or not handle.accepted:
        print(f"[pick-place] {label}: arm goal rejected")
        return False
    res_fut = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res_fut)
    code = int(res_fut.result().result.error_code)
    if code == 0:
        print(f"[pick-place] {label}: arm success")
        return True
    print(f"[pick-place] {label}: arm failed error_code={code}")
    return False


def _send_gripper(node, rclpy, client, position: float, max_effort: float, label) -> bool:
    """GripperCommand 전송 → True=reached_goal, False=실패."""
    from control_msgs.action import GripperCommand

    goal = GripperCommand.Goal()
    goal.command.position = float(position)
    goal.command.max_effort = float(max_effort)
    fut = client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut)
    handle = fut.result()
    if handle is None or not handle.accepted:
        print(f"[pick-place] {label}: gripper goal rejected")
        return False
    res_fut = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res_fut)
    r = res_fut.result().result
    print(
        f"[pick-place] {label}: gripper pos={r.position:.4f} "
        f"effort={r.effort:.4f} stalled={r.stalled} reached={r.reached_goal}")
    return bool(r.reached_goal)


# ─────────────────────────────────────────────────────────────────────────────
# MoveIt planning (plan_only=True + joint5 path constraint)
# ─────────────────────────────────────────────────────────────────────────────

def _plan_arm_pose(
    *,
    node, rclpy, mg_client,
    x: float, y: float, z: float,
    quat: np.ndarray,
    workspace: dict[str, float],
    group_name: str,
    base_frame: str,
    ee_frame: str,
    planning_time: float,
    attempts: int,
    pos_tol: float,
    ori_tol: float,
    velocity_scale: float,
    acceleration_scale: float,
    max_joint_delta: float,
    max_segment_delta: float,
    constrain_joint5: bool,
    joint5_tol: float,
    current_arm: np.ndarray,
    label: str,
) -> tuple[bool, np.ndarray | None]:
    """MoveIt plan_only=True → goal 관절값 추출 + delta guard.

    반환: (ok, goal_arm_joints)  ok=False 면 goal_arm_joints=None
    """
    from real_moveit_common import make_move_group_goal, trajectory_delta_report
    from moveit_msgs.msg import Constraints, JointConstraint

    pos = np.array([x, y, z], dtype=np.float64)
    goal = make_move_group_goal(
        frame_id=base_frame,
        group_name=group_name,
        link_name=ee_frame,
        pos=pos,
        quat_wxyz=quat,
        workspace=workspace,
        planning_time=planning_time,
        attempts=attempts,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
        velocity_scale=velocity_scale,
        acceleration_scale=acceleration_scale,
        plan_only=True,
        keep_orientation=True,
        planner_id="",
    )

    if constrain_joint5:
        path = Constraints()
        c = JointConstraint()
        c.joint_name = "joint5"
        c.position = JOINT5_VERTICAL
        c.tolerance_above = joint5_tol
        c.tolerance_below = joint5_tol
        c.weight = 1.0
        path.joint_constraints.append(c)
        goal.request.path_constraints = path

    fut = mg_client.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, fut)
    handle = fut.result()
    if handle is None or not handle.accepted:
        print(f"[pick-place] {label}: plan goal rejected")
        return False, None

    res_fut = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res_fut)
    result = res_fut.result().result
    code = int(result.error_code.val)

    report = trajectory_delta_report(result.planned_trajectory)
    plan_ok = (
        code == MOVEIT_SUCCESS
        and float(report["max_start_to_goal_delta"]) <= max_joint_delta
        and float(report["max_segment_delta"]) <= max_segment_delta
    )
    print(
        f"[pick-place] {label}: plan ok={plan_ok} "
        f"error_code={code} "
        f"plan_delta={float(report['max_start_to_goal_delta']):.4f} rad "
        f"segment_delta={float(report['max_segment_delta']):.4f} rad "
        f"worst={report.get('max_joint', '')}")

    if not plan_ok:
        return False, None

    # plan에서 ARM_JOINTS 순서로 목표 관절값 추출
    name_to_pos = dict(zip(report["joint_names"], report["goal_positions"]))
    missing = [j for j in ARM_JOINTS if j not in name_to_pos]
    if missing:
        print(f"[pick-place] {label}: plan missing joints {missing}")
        return False, None
    goal_arm = np.array([name_to_pos[j] for j in ARM_JOINTS], dtype=np.float64)

    # 현재 실제 자세 기준 delta 재검증
    deltas = np.abs(goal_arm - current_arm)
    max_d = float(np.max(deltas))
    max_j = ARM_JOINTS[int(np.argmax(deltas))]
    if max_d > max_joint_delta:
        print(
            f"[pick-place] {label}: actual delta {max_d:.4f} rad at {max_j} "
            f"> limit {max_joint_delta:.4f} rad → 거부")
        return False, None

    print(
        f"[pick-place] {label}: actual max_delta={max_d:.4f} rad at {max_j} ✓")
    return True, goal_arm


# ─────────────────────────────────────────────────────────────────────────────
# 단계별 확인
# ─────────────────────────────────────────────────────────────────────────────

def _confirm(step_name: str) -> None:
    """step 이름을 직접 타이핑해야 실행 진행."""
    typed = input(
        f"\n[pick-place] '{step_name}' 실행하려면 이름 그대로 입력 (중단: Enter): "
    ).strip()
    if typed != step_name:
        raise RuntimeError(
            f"'{step_name}' 실행 거부 (입력값: '{typed}')")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    scripts = str(pathlib.Path(__file__).resolve().parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)

    from real_moveit_common import assert_in_workspace, joint_state_once

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory, GripperCommand
    from moveit_msgs.action import MoveGroup

    parser = argparse.ArgumentParser(
        description="(x1,y1) 에서 집어서 (x2,y2) 로 옮기는 9단계 pick-and-place.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── 좌표 입력 (생략 시 YAML waypoints['2']/'4' x,y 자동 사용) ────────────
    parser.add_argument("--pick-x", type=float, default=None,
                        help="집을 위치 x [m] (link0 기준). 미지정 시 YAML ['2'] 사용")
    parser.add_argument("--pick-y", type=float, default=None,
                        help="집을 위치 y [m]. 미지정 시 YAML ['2'] 사용")
    parser.add_argument("--place-x", type=float, default=None,
                        help="놓을 위치 x [m]. 미지정 시 YAML ['4'] 사용")
    parser.add_argument("--place-y", type=float, default=None,
                        help="놓을 위치 y [m]. 미지정 시 YAML ['4'] 사용")

    # ── Z 좌표 오버라이드 ─────────────────────────────────────────────────────
    parser.add_argument("--approach-z", type=float, default=None,
                        help="호버 높이 [m]. 미지정 시 YAML waypoints['1'] z 값 사용")
    parser.add_argument("--grasp-z", type=float, default=None,
                        help="파지 높이 [m]. 미지정 시 YAML waypoints['2'] z 값 사용")
    parser.add_argument("--lift-offset", type=float, default=0.06,
                        help="approach_z 위로 추가로 올리는 높이 [m] → lift_z = approach_z + offset")

    # ── YAML / action 설정 ───────────────────────────────────────────────────
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--home-waypoint", default="start",
                        help="홈 복귀에 사용할 YAML waypoint 이름")
    parser.add_argument("--move-group-action", default=DEFAULT_MOVE_GROUP_ACTION)
    parser.add_argument("--arm-action", default=DEFAULT_ARM_ACTION)
    parser.add_argument("--gripper-action", default=DEFAULT_GRIPPER_ACTION)
    parser.add_argument("--group-name", default=DEFAULT_GROUP_NAME)
    parser.add_argument("--base-frame", default=DEFAULT_BASE_FRAME)
    parser.add_argument("--ee-frame", default=DEFAULT_EE_FRAME)

    # ── 이동 파라미터 ────────────────────────────────────────────────────────
    parser.add_argument("--duration", type=float, default=6.0,
                        help="암 이동 시간 [s] (최소 4.0)")
    parser.add_argument("--home-duration", type=float, default=9.0,
                        help="홈 복귀 이동 시간 [s] (더 느리게 설정 권장)")
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--velocity-scale", type=float, default=0.03)
    parser.add_argument("--acceleration-scale", type=float, default=0.03)
    parser.add_argument("--gripper-max-effort", type=float, default=0.0)

    # ── 허용 오차 ────────────────────────────────────────────────────────────
    parser.add_argument("--position-tolerance", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance", type=float, default=0.15)
    parser.add_argument("--max-joint-delta", type=float, default=0.35)
    parser.add_argument("--max-segment-delta", type=float, default=0.12)

    # ── joint5 constraint ────────────────────────────────────────────────────
    parser.add_argument("--no-constrain-joint5", dest="constrain_joint5",
                        action="store_false", default=True)
    parser.add_argument("--joint5-tolerance", type=float, default=JOINT5_TOLERANCE)

    # ── 작업 공간 ────────────────────────────────────────────────────────────
    parser.add_argument("--x-min", type=float, default=DEFAULT_WORKSPACE["x_min"])
    parser.add_argument("--x-max", type=float, default=DEFAULT_WORKSPACE["x_max"])
    parser.add_argument("--y-min", type=float, default=DEFAULT_WORKSPACE["y_min"])
    parser.add_argument("--y-max", type=float, default=DEFAULT_WORKSPACE["y_max"])
    parser.add_argument("--z-min", type=float, default=DEFAULT_WORKSPACE["z_min"])
    parser.add_argument("--z-max", type=float, default=DEFAULT_WORKSPACE["z_max"])

    # ── 실행 제어 ────────────────────────────────────────────────────────────
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--no-home", action="store_true",
                        help="9번 홈 복귀 단계 생략")
    parser.add_argument("--no-step-prompts", action="store_true",
                        help="단계별 타이핑 확인 생략 (권장하지 않음)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")

    args = parser.parse_args()

    # ── 전제 조건 검사 ────────────────────────────────────────────────────────
    if args.execute and args.confirm != CONFIRM_TEXT:
        raise RuntimeError(
            f"Refusing to execute. --confirm {CONFIRM_TEXT} 필요")
    if args.duration < 4.0:
        raise ValueError("--duration >= 4.0 s 필요")
    if args.home_duration < 4.0:
        raise ValueError("--home-duration >= 4.0 s 필요")

    workspace = {
        "x_min": args.x_min, "x_max": args.x_max,
        "y_min": args.y_min, "y_max": args.y_max,
        "z_min": args.z_min, "z_max": args.z_max,
    }

    # ── YAML 로드 → 좌표 / Z 값 결정 ─────────────────────────────────────
    data = _load_yaml(_resolve_path(args.config))

    # x,y 좌표: CLI 지정 값 우선, 없으면 YAML ee_pose 에서 읽음
    if args.pick_x is not None and args.pick_y is not None:
        pick_x, pick_y = args.pick_x, args.pick_y
        pick_src = "CLI"
    else:
        pick_x, pick_y = _read_xy_from_waypoint(data, "2")
        pick_src = "YAML['2']"
        print(f"[pick-place] pick x,y: YAML waypoints['2'] → x={pick_x:.6f}  y={pick_y:.6f}")

    if args.place_x is not None and args.place_y is not None:
        place_x, place_y = args.place_x, args.place_y
        place_src = "CLI"
    else:
        place_x, place_y = _read_xy_from_waypoint(data, "4")
        place_src = "YAML['4']"
        print(f"[pick-place] place x,y: YAML waypoints['4'] → x={place_x:.6f}  y={place_y:.6f}")

    if args.approach_z is not None:
        approach_z = args.approach_z
    else:
        approach_z = _read_z_from_waypoint(data, "1")
        print(f"[pick-place] approach_z: YAML waypoints['1'] z = {approach_z:.6f} m")

    if args.grasp_z is not None:
        grasp_z = args.grasp_z
    else:
        grasp_z = _read_z_from_waypoint(data, "2")
        print(f"[pick-place] grasp_z:    YAML waypoints['2'] z = {grasp_z:.6f} m")

    lift_z = approach_z + args.lift_offset
    place_z = grasp_z  # 놓는 높이 = 집는 높이

    close_val, open_val = _read_gripper_targets(data)
    home_joints = None if args.no_home else _read_home_joints(data, args.home_waypoint)

    # ── 시퀀스 출력 ───────────────────────────────────────────────────────
    print("\n[pick-place] ─── 실행 계획 ──────────────────────────────────────")
    print(f"[pick-place] execute : {args.execute}")
    print(f"[pick-place] pick    : x={pick_x:.4f}  y={pick_y:.4f}  ({pick_src})")
    print(f"[pick-place] place   : x={place_x:.4f}  y={place_y:.4f}  ({place_src})")
    print(f"[pick-place] z 높이  : approach={approach_z:.4f}  grasp={grasp_z:.4f}  "
          f"lift={lift_z:.4f}  place={place_z:.4f}  [m]")
    print(f"[pick-place] gripper : close={close_val:.3f}  open={open_val:.3f}")
    seq = [
        ("1. pre_grasp    ", f"arm  ({pick_x:.4f}, {pick_y:.4f}, {approach_z:.4f})"),
        ("2. grasp        ", f"arm  ({pick_x:.4f}, {pick_y:.4f}, {grasp_z:.4f})"),
        ("3. close_gripper", f"grip  position={close_val:.3f}"),
        ("4. lift         ", f"arm  ({pick_x:.4f}, {pick_y:.4f}, {lift_z:.4f})"),
        ("5. transport    ", f"arm  ({place_x:.4f}, {place_y:.4f}, {lift_z:.4f})"),
        ("6. place_descend", f"arm  ({place_x:.4f}, {place_y:.4f}, {place_z:.4f})"),
        ("7. open_gripper ", f"grip  position={open_val:.3f}"),
        ("8. retract      ", f"arm  ({place_x:.4f}, {place_y:.4f}, {lift_z:.4f})"),
    ]
    if not args.no_home:
        seq.append(("9. home         ",
                    f"teach-replay 'start'  ({args.home_waypoint})"))
    for name, desc in seq:
        print(f"[pick-place]   step {name}: {desc}")
    print("[pick-place] ──────────────────────────────────────────────────────\n")

    # 좌표 workspace 검증
    for label, x, y, z in [
        ("pick/approach",  pick_x,  pick_y,  approach_z),
        ("pick/grasp",     pick_x,  pick_y,  grasp_z),
        ("pick/lift",      pick_x,  pick_y,  lift_z),
        ("place/transport",place_x, place_y, lift_z),
        ("place/descend",  place_x, place_y, place_z),
        ("place/retract",  place_x, place_y, lift_z),
    ]:
        assert_in_workspace(label, np.array([x, y, z]), workspace)

    # ── ROS2 초기화 ───────────────────────────────────────────────────────
    rclpy.init(args=None)
    node = Node("motion2_run_pick_place")
    mg_client = ActionClient(node, MoveGroup, args.move_group_action)
    arm_client = ActionClient(node, FollowJointTrajectory, args.arm_action)
    gripper_client = ActionClient(node, GripperCommand, args.gripper_action)

    try:
        # 현재 관절 상태 취득 (planning delta guard 용)
        current_joints = joint_state_once(node, rclpy, args.joint_state_timeout)
        missing = [j for j in ARM_JOINTS if j not in current_joints]
        if missing:
            raise RuntimeError(f"/joint_states 에 관절 없음: {missing}")
        current_arm = np.array([current_joints[j] for j in ARM_JOINTS], dtype=np.float64)

        if args.execute:
            print(f"[pick-place] MoveGroup 대기: {args.move_group_action}")
            if not mg_client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"MoveGroup 없음: {args.move_group_action}")
            print(f"[pick-place] arm action 대기: {args.arm_action}")
            if not arm_client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"arm action 없음: {args.arm_action}")
            print(f"[pick-place] gripper action 대기: {args.gripper_action}")
            if not gripper_client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"gripper action 없음: {args.gripper_action}")
        else:
            # dry-run에서도 MoveGroup은 planning에 필요
            print(f"[pick-place] MoveGroup 대기 (dry-run planning 용)")
            if not mg_client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"MoveGroup 없음: {args.move_group_action}")

        # 공통 planning 파라미터
        plan_kwargs: dict[str, Any] = dict(
            node=node, rclpy=rclpy, mg_client=mg_client,
            quat=VERTICAL_GRIP_QUAT,
            workspace=workspace,
            group_name=args.group_name,
            base_frame=args.base_frame,
            ee_frame=args.ee_frame,
            planning_time=args.planning_time,
            attempts=args.attempts,
            pos_tol=args.position_tolerance,
            ori_tol=args.orientation_tolerance,
            velocity_scale=args.velocity_scale,
            acceleration_scale=args.acceleration_scale,
            max_joint_delta=args.max_joint_delta,
            max_segment_delta=args.max_segment_delta,
            constrain_joint5=args.constrain_joint5,
            joint5_tol=args.joint5_tolerance,
        )

        def do_arm_step(label: str, x: float, y: float, z: float) -> bool:
            """plan → (실행모드: confirm → execute). 실패 시 False."""
            nonlocal current_arm
            print(f"\n[pick-place] ▶ {label}  target=({x:.4f}, {y:.4f}, {z:.4f})")
            ok, goal_joints = _plan_arm_pose(
                **plan_kwargs, x=x, y=y, z=z,
                current_arm=current_arm, label=label)
            if not ok:
                return False
            if not args.execute:
                print(f"[pick-place] {label}: dry-run OK (command_sent=false)")
                return True
            if not args.no_step_prompts:
                _confirm(label)
            success = _send_arm_traj(
                node, rclpy, arm_client,
                ARM_JOINTS, goal_joints.tolist(), args.duration, label)
            if success:
                # 실행 후 current_arm 갱신 (다음 step의 delta guard 기준)
                current_arm = goal_joints.copy()
            return success

        def do_gripper_step(label: str, position: float) -> bool:
            """그리퍼 열기/닫기. 실패 시 False."""
            print(f"\n[pick-place] ▶ {label}  position={position:.3f}")
            if not args.execute:
                print(f"[pick-place] {label}: dry-run OK (command_sent=false)")
                return True
            if not args.no_step_prompts:
                _confirm(label)
            return _send_gripper(
                node, rclpy, gripper_client, position, args.gripper_max_effort, label)

        def do_home_step() -> bool:
            """YAML 'start' 관절값으로 직접 복귀. delta guard 없음."""
            label = f"home({args.home_waypoint})"
            print(f"\n[pick-place] ▶ {label}  (teach-replay, delta guard 없음)")
            for j, v in zip(ARM_JOINTS, home_joints):
                print(f"  {j}: {v:.6f} rad")
            if not args.execute:
                print(f"[pick-place] {label}: dry-run OK (command_sent=false)")
                return True
            if not args.no_step_prompts:
                _confirm(label)
            return _send_arm_traj(
                node, rclpy, arm_client,
                ARM_JOINTS, home_joints, args.home_duration, label)

        # ── 9단계 순서 실행 ──────────────────────────────────────────────
        steps = [
            lambda: do_arm_step("1.pre_grasp",     pick_x,  pick_y,  approach_z),
            lambda: do_arm_step("2.grasp",          pick_x,  pick_y,  grasp_z),
            lambda: do_gripper_step("3.close_gripper", close_val),
            lambda: do_arm_step("4.lift",           pick_x,  pick_y,  lift_z),
            lambda: do_arm_step("5.transport",      place_x, place_y, lift_z),
            lambda: do_arm_step("6.place_descend",  place_x, place_y, place_z),
            lambda: do_gripper_step("7.open_gripper",  open_val),
            lambda: do_arm_step("8.retract",        place_x, place_y, lift_z),
        ]
        if not args.no_home:
            steps.append(do_home_step)

        for step_fn in steps:
            if not step_fn():
                print("\n[pick-place] FAILED — 시퀀스 중단 command_sent=false")
                return 2

        status = "SUCCESS" if args.execute else "DRY-RUN COMPLETE"
        print(f"\n[pick-place] {status}  command_sent={args.execute}")
        return 0

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
