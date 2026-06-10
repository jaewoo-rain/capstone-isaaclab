"""rl/run_grasp.py — grasp RL 정책으로 실제 OMY-F3M 박스 파지 (sim2real).

박스 좌표(base link0 기준 x, y, yaw)를 수동 입력받아:
  1. 현재 EE pose(TF2) 읽기 → 롤아웃 시작점
  2. grasp 정책 내부 롤아웃 → 최종 정렬 EE xy/yaw 계산 (MoveIt 폐루프 불가 회피)
  3. 안전 검증 (학습 분포 / workspace / delta)
  4. 단일 이동 시퀀스: pre_grasp → grasp → close_gripper → lift

좌표는 지금은 수동 입력. 나중에 카메라가 같은 (x, y, yaw) 자리를 채운다.

기본 동작은 dry-run (MoveIt plan 만, 로봇 명령 X). 실제 실행:
    --execute --confirm EXECUTE_GRASP

사전 조건 (별도 터미널):
    ssh root@omy-SNPR44B1021.local
    ros2 launch open_manipulator_bringup omy_f3m.launch.py

⚠️ 실행 전 README.md 의 sim2real 검증 순서(yaw=0 먼저)를 반드시 읽을 것.

사용 예시:
    # dry-run (박스 x=0.45 y=-0.10 yaw=0.3rad)
    python3 run_grasp.py --box-x 0.45 --box-y -0.10 --box-yaw 0.3

    # 1단계 검증: yaw 정렬 끄고 자세부터 확인 (실제 실행)
    python3 run_grasp.py --box-x 0.45 --box-y -0.10 --box-yaw 0.0 \\
        --no-align-yaw --execute --confirm EXECUTE_GRASP

    # 전체 (yaw 정렬 포함)
    python3 run_grasp.py --box-x 0.45 --box-y -0.10 --box-yaw 0.3 \\
        --execute --confirm EXECUTE_GRASP
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

# rl/grasp/ 자신은 스크립트 디렉토리라 자동 import. 상위 디렉토리들을 path 에 추가.
_THIS_DIR = pathlib.Path(__file__).resolve().parent   # .../rl/grasp
_JAEWOO_DIR = _THIS_DIR.parents[1]        # .../scripts/jaewoo  (run_pick_place)
_SCRIPTS_DIR = _THIS_DIR.parents[2]       # .../scripts         (real_moveit_common)
for _p in (str(_JAEWOO_DIR), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C
from grasp_policy import GraspPolicy
from grasp_rollout import rollout_grasp, wrap_to_pi


# ── 단계별 이동 시간 [s] (run_pick_place 와 동일 스타일, 최소 4.0) ──
DURATION_PRE_GRASP = 8.0
DURATION_GRASP = 8.0
DURATION_LIFT = 8.0


def _compose_yaw_quat(ee_yaw: float, base_quat: np.ndarray) -> np.ndarray:
    """R_z(ee_yaw) ⊗ base_quat — sim grasp_env 의 target quat 합성과 동일.

    sim: target_quat_w = quat_mul(yaw_quat, base_ee_quat).
    실제 로봇 base_quat = VERTICAL_GRIP_QUAT (수직 파지). yaw 회전축은 world Z (동일).
    """
    from real_moveit_common import quat_from_z_yaw, quat_mul

    return quat_mul(quat_from_z_yaw(ee_yaw), base_quat)


def _check_box_distribution(box_xy: np.ndarray, box_yaw: float) -> list[str]:
    """박스 입력이 정책 학습 분포 안인지 검사. 경고 메시지 리스트 반환."""
    warns: list[str] = []
    cx, cy = C.BOX_DIST_CENTER_XY
    margin = C.BOX_DIST_XY_NOISE + 0.05  # 분포 + 5cm 여유
    if not (cx - margin <= box_xy[0] <= cx + margin):
        warns.append(
            f"box_x={box_xy[0]:.3f} 가 학습 분포 [{cx-C.BOX_DIST_XY_NOISE:.2f}, "
            f"{cx+C.BOX_DIST_XY_NOISE:.2f}] 밖 → 정책 정렬 정확도 저하 가능")
    if not (cy - margin <= box_xy[1] <= cy + margin):
        warns.append(
            f"box_y={box_xy[1]:.3f} 가 학습 분포 [{cy-C.BOX_DIST_XY_NOISE:.2f}, "
            f"{cy+C.BOX_DIST_XY_NOISE:.2f}] 밖 → 정책 정렬 정확도 저하 가능")
    if abs(box_yaw) > C.BOX_DIST_YAW_MAX:
        warns.append(
            f"box_yaw={math.degrees(box_yaw):.1f}° 가 학습 범위 "
            f"±{math.degrees(C.BOX_DIST_YAW_MAX):.0f}° 밖 → yaw 정렬 실패 가능")
    return warns


def main() -> int:
    parser = argparse.ArgumentParser(
        description="grasp RL 정책으로 실제 OMY-F3M 박스 파지 (내부 롤아웃 → 단일 이동).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── 박스 좌표 (base link0 기준) — 지금은 수동, 나중에 카메라 ──
    parser.add_argument("--box-x", type=float, required=True, help="박스 중심 x [m] (link0)")
    parser.add_argument("--box-y", type=float, required=True, help="박스 중심 y [m] (link0)")
    parser.add_argument("--box-yaw", type=float, default=0.0, help="박스 yaw [rad]")

    # ── yaw 정렬 on/off (sim2real 1단계 검증: --no-align-yaw 로 자세부터) ──
    parser.add_argument("--align-yaw", dest="align_yaw", action="store_true", default=True,
                        help="박스 yaw 에 EE 회전 정렬 (기본 활성화)")
    parser.add_argument("--no-align-yaw", dest="align_yaw", action="store_false",
                        help="yaw 정렬 끄고 수직 파지 기본 자세 유지 (검증 1단계)")

    # ── z 높이 (정책 무관, YAML 또는 CLI) ──
    parser.add_argument("--approach-z", type=float, default=None,
                        help="호버 높이 [m]. 미지정 시 YAML waypoints['1'] z")
    parser.add_argument("--grasp-z", type=float, default=None,
                        help="파지 하강 높이 [m]. 미지정 시 YAML waypoints['2'] z")
    parser.add_argument("--lift-offset", type=float, default=C.LIFT_OFFSET,
                        help="approach_z 위로 추가로 들어올리는 높이 [m]")
    parser.add_argument("--no-lift", action="store_true", help="파지 후 들어올리기 생략")

    # ── ROS2 / action ──
    parser.add_argument("--config", default="motion2/config/teach_pick_place_waypoints.yaml")
    parser.add_argument("--move-group-action", default="/move_action")
    parser.add_argument("--arm-action", default="/arm_controller/follow_joint_trajectory")
    parser.add_argument("--gripper-action", default="/gripper_controller/gripper_cmd")
    parser.add_argument("--group-name", default="arm")
    parser.add_argument("--base-frame", default="link0")
    parser.add_argument("--ee-frame", default="link6")

    # ── planning / 허용오차 / guard (run_pick_place 기본값과 동일) ──
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--velocity-scale", type=float, default=0.03)
    parser.add_argument("--acceleration-scale", type=float, default=0.03)
    parser.add_argument("--position-tolerance", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance", type=float, default=0.15)
    parser.add_argument("--max-joint-delta", type=float, default=0.35)
    parser.add_argument("--max-segment-delta", type=float, default=0.12)
    parser.add_argument("--no-constrain-joint5", dest="constrain_joint5",
                        action="store_false", default=True)
    parser.add_argument("--joint5-tolerance", type=float, default=0.08)
    parser.add_argument("--gripper-max-effort", type=float, default=0.0)

    # ── 실행 제어 ──
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--no-step-prompts", action="store_true",
                        help="단계별 타이핑 확인 생략 (권장하지 않음)")
    parser.add_argument("--force", action="store_true",
                        help="롤아웃 미수렴/분포 경고를 무시하고 진행 (주의)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if args.execute and args.confirm != C.CONFIRM_TEXT:
        raise RuntimeError(f"Refusing to execute. --confirm {C.CONFIRM_TEXT} 필요")

    # run_pick_place / real_moveit_common 헬퍼 재사용
    from run_pick_place import (
        DEFAULT_WORKSPACE, VERTICAL_GRIP_QUAT, ARM_JOINTS,
        _plan_arm_pose, _send_arm_traj, _send_gripper, _confirm,
        _load_yaml, _resolve_path, _read_z_from_waypoint, _read_gripper_targets,
    )
    from real_moveit_common import assert_in_workspace, joint_state_once, lookup_current_pose

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory, GripperCommand
    from moveit_msgs.action import MoveGroup

    box_xy = np.array([args.box_x, args.box_y], dtype=np.float64)
    box_yaw = float(args.box_yaw)

    # ── z 높이 결정 (YAML 또는 CLI) ──
    data = _load_yaml(_resolve_path(args.config))
    try:
        approach_z = args.approach_z if args.approach_z is not None else _read_z_from_waypoint(data, "1")
        grasp_z = args.grasp_z if args.grasp_z is not None else _read_z_from_waypoint(data, "2")
    except Exception as exc:
        approach_z = args.approach_z if args.approach_z is not None else C.APPROACH_Z_FALLBACK
        grasp_z = args.grasp_z if args.grasp_z is not None else C.GRASP_Z_FALLBACK
        print(f"[run_grasp] YAML z 읽기 실패 ({exc}) → fallback "
              f"approach={approach_z:.3f} grasp={grasp_z:.3f}")
    lift_z = approach_z + args.lift_offset
    close_val, _open_val = _read_gripper_targets(data)

    workspace = dict(DEFAULT_WORKSPACE)

    # ── 박스 분포 경고 ──
    box_warns = _check_box_distribution(box_xy, box_yaw)
    print("\n[run_grasp] ─── 입력 ───────────────────────────────────────────")
    print(f"[run_grasp] box        : x={box_xy[0]:.4f} y={box_xy[1]:.4f} "
          f"yaw={box_yaw:.4f} rad ({math.degrees(box_yaw):.1f}°)")
    print(f"[run_grasp] align_yaw  : {args.align_yaw}")
    print(f"[run_grasp] z heights  : approach={approach_z:.4f} grasp={grasp_z:.4f} "
          f"lift={lift_z:.4f} (no_lift={args.no_lift})")
    print(f"[run_grasp] gripper    : close={close_val:.3f}")
    for w in box_warns:
        print(f"[run_grasp] ⚠️  {w}")

    # ── 정책 로드 (ROS 연결 전 — 빠르게 실패 가능) ──
    print("[run_grasp] loading grasp policy ...")
    policy = GraspPolicy()

    # ── ROS2 init + 현재 상태 ──
    rclpy.init(args=None)
    node = Node("motion2_run_grasp")
    mg_client = ActionClient(node, MoveGroup, args.move_group_action)
    arm_client = ActionClient(node, FollowJointTrajectory, args.arm_action)
    gripper_client = ActionClient(node, GripperCommand, args.gripper_action)

    try:
        print("[run_grasp] reading current EE pose (TF2) + joint states ...")
        cur_pos, _cur_quat = lookup_current_pose(
            node, rclpy, args.base_frame, args.ee_frame, args.joint_state_timeout)
        current_joints = joint_state_once(node, rclpy, args.joint_state_timeout)
        missing = [j for j in ARM_JOINTS if j not in current_joints]
        if missing:
            raise RuntimeError(f"/joint_states 에 관절 없음: {missing}")
        current_arm = np.array([current_joints[j] for j in ARM_JOINTS], dtype=np.float64)

        # link6 TF → grip center (offset 보정). 롤아웃은 grip center 기준.
        off = np.array(C.GRIP_CENTER_OFFSET_XY, dtype=np.float64)
        start_grip_xy = cur_pos[:2] + off
        start_yaw = 0.0  # sim reset 기준 (실제 로봇은 수직 파지 자세에서 시작 가정)

        start_dist = float(np.linalg.norm(box_xy - start_grip_xy))
        print(f"[run_grasp] current EE : link6=({cur_pos[0]:.4f}, {cur_pos[1]:.4f}, "
              f"{cur_pos[2]:.4f})  grip_xy=({start_grip_xy[0]:.4f}, {start_grip_xy[1]:.4f})")
        print(f"[run_grasp] start dist : {start_dist*100:.1f} cm (EE→box)")
        if start_dist > C.FAIL_XY_THRESHOLD:
            print(f"[run_grasp] ⚠️  시작 EE가 박스에서 {start_dist*100:.1f}cm > "
                  f"{C.FAIL_XY_THRESHOLD*100:.0f}cm — 학습 분포 밖. EE를 박스 근처로 먼저 이동 권장.")

        # ── 내부 롤아웃 ──
        print("[run_grasp] rolling out grasp policy (box fixed, EE virtual integrate) ...")
        res = rollout_grasp(
            policy, box_xy, box_yaw, start_grip_xy, start_yaw, align_yaw=args.align_yaw)
        print(f"[run_grasp] rollout    : status={res.status} steps={res.n_steps}")
        print(f"[run_grasp]   final grip xy = ({res.ee_xy[0]:.4f}, {res.ee_xy[1]:.4f}) "
              f"yaw = {res.ee_yaw:.4f} rad ({math.degrees(res.ee_yaw):.1f}°)")
        print(f"[run_grasp]   align err: xy={np.round(res.final_xy_err*1000,2).tolist()} mm "
              f"yaw={math.degrees(res.final_yaw_err):.2f}°")

        if not res.converged:
            print(f"[run_grasp] ❌ 롤아웃 미수렴 (status={res.status}). "
                  f"박스 좌표/시작 자세를 확인하세요.")
            if not args.force:
                print("[run_grasp] 중단 (강제로 진행하려면 --force). command_sent=false")
                return 2

        if box_warns and not args.force and args.execute:
            print("[run_grasp] ⚠️  분포 경고가 있어 실행을 막습니다 (--force 로 무시). command_sent=false")
            return 2

        # ── grip center target → link6 target (offset 역보정) ──
        target_xy = res.ee_xy - off
        ee_yaw = res.ee_yaw if args.align_yaw else 0.0
        # orientation: yaw 정렬 시 R_z(yaw)⊗VERTICAL, 아니면 VERTICAL 그대로
        target_quat = (
            _compose_yaw_quat(ee_yaw, VERTICAL_GRIP_QUAT)
            if args.align_yaw else VERTICAL_GRIP_QUAT.copy()
        )

        # ── 시퀀스 좌표 workspace 검증 ──
        for label, z in [("pre_grasp", approach_z), ("grasp", grasp_z), ("lift", lift_z)]:
            if label == "lift" and args.no_lift:
                continue
            assert_in_workspace(label, np.array([target_xy[0], target_xy[1], z]), workspace)

        print("\n[run_grasp] ─── 시퀀스 ─────────────────────────────────────────")
        print(f"[run_grasp]   1. pre_grasp : ({target_xy[0]:.4f}, {target_xy[1]:.4f}, {approach_z:.4f})")
        print(f"[run_grasp]   2. grasp     : ({target_xy[0]:.4f}, {target_xy[1]:.4f}, {grasp_z:.4f})")
        print(f"[run_grasp]   3. close_gripper : {close_val:.3f}")
        if not args.no_lift:
            print(f"[run_grasp]   4. lift      : ({target_xy[0]:.4f}, {target_xy[1]:.4f}, {lift_z:.4f})")
        print(f"[run_grasp]   target quat (wxyz): {np.round(target_quat,4).tolist()}")
        print("[run_grasp] ────────────────────────────────────────────────────\n")

        # ── action server 대기 ──
        if not mg_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"MoveGroup 없음: {args.move_group_action}")
        if args.execute:
            if not arm_client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"arm action 없음: {args.arm_action}")
            if not gripper_client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"gripper action 없음: {args.gripper_action}")

        plan_kwargs = dict(
            node=node, rclpy=rclpy, mg_client=mg_client,
            quat=target_quat, workspace=workspace,
            group_name=args.group_name, base_frame=args.base_frame, ee_frame=args.ee_frame,
            planning_time=args.planning_time, attempts=args.attempts,
            pos_tol=args.position_tolerance, ori_tol=args.orientation_tolerance,
            velocity_scale=args.velocity_scale, acceleration_scale=args.acceleration_scale,
            max_joint_delta=args.max_joint_delta, max_segment_delta=args.max_segment_delta,
            constrain_joint5=args.constrain_joint5, joint5_tol=args.joint5_tolerance,
        )

        def do_arm(label, x, y, z, duration):
            nonlocal current_arm
            print(f"\n[run_grasp] ▶ {label}  target=({x:.4f}, {y:.4f}, {z:.4f})")
            ok, goal_joints = _plan_arm_pose(
                **plan_kwargs, x=x, y=y, z=z, current_arm=current_arm, label=label)
            if not ok:
                return False
            if not args.execute:
                print(f"[run_grasp] {label}: dry-run OK (command_sent=false)")
                return True
            if not args.no_step_prompts:
                _confirm(label)
            success = _send_arm_traj(
                node, rclpy, arm_client, ARM_JOINTS, goal_joints.tolist(), duration, label)
            if success:
                current_arm = goal_joints.copy()
            return success

        def do_gripper(label, position):
            print(f"\n[run_grasp] ▶ {label}  position={position:.3f}")
            if not args.execute:
                print(f"[run_grasp] {label}: dry-run OK (command_sent=false)")
                return True
            if not args.no_step_prompts:
                _confirm(label)
            return _send_gripper(
                node, rclpy, gripper_client, position, args.gripper_max_effort, label)

        steps = [
            lambda: do_arm("1.pre_grasp", target_xy[0], target_xy[1], approach_z, DURATION_PRE_GRASP),
            lambda: do_arm("2.grasp", target_xy[0], target_xy[1], grasp_z, DURATION_GRASP),
            lambda: do_gripper("3.close_gripper", close_val),
        ]
        if not args.no_lift:
            steps.append(
                lambda: do_arm("4.lift", target_xy[0], target_xy[1], lift_z, DURATION_LIFT))

        for step_fn in steps:
            if not step_fn():
                print("\n[run_grasp] FAILED — 시퀀스 중단 command_sent=false")
                return 2

        status = "SUCCESS" if args.execute else "DRY-RUN COMPLETE"
        print(f"\n[run_grasp] {status}  command_sent={args.execute}")
        return 0

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
