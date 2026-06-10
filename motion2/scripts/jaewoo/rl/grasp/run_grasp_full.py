"""rl/grasp/run_grasp_full.py — 홈 자세에서 박스 파지까지 통합 플로우.

박스 좌표(x, y, yaw)만 입력하면 순서대로 실행:
  0. approach   — 박스 위 대략적 위치로 먼저 이동 (홈에서 긴 거리, yaw 없이)
  1. RL rollout — grasp 정책으로 정밀 정렬 위치 계산
  2. pre_grasp  — 정밀 정렬 위치 + yaw 로 이동
  3. grasp      — grasp_z 로 하강
  4. close      — 그리퍼 닫기
  5. lift       — 들어올리기

기존 run_grasp.py 와 차이:
  - 0.approach 단계 추가 → 홈 자세에서 바로 실행 가능 (30cm 이내 제약 없음)
  - approach 완료 후 TF2 로 EE 위치 재측정 → rollout 시작점 정확하게

사용 예시:
    # dry-run
    python3 run_grasp_full.py --box-x 0.45 --box-y -0.10 --box-yaw 0.0

    # 실제 실행 (yaw 정렬 없이 먼저 검증)
    python3 run_grasp_full.py --box-x 0.45 --box-y -0.10 --box-yaw 0.0 \\
        --no-align-yaw --execute --confirm EXECUTE_GRASP \\
        --no-constrain-joint5 --max-joint-delta 3.0

    # 실제 실행 (yaw 정렬 포함)
    python3 run_grasp_full.py --box-x 0.45 --box-y -0.10 --box-yaw 0.3 \\
        --execute --confirm EXECUTE_GRASP \\
        --no-constrain-joint5 --max-joint-delta 3.0
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

_THIS_DIR   = pathlib.Path(__file__).resolve().parent   # .../rl/grasp
_JAEWOO_DIR = _THIS_DIR.parents[1]                      # .../scripts/jaewoo
_SCRIPTS_DIR = _THIS_DIR.parents[2]                     # .../scripts
for _p in (str(_JAEWOO_DIR), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C
from grasp_policy import GraspPolicy
from grasp_rollout import rollout_grasp

_rng = np.random.default_rng()


DURATION_APPROACH  = 10.0   # 홈→박스 위, 먼 거리라 여유있게
DURATION_PRE_GRASP = 8.0
DURATION_GRASP     = 8.0
DURATION_LIFT      = 8.0


def _compose_yaw_quat(ee_yaw: float, base_quat: np.ndarray) -> np.ndarray:
    from real_moveit_common import quat_from_z_yaw, quat_mul
    return quat_mul(quat_from_z_yaw(ee_yaw), base_quat)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="홈 자세에서 박스 파지까지 통합 플로우 (approach → RL → grasp → lift).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 박스 좌표
    parser.add_argument("--box-x",    type=float, required=True,  help="박스 중심 x [m] (link0)")
    parser.add_argument("--box-y",    type=float, required=True,  help="박스 중심 y [m] (link0)")
    parser.add_argument("--box-yaw",  type=float, default=0.0,    help="박스 yaw [rad]")

    # approach 노이즈 — sim 학습 분포(EE 박스에서 3~5cm 오프셋)에 맞추기 위해 의도적으로 추가
    parser.add_argument("--approach-noise", type=float, default=0.04,
                        help="approach 목표에 추가할 xy 노이즈 반경 [m]. "
                             "sim 학습 분포(3~5cm)와 맞추기 위한 의도적 오프셋. "
                             "0.0 이면 박스 정중앙으로 이동 (노이즈 없음)")

    # yaw 정렬
    parser.add_argument("--align-yaw",    dest="align_yaw", action="store_true",  default=True)
    parser.add_argument("--no-align-yaw", dest="align_yaw", action="store_false",
                        help="yaw 정렬 끄기 (검증 1단계)")

    # z 높이
    parser.add_argument("--approach-z",  type=float, default=None,
                        help="호버 높이 [m]. 미지정 시 YAML waypoints['1'] z")
    parser.add_argument("--grasp-z",     type=float, default=None,
                        help="파지 높이 [m]. 미지정 시 YAML waypoints['2'] z")
    parser.add_argument("--lift-offset", type=float, default=C.LIFT_OFFSET,
                        help="approach_z 위로 추가 높이 [m]")
    parser.add_argument("--no-lift",     action="store_true", help="파지 후 들어올리기 생략")

    # ROS2 / action
    parser.add_argument("--config",             default="motion2/config/teach_pick_place_waypoints.yaml")
    parser.add_argument("--move-group-action",  default="/move_action")
    parser.add_argument("--arm-action",         default="/arm_controller/follow_joint_trajectory")
    parser.add_argument("--gripper-action",     default="/gripper_controller/gripper_cmd")
    parser.add_argument("--group-name",         default="arm")
    parser.add_argument("--base-frame",         default="link0")
    parser.add_argument("--ee-frame",           default="link6")

    # planning / guard
    parser.add_argument("--planning-time",          type=float, default=5.0)
    parser.add_argument("--attempts",               type=int,   default=5)
    parser.add_argument("--velocity-scale",         type=float, default=0.03)
    parser.add_argument("--acceleration-scale",     type=float, default=0.03)
    parser.add_argument("--position-tolerance",     type=float, default=0.01)
    parser.add_argument("--orientation-tolerance",  type=float, default=0.15)
    parser.add_argument("--max-joint-delta",        type=float, default=0.35)
    parser.add_argument("--max-segment-delta",      type=float, default=0.12)
    parser.add_argument("--joint5-tolerance",       type=float, default=0.08)
    parser.add_argument("--gripper-max-effort",     type=float, default=0.0)

    # 실행 제어
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--no-step-prompts",     action="store_true")
    parser.add_argument("--force",               action="store_true",
                        help="롤아웃 미수렴/분포 경고 무시")
    parser.add_argument("--execute",             action="store_true")
    parser.add_argument("--confirm",             default="")
    args = parser.parse_args()

    if args.execute and args.confirm != C.CONFIRM_TEXT:
        raise RuntimeError(f"Refusing to execute. --confirm {C.CONFIRM_TEXT} 필요")

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

    box_xy  = np.array([args.box_x, args.box_y], dtype=np.float64)
    box_yaw = float(args.box_yaw)

    # z 높이 결정
    data = _load_yaml(_resolve_path(args.config))
    try:
        approach_z = args.approach_z if args.approach_z is not None else _read_z_from_waypoint(data, "1")
        grasp_z    = args.grasp_z    if args.grasp_z    is not None else _read_z_from_waypoint(data, "2")
    except Exception as exc:
        approach_z = args.approach_z if args.approach_z is not None else C.APPROACH_Z_FALLBACK
        grasp_z    = args.grasp_z    if args.grasp_z    is not None else C.GRASP_Z_FALLBACK
        print(f"[run_grasp_full] YAML z 읽기 실패 ({exc}) → fallback "
              f"approach={approach_z:.3f} grasp={grasp_z:.3f}")
    lift_z       = approach_z + args.lift_offset
    close_val, _ = _read_gripper_targets(data)
    workspace    = dict(DEFAULT_WORKSPACE)
    off          = np.array(C.GRIP_CENTER_OFFSET_XY, dtype=np.float64)

    print("\n[run_grasp_full] ─── 입력 ──────────────────────────────────────────")
    print(f"[run_grasp_full] box       : x={box_xy[0]:.4f} y={box_xy[1]:.4f} "
          f"yaw={box_yaw:.4f} rad ({math.degrees(box_yaw):.1f}°)")
    print(f"[run_grasp_full] z heights : approach={approach_z:.4f} grasp={grasp_z:.4f} "
          f"lift={lift_z:.4f}  (no_lift={args.no_lift})")
    print(f"[run_grasp_full] align_yaw : {args.align_yaw}")
    print(f"[run_grasp_full] gripper   : close={close_val:.3f}")

    print("[run_grasp_full] loading grasp policy ...")
    policy = GraspPolicy()

    rclpy.init(args=None)
    node = Node("motion2_run_grasp_full")
    mg_client      = ActionClient(node, MoveGroup,             args.move_group_action)
    arm_client     = ActionClient(node, FollowJointTrajectory, args.arm_action)
    gripper_client = ActionClient(node, GripperCommand,        args.gripper_action)

    try:
        print("[run_grasp_full] reading current EE + joints ...")
        cur_pos, _ = lookup_current_pose(
            node, rclpy, args.base_frame, args.ee_frame, args.joint_state_timeout)
        current_joints = joint_state_once(node, rclpy, args.joint_state_timeout)
        missing = [j for j in ARM_JOINTS if j not in current_joints]
        if missing:
            raise RuntimeError(f"/joint_states 에 관절 없음: {missing}")
        current_arm = np.array([current_joints[j] for j in ARM_JOINTS], dtype=np.float64)
        print(f"[run_grasp_full] current EE : ({cur_pos[0]:.4f}, {cur_pos[1]:.4f}, {cur_pos[2]:.4f})")

        if not mg_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"MoveGroup 없음: {args.move_group_action}")
        if args.execute:
            if not arm_client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"arm action 없음: {args.arm_action}")
            if not gripper_client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"gripper action 없음: {args.gripper_action}")

        # plan_kwargs 기본: 수직파지 orientation (approach 용)
        plan_kwargs = dict(
            node=node, rclpy=rclpy, mg_client=mg_client,
            quat=VERTICAL_GRIP_QUAT,
            workspace=workspace,
            group_name=args.group_name, base_frame=args.base_frame, ee_frame=args.ee_frame,
            planning_time=args.planning_time, attempts=args.attempts,
            pos_tol=args.position_tolerance, ori_tol=args.orientation_tolerance,
            velocity_scale=args.velocity_scale, acceleration_scale=args.acceleration_scale,
            max_joint_delta=args.max_joint_delta, max_segment_delta=args.max_segment_delta,
            constrain_joint5=False, joint5_tol=args.joint5_tolerance,
        )

        def do_arm(label, x, y, z, duration, quat=None):
            nonlocal current_arm
            kw = dict(plan_kwargs)
            if quat is not None:
                kw["quat"] = quat
            print(f"\n[run_grasp_full] ▶ {label}  target=({x:.4f}, {y:.4f}, {z:.4f})")
            ok, goal_joints = _plan_arm_pose(
                **kw, x=x, y=y, z=z, current_arm=current_arm, label=label)
            if not ok:
                return False
            if not args.execute:
                print(f"[run_grasp_full] {label}: dry-run OK (command_sent=false)")
                return True
            if not args.no_step_prompts:
                _confirm(label)
            success = _send_arm_traj(
                node, rclpy, arm_client, ARM_JOINTS, goal_joints.tolist(), duration, label)
            if success:
                current_arm = goal_joints.copy()
            return success

        def do_gripper(label, position):
            print(f"\n[run_grasp_full] ▶ {label}  position={position:.3f}")
            if not args.execute:
                print(f"[run_grasp_full] {label}: dry-run OK (command_sent=false)")
                return True
            if not args.no_step_prompts:
                _confirm(label)
            return _send_gripper(
                node, rclpy, gripper_client, position, args.gripper_max_effort, label)

        # ─────────────────────────────────────────────────────────────────
        # 0. approach — 박스 위로 대략적 이동 (yaw 없이, 기본 수직파지)
        #   의도적 노이즈: sim 학습 분포(EE 박스에서 3~5cm 오프셋)에 맞추기 위해
        #   박스 정중앙이 아니라 approach_noise 반경 안의 랜덤 오프셋 위치로 이동.
        # ─────────────────────────────────────────────────────────────────
        print("\n[run_grasp_full] ══ Step 0: approach ══════════════════════════")
        if args.approach_noise > 0.0:
            angle  = _rng.uniform(-math.pi, math.pi)
            radius = _rng.uniform(args.approach_noise * 0.6, args.approach_noise)
            approach_xy = box_xy + np.array([radius * math.cos(angle),
                                             radius * math.sin(angle)])
            print(f"[run_grasp_full] approach noise: r={radius*100:.1f}cm "
                  f"angle={math.degrees(angle):.1f}°  "
                  f"→ ({approach_xy[0]:.4f}, {approach_xy[1]:.4f})")
        else:
            approach_xy = box_xy.copy()
            print("[run_grasp_full] approach noise: 없음 (박스 정중앙)")

        assert_in_workspace("approach", np.array([approach_xy[0], approach_xy[1], approach_z]), workspace)
        if not do_arm("0.approach", approach_xy[0], approach_xy[1], approach_z, DURATION_APPROACH):
            print("[run_grasp_full] FAILED at 0.approach — 중단")
            return 2

        # approach 완료 후 실제 EE 위치 재측정 (rollout 시작점)
        if args.execute:
            cur_pos2, _ = lookup_current_pose(
                node, rclpy, args.base_frame, args.ee_frame, args.joint_state_timeout)
            start_grip_xy = cur_pos2[:2] + off
            print(f"[run_grasp_full] EE after approach: ({cur_pos2[0]:.4f}, {cur_pos2[1]:.4f})")
        else:
            start_grip_xy = box_xy.copy()  # dry-run: approach 정확하다고 가정

        # ─────────────────────────────────────────────────────────────────
        # 1. RL rollout — 정밀 정렬 위치 계산
        # ─────────────────────────────────────────────────────────────────
        print("\n[run_grasp_full] ══ Step 1: RL rollout ═════════════════════════")
        res = rollout_grasp(policy, box_xy, box_yaw, start_grip_xy, 0.0, align_yaw=args.align_yaw)
        print(f"[run_grasp_full] rollout : status={res.status} steps={res.n_steps}")
        print(f"[run_grasp_full]   target : xy=({res.ee_xy[0]:.4f}, {res.ee_xy[1]:.4f}) "
              f"yaw={math.degrees(res.ee_yaw):.1f}°")
        print(f"[run_grasp_full]   err    : xy={np.round(res.final_xy_err*1000,2).tolist()} mm "
              f"yaw={math.degrees(res.final_yaw_err):.2f}°")

        if not res.converged and not args.force:
            print("[run_grasp_full] ❌ 롤아웃 미수렴. --force 로 강제 진행 가능.")
            return 2

        target_xy   = res.ee_xy - off
        ee_yaw      = res.ee_yaw if args.align_yaw else 0.0
        target_quat = (
            _compose_yaw_quat(ee_yaw, VERTICAL_GRIP_QUAT)
            if args.align_yaw else VERTICAL_GRIP_QUAT.copy()
        )

        # workspace 검증
        for label, z in [("pre_grasp", approach_z), ("grasp", grasp_z), ("lift", lift_z)]:
            if label == "lift" and args.no_lift:
                continue
            assert_in_workspace(label, np.array([target_xy[0], target_xy[1], z]), workspace)

        print(f"\n[run_grasp_full] ══ Step 2~5: pre_grasp → grasp → close → lift ══")
        print(f"[run_grasp_full]   target xy=({target_xy[0]:.4f}, {target_xy[1]:.4f}) "
              f"yaw={math.degrees(ee_yaw):.1f}°")

        # ─────────────────────────────────────────────────────────────────
        # 2~5. pre_grasp → grasp → close → lift
        # ─────────────────────────────────────────────────────────────────
        steps = [
            lambda: do_arm("2.pre_grasp", target_xy[0], target_xy[1], approach_z,
                           DURATION_PRE_GRASP, quat=target_quat),
            lambda: do_arm("3.grasp",     target_xy[0], target_xy[1], grasp_z,
                           DURATION_GRASP,     quat=target_quat),
            lambda: do_gripper("4.close_gripper", close_val),
        ]
        if not args.no_lift:
            steps.append(
                lambda: do_arm("5.lift", target_xy[0], target_xy[1], lift_z,
                               DURATION_LIFT, quat=target_quat))

        for step_fn in steps:
            if not step_fn():
                print("\n[run_grasp_full] FAILED — 시퀀스 중단")
                return 2

        status = "SUCCESS" if args.execute else "DRY-RUN COMPLETE"
        print(f"\n[run_grasp_full] {status}  command_sent={args.execute}")
        return 0

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
