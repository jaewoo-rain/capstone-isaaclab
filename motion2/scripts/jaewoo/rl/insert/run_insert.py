"""rl/insert/run_insert.py — insert RL 정책으로 셀 위 yaw 정렬 + place 하강 (sim2real).

⚠️⚠️ 실험적 / 미검증 골격. grasp(run_grasp.py)만큼 완성되지 않았다. README.md 의
   "단독 실행 한계" 를 반드시 읽고, 충분히 dry-run 으로 검증한 뒤에만 실행할 것.

insert 는 yaw_only RL: xy 는 셀에 고정(IK), 정책은 손목 yaw 만 정렬한다.
전제: 박스를 이미 **잡고**, 뒤로 **돌려**, 셀 위 **호버 자세**가 된 상태에서 시작
      (앞단 grasp→lift→turn 은 이 스크립트 밖. 별도로 만들어야 함).

시퀀스:
  1. 현재 EE yaw(TF2) 읽기 → insert 정책 yaw 롤아웃 → 최종 정렬 yaw
  2. align : 호버 높이에서 (cell_x, cell_y, hover_z) + 정렬 yaw 로 손목 회전
  3. place : (cell_x, cell_y, place_z) 하강 ← ⚠️ sim 에서도 xy 드리프트 병목 단계

기본 dry-run. 실제 실행: --execute --confirm EXECUTE_INSERT

⚠️ 셀이 로봇 뒤쪽(-x)이면 기본 workspace(x_min=-0.10)를 벗어난다. --x-min 등으로 조정.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent   # .../rl/insert
_JAEWOO_DIR = _THIS_DIR.parents[1]        # .../scripts/jaewoo
_SCRIPTS_DIR = _THIS_DIR.parents[2]       # .../scripts
for _p in (str(_JAEWOO_DIR), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C
from insert_policy import InsertPolicy
from insert_rollout import rollout_insert, fold_yaw_sym


DURATION_ALIGN = 8.0
DURATION_PLACE = 8.0


def _compose_yaw_quat(ee_yaw: float, base_quat: np.ndarray) -> np.ndarray:
    """R_z(ee_yaw) ⊗ base_quat — insert_env 의 target quat 합성과 동일."""
    from real_moveit_common import quat_from_z_yaw, quat_mul
    return quat_mul(quat_from_z_yaw(ee_yaw), base_quat)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="insert RL(yaw-only) — 셀 위 yaw 정렬 + place 하강. 실험적.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── 셀 좌표 (base link0 기준) — 지금은 수동, 나중에 비전 ──
    parser.add_argument("--cell-x", type=float, required=True, help="셀 중심 x [m] (link0)")
    parser.add_argument("--cell-y", type=float, required=True, help="셀 중심 y [m] (link0)")
    parser.add_argument("--cell-yaw", type=float, default=0.0, help="셀 yaw [rad]")

    # ── z 높이 ──
    parser.add_argument("--hover-z", type=float, default=C.EE_FIXED_Z,
                        help="정렬 호버 높이 [m] (insert_env ee_fixed_z=0.20)")
    parser.add_argument("--place-z", type=float, default=None,
                        help="place 하강 높이 [m]. 미지정 시 하강 단계 생략")

    # ── ROS2 / action ──
    parser.add_argument("--move-group-action", default="/move_action")
    parser.add_argument("--arm-action", default="/arm_controller/follow_joint_trajectory")
    parser.add_argument("--group-name", default="arm")
    parser.add_argument("--base-frame", default="link0")
    parser.add_argument("--ee-frame", default="link6")

    # ── planning / guard ──
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

    # ── workspace (셀이 뒤쪽이면 조정 필요) ──
    parser.add_argument("--x-min", type=float, default=-0.10)
    parser.add_argument("--x-max", type=float, default=0.55)
    parser.add_argument("--y-min", type=float, default=-0.45)
    parser.add_argument("--y-max", type=float, default=0.20)
    parser.add_argument("--z-min", type=float, default=0.05)
    parser.add_argument("--z-max", type=float, default=0.55)

    # ── 실행 제어 ──
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--no-step-prompts", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if args.execute and args.confirm != C.CONFIRM_TEXT:
        raise RuntimeError(f"Refusing to execute. --confirm {C.CONFIRM_TEXT} 필요")

    from run_pick_place import (
        VERTICAL_GRIP_QUAT, ARM_JOINTS,
        _plan_arm_pose, _send_arm_traj, _confirm,
    )
    from real_moveit_common import assert_in_workspace, joint_state_once, lookup_current_pose

    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory
    from moveit_msgs.action import MoveGroup

    cell_xy = np.array([args.cell_x, args.cell_y], dtype=np.float64)
    cell_yaw = float(args.cell_yaw)
    workspace = {
        "x_min": args.x_min, "x_max": args.x_max,
        "y_min": args.y_min, "y_max": args.y_max,
        "z_min": args.z_min, "z_max": args.z_max,
    }

    print("\n[run_insert] ⚠️  실험적 스크립트 — 전제: 박스 잡고 셀 위 호버 자세에서 시작")
    print(f"[run_insert] cell      : x={cell_xy[0]:.4f} y={cell_xy[1]:.4f} "
          f"yaw={cell_yaw:.4f} rad ({math.degrees(cell_yaw):.1f}°)")
    if abs(cell_yaw) > C.CELL_YAW_MAX:
        print(f"[run_insert] ⚠️  cell_yaw 가 학습 범위 ±{math.degrees(C.CELL_YAW_MAX):.0f}° 밖")

    print("[run_insert] loading insert policy ...")
    policy = InsertPolicy()

    rclpy.init(args=None)
    node = Node("motion2_run_insert")
    mg_client = ActionClient(node, MoveGroup, args.move_group_action)
    arm_client = ActionClient(node, FollowJointTrajectory, args.arm_action)

    try:
        print("[run_insert] reading current EE pose (TF2) + joints ...")
        _cur_pos, _cur_quat = lookup_current_pose(
            node, rclpy, args.base_frame, args.ee_frame, args.joint_state_timeout)
        current_joints = joint_state_once(node, rclpy, args.joint_state_timeout)
        missing = [j for j in ARM_JOINTS if j not in current_joints]
        if missing:
            raise RuntimeError(f"/joint_states 에 관절 없음: {missing}")
        current_arm = np.array([current_joints[j] for j in ARM_JOINTS], dtype=np.float64)

        # 현재 EE yaw 추출 (link6 quat → z yaw)
        w, x, y, z = (_cur_quat[0], _cur_quat[1], _cur_quat[2], _cur_quat[3])
        start_ee_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        print(f"[run_insert] start ee yaw : {start_ee_yaw:.4f} rad ({math.degrees(start_ee_yaw):.1f}°)")

        # ── yaw 롤아웃 ──
        res = rollout_insert(policy, cell_yaw, start_ee_yaw, cell_xy=cell_xy)
        print(f"[run_insert] rollout    : status={res.status} steps={res.n_steps}")
        print(f"[run_insert]   final ee yaw = {res.ee_yaw:.4f} rad ({math.degrees(res.ee_yaw):.1f}°)  "
              f"yaw_err={math.degrees(res.final_yaw_err):.2f}°")
        if not res.converged:
            print("[run_insert] ❌ yaw 롤아웃 미수렴 — 시작 자세/셀 yaw 확인. command_sent=false")
            return 2

        target_quat = _compose_yaw_quat(res.ee_yaw, VERTICAL_GRIP_QUAT)

        # workspace 검증
        assert_in_workspace("align", np.array([cell_xy[0], cell_xy[1], args.hover_z]), workspace)
        if args.place_z is not None:
            assert_in_workspace("place", np.array([cell_xy[0], cell_xy[1], args.place_z]), workspace)

        print("\n[run_insert] ─── 시퀀스 ───────────────────────────────────────")
        print(f"[run_insert]   1. align : ({cell_xy[0]:.4f}, {cell_xy[1]:.4f}, {args.hover_z:.4f}) yaw={math.degrees(res.ee_yaw):.1f}°")
        if args.place_z is not None:
            print(f"[run_insert]   2. place : ({cell_xy[0]:.4f}, {cell_xy[1]:.4f}, {args.place_z:.4f}) ⚠️ 드리프트 주의")
        print(f"[run_insert]   target quat(wxyz): {np.round(target_quat,4).tolist()}")
        print("[run_insert] ──────────────────────────────────────────────────\n")

        if not mg_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"MoveGroup 없음: {args.move_group_action}")
        if args.execute and not arm_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"arm action 없음: {args.arm_action}")

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
            print(f"\n[run_insert] ▶ {label}  target=({x:.4f}, {y:.4f}, {z:.4f})")
            ok, goal_joints = _plan_arm_pose(
                **plan_kwargs, x=x, y=y, z=z, current_arm=current_arm, label=label)
            if not ok:
                return False
            if not args.execute:
                print(f"[run_insert] {label}: dry-run OK (command_sent=false)")
                return True
            if not args.no_step_prompts:
                _confirm(label)
            success = _send_arm_traj(
                node, rclpy, arm_client, ARM_JOINTS, goal_joints.tolist(), duration, label)
            if success:
                current_arm = goal_joints.copy()
            return success

        steps = [lambda: do_arm("1.align", cell_xy[0], cell_xy[1], args.hover_z, DURATION_ALIGN)]
        if args.place_z is not None:
            steps.append(lambda: do_arm("2.place", cell_xy[0], cell_xy[1], args.place_z, DURATION_PLACE))

        for step_fn in steps:
            if not step_fn():
                print("\n[run_insert] FAILED — 중단 command_sent=false")
                return 2

        status = "SUCCESS" if args.execute else "DRY-RUN COMPLETE"
        print(f"\n[run_insert] {status}  command_sent={args.execute}")
        return 0

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
