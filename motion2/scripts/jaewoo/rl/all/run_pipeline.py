"""rl/all/run_pipeline.py — grasp+insert 전체 적재 파이프라인 (sim2real, 실험적).

motion3 통합 chain 을 실제 OMY-F3M 로봇에서 한 번에 실행한다:
  1.다가감 → 2.grasp RL → 3a.하강 → 3b.잡기 → 3c.들기 →
  3c2.turn(joint1) → 3d.셀 위로/호버하강 → 4.insert RL →
  5a.place하강 → 5b.열기 → 6a.올라오기 → 6b.home복귀

⚠️⚠️ 실험적 / 미검증 골격(청사진). grasp(run_grasp.py) 외 단계는 실물 검증 안 됨.
   특히 ① z 높이·turn/home 관절(config 의 SIM 값)은 실측·보정 필요,
        ② 뒤쪽 셀 좌표계/자세 매핑 미검증, ③ place 하강 xy 드리프트 병목.
   반드시 dry-run 으로 모든 plan 을 확인하고, 단계별로 끊어 검증하며 진행할 것.

all/ 은 독립 모듈 — grasp/insert 폴더 config 와 충돌 피하려고 정책 로드/롤아웃을 자체 포함.
정책 zip/pkl 만 grasp/insert 폴더에서 참조한다.

기본 dry-run. 실제 실행: --execute --confirm EXECUTE_PIPELINE
"""
from __future__ import annotations

import argparse
import math
import pathlib
import pickle
import sys

import numpy as np

_THIS_DIR = pathlib.Path(__file__).resolve().parent   # .../rl/all
_JAEWOO_DIR = _THIS_DIR.parents[2]        # .../scripts/jaewoo
_SCRIPTS_DIR = _THIS_DIR.parents[3]       # .../scripts
for _p in (str(_JAEWOO_DIR), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config as C


# ─────────────────────────────────────────────────────────────────────────────
# 정책 로드 + obs 정규화 (grasp/insert 공통 — env-free)
# ─────────────────────────────────────────────────────────────────────────────
def load_policy(ckpt_path, vecnorm_path, expect_obs: int):
    from stable_baselines3 import PPO

    model = PPO.load(str(ckpt_path), device="cpu")
    with open(vecnorm_path, "rb") as f:
        vn = pickle.load(f)
    mean = np.asarray(vn.obs_rms.mean, dtype=np.float64)
    var = np.asarray(vn.obs_rms.var, dtype=np.float64)
    eps = float(getattr(vn, "epsilon", 1e-8))
    clip = float(getattr(vn, "clip_obs", 10.0))
    if model.observation_space.shape[0] != expect_obs:
        raise RuntimeError(
            f"obs dim {model.observation_space.shape[0]} != 기대 {expect_obs}: {ckpt_path}")

    def predict(obs):
        norm = np.clip((np.asarray(obs, np.float64) - mean) / np.sqrt(var + eps), -clip, clip)
        action, _ = model.predict(norm.astype(np.float32), deterministic=True)
        return np.clip(np.asarray(action, np.float64), -1.0, 1.0)

    return predict


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def fold_yaw_sym(a: float) -> float:
    return (a + math.pi / 2.0) % math.pi - math.pi / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# 롤아웃 (내부 가상 — 실제 로봇 폐루프 불가 회피). grasp/insert 폴더와 동일 로직.
# ─────────────────────────────────────────────────────────────────────────────
def rollout_grasp(predict, box_xy, box_yaw, start_xy, start_yaw, align_yaw=True):
    box_xy = np.asarray(box_xy, np.float64)
    ee_xy = np.asarray(start_xy, np.float64).copy()
    ee_yaw = float(start_yaw)
    prev_xy, prev_yaw = ee_xy.copy(), ee_yaw
    cnt = 0
    status = "max_steps"
    for step in range(C.ROLLOUT_MAX_STEPS):
        rel = box_xy - ee_xy
        yaw_err = wrap_to_pi(box_yaw - ee_yaw)
        vel = (ee_xy - prev_xy) / C.CONTROL_DT
        yaw_vel = (ee_yaw - prev_yaw) / C.CONTROL_DT
        obs = np.array([rel[0], rel[1], yaw_err, vel[0], vel[1], yaw_vel])
        if abs(rel[0]) < C.GRASP_ALIGN_XY and abs(rel[1]) < C.GRASP_ALIGN_XY and abs(yaw_err) < C.GRASP_ALIGN_YAW:
            cnt += 1
        else:
            cnt = 0
        if cnt >= C.GRASP_HOLD_STEPS:
            status = "converged"; break
        if abs(rel[0]) > C.GRASP_FAIL_XY or abs(rel[1]) > C.GRASP_FAIL_XY:
            status = "diverged"; break
        a = predict(obs)
        prev_xy, prev_yaw = ee_xy.copy(), ee_yaw
        ee_xy = ee_xy + a[:2] * C.GRASP_ACTION_SCALE_XY
        if align_yaw:
            ee_yaw = float(np.clip(ee_yaw + a[2] * C.GRASP_ACTION_SCALE_YAW, C.GRASP_EE_YAW_MIN, C.GRASP_EE_YAW_MAX))
    return ee_xy, ee_yaw, status, step + 1, np.abs(box_xy - ee_xy)


def rollout_insert(predict, cell_yaw, start_yaw, cell_xy=(0.0, 0.0)):
    cell_xy = np.asarray(cell_xy, np.float64)
    ee_xy = cell_xy.copy()           # yaw-only: xy 는 IK 가 셀 고정 → slot_rel ≈ 0
    ee_yaw = float(start_yaw)
    yaw_ref = float(start_yaw)
    prev_yaw = ee_yaw
    cnt = 0
    status = "max_steps"
    for step in range(C.ROLLOUT_MAX_STEPS):
        rel = cell_xy - ee_xy
        yaw_err = fold_yaw_sym(cell_yaw - ee_yaw)
        yaw_vel = (ee_yaw - prev_yaw) / C.CONTROL_DT
        obs = np.array([rel[0], rel[1], yaw_err, 1.0, 0.0, 0.0, yaw_vel])
        if abs(yaw_err) < C.INSERT_ALIGN_YAW:
            cnt += 1
        else:
            cnt = 0
        if cnt >= C.INSERT_HOLD_STEPS:
            status = "converged"; break
        a = predict(obs)
        prev_yaw = ee_yaw
        ee_yaw = float(np.clip(ee_yaw + a[2] * C.INSERT_ACTION_SCALE_YAW, yaw_ref - C.INSERT_YAW_MARGIN, yaw_ref + C.INSERT_YAW_MARGIN))
    return ee_yaw, status, step + 1, abs(fold_yaw_sym(cell_yaw - ee_yaw))


def _ee_yaw_from_quat(q_wxyz) -> float:
    w, x, y, z = q_wxyz
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="grasp+insert 전체 적재 파이프라인 (sim2real, 실험적).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # 앞쪽 박스 / 뒤쪽 셀 좌표 (base link0)
    parser.add_argument("--box-x", type=float, required=True)
    parser.add_argument("--box-y", type=float, required=True)
    parser.add_argument("--box-yaw", type=float, default=0.0)
    parser.add_argument("--cell-x", type=float, required=True)
    parser.add_argument("--cell-y", type=float, required=True)
    parser.add_argument("--cell-yaw", type=float, default=0.0)

    # z 높이 오버라이드 (미지정 시 config SIM 값 — ⚠️ 실측 권장)
    parser.add_argument("--pre-grasp-z", type=float, default=C.PRE_GRASP_Z)
    parser.add_argument("--grasp-z", type=float, default=C.GRASP_Z)
    parser.add_argument("--lift-z", type=float, default=C.LIFT_Z)
    parser.add_argument("--hover-z", type=float, default=C.HOVER_Z)
    parser.add_argument("--place-z", type=float, default=C.PLACE_Z)
    parser.add_argument("--retract-z", type=float, default=C.RETRACT_Z)

    # 단계 선택 (검증용으로 일부만)
    parser.add_argument("--stop-after", default="",
                        help="이 단계까지만 실행 (예: grasp_lift / turn / insert / place)")

    # ROS / action
    parser.add_argument("--move-group-action", default="/move_action")
    parser.add_argument("--arm-action", default="/arm_controller/follow_joint_trajectory")
    parser.add_argument("--gripper-action", default="/gripper_controller/gripper_cmd")
    parser.add_argument("--group-name", default="arm")
    parser.add_argument("--base-frame", default="link0")
    parser.add_argument("--ee-frame", default="link6")

    # planning / guard
    parser.add_argument("--planning-time", type=float, default=5.0)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--velocity-scale", type=float, default=0.03)
    parser.add_argument("--acceleration-scale", type=float, default=0.03)
    parser.add_argument("--position-tolerance", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance", type=float, default=0.15)
    parser.add_argument("--max-joint-delta", type=float, default=0.35)
    parser.add_argument("--max-segment-delta", type=float, default=0.12)
    parser.add_argument("--turn-max-joint-delta", type=float, default=3.2,
                        help="turn/home joint move 허용 변위 (큰 회전이라 큼)")
    parser.add_argument("--no-constrain-joint5", dest="constrain_joint5",
                        action="store_false", default=True)
    parser.add_argument("--joint5-tolerance", type=float, default=0.08)
    parser.add_argument("--gripper-max-effort", type=float, default=0.0)

    # workspace (앞+뒤 모두 커버. 셀이 뒤쪽이면 x_min 음수)
    parser.add_argument("--x-min", type=float, default=-0.55)
    parser.add_argument("--x-max", type=float, default=0.55)
    parser.add_argument("--y-min", type=float, default=-0.45)
    parser.add_argument("--y-max", type=float, default=0.45)
    parser.add_argument("--z-min", type=float, default=0.05)
    parser.add_argument("--z-max", type=float, default=0.60)

    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--no-step-prompts", action="store_true")
    parser.add_argument("--no-align-yaw", dest="align_yaw", action="store_false", default=True,
                        help="grasp yaw 정렬 끄기 (검증용)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if args.execute and args.confirm != C.CONFIRM_TEXT:
        raise RuntimeError(f"Refusing to execute. --confirm {C.CONFIRM_TEXT} 필요")

    from run_pick_place import (
        VERTICAL_GRIP_QUAT, ARM_JOINTS,
        _plan_arm_pose, _send_arm_traj, _send_gripper, _confirm,
    )
    from real_moveit_common import (
        assert_in_workspace, joint_state_once, lookup_current_pose,
        quat_from_z_yaw, quat_mul,
    )
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory, GripperCommand
    from moveit_msgs.action import MoveGroup

    box_xy = np.array([args.box_x, args.box_y], np.float64)
    cell_xy = np.array([args.cell_x, args.cell_y], np.float64)
    workspace = {"x_min": args.x_min, "x_max": args.x_max, "y_min": args.y_min,
                 "y_max": args.y_max, "z_min": args.z_min, "z_max": args.z_max}

    print("\n[pipeline] ⚠️  실험적 전체 파이프라인 — z/turn/home 은 SIM 값(실측 필요).")
    print(f"[pipeline] box  : ({box_xy[0]:.3f}, {box_xy[1]:.3f}, yaw {math.degrees(args.box_yaw):.0f}°)")
    print(f"[pipeline] cell : ({cell_xy[0]:.3f}, {cell_xy[1]:.3f}, yaw {math.degrees(args.cell_yaw):.0f}°)")

    print("[pipeline] loading policies ...")
    grasp_predict = load_policy(C.GRASP_CKPT, C.GRASP_VECNORM, 6)
    insert_predict = load_policy(C.INSERT_CKPT, C.INSERT_VECNORM, 7)

    rclpy.init(args=None)
    node = Node("motion2_run_pipeline")
    mg = ActionClient(node, MoveGroup, args.move_group_action)
    arm = ActionClient(node, FollowJointTrajectory, args.arm_action)
    grip = ActionClient(node, GripperCommand, args.gripper_action)

    try:
        cur_pos, _ = lookup_current_pose(node, rclpy, args.base_frame, args.ee_frame, args.joint_state_timeout)
        cj = joint_state_once(node, rclpy, args.joint_state_timeout)
        if [j for j in ARM_JOINTS if j not in cj]:
            raise RuntimeError("관절 누락")
        current_arm = np.array([cj[j] for j in ARM_JOINTS], np.float64)

        # ── grasp 롤아웃 (현재 EE → 박스 정렬) ──
        gxy, gyaw, gstatus, gsteps, gerr = rollout_grasp(
            grasp_predict, box_xy, args.box_yaw, cur_pos[:2], 0.0, align_yaw=args.align_yaw)
        print(f"[pipeline] grasp rollout: {gstatus} steps={gsteps} "
              f"final=({gxy[0]:.4f},{gxy[1]:.4f}) yaw={math.degrees(gyaw):.1f}° err={np.round(gerr*1000,1).tolist()}mm")
        if gstatus != "converged":
            print("[pipeline] ❌ grasp 미수렴 — 중단. command_sent=false")
            return 2

        grasp_quat = quat_mul(quat_from_z_yaw(gyaw if args.align_yaw else 0.0), VERTICAL_GRIP_QUAT)

        if not mg.wait_for_server(timeout_sec=10.0):
            raise RuntimeError("MoveGroup 없음")
        if args.execute:
            for cl, nm in [(arm, "arm"), (grip, "gripper")]:
                if not cl.wait_for_server(timeout_sec=10.0):
                    raise RuntimeError(f"{nm} action 없음")

        common = dict(node=node, rclpy=rclpy, mg_client=mg, workspace=workspace,
                      group_name=args.group_name, base_frame=args.base_frame, ee_frame=args.ee_frame,
                      planning_time=args.planning_time, attempts=args.attempts,
                      pos_tol=args.position_tolerance, ori_tol=args.orientation_tolerance,
                      velocity_scale=args.velocity_scale, acceleration_scale=args.acceleration_scale,
                      max_segment_delta=args.max_segment_delta,
                      constrain_joint5=args.constrain_joint5, joint5_tol=args.joint5_tolerance)

        def do_arm(label, x, y, z, quat, dur, max_delta=None):
            nonlocal current_arm
            print(f"\n[pipeline] ▶ {label}  ({x:.4f}, {y:.4f}, {z:.4f})")
            assert_in_workspace(label, np.array([x, y, z]), workspace)
            ok, goal = _plan_arm_pose(**common, quat=quat, x=x, y=y, z=z,
                                      max_joint_delta=max_delta or args.max_joint_delta,
                                      current_arm=current_arm, label=label)
            if not ok:
                return False
            if not args.execute:
                print(f"[pipeline] {label}: dry-run OK"); return True
            if not args.no_step_prompts:
                _confirm(label)
            if _send_arm_traj(node, rclpy, arm, ARM_JOINTS, goal.tolist(), dur, label):
                current_arm = goal.copy(); return True
            return False

        def do_joint(label, joint_dict, dur):
            """관절 직접 이동 (MoveIt 미사용, turn/home). delta guard 는 호출 전 책임."""
            nonlocal current_arm
            target = np.array([joint_dict.get(j, current_arm[i]) for i, j in enumerate(ARM_JOINTS)], np.float64)
            dmax = float(np.max(np.abs(target - current_arm)))
            print(f"\n[pipeline] ▶ {label}  (joint move, max Δ={dmax:.3f} rad)")
            if dmax > args.turn_max_joint_delta:
                print(f"[pipeline] {label}: Δ {dmax:.3f} > {args.turn_max_joint_delta} 거부"); return False
            if not args.execute:
                print(f"[pipeline] {label}: dry-run OK"); return True
            if not args.no_step_prompts:
                _confirm(label)
            if _send_arm_traj(node, rclpy, arm, ARM_JOINTS, target.tolist(), dur, label):
                current_arm = target.copy(); return True
            return False

        def do_grip(label, pos):
            print(f"\n[pipeline] ▶ {label}  gripper={pos:.2f}")
            if not args.execute:
                print(f"[pipeline] {label}: dry-run OK"); return True
            if not args.no_step_prompts:
                _confirm(label)
            return _send_gripper(node, rclpy, grip, pos, args.gripper_max_effort, label)

        STOP = args.stop_after

        # ════ 앞쪽: grasp → lift ════
        if not do_arm("1.approach", gxy[0], gxy[1], args.pre_grasp_z, grasp_quat, C.DUR_APPROACH): return 2
        if not do_arm("3a.descend", gxy[0], gxy[1], args.grasp_z, grasp_quat, C.DUR_DESCEND): return 2
        if not do_grip("3b.close", C.GRIPPER_CLOSE_CMD): return 2
        if not do_arm("3c.lift", gxy[0], gxy[1], args.lift_z, grasp_quat, C.DUR_LIFT): return 2
        if STOP == "grasp_lift":
            print("\n[pipeline] stop-after=grasp_lift 완료"); return 0

        # ════ 뒤로 돌기 (joint1) ════
        if not do_joint("3c2.turn", {"joint1": C.TURN_JOINT1}, C.DUR_TURN): return 2
        if STOP == "turn":
            print("\n[pipeline] stop-after=turn 완료"); return 0

        # turn 후 현재 EE yaw 재측정 (insert 롤아웃 시작점)
        turn_pos, turn_quat = lookup_current_pose(node, rclpy, args.base_frame, args.ee_frame, args.joint_state_timeout)
        turn_ee_yaw = _ee_yaw_from_quat(turn_quat)

        # ════ 셀 위로 이동 → 호버 하강 ════
        # ⚠️ turn 후 자세(turn_quat)로 이동. 정렬 전이라 자세 유지.
        if not do_arm("3d1.above_cell", cell_xy[0], cell_xy[1], args.lift_z,
                      quat_mul(quat_from_z_yaw(turn_ee_yaw), VERTICAL_GRIP_QUAT), C.DUR_TRANSPORT): return 2
        if not do_arm("3d2.hover", cell_xy[0], cell_xy[1], args.hover_z,
                      quat_mul(quat_from_z_yaw(turn_ee_yaw), VERTICAL_GRIP_QUAT), C.DUR_HOVER_DESCEND): return 2

        # ════ insert 롤아웃 (셀 yaw 정렬) ════
        iyaw, istatus, isteps, ierr = rollout_insert(insert_predict, args.cell_yaw, turn_ee_yaw, cell_xy=cell_xy)
        print(f"[pipeline] insert rollout: {istatus} steps={isteps} "
              f"final yaw={math.degrees(iyaw):.1f}° err={math.degrees(ierr):.2f}°")
        if istatus != "converged":
            print("[pipeline] ❌ insert 미수렴 — 중단. command_sent=false"); return 2
        insert_quat = quat_mul(quat_from_z_yaw(iyaw), VERTICAL_GRIP_QUAT)

        if not do_arm("4.insert_align", cell_xy[0], cell_xy[1], args.hover_z, insert_quat, C.DUR_HOVER_DESCEND): return 2
        if STOP == "insert":
            print("\n[pipeline] stop-after=insert 완료"); return 0

        # ════ place 하강 → 열기 → 복귀 ════
        # ⚠️ place 하강은 sim 에서도 xy 8~10cm 드리프트 병목. 충분히 검증 후.
        if not do_arm("5a.place", cell_xy[0], cell_xy[1], args.place_z, insert_quat, C.DUR_PLACE): return 2
        if not do_grip("5b.release", C.GRIPPER_OPEN_CMD): return 2
        if not do_arm("6a.retract", cell_xy[0], cell_xy[1], args.retract_z, insert_quat, C.DUR_RETRACT): return 2
        if not do_joint("6b.home", C.HOME_JOINTS, C.DUR_HOME): return 2

        status = "SUCCESS" if args.execute else "DRY-RUN COMPLETE"
        print(f"\n[pipeline] {status}  command_sent={args.execute}")
        return 0

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
