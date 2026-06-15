"""rl/grasp/run_grasp_vision.py — 카메라→RL→실물 grasp 단일프로세스 오케스트레이터 (S1~S4).

천장캠 coarse 1회 + 손목캠 fine 반복수렴을 **한 프로세스 안에서** 돌려 실제
OMY-F3M 박스 파지를 자동 수행한다. 기존 얇은 subprocess 버전(run_grasp.py 호출)을
대체한다 — fine 반복루프는 [구독+롤아웃+MoveIt이동]을 긴밀히 묶어야 하므로
subprocess 로는 불가능하기 때문이다 (ARCHITECTURE §8 "코드 구조 함의").

run_grasp.py 의 검증된 단일샷 primitive([현재EE읽기→롤아웃→MoveIt이동→하강/close/lift])
를 그대로 재사용한다. run_grasp.py / run_pick_place.py / real_moveit_common.py 의
헬퍼와 config.py 상수만 import 하고, 그 위에 S1~S4 상태머신을 얹는다.

────────────────────────────────────────────────────────────────────────────
스테이지 (ARCHITECTURE §8 그대로)
  S1 SENSE_COARSE  : /vision/box_target 1프레임 → coarse (box_x, box_y, box_yaw)
                     없으면 중단(command_sent=false). --dry-coords 로 비전 우회.
  S2 APPROACH      : MoveIt 으로 (box_x, box_y, hover_z) 수직파지로 이동.
  S3 FINE_REFINE   : box_est=coarse 로 시작. for i in range(N):
                       /vision/box_fine 1프레임 → 없으면 no_det++ (K회면 coarse 진행)
                       box_est=fine → rollout_grasp(box_est, start=현재 grip xy/yaw)
                       residual=|target.xy - 현재EE.xy| → hover_z 에서 target 이동
                       현재EE 갱신 → residual<xy_tol and dyaw<yaw_tol 면 break
                       롤아웃 발산(status!=converged)이면 중단.
  S4 GRASP         : 최종 EE 에서 grasp_z 하강 → close → lift_z 들기.

★ hover_z 는 approach_z/grasp_z 와 별도 CLI 인자(--hover-z, 기본 = approach_z).
  이유(ARCHITECTURE §8 물리 미지수 1): 손목 D405 는 근접센서라 호버 높이에서
  박스가 FOV/depth 범위에 들어와야 fine 검출이 된다. 현장에서 "손목이 박스를
  잘 보는 높이"를 측정해 --hover-z 로 따로 잡는다. (occlusion: §8 미지수 2도 현장확인.)

────────────────────────────────────────────────────────────────────────────
안전 (실제 로봇 — run_grasp.py 와 동일)
  - 기본 dry-run. --execute + --confirm EXECUTE_GRASP 둘 다 있어야 실제 이동.
  - 각 cartesian 이동 전 3단계 guard (workspace / plan delta / actual delta).
  - 단계별 타이핑 확인(_confirm) 기본 on, --no-step-prompts 로 off.
  - 분포/미수렴 경고는 --force 로만 무시.

사용 예시:
  # 1) dry-run (비전 우회 — ROS/비전 없이 배선·정책 로드 확인까지)
  python3 run_grasp_vision.py --dry-coords 0.45 -0.10 0.30

  # 2) 자세검증 (yaw 정렬 끔, fine 끔 = coarse only) 실제 실행
  python3 run_grasp_vision.py --no-fine --no-align-yaw \\
      --execute --confirm EXECUTE_GRASP --no-constrain-joint5 --max-joint-delta 3.0

  # 3) 전체 (천장 coarse + 손목 fine 3회 반복수렴 + yaw 정렬) 실제 실행
  python3 run_grasp_vision.py \\
      --execute --confirm EXECUTE_GRASP --no-constrain-joint5 --max-joint-delta 3.0

사전 조건 (별도 터미널):
  ssh root@omy-SNPR44B1021.local
  ros2 launch open_manipulator_bringup omy_f3m.launch.py
  # + ceiling_detector --ros (box_target), wrist_detector --ros (box_fine)
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

# result/grasp/ 는 self-contained 폴더 — 필요한 모든 모듈(config, grasp_policy, grasp_rollout,
# run_grasp, run_pick_place, real_moveit_common)과 checkpoints/·waypoints YAML 을 이 폴더에
# 복사해 두어, 다른 폴더 의존 없이 여기서 바로 import 한다.
# (배포: 이 폴더만 통째로 scp/docker cp 하면 됨. 원본은 rl/grasp·jaewoo·scripts 에 그대로 둠.)
_THIS_DIR = pathlib.Path(__file__).resolve().parent     # .../result/grasp
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import config as C                                        # noqa: E402
from grasp_policy import GraspPolicy                       # noqa: E402
from grasp_rollout import rollout_grasp, wrap_to_pi        # noqa: E402

# run_grasp.py 의 검증된 분포검사를 그대로 재사용 (동일 기준)
from run_grasp import _check_box_distribution, _compose_yaw_quat  # noqa: E402


# ── 단계별 이동 시간 [s] (run_grasp 와 동일, 최소 4.0) ──
DURATION_APPROACH = 8.0
DURATION_FINE = 8.0
DURATION_GRASP = 8.0
DURATION_LIFT = 8.0


# ─────────────────────────────────────────────────────────────────────────────
# 비전 토픽 1프레임 디코드 (PoseStamped → x, y, yaw)
#   Pose 인코딩: position.x/y = link0 xy, orientation = quat_from_z_yaw(yaw),
#   position.z = 카메라 depth (무시). ARCHITECTURE §2 Pose 인코딩 규약.
# ─────────────────────────────────────────────────────────────────────────────

def _yaw_from_quat(q) -> float:
    """quaternion → z축 yaw [rad] (quat_from_z_yaw 의 pure-z 도 정확히 복원)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _read_pose_once(node, rclpy, topic: str, timeout: float):
    """주어진 node 에서 `topic`(PoseStamped) 최신 1프레임을 timeout 내에 받는다.

    Returns: (x, y, yaw) or None (timeout). z(depth)는 무시한다.
    이미 떠 있는 node 를 재사용하므로 rclpy.init/shutdown 은 호출자 책임.
    """
    from geometry_msgs.msg import PoseStamped
    from rclpy.duration import Duration

    holder = {"msg": None}
    sub = node.create_subscription(
        PoseStamped, topic, lambda m: holder.__setitem__("msg", m), 10)
    try:
        deadline = node.get_clock().now() + Duration(seconds=timeout)
        while rclpy.ok() and holder["msg"] is None and node.get_clock().now() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_subscription(sub)

    msg = holder["msg"]
    if msg is None:
        return None
    p = msg.pose.position
    return float(p.x), float(p.y), _yaw_from_quat(msg.pose.orientation)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="카메라→RL→실물 grasp 오케스트레이터 (S1 coarse + S3 fine 반복수렴).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── S1 비전 (천장 coarse) ──
    parser.add_argument("--box-topic", default="/vision/box_target",
                        help="S1 천장캠 coarse 박스 토픽 (PoseStamped)")
    parser.add_argument("--coarse-timeout", type=float, default=10.0,
                        help="S1 box_target 대기 타임아웃 [s]")
    parser.add_argument("--dry-coords", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "YAW"),
                        help="S1 비전 우회: coarse 좌표 직접 지정(배선 테스트용)")

    # ── S3 비전 (손목 fine) 반복수렴 ──
    parser.add_argument("--no-fine", action="store_true",
                        help="S3 fine 반복수렴 생략 — coarse 만으로 파지")
    parser.add_argument("--fine-topic", default="/vision/box_fine",
                        help="S3 손목캠 fine 박스 토픽 (PoseStamped)")
    parser.add_argument("--fine-iters", type=int, default=3,
                        help="S3 fine 반복 최대 횟수 N_MAX")
    parser.add_argument("--fine-no-det-max", type=int, default=3,
                        help="S3 손목 무검출 누적 K회면 coarse 로 진행(경고)")
    parser.add_argument("--fine-timeout", type=float, default=2.0,
                        help="S3 box_fine 1프레임 대기 타임아웃 [s]")
    parser.add_argument("--refine-xy-tol", type=float, default=0.005,
                        help="S3 수렴 판정 residual xy 임계 [m] (5mm)")
    parser.add_argument("--refine-yaw-tol", type=float, default=0.05,
                        help="S3 수렴 판정 dyaw 임계 [rad] (~2.9°)")

    # ── yaw 정렬 (run_grasp 와 동일) ──
    parser.add_argument("--align-yaw", dest="align_yaw", action="store_true", default=True,
                        help="박스 yaw 에 EE 회전 정렬 (기본 활성화)")
    parser.add_argument("--no-align-yaw", dest="align_yaw", action="store_false",
                        help="yaw 정렬 끄고 수직 파지 기본 자세 유지 (검증 1단계)")

    # ── z 높이 ──
    #   hover_z 는 approach_z 와 별도 — 손목 D405 근접센서 작동거리 때문(§8 미지수 1).
    #   기본값은 approach_z 와 같게(=YAML['1']) 두고, 현장에서 손목이 박스를 잘 보는
    #   높이를 측정해 --hover-z 로 덮어쓴다.
    parser.add_argument("--hover-z", type=float, default=None,
                        help="S2/S3 호버 높이 [m]. 미지정 시 approach_z 와 동일. "
                             "손목캠이 박스를 잘 보는 높이로 별도 조정(§8).")
    parser.add_argument("--approach-z", type=float, default=None,
                        help="호버 기준 높이 [m]. 미지정 시 YAML waypoints['1'] z")
    parser.add_argument("--grasp-z", type=float, default=None,
                        help="파지 하강 높이 [m]. 미지정 시 YAML waypoints['2'] z")
    parser.add_argument("--lift-offset", type=float, default=C.LIFT_OFFSET,
                        help="approach_z 위로 추가로 들어올리는 높이 [m]")
    parser.add_argument("--no-lift", action="store_true", help="파지 후 들어올리기 생략")

    # ── ROS2 / action (run_grasp 와 동일) ──
    # 기본값 = 이 폴더에 복사된 로컬 waypoints YAML(절대경로) — _resolve_path 가 절대경로는
    # 그대로 쓰므로 다른 폴더 의존 없이 z 높이/그리퍼 값을 읽는다.
    parser.add_argument("--config", default=str(_THIS_DIR / "teach_pick_place_waypoints.yaml"))
    parser.add_argument("--move-group-action", default="/move_action")
    parser.add_argument("--arm-action", default="/arm_controller/follow_joint_trajectory")
    parser.add_argument("--gripper-action", default="/gripper_controller/gripper_cmd")
    parser.add_argument("--group-name", default="arm")
    parser.add_argument("--base-frame", default="link0")
    parser.add_argument("--ee-frame", default="link6")

    # ── planning / 허용오차 / guard (run_grasp 기본값과 동일) ──
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

    # ── 실행 제어 (run_grasp 와 동일) ──
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

    # ── 헬퍼 import (run_pick_place / real_moveit_common 은 ROS 없이도 import 됨;
    #    실제 rclpy/control_msgs/moveit_msgs 는 아래에서 따로 지연 import) ──
    from run_pick_place import (
        DEFAULT_WORKSPACE, VERTICAL_GRIP_QUAT, ARM_JOINTS,
        _plan_arm_pose, _send_arm_traj, _send_gripper, _confirm,
        _load_yaml, _resolve_path, _read_z_from_waypoint, _read_gripper_targets,
    )
    from real_moveit_common import assert_in_workspace, joint_state_once, lookup_current_pose

    # ── z 높이 결정 (YAML 또는 CLI) ──
    data = _load_yaml(_resolve_path(args.config))
    try:
        approach_z = args.approach_z if args.approach_z is not None else _read_z_from_waypoint(data, "1")
        grasp_z = args.grasp_z if args.grasp_z is not None else _read_z_from_waypoint(data, "2")
    except Exception as exc:
        approach_z = args.approach_z if args.approach_z is not None else C.APPROACH_Z_FALLBACK
        grasp_z = args.grasp_z if args.grasp_z is not None else C.GRASP_Z_FALLBACK
        print(f"[grasp_vision] YAML z 읽기 실패 ({exc}) → fallback "
              f"approach={approach_z:.3f} grasp={grasp_z:.3f}")
    hover_z = args.hover_z if args.hover_z is not None else approach_z
    lift_z = approach_z + args.lift_offset
    close_val, _open_val = _read_gripper_targets(data)
    workspace = dict(DEFAULT_WORKSPACE)

    off = np.array(C.GRIP_CENTER_OFFSET_XY, dtype=np.float64)

    print("\n[grasp_vision] ─── 설정 ─────────────────────────────────────────")
    print(f"[grasp_vision] execute   : {args.execute}")
    print(f"[grasp_vision] align_yaw : {args.align_yaw}")
    print(f"[grasp_vision] fine      : {'OFF (coarse only)' if args.no_fine else f'ON (N_MAX={args.fine_iters}, K={args.fine_no_det_max})'}")
    print(f"[grasp_vision] z heights : hover={hover_z:.4f} approach={approach_z:.4f} "
          f"grasp={grasp_z:.4f} lift={lift_z:.4f} (no_lift={args.no_lift})")
    print(f"[grasp_vision] refine tol: xy<{args.refine_xy_tol*1000:.1f}mm "
          f"yaw<{math.degrees(args.refine_yaw_tol):.1f}°")

    # ── 정책 로드 (ROS 연결 전 — 빠르게 실패 가능. dry-coords 배선 테스트도 여기까지 도달) ──
    print("[grasp_vision] loading grasp policy ...")
    policy = GraspPolicy()

    # ── 무거운 ROS import 는 정책 로드 후로 지연 (run_grasp 패턴 — ROS 없으면 여기서 멈춤) ──
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory, GripperCommand
    from moveit_msgs.action import MoveGroup

    # ── ROS2 init + 노드/클라이언트 (단일 프로세스 — 루프 내내 재사용) ──
    rclpy.init(args=None)
    node = Node("motion2_run_grasp_vision")
    mg_client = ActionClient(node, MoveGroup, args.move_group_action)
    arm_client = ActionClient(node, FollowJointTrajectory, args.arm_action)
    gripper_client = ActionClient(node, GripperCommand, args.gripper_action)

    # current_arm 은 do_arm 이 갱신하므로 클로저 공유 상태로 둔다.
    state = {"current_arm": None}

    plan_kwargs = dict(
        node=node, rclpy=rclpy, mg_client=mg_client,
        workspace=workspace,
        group_name=args.group_name, base_frame=args.base_frame, ee_frame=args.ee_frame,
        planning_time=args.planning_time, attempts=args.attempts,
        pos_tol=args.position_tolerance, ori_tol=args.orientation_tolerance,
        velocity_scale=args.velocity_scale, acceleration_scale=args.acceleration_scale,
        max_joint_delta=args.max_joint_delta, max_segment_delta=args.max_segment_delta,
        constrain_joint5=args.constrain_joint5, joint5_tol=args.joint5_tolerance,
    )

    def do_arm(label, x, y, z, quat, duration) -> bool:
        """단일 cartesian 이동 primitive (run_grasp.do_arm 과 동일 구조).

        3단계 guard(workspace assert + _plan_arm_pose 의 plan/actual delta) →
        dry-run 이면 plan 만, execute 면 타이핑 확인 후 FollowJointTrajectory.
        성공 시 state['current_arm'] 갱신.
        """
        assert_in_workspace(label, np.array([x, y, z]), workspace)
        print(f"\n[grasp_vision] ▶ {label}  target=({x:.4f}, {y:.4f}, {z:.4f})")
        ok, goal_joints = _plan_arm_pose(
            **plan_kwargs, quat=quat, x=x, y=y, z=z,
            current_arm=state["current_arm"], label=label)
        if not ok:
            return False
        if not args.execute:
            print(f"[grasp_vision] {label}: dry-run OK (command_sent=false)")
            return True
        if not args.no_step_prompts:
            _confirm(label)
        success = _send_arm_traj(
            node, rclpy, arm_client, ARM_JOINTS, goal_joints.tolist(), duration, label)
        if success:
            state["current_arm"] = goal_joints.copy()
        return success

    def do_gripper(label, position) -> bool:
        print(f"\n[grasp_vision] ▶ {label}  position={position:.3f}")
        if not args.execute:
            print(f"[grasp_vision] {label}: dry-run OK (command_sent=false)")
            return True
        if not args.no_step_prompts:
            _confirm(label)
        return _send_gripper(
            node, rclpy, gripper_client, position, args.gripper_max_effort, label)

    try:
        # ── 현재 EE/joints 읽기 (롤아웃 시작점 + delta guard 기준) ──
        print("[grasp_vision] reading current EE pose (TF2) + joint states ...")
        cur_pos, _cur_quat = lookup_current_pose(
            node, rclpy, args.base_frame, args.ee_frame, args.joint_state_timeout)
        current_joints = joint_state_once(node, rclpy, args.joint_state_timeout)
        missing = [j for j in ARM_JOINTS if j not in current_joints]
        if missing:
            raise RuntimeError(f"/joint_states 에 관절 없음: {missing}")
        state["current_arm"] = np.array(
            [current_joints[j] for j in ARM_JOINTS], dtype=np.float64)

        # link6 TF → grip center (offset 보정). 롤아웃/residual 은 grip center 기준.
        cur_grip_xy = cur_pos[:2].astype(np.float64) + off
        cur_yaw = 0.0  # 수직 파지 시작 자세 가정 (run_grasp 와 동일)

        # ── action server 대기 ──
        if not mg_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"MoveGroup 없음: {args.move_group_action}")
        if args.execute:
            if not arm_client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"arm action 없음: {args.arm_action}")
            if not gripper_client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"gripper action 없음: {args.gripper_action}")

        # ════════════════════════════════════════════════════════════════════
        # S1 SENSE_COARSE — /vision/box_target 1프레임 (or --dry-coords)
        # ════════════════════════════════════════════════════════════════════
        print("\n[grasp_vision] ═══ S1 SENSE_COARSE ═══════════════════════════")
        if args.dry_coords is not None:
            box_x, box_y, box_yaw = args.dry_coords
            print(f"[grasp_vision] S1 --dry-coords (비전 우회): "
                  f"x={box_x:.4f} y={box_y:.4f} yaw={box_yaw:.4f}")
        else:
            print(f"[grasp_vision] S1 {args.box_topic} 구독 (최대 {args.coarse_timeout:.0f}s) ...")
            got = _read_pose_once(node, rclpy, args.box_topic, args.coarse_timeout)
            if got is None:
                print(f"[grasp_vision] ❌ S1 {args.coarse_timeout:.0f}s 내 {args.box_topic} 없음.\n"
                      f"  확인: ceiling_detector(--ros) 가 떠 있고 박스가 보이는가?\n"
                      f"        ros2 topic echo {args.box_topic} --once\n"
                      f"  command_sent=false")
                return 2
            box_x, box_y, box_yaw = got
            print(f"[grasp_vision] S1 coarse 수신: x={box_x:.4f} y={box_y:.4f} "
                  f"yaw={box_yaw:.4f} rad ({math.degrees(box_yaw):.1f}°)")

        coarse_xy = np.array([box_x, box_y], dtype=np.float64)
        coarse_yaw = float(box_yaw)

        # 분포 경고 (run_grasp 와 동일 기준)
        box_warns = _check_box_distribution(coarse_xy, coarse_yaw)
        for w in box_warns:
            print(f"[grasp_vision] ⚠️  {w}")
        if box_warns and not args.force and args.execute:
            print("[grasp_vision] ⚠️  분포 경고로 실행 차단 (--force 로 무시). command_sent=false")
            return 2

        start_dist = float(np.linalg.norm(coarse_xy - cur_grip_xy))
        print(f"[grasp_vision] S1 start dist (EE→box): {start_dist*100:.1f} cm")
        if start_dist > C.FAIL_XY_THRESHOLD:
            print(f"[grasp_vision] ⚠️  시작 EE가 박스에서 {start_dist*100:.1f}cm > "
                  f"{C.FAIL_XY_THRESHOLD*100:.0f}cm — 학습 분포 밖. 호버로 먼저 이동 권장.")

        def align_and_quat(box_xy, box_yaw, start_xy, start_yaw):
            """rollout_grasp → (target link6 xy, ee_yaw, target_quat, residual, dyaw, res).

            residual/dyaw 는 grip center 기준 (start_xy, start_yaw 대비).
            """
            res = rollout_grasp(
                policy, box_xy, box_yaw, start_xy, start_yaw, align_yaw=args.align_yaw)
            ee_yaw = res.ee_yaw if args.align_yaw else 0.0
            target_xy = res.ee_xy - off  # grip center → link6
            quat = (_compose_yaw_quat(ee_yaw, VERTICAL_GRIP_QUAT)
                    if args.align_yaw else VERTICAL_GRIP_QUAT.copy())
            residual = float(np.linalg.norm(res.ee_xy - start_xy))
            dyaw = abs(wrap_to_pi(ee_yaw - start_yaw))
            return target_xy, ee_yaw, quat, residual, dyaw, res

        # ════════════════════════════════════════════════════════════════════
        # S2 APPROACH — coarse 롤아웃으로 (box_x, box_y, hover_z) 수직파지 이동
        # ════════════════════════════════════════════════════════════════════
        print("\n[grasp_vision] ═══ S2 APPROACH ═══════════════════════════════")
        target_xy, ee_yaw, target_quat, residual, dyaw, res = align_and_quat(
            coarse_xy, coarse_yaw, cur_grip_xy, cur_yaw)
        print(f"[grasp_vision] S2 rollout: status={res.status} steps={res.n_steps} "
              f"final grip=({res.ee_xy[0]:.4f}, {res.ee_xy[1]:.4f}) yaw={math.degrees(ee_yaw):.1f}°")
        if not res.converged and not args.force:
            print(f"[grasp_vision] ❌ S2 롤아웃 미수렴 (status={res.status}). "
                  f"중단 (--force 로 강제). command_sent=false")
            return 2

        if not do_arm("S2.approach", target_xy[0], target_xy[1], hover_z,
                      target_quat, DURATION_APPROACH):
            print("[grasp_vision] FAILED S2 — 중단 command_sent=false")
            return 2
        # 이동 후 grip center / yaw 갱신 (다음 롤아웃 시작점)
        cur_grip_xy = target_xy + off
        cur_yaw = ee_yaw

        # ════════════════════════════════════════════════════════════════════
        # S3 FINE_REFINE — /vision/box_fine 반복수렴 (ARCHITECTURE §8 의사코드)
        # ════════════════════════════════════════════════════════════════════
        if args.no_fine:
            print("\n[grasp_vision] ═══ S3 FINE_REFINE: SKIP (--no-fine) ════════")
        else:
            print("\n[grasp_vision] ═══ S3 FINE_REFINE ════════════════════════════")
            no_det = 0
            for i in range(args.fine_iters):
                print(f"\n[grasp_vision] S3 iter {i+1}/{args.fine_iters} "
                      f"{args.fine_topic} 1프레임 (≤{args.fine_timeout:.1f}s) ...")
                fine = _read_pose_once(node, rclpy, args.fine_topic, args.fine_timeout)
                if fine is None:
                    no_det += 1
                    print(f"[grasp_vision] S3 손목 무검출 ({no_det}/{args.fine_no_det_max}) "
                          f"— 손가락 occlusion/높이(§8) 확인.")
                    if no_det >= args.fine_no_det_max:
                        print(f"[grasp_vision] ⚠️  S3 손목 {no_det}회 연속 무검출 "
                              f"→ coarse 로 진행(break).")
                        break
                    continue

                box_est = np.array([fine[0], fine[1]], dtype=np.float64)
                box_est_yaw = float(fine[2])
                print(f"[grasp_vision] S3 fine 수신: x={box_est[0]:.4f} y={box_est[1]:.4f} "
                      f"yaw={math.degrees(box_est_yaw):.1f}°")

                target_xy, ee_yaw, target_quat, residual, dyaw, res = align_and_quat(
                    box_est, box_est_yaw, cur_grip_xy, cur_yaw)
                print(f"[grasp_vision] S3 rollout: status={res.status} "
                      f"residual={residual*1000:.1f}mm dyaw={math.degrees(dyaw):.1f}°")

                if not res.converged:
                    if not args.force:
                        print(f"[grasp_vision] ❌ S3 롤아웃 발산 (status={res.status}). "
                              f"중단 (command_sent=false)")
                        return 2
                    # --force 라도 발산 target 으로는 이동하지 않는다(안전): fine 중단하고
                    # 직전 안전 위치에서 S4 로 진행. (발산 target hover 이동 강행 금지)
                    print(f"[grasp_vision] ⚠️  S3 롤아웃 발산 (status={res.status}) + --force "
                          f"→ 이 iter 이동 생략, fine 중단하고 현재 위치에서 S4 진행.")
                    break

                if not do_arm(f"S3.fine[{i+1}]", target_xy[0], target_xy[1], hover_z,
                              target_quat, DURATION_FINE):
                    print("[grasp_vision] FAILED S3 — 중단 command_sent=false")
                    return 2
                cur_grip_xy = target_xy + off
                cur_yaw = ee_yaw

                if residual < args.refine_xy_tol and dyaw < args.refine_yaw_tol:
                    print(f"[grasp_vision] ✓ S3 수렴 (residual<{args.refine_xy_tol*1000:.1f}mm, "
                          f"dyaw<{math.degrees(args.refine_yaw_tol):.1f}°) @ iter {i+1}")
                    break
            else:
                print(f"[grasp_vision] S3 N_MAX={args.fine_iters} 소진 → 최종 정렬로 진행.")

        # ════════════════════════════════════════════════════════════════════
        # S4 GRASP — 최종 EE 에서 grasp_z 하강 → close → lift
        # ════════════════════════════════════════════════════════════════════
        print("\n[grasp_vision] ═══ S4 GRASP ══════════════════════════════════")
        # 최종 link6 target xy = 현재 grip center - offset
        final_xy = cur_grip_xy - off
        final_quat = (_compose_yaw_quat(cur_yaw, VERTICAL_GRIP_QUAT)
                      if args.align_yaw else VERTICAL_GRIP_QUAT.copy())
        print(f"[grasp_vision] S4 final EE xy=({final_xy[0]:.4f}, {final_xy[1]:.4f}) "
              f"yaw={math.degrees(cur_yaw):.1f}°  gripper close={close_val:.3f}")

        s4_steps = [
            lambda: do_arm("S4.grasp", final_xy[0], final_xy[1], grasp_z,
                           final_quat, DURATION_GRASP),
            lambda: do_gripper("S4.close_gripper", close_val),
        ]
        if not args.no_lift:
            s4_steps.append(
                lambda: do_arm("S4.lift", final_xy[0], final_xy[1], lift_z,
                               final_quat, DURATION_LIFT))
        for step_fn in s4_steps:
            if not step_fn():
                print("[grasp_vision] FAILED S4 — 중단 command_sent=false")
                return 2

        status = "SUCCESS" if args.execute else "DRY-RUN COMPLETE"
        print(f"\n[grasp_vision] {status}  command_sent={args.execute}")
        return 0

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
