"""result/insert/run_insert_vision.py — 카메라(천장 셀)→RL→실물 insert 오케스트레이터 (I1~I3).

╔══════════════════════════════════════════════════════════════════════════════╗
║ ⚠️ 전제 (반드시 읽을 것) — 이 스크립트는 grasp/turn 을 하지 않는다.              ║
║   박스를 이미 **잡고(close)**, 뒤로 **돌려(turn)**, 셀 위 **호버 자세**가 된     ║
║   상태에서 시작한다고 **전제**한다. 여기서 하는 일은:                            ║
║     ① 천장캠으로 셀 좌표/yaw 감지 → ② 손목 yaw 정렬 → ③ (옵션) place 하강·release ║
║   앞단 chain(grasp→lift→turn)은 이 폴더 밖이며 별도로 선행돼야 한다.             ║
╚══════════════════════════════════════════════════════════════════════════════╝

insert 는 yaw-only RL: xy 는 셀에 고정(IK), 정책은 손목 yaw 만 정렬한다.
grasp(run_grasp_vision.py)의 검증된 단일프로세스 패턴(self-contained sys.path,
_read_pose_once, do_arm 클로저, dry-run 가드, 안전·로그 스타일)을 그대로 본떠,
rl/insert 의 검증된 primitive(rollout_insert)를 재사용한다.

────────────────────────────────────────────────────────────────────────────
⚠️ 단독 실행 3대 한계 (rl/insert/README.md — grasp 와 결정적으로 다른 점)
  1. 앞단 선행 필요 — "박스 잡고·뒤로 돌려·셀 위 호버"가 된 상태에서 시작. 그
     앞단(grasp→lift→turn)은 이 폴더 밖이라 별도 통합 필요. 이 스크립트는 그
     호버 자세가 이미 됐다고 전제한다.
  2. 뒤쪽 좌표계 / workspace — 셀은 로봇 **뒤쪽(-x)** 에 있다. 앞쪽(+x) 기준
     기본 workspace 가 그대로 안 맞아 --x-min 기본을 -0.55 로 둔다(README 따라).
  3. place 하강 드리프트 — yaw 를 잘 맞춰도 셀로 하강할 때 xy 가 8~10cm 드리프트
     하는 게 sim 에서도 미해결 병목(IK/모션, RL 아님). 실물에선 더 클 수 있으니
     --place-z 하강은 충분히 검증 후에만.

────────────────────────────────────────────────────────────────────────────
스테이지
  I1 SENSE_CELL : /vision/cell_coarse (PoseArray, 모든 셀, link0) 1프레임 구독 →
                  --cell-index(기본 0)로 타깃 슬롯 1개 선택 → (cell_x, cell_y, cell_yaw).
                  --dry-coords X Y YAW 로 비전 우회. cell_yaw 가 config.CELL_YAW_MAX
                  밖이면 경고.
  I2 ALIGN      : 현재 EE yaw(TF lookup_current_pose→quat→z yaw) → rollout_insert(
                  cell_yaw, start_ee_yaw, cell_xy) → 미수렴이면 중단(--force 로만 진행).
                  수렴 시 _compose_yaw_quat(ee_yaw, VERTICAL_GRIP_QUAT) → target_quat →
                  do_arm("I2.align", cell_x, cell_y, hover_z) 로 손목 yaw 회전.
                  (xy 는 셀 고정 — yaw-only.)
                  ※ 손목 슬롯-yaw fine 반복루프는 만들지 않는다(슬롯-yaw 토픽 미구현).
                    향후 /vision/cell_fine 이 생기면 grasp S3 식 yaw 반복루프 추가 가능.
  I3 PLACE      : (옵션, --place-z 줄 때만) ⚠️ place 하강 xy 드리프트 병목 경고 출력 →
                  do_arm("I3.place", cell_x, cell_y, place_z) 하강 → 그리퍼 open(release;
                  --no-open 으로 생략 가능) → do_arm("I3.retract", cell_x, cell_y, hover_z) 복귀.

────────────────────────────────────────────────────────────────────────────
안전 (실제 로봇 — run_grasp_vision.py 와 동일)
  - 기본 dry-run. --execute + --confirm EXECUTE_INSERT (config.CONFIRM_TEXT) 둘 다
    있어야 실제 이동. confirm 틀리면 ROS 연결 전에 차단.
  - 모든 cartesian 이동은 do_arm 단일 통로 → 3단계 guard(assert_in_workspace +
    _plan_arm_pose 의 plan/actual delta). dry-run 이면 plan-only, command_sent=false.
  - 단계별 타이핑 확인 기본 on(--no-step-prompts 로 off).
  - 발산 target 으로 이동 금지: 롤아웃 미수렴이면 --force 라도 이동 강행하지 않는다.

사용 예시:
  # 1) dry-run (비전 우회 — ROS/비전 없이 인자파싱·정책 로드까지)
  python3 run_insert_vision.py --dry-coords -0.45 0.10 0.0

  # 2) align(yaw 회전)만 실제 실행 (place 생략)
  python3 run_insert_vision.py \\
      --execute --confirm EXECUTE_INSERT

  # 3) place 하강·release 까지 (⚠️ 드리프트 검증 후)
  python3 run_insert_vision.py --place-z 0.07 \\
      --execute --confirm EXECUTE_INSERT

사전 조건 (별도 터미널):
  ssh root@omy-SNPR44B1021.local
  ros2 launch open_manipulator_bringup omy_f3m.launch.py
  # + ceiling_detector --ros (/vision/cell_coarse PoseArray)
  # + 박스를 이미 잡고 셀 위 호버 자세로 만들어 둔 앞단 chain
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

# result/insert/ 는 self-contained 폴더 — 필요한 모든 모듈(config, insert_policy,
# insert_rollout, run_pick_place, real_moveit_common)과 checkpoints/·waypoints YAML 을
# 이 폴더에 복사해 두어, 다른 폴더 의존 없이 여기서 바로 import 한다.
# (배포: 이 폴더만 통째로 scp/docker cp 하면 됨. 원본은 rl/insert·jaewoo·scripts 에 그대로 둠.)
_THIS_DIR = pathlib.Path(__file__).resolve().parent     # .../result/insert
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import config as C                                        # noqa: E402
from insert_policy import InsertPolicy                     # noqa: E402
from insert_rollout import rollout_insert                  # noqa: E402


# ── 단계별 이동 시간 [s] (run_grasp_vision 과 동일, 최소 4.0) ──
DURATION_ALIGN = 8.0
DURATION_PLACE = 8.0
DURATION_RETRACT = 8.0


def _compose_yaw_quat(ee_yaw: float, base_quat: np.ndarray) -> np.ndarray:
    """R_z(ee_yaw) ⊗ base_quat — insert_env 의 target quat 합성과 동일 (run_insert.py 와 동일)."""
    from real_moveit_common import quat_from_z_yaw, quat_mul
    return quat_mul(quat_from_z_yaw(ee_yaw), base_quat)


# ─────────────────────────────────────────────────────────────────────────────
# 비전 토픽 1프레임 디코드 (PoseArray → 셀 목록)
#   Pose 인코딩(ARCHITECTURE §2): position.x/y = link0 xy, orientation = quat_from_z_yaw(yaw),
#   position.z = 카메라 depth (무시).
#   셀이 여러 개 — 그중 타깃 슬롯 선택은 이 insert bridge 의 책임(ARCHITECTURE §5).
# ─────────────────────────────────────────────────────────────────────────────

def _yaw_from_quat(q) -> float:
    """quaternion → z축 yaw [rad] (quat_from_z_yaw 의 pure-z 도 정확히 복원)."""
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _read_cells_once(node, rclpy, topic: str, timeout: float):
    """주어진 node 에서 `topic`(PoseArray) 최신 1프레임을 timeout 내에 받는다.

    Returns: [(x, y, yaw), ...] (모든 셀) or None (timeout). z(depth)는 무시한다.
    이미 떠 있는 node 를 재사용하므로 rclpy.init/shutdown 은 호출자 책임.
    """
    from geometry_msgs.msg import PoseArray
    from rclpy.duration import Duration

    holder = {"msg": None}
    sub = node.create_subscription(
        PoseArray, topic, lambda m: holder.__setitem__("msg", m), 10)
    try:
        deadline = node.get_clock().now() + Duration(seconds=timeout)
        while rclpy.ok() and holder["msg"] is None and node.get_clock().now() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_subscription(sub)

    msg = holder["msg"]
    if msg is None:
        return None
    cells = []
    for pose in msg.poses:
        p = pose.position
        cells.append((float(p.x), float(p.y), _yaw_from_quat(pose.orientation)))
    return cells


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="카메라(천장 셀)→RL→실물 insert 오케스트레이터 (I1 sense + I2 yaw align + I3 place).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── I1 비전 (천장 셀 PoseArray) ──
    parser.add_argument("--cell-topic", default="/vision/cell_coarse",
                        help="I1 천장캠 셀 토픽 (PoseArray, 모든 셀, link0)")
    parser.add_argument("--cell-index", type=int, default=0,
                        help="I1 셀 목록에서 선택할 타깃 슬롯 인덱스 (범위밖이면 에러)")
    parser.add_argument("--cell-timeout", type=float, default=10.0,
                        help="I1 cell_coarse 대기 타임아웃 [s]")
    parser.add_argument("--dry-coords", nargs=3, type=float, default=None,
                        metavar=("X", "Y", "YAW"),
                        help="I1 비전 우회: 셀 좌표 직접 지정(배선 테스트용)")

    # ── z 높이 ──
    parser.add_argument("--hover-z", type=float, default=C.EE_FIXED_Z,
                        help="I2 정렬 호버 높이 [m] (config.EE_FIXED_Z=0.20)")
    parser.add_argument("--place-z", type=float, default=None,
                        help="I3 place 하강 높이 [m]. 미지정 시 place 단계 생략")
    parser.add_argument("--place-z-floor", type=float, default=0.08,
                        help="I3 place 하강 최저 높이 하한 [m] (셀/바닥 충돌 방어). "
                             "place_z 가 이보다 낮거나 hover_z 이상이면 중단")
    parser.add_argument("--place-drift-tol", type=float, default=0.03,
                        help="place 하강 직전 실제 EE-셀 xy 편차 허용 [m]. "
                             "초과 시 중단(드리프트 충돌 방지). --force 로만 강행")
    parser.add_argument("--no-open", action="store_true",
                        help="I3 place 후 그리퍼 open(release) 생략")

    # ── ROS2 / action (run_grasp_vision 과 동일) ──
    # 기본값 = 이 폴더에 복사된 로컬 waypoints YAML(절대경로) — _resolve_path 가 절대경로는
    # 그대로 쓰므로 다른 폴더 의존 없이 그리퍼 open 값을 읽는다. (★ 복사된 run_pick_place 의
    #  _repo_root()=parents[3] 가 깨지므로 반드시 로컬 절대경로를 넘긴다.)
    parser.add_argument("--config", default=str(_THIS_DIR / "teach_pick_place_waypoints.yaml"))
    parser.add_argument("--move-group-action", default="/move_action")
    parser.add_argument("--arm-action", default="/arm_controller/follow_joint_trajectory")
    parser.add_argument("--gripper-action", default="/gripper_controller/gripper_cmd")
    parser.add_argument("--group-name", default="arm")
    parser.add_argument("--base-frame", default="link0")
    parser.add_argument("--ee-frame", default="link6")

    # ── planning / 허용오차 / guard (run_insert 기본값과 동일) ──
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

    # ── workspace (셀이 뒤쪽 -x → x_min 기본 -0.55; README 따라) ──
    parser.add_argument("--x-min", type=float, default=-0.55)
    parser.add_argument("--x-max", type=float, default=0.55)
    parser.add_argument("--y-min", type=float, default=-0.45)
    parser.add_argument("--y-max", type=float, default=0.20)
    parser.add_argument("--z-min", type=float, default=0.10)  # grasp DEFAULT_WORKSPACE 와 동일(충돌 방어선)
    parser.add_argument("--z-max", type=float, default=0.55)

    # ── 실행 제어 (run_grasp_vision 과 동일) ──
    parser.add_argument("--joint-state-timeout", type=float, default=5.0)
    parser.add_argument("--no-step-prompts", action="store_true",
                        help="단계별 타이핑 확인 생략 (권장하지 않음)")
    parser.add_argument("--force", action="store_true",
                        help="롤아웃 미수렴 경고를 무시하고 진행 (주의 — 발산 target 이동은 여전히 금지)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    args = parser.parse_args()

    if args.execute and args.confirm != C.CONFIRM_TEXT:
        raise RuntimeError(f"Refusing to execute. --confirm {C.CONFIRM_TEXT} 필요")

    # ── 헬퍼 import (run_pick_place / real_moveit_common 은 ROS 없이도 import 됨;
    #    실제 rclpy/control_msgs/moveit_msgs 는 아래에서 따로 지연 import) ──
    from run_pick_place import (
        VERTICAL_GRIP_QUAT, ARM_JOINTS,
        _plan_arm_pose, _send_arm_traj, _send_gripper, _confirm,
        _load_yaml, _resolve_path, _read_gripper_targets,
    )
    from real_moveit_common import assert_in_workspace, joint_state_once, lookup_current_pose

    # ── 그리퍼 open(release) 값 (로컬 절대경로 YAML — self-contained) ──
    #   ★ 복사된 run_pick_place._repo_root() 가 깨지므로 절대경로를 _resolve_path 에 넘긴다.
    try:
        data = _load_yaml(_resolve_path(args.config))
        _close_val, open_val = _read_gripper_targets(data)
    except Exception as exc:
        open_val = 0.0
        print(f"[insert_vision] YAML 그리퍼 값 읽기 실패 ({exc}) → open fallback={open_val:.3f}")

    workspace = {
        "x_min": args.x_min, "x_max": args.x_max,
        "y_min": args.y_min, "y_max": args.y_max,
        "z_min": args.z_min, "z_max": args.z_max,
    }

    print("\n[insert_vision] ─── 설정 ─────────────────────────────────────────")
    print("[insert_vision] ⚠️  실험적 — 전제: 박스 잡고·뒤로 돌려·셀 위 호버 자세에서 시작 (grasp/turn 안 함)")
    print(f"[insert_vision] execute   : {args.execute}")
    print(f"[insert_vision] cell-index: {args.cell_index}")
    print(f"[insert_vision] z heights : hover={args.hover_z:.4f}"
          + (f" place={args.place_z:.4f} (no_open={args.no_open})"
             if args.place_z is not None else "  place=SKIP (--place-z 미지정)"))
    print(f"[insert_vision] workspace : x[{args.x_min:.2f},{args.x_max:.2f}] "
          f"y[{args.y_min:.2f},{args.y_max:.2f}] z[{args.z_min:.2f},{args.z_max:.2f}]")

    # ── 정책 로드 (ROS 연결 전 — 빠르게 실패 가능. dry-coords 배선 테스트도 여기까지 도달) ──
    print("[insert_vision] loading insert policy ...")
    policy = InsertPolicy()

    # ── 무거운 ROS import 는 정책 로드 후로 지연 (run_grasp_vision 패턴 — ROS 없으면 여기서 멈춤) ──
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from control_msgs.action import FollowJointTrajectory, GripperCommand
    from moveit_msgs.action import MoveGroup

    # ── ROS2 init + 노드/클라이언트 (단일 프로세스 — 루프 내내 재사용) ──
    rclpy.init(args=None)
    node = Node("motion2_run_insert_vision")
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
        """단일 cartesian 이동 primitive (run_grasp_vision.do_arm 과 동일 구조).

        3단계 guard(workspace assert + _plan_arm_pose 의 plan/actual delta) →
        dry-run 이면 plan 만, execute 면 타이핑 확인 후 FollowJointTrajectory.
        성공 시 state['current_arm'] 갱신.
        """
        assert_in_workspace(label, np.array([x, y, z]), workspace)
        print(f"\n[insert_vision] ▶ {label}  target=({x:.4f}, {y:.4f}, {z:.4f})")
        ok, goal_joints = _plan_arm_pose(
            **plan_kwargs, quat=quat, x=x, y=y, z=z,
            current_arm=state["current_arm"], label=label)
        if not ok:
            return False
        if not args.execute:
            print(f"[insert_vision] {label}: dry-run OK (command_sent=false)")
            return True
        if not args.no_step_prompts:
            _confirm(label)
        success = _send_arm_traj(
            node, rclpy, arm_client, ARM_JOINTS, goal_joints.tolist(), duration, label)
        if success:
            state["current_arm"] = goal_joints.copy()
        return success

    def do_gripper(label, position) -> bool:
        print(f"\n[insert_vision] ▶ {label}  position={position:.3f}")
        if not args.execute:
            print(f"[insert_vision] {label}: dry-run OK (command_sent=false)")
            return True
        if not args.no_step_prompts:
            _confirm(label)
        return _send_gripper(
            node, rclpy, gripper_client, position, args.gripper_max_effort, label)

    try:
        # ── 현재 EE/joints 읽기 (yaw 롤아웃 시작점 + delta guard 기준) ──
        print("[insert_vision] reading current EE pose (TF2) + joint states ...")
        _cur_pos, cur_quat = lookup_current_pose(
            node, rclpy, args.base_frame, args.ee_frame, args.joint_state_timeout)
        current_joints = joint_state_once(node, rclpy, args.joint_state_timeout)
        missing = [j for j in ARM_JOINTS if j not in current_joints]
        if missing:
            raise RuntimeError(f"/joint_states 에 관절 없음: {missing}")
        state["current_arm"] = np.array(
            [current_joints[j] for j in ARM_JOINTS], dtype=np.float64)

        # 현재 EE yaw 추출 (link6 quat → z yaw; run_insert.py:148-150 방식)
        w, x, y, z = (cur_quat[0], cur_quat[1], cur_quat[2], cur_quat[3])
        start_ee_yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        print(f"[insert_vision] start ee yaw : {start_ee_yaw:.4f} rad "
              f"({math.degrees(start_ee_yaw):.1f}°)")

        # ── action server 대기 ──
        if not mg_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"MoveGroup 없음: {args.move_group_action}")
        if args.execute:
            if not arm_client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError(f"arm action 없음: {args.arm_action}")
            if (args.place_z is not None and not args.no_open
                    and not gripper_client.wait_for_server(timeout_sec=10.0)):
                raise RuntimeError(f"gripper action 없음: {args.gripper_action}")

        # ════════════════════════════════════════════════════════════════════
        # I1 SENSE_CELL — /vision/cell_coarse (PoseArray) 1프레임 → 슬롯 1개 선택
        # ════════════════════════════════════════════════════════════════════
        print("\n[insert_vision] ═══ I1 SENSE_CELL ═════════════════════════════")
        if args.dry_coords is not None:
            cell_x, cell_y, cell_yaw = args.dry_coords
            print(f"[insert_vision] I1 --dry-coords (비전 우회): "
                  f"x={cell_x:.4f} y={cell_y:.4f} yaw={cell_yaw:.4f}")
        else:
            print(f"[insert_vision] I1 {args.cell_topic} 구독 (최대 {args.cell_timeout:.0f}s) ...")
            cells = _read_cells_once(node, rclpy, args.cell_topic, args.cell_timeout)
            if cells is None:
                print(f"[insert_vision] ❌ I1 {args.cell_timeout:.0f}s 내 {args.cell_topic} 없음.\n"
                      f"  확인: ceiling_detector(--ros) 가 떠 있고 셀이 보이는가?\n"
                      f"        ros2 topic echo {args.cell_topic} --once\n"
                      f"  command_sent=false")
                return 2
            if len(cells) == 0:
                print(f"[insert_vision] ❌ I1 {args.cell_topic} 셀 0개 (PoseArray 비어 있음). "
                      f"셀이 FOV 안에 있는지 확인. command_sent=false")
                return 2
            print(f"[insert_vision] I1 셀 {len(cells)}개 수신:")
            for i, (cx, cy, cyaw) in enumerate(cells):
                mark = " ← 선택" if i == args.cell_index else ""
                print(f"[insert_vision]     [{i}] x={cx:.4f} y={cy:.4f} "
                      f"yaw={math.degrees(cyaw):.1f}°{mark}")
            if not (0 <= args.cell_index < len(cells)):
                print(f"[insert_vision] ❌ I1 --cell-index {args.cell_index} 가 셀 "
                      f"개수({len(cells)}) 범위 밖. command_sent=false")
                return 2
            cell_x, cell_y, cell_yaw = cells[args.cell_index]
            print(f"[insert_vision] I1 선택 슬롯[{args.cell_index}]: x={cell_x:.4f} "
                  f"y={cell_y:.4f} yaw={cell_yaw:.4f} rad ({math.degrees(cell_yaw):.1f}°)")

        cell_xy = np.array([cell_x, cell_y], dtype=np.float64)
        cell_yaw = float(cell_yaw)
        if abs(cell_yaw) > C.CELL_YAW_MAX:
            print(f"[insert_vision] ⚠️  cell_yaw {math.degrees(cell_yaw):.1f}° 가 학습 범위 "
                  f"±{math.degrees(C.CELL_YAW_MAX):.0f}° 밖 — 정렬 품질 저하 가능.")

        # ════════════════════════════════════════════════════════════════════
        # I2 ALIGN — 현재 EE yaw → rollout_insert → 손목 yaw 회전 (xy 는 셀 고정)
        # ════════════════════════════════════════════════════════════════════
        print("\n[insert_vision] ═══ I2 ALIGN ══════════════════════════════════")
        res = rollout_insert(policy, cell_yaw, start_ee_yaw, cell_xy=cell_xy)
        print(f"[insert_vision] I2 rollout: status={res.status} steps={res.n_steps}")
        print(f"[insert_vision]   final ee yaw = {res.ee_yaw:.4f} rad "
              f"({math.degrees(res.ee_yaw):.1f}°)  yaw_err={math.degrees(res.final_yaw_err):.2f}°")
        if not res.converged:
            print(f"[insert_vision] ❌ I2 yaw 롤아웃 미수렴 (status={res.status}). "
                  f"시작 자세/셀 yaw 확인. "
                  + ("--force 라도 발산 target 이동은 금지 — 중단. "
                     if args.force else "중단 (--force 로 무시 가능). ")
                  + "command_sent=false")
            return 2

        target_quat = _compose_yaw_quat(res.ee_yaw, VERTICAL_GRIP_QUAT)
        if not do_arm("I2.align", cell_xy[0], cell_xy[1], args.hover_z,
                      target_quat, DURATION_ALIGN):
            print("[insert_vision] FAILED I2 — 중단 command_sent=false")
            return 2

        # ════════════════════════════════════════════════════════════════════
        # I3 PLACE — (옵션) 셀로 하강 → release → 호버 복귀
        # ════════════════════════════════════════════════════════════════════
        if args.place_z is None:
            print("\n[insert_vision] ═══ I3 PLACE: SKIP (--place-z 미지정) ════════")
        else:
            print("\n[insert_vision] ═══ I3 PLACE ══════════════════════════════════")
            print("[insert_vision] ⚠️  place 하강 xy 드리프트 병목 — sim 에서도 8~10cm 미해결. "
                  "실물에선 더 클 수 있음. 충분히 검증 후에만.")

            # (a) place-z 하한/상한 sanity — 셀/바닥 충돌 방어
            if args.place_z < args.place_z_floor or args.place_z >= args.hover_z:
                print(f"[insert_vision] ❌ --place-z={args.place_z:.3f} 가 허용범위 "
                      f"[{args.place_z_floor:.3f}, {args.hover_z:.3f}) 밖. "
                      f"중단 command_sent=false")
                return 2

            # (b) 하강 직전 실제 EE xy 재조회 → 셀과 편차 크면(드리프트) 중단(충돌 방지).
            #     delta guard 는 관절변위만 봐서 xy 드리프트를 못 막으므로 여기서 별도 차단.
            cur_pos_now, _q_now = lookup_current_pose(
                node, rclpy, args.base_frame, args.ee_frame, args.joint_state_timeout)
            drift = float(np.linalg.norm(np.asarray(cur_pos_now[:2]) - cell_xy))
            print(f"[insert_vision] place 직전 EE-셀 xy 편차 = {drift*1000:.1f}mm "
                  f"(허용 {args.place_drift_tol*1000:.0f}mm)")
            if drift > args.place_drift_tol and not args.force:
                print(f"[insert_vision] ❌ EE 가 셀에서 {drift*1000:.1f}mm 벗어남(드리프트) "
                      f"→ 하강 중단(충돌 방지). --force 로만 강행. command_sent=false")
                return 2

            i3_steps = [
                lambda: do_arm("I3.place", cell_xy[0], cell_xy[1], args.place_z,
                               target_quat, DURATION_PLACE),
            ]
            if not args.no_open:
                i3_steps.append(lambda: do_gripper("I3.open_gripper", open_val))
            i3_steps.append(
                lambda: do_arm("I3.retract", cell_xy[0], cell_xy[1], args.hover_z,
                               target_quat, DURATION_RETRACT))
            for step_fn in i3_steps:
                if not step_fn():
                    print("[insert_vision] FAILED I3 — 중단 command_sent=false")
                    return 2

        status = "SUCCESS" if args.execute else "DRY-RUN COMPLETE"
        print(f"\n[insert_vision] {status}  command_sent={args.execute}")
        return 0

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
