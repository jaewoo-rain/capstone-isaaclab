"""motion1 — Motion-only pipeline prototype (6단계).

  1. 물체 위 이동      : home → 박스 xy + 1cm random offset (PRE_GRASP_Z)  [motion]
  2. grasp 미세조정    : 1cm 정렬 → 정확한 박스 xy                           [RL placeholder]
  3. 파지 + 적재 이동  : 수직 하강 → close → lift → transport (셀 + 1cm off) [motion]
  4. insert 미세조정   : 1cm 정렬 → 정확한 셀 xy                              [RL placeholder]
  5. 삽입              : 수직 하강 (PLACE_Z) → release                        [motion]
  6. 복귀              : retract up → home                                    [motion]

각 단계: cartesian 직선 보간 + DifferentialIKController(dls).
EE = 양 finger body(rh_p12_rn_l2 / rh_p12_rn_r2) 평균 (jacobian / pose 모두 평균).

박스 1개 / 셀 1개. 단계 2 / 4 는 RL 학습 자리지만 현재는 motion 직선으로 placeholder.
RL 도입 시 박스 위치 / yaw randomize 추가 예정.

실행:
    ./isaaclab.sh -p source/motion1/scripts/play_motion_chain.py
    ./isaaclab.sh -p source/motion1/scripts/play_motion_chain.py --gripper_close 0.7
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# -------------------- argparse / app launcher --------------------
parser = argparse.ArgumentParser(description="motion1: motion-only pick-and-place pipeline.")
parser.add_argument("--gripper_close", type=float, default=0.8,
                    help="박스 잡을 때 gripper 4-joint 동일 명령 (rad). 박스 슬립 시 키울 것.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--repeat", type=int, default=1,
                    help="파이프라인을 몇 회 반복하고 종료할지 (default 1).")
parser.add_argument("--hold_s", type=float, default=5.0,
                    help="모든 반복 후 hold 시간(초). 끝나면 자동 종료. 0=즉시, -1=무한 hold(GUI 수동 종료).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------- imports (after app start) --------------------
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms, quat_slerp

from source.omy.omy_robot_cfg import OMY_OFF_SELF_COLLISION_CFG

# -------------------- constants (env-relative coords) --------------------
BOX_SPAWN = (0.45, -0.10, 0.07)  # z=0.07 → 박스 바닥 ground 위 1.1cm (spawn 안정)
BOX_SIZE = (0.139, 0.044, 0.118)
BOX_MASS = 0.3

CELL_CENTER_X = 0.25
CELL_CENTER_Y = -0.45
# 박스 long edge=X(0.139), short edge=Y(0.044) 그대로. 셀도 X 가 길쭉.
CELL_INNER_X = 0.16    # 박스 long edge(0.139) + 여유
CELL_INNER_Y = 0.065   # 박스 short edge(0.044) + 여유
WALL_THICKNESS = 0.008
WALL_HEIGHT = 0.12

# 박스 spawn rot — yaw=0 (회전 없음). default home 에서 finger sep Y축 → 박스 short edge(Y) 잡음.
BOX_SPAWN_ROT = (1.0, 0.0, 0.0, 0.0)

# z 좌표
PRE_GRASP_Z = BOX_SPAWN[2] + 0.1       # 박스 위 5cm  (≈0.120)
GRASP_Z = BOX_SPAWN[2]+0.06            # 박스 중심   (=0.070)
LIFT_Z = 0.30                          # 충분히 들어올림
TRANSPORT_Z = 0.25                     # 셀 위
PLACE_Z = BOX_SPAWN[2] + 0.095          # 셀 안 박스 중심 (박스 바닥 셀 바닥 위 1.1cm)
RETRACT_Z = 0.25

# 단계별 지속시간(초). dt 로 나눠서 step 수 계산.
# IK가 trajectory 따라가는 시간이 충분히 필요하므로 넉넉하게.
STAGE_DURATION_S: dict[str, float] = {
    "move_above_box": 2.0,    # 1. home → pre_grasp_offset (박스 위 1cm offset)
    "align_grasp":    0.6,    # 2. RL placeholder — 1cm 정렬
    "descend":        1.0,    # 3a. PRE_GRASP_Z → GRASP_Z (수직 하강)
    "close":          1.5,    # 3b. gripper close + hold
    "lift":           2.0,    # 3c. GRASP_Z → LIFT_Z
    "transport":      3.0,    # 3d. lift → transport_offset (셀 위 1cm offset)
    "align_insert":   0.6,    # 4. RL placeholder — 1cm 정렬
    "insert":         1.5,    # 5a. TRANSPORT_Z → PLACE_Z
    "release":        0.7,    # 5b. gripper open
    "retract_up":     1.5,    # 6a. PLACE_Z → RETRACT_Z
    "retract_home":   3.0,    # 6b. retract → home
}

# 정렬 단계 placeholder 의 의도된 xy 오차 (RL 학습 시 박스/셀 위치 noise 시뮬용)
ALIGN_OFFSET_M = 0.03   # 1cm

# 각 stage 끝의 settle 시간 — IK 가 trajectory 끝점에 수렴할 시간.
SETTLE_S = 0.8

GRIPPER_OPEN = 0.0

# 셀 벽 사이즈 / 위치 (모듈 레벨, configclass 밖 — dataclass field 충돌 방지)
_V_WALL_SIZE = (WALL_THICKNESS, CELL_INNER_Y + 2 * WALL_THICKNESS, WALL_HEIGHT)
_H_WALL_SIZE = (CELL_INNER_X + 2 * WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT)
_WALL_Z = WALL_HEIGHT / 2


# -------------------- Scene cfg --------------------
@configclass
class MotionSceneCfg(InteractiveSceneCfg):
    """1박스 + 1셀(4벽) + OMY 로봇."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
    )

    robot = OMY_OFF_SELF_COLLISION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.7, 0.72), metallic=0.5, roughness=0.4,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",   # 박스 finger 사이 마찰: 두 material 중 큰 값 사용
                static_friction=3.0,
                dynamic_friction=3.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_SPAWN, rot=BOX_SPAWN_ROT),
    )

    # 셀 4벽 (수직 2 + 수평 2). 셀 1개 hard-coded.
    wall_v_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_VL",
        spawn=sim_utils.CuboidCfg(
            size=_V_WALL_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CELL_CENTER_X - CELL_INNER_X / 2 - WALL_THICKNESS / 2,
                 CELL_CENTER_Y, _WALL_Z),
        ),
    )
    wall_v_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_VR",
        spawn=sim_utils.CuboidCfg(
            size=_V_WALL_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CELL_CENTER_X + CELL_INNER_X / 2 + WALL_THICKNESS / 2,
                 CELL_CENTER_Y, _WALL_Z),
        ),
    )
    wall_h_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_HF",
        spawn=sim_utils.CuboidCfg(
            size=_H_WALL_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CELL_CENTER_X,
                 CELL_CENTER_Y - CELL_INNER_Y / 2 - WALL_THICKNESS / 2, _WALL_Z),
        ),
    )
    wall_h_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_HB",
        spawn=sim_utils.CuboidCfg(
            size=_H_WALL_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CELL_CENTER_X,
                 CELL_CENTER_Y + CELL_INNER_Y / 2 + WALL_THICKNESS / 2, _WALL_Z),
        ),
    )


# -------------------- helpers --------------------
def grip_center_pos(robot, l_id: int, r_id: int) -> torch.Tensor:
    """월드 좌표에서 grip center (양 finger 평균). 반환 (num_envs, 3)."""
    return 0.5 * (robot.data.body_pos_w[:, l_id] + robot.data.body_pos_w[:, r_id])


def grip_center_quat(robot, l_id: int) -> torch.Tensor:
    """grip center quaternion (왼쪽 finger 사용 — 양쪽 거의 동일). (num_envs, 4)."""
    return robot.data.body_quat_w[:, l_id]


def grip_center_jacobian(robot, l_jac_idx: int, r_jac_idx: int,
                         joint_ids: list[int]) -> torch.Tensor:
    """양 finger 의 jacobian 평균. (num_envs, 6, num_joints)."""
    J = robot.root_physx_view.get_jacobians()
    j_l = J[:, l_jac_idx, :, :][:, :, joint_ids]
    j_r = J[:, r_jac_idx, :, :][:, :, joint_ids]
    return 0.5 * (j_l + j_r)


def cartesian_lerp(start: torch.Tensor, end: torch.Tensor, num_steps: int) -> torch.Tensor:
    """선형 cartesian 보간. start, end: (3,). 반환 (num_steps, 3)."""
    alphas = torch.linspace(0, 1, num_steps, device=start.device).unsqueeze(-1)
    return start.unsqueeze(0) * (1 - alphas) + end.unsqueeze(0) * alphas


# -------------------- pipeline --------------------
def run_pipeline(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot = scene["robot"]
    box = scene["box"]
    device = sim.device
    dt = sim.get_physics_dt()
    duration_to_steps = lambda s: max(1, int(s / dt))

    # --- joint / body indices ---
    arm_names = [f"joint{i}" for i in range(1, 7)]
    gripper_names = ["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"]
    arm_ids = [robot.find_joints(n)[0][0] for n in arm_names]
    gripper_ids = [robot.find_joints(n)[0][0] for n in gripper_names]
    all_joint_ids = arm_ids + gripper_ids

    left_id = robot.find_bodies("rh_p12_rn_l2")[0][0]
    right_id = robot.find_bodies("rh_p12_rn_r2")[0][0]
    print(f"[motion1] body names: {robot.body_names}")
    print(f"[motion1] joint names: {robot.joint_names}")
    print(f"[motion1] arm ids: {arm_ids}, gripper ids: {gripper_ids}")
    print(f"[motion1] finger body ids: L={left_id} R={right_id}")
    print(f"[motion1] is_fixed_base: {robot.is_fixed_base}")

    # fixed-base → jacobian idx = body_id - 1
    if robot.is_fixed_base:
        l_jac, r_jac = left_id - 1, right_id - 1
    else:
        l_jac, r_jac = left_id, right_id

    # --- IK controller ---
    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=device)

    # --- 1) HOME 자세: OMY_CFG init_state 명시 ---
    # default_joint_pos 가 박스 collision 으로 영향받을 가능성 → cfg 값 직접 적용
    HOME_JOINT_POS = {
        "joint1": 0.0, "joint2": -1.55, "joint3": 2.66,
        "joint4": -1.1, "joint5": 1.6, "joint6": 0.0,
        "rh_r1_joint": 0.0, "rh_r2": 0.0, "rh_l1": 0.0, "rh_l2": 0.0,
    }
    home_q = torch.zeros((scene.num_envs, robot.num_joints), device=device)
    for n, v in HOME_JOINT_POS.items():
        jid = robot.find_joints(n)[0][0]
        home_q[:, jid] = v
    joint_vel = torch.zeros_like(home_q)
    robot.write_joint_state_to_sim(home_q, joint_vel)
    # PD 가 init_state target 으로 끌어가지 않도록 target 도 명시적으로 설정
    robot.set_joint_position_target(home_q)
    robot.reset()
    box.reset()
    for _ in range(60):
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

    env_origin = scene.env_origins[0]  # (3,)
    home_grip_w = grip_center_pos(robot, left_id, right_id)[0]
    home_grip_quat_w = grip_center_quat(robot, left_id)[0]
    home_grip_env = home_grip_w - env_origin
    l_h = robot.data.body_pos_w[0, left_id]
    r_h = robot.data.body_pos_w[0, right_id]
    sep_h = r_h - l_h
    print(f"[motion1] home grip (env-rel): {home_grip_env.tolist()}")
    print(f"[motion1] home grip quat (wxyz): {home_grip_quat_w.tolist()}")
    print(f"[motion1] home finger sep (R-L): {sep_h.tolist()} | norm={sep_h.norm().item():.4f}")
    print(f"[motion1]   -> finger sep 은 X축. 박스 yaw=90° spawn 으로 박스 short edge(X)를 잡음.")

    # --- target EE orientation 직접 명시 ---
    # 이전: grasp pose 로 워프해서 EE quat 측정 → hardcoded joint 자세에 의존, 부정확.
    # 신규: world frame 에서 "수직 아래 + finger Y 양옆" 자세를 quat 으로 직접 정의.
    # IK 가 알아서 joint 풀어줌.
    #
    # URDF 분석: finger body 의 local frame 은 wrist (rh_p12_rn_*1) 와 동일 (rpy=0).
    # finger sep 방향 = local Y 축. → world frame 에서 ee local Y = world ±Y 가 되어야 함.
    # ee local Z = world -Z (수직 아래) 이어야 함.
    #
    # 시도/에러 후보 (시도 1: (0, 1, 0, 0) = 180° around world X):
    #   - 결과: ee z flip (world -z), ee y flip, ee x 그대로
    # 만약 결과가 finger 가 박스 short edge 잡지 않으면 (0, 0, 1, 0) 으로 변경 시도.
    EE_DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)   # (w, x, y, z)
    ee_quat_target_w = torch.tensor([list(EE_DOWN_QUAT)], device=device)  # (1, 4)
    print(f"[motion1] target quat (hardcoded vertical-down): {ee_quat_target_w[0].tolist()}")

    # --- waypoints (env-relative xyz) ---
    bx, by, _ = BOX_SPAWN
    cx, cy = CELL_CENTER_X, CELL_CENTER_Y

    # 정렬 단계 (RL placeholder) 의 시작점 — 박스/셀 xy 에서 임의 방향 1cm 빗나간 곳.
    # 매 run 마다 random 방향 (RL 학습 시 generalize 용 시뮬).
    import math as _math
    ang_g = float(torch.rand(1).item() * 2 * _math.pi)
    ang_i = float(torch.rand(1).item() * 2 * _math.pi)
    off_grasp = (ALIGN_OFFSET_M * _math.cos(ang_g), ALIGN_OFFSET_M * _math.sin(ang_g))
    off_insert = (ALIGN_OFFSET_M * _math.cos(ang_i), ALIGN_OFFSET_M * _math.sin(ang_i))
    print(f"[motion1] random align offset (grasp,  1cm): dxy=({off_grasp[0]*100:+.2f}, {off_grasp[1]*100:+.2f}) cm")
    print(f"[motion1] random align offset (insert, 1cm): dxy=({off_insert[0]*100:+.2f}, {off_insert[1]*100:+.2f}) cm")

    waypoints = {
        "home":              home_grip_env,
        # --- grasp side ---
        "pre_grasp_offset":  torch.tensor([bx + off_grasp[0], by + off_grasp[1], PRE_GRASP_Z], device=device),
        "pre_grasp":         torch.tensor([bx, by, PRE_GRASP_Z], device=device),  # 정확한 박스 위
        "grasp":             torch.tensor([bx, by, GRASP_Z],     device=device),  # 박스 잡는 깊이
        "lift":              torch.tensor([bx, by, LIFT_Z],      device=device),
        # --- insert side ---
        "transport_offset":  torch.tensor([cx + off_insert[0], cy + off_insert[1], TRANSPORT_Z], device=device),
        "transport":         torch.tensor([cx, cy, TRANSPORT_Z], device=device),  # 정확한 셀 위
        "place":             torch.tensor([cx, cy, PLACE_Z],     device=device),  # release z
        "retract":           torch.tensor([cx, cy, RETRACT_Z],   device=device),
    }
    for k, v in waypoints.items():
        print(f"[motion1] waypoint {k:>18s}: {v.tolist()}")

    gripper_close = float(args_cli.gripper_close)
    print(f"[motion1] gripper close cmd: {gripper_close:.3f} rad (open=0.0)")

    # --- one IK control step ---
    def control_step(target_pos_env: torch.Tensor, gripper_value: float,
                     target_quat_w: torch.Tensor | None = None):
        """한 sim step 동안 ee 가 target_pos_env (월드로 변환됨) 향해 이동.

        target_quat_w: (1, 4) 또는 None. None 이면 모듈 default(ee_quat_target_w) 사용.
        """
        if target_quat_w is None:
            target_quat_w = ee_quat_target_w
        ee_pos_w = grip_center_pos(robot, left_id, right_id)  # (1, 3)
        ee_quat_w = grip_center_quat(robot, left_id)          # (1, 4)
        cur_arm_q = robot.data.joint_pos[:, arm_ids]          # (1, 6)
        jac = grip_center_jacobian(robot, l_jac, r_jac, arm_ids)  # (1, 6, 6)

        target_pos_w = target_pos_env.unsqueeze(0) + env_origin.unsqueeze(0)  # (1, 3)
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w

        # world → root frame
        tgt_pos_b, tgt_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w
        )
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
        )

        ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))
        arm_target = ik.compute(ee_pos_b, ee_quat_b, jac, cur_arm_q)  # (1, 6)

        # gripper_names 순서: ["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"]
        # 끝 link (l2/r2) 는 기저 link (l1/r1) 의 약 2.3× 로 굽혀야 finger 가 안쪽 모임
        # (USD mimic 깨짐 → 코드 레벨에서 비율 명령. example7 fallback 자세에서 추정.)
        tip_ratio = 2.3
        gripper_target = torch.tensor(
            [[gripper_value, gripper_value * tip_ratio, gripper_value, gripper_value * tip_ratio]],
            device=device,
        ).expand(scene.num_envs, -1)
        full_target = torch.cat([arm_target, gripper_target], dim=-1)  # (1, 10)

        robot.set_joint_position_target(full_target, joint_ids=all_joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)

    def report(tag: str):
        ee = (grip_center_pos(robot, left_id, right_id)[0] - env_origin).tolist()
        bx_w = box.data.root_pos_w[0] - env_origin
        bx = bx_w.tolist()
        d_ee_box = (grip_center_pos(robot, left_id, right_id)[0] - box.data.root_pos_w[0]).norm().item()
        print(f"  [status @ {tag:24s}] ee=({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f}) "
              f"box=({bx[0]:+.3f},{bx[1]:+.3f},{bx[2]:+.3f}) ee<->box={d_ee_box*100:.1f}cm")

    def stage_move(start_key: str, end_key: str, dur_s: float, gripper_val: float, label: str,
                   start_quat_w: torch.Tensor | None = None,
                   end_quat_w: torch.Tensor | None = None):
        """cartesian 위치 보간 + 옵션으로 quat slerp.

        start_quat_w / end_quat_w 둘 다 명시하면 매 step quat slerp.
        둘 다 None 이면 ee_quat_target_w (고정) 사용.
        """
        n = duration_to_steps(dur_s)
        s = duration_to_steps(SETTLE_S)
        traj = cartesian_lerp(waypoints[start_key], waypoints[end_key], n)  # (n, 3)
        end_target = waypoints[end_key]
        do_slerp = (start_quat_w is not None) and (end_quat_w is not None)
        slerp_tag = "+slerp" if do_slerp else ""
        print(f"[stage] {label:30s} {start_key} -> {end_key} | move={n} settle={s} grip={gripper_val:.2f}{slerp_tag}")
        for i in range(n):
            if do_slerp:
                assert start_quat_w is not None and end_quat_w is not None  # narrowing for type checker
                tau = (i + 1) / n  # 0~1
                q_i = quat_slerp(start_quat_w[0], end_quat_w[0], tau).unsqueeze(0)  # (1, 4)
            else:
                q_i = None
            control_step(traj[i], gripper_val, target_quat_w=q_i)
        # IK가 trajectory 끝점에 수렴할 때까지 hold (끝 quat 으로 고정)
        end_q = end_quat_w if do_slerp else None
        for _ in range(s):
            control_step(end_target, gripper_val, target_quat_w=end_q)
        report(label)

    def stage_hold(at_key: str, dur_s: float, gripper_val: float, label: str):
        n = duration_to_steps(dur_s)
        target = waypoints[at_key]
        print(f"[stage] {label:30s} HOLD at {at_key} | steps={n} grip={gripper_val:.2f}")
        for _ in range(n):
            control_step(target, gripper_val)
        report(label)

    # ============= 6 단계 실행 (--repeat 회 반복) =============
    # 1. 물체 위 이동 (1cm offset 도착)         — motion
    # 2. grasp 미세조정 (1cm 정렬)               — RL placeholder (현재는 motion 직선)
    # 3. 파지 + 적재 이동 (수직 하강 → close → lift → transport, 셀 1cm offset 도착) — motion
    # 4. insert 미세조정 (1cm 정렬)              — RL placeholder (현재는 motion 직선)
    # 5. 삽입 (수직 하강 → release)               — motion
    # 6. 복귀 (retract → home)                  — motion
    n_repeat = max(1, int(args_cli.repeat))
    for rep in range(n_repeat):
        print(f"\n========== motion1 pipeline START (run {rep+1}/{n_repeat}) ==========")

        # 매 반복마다 home 자세 + 박스 spawn 위치로 reset
        if rep > 0:
            robot.write_joint_state_to_sim(home_q, joint_vel)
            robot.reset()
            box.reset()
            for _ in range(30):
                scene.write_data_to_sim()
                sim.step()
                scene.update(dt)

        # ----- 1. 물체 위 이동 (motion) -----
        # home → 박스 xy + 1cm offset, z=PRE_GRASP_Z. quat slerp 로 회전 부드럽게.
        stage_move("home", "pre_grasp_offset",
                   STAGE_DURATION_S["move_above_box"], GRIPPER_OPEN,
                   "1. Move above box (1cm off)",
                   start_quat_w=home_grip_quat_w.unsqueeze(0),
                   end_quat_w=ee_quat_target_w)

        # ----- 2. grasp 미세조정 (RL placeholder) -----
        # 1cm offset → 정확한 박스 xy. 현재는 motion 직선 보간.
        # TODO(RL): Grasp RL policy 로 교체. action=(Δx,Δy,Δz,Δyaw,gripper).
        stage_move("pre_grasp_offset", "pre_grasp",
                   STAGE_DURATION_S["align_grasp"], GRIPPER_OPEN,
                   "2. Fine-align grasp [RL placeholder]")

        # ----- 3. 파지 + 적재 이동 (motion) -----
        # 3a) 수직 하강: PRE_GRASP_Z → GRASP_Z
        stage_move("pre_grasp", "grasp",
                   STAGE_DURATION_S["descend"], GRIPPER_OPEN,
                   "3a. Descend to grasp depth")
        # 3b) 그리퍼 close + hold
        stage_hold("grasp",
                   STAGE_DURATION_S["close"], gripper_close,
                   "3b. Grasp close+hold")
        # 3c) Lift: GRASP_Z → LIFT_Z
        stage_move("grasp", "lift",
                   STAGE_DURATION_S["lift"], gripper_close,
                   "3c. Lift")
        # 3d) Transport: lift → transport_offset (셀 위 1cm offset)
        stage_move("lift", "transport_offset",
                   STAGE_DURATION_S["transport"], gripper_close,
                   "3d. Transport over cell (1cm off)")

        # ----- 4. insert 미세조정 (RL placeholder) -----
        # 1cm offset → 정확한 셀 xy. 현재는 motion 직선 보간.
        # TODO(RL): Insert RL policy 로 교체. holding_box state 필수.
        stage_move("transport_offset", "transport",
                   STAGE_DURATION_S["align_insert"], gripper_close,
                   "4. Fine-align insert [RL placeholder]")

        # ----- 5. 삽입 (motion) -----
        # 5a) 수직 하강: TRANSPORT_Z → PLACE_Z
        stage_move("transport", "place",
                   STAGE_DURATION_S["insert"], gripper_close,
                   "5a. Insert descend")
        # 5b) 그리퍼 open (release)
        stage_hold("place",
                   STAGE_DURATION_S["release"], GRIPPER_OPEN,
                   "5b. Release")

        # ----- 6. 복귀 (motion) -----
        stage_move("place", "retract",
                   STAGE_DURATION_S["retract_up"], GRIPPER_OPEN,
                   "6a. Retract up")
        stage_move("retract", "home",
                   STAGE_DURATION_S["retract_home"], GRIPPER_OPEN,
                   "6b. Retract home")

        # 박스 위치 보고
        obj_pos_w = box.data.root_pos_w[0] - env_origin
        print(f"\n[motion1] run {rep+1} FINAL box pos (env-rel): {obj_pos_w.tolist()}")
        cell_xy_dist = ((obj_pos_w[0] - cx) ** 2 + (obj_pos_w[1] - cy) ** 2).sqrt().item()
        print(f"[motion1] run {rep+1} FINAL box xy-dist to cell center: {cell_xy_dist*100:.2f} cm")
        print(f"[motion1] run {rep+1} FINAL box z: {obj_pos_w[2].item()*100:.2f} cm")
        print(f"========== motion1 pipeline DONE (run {rep+1}/{n_repeat}) ==========\n")

    # --- hold 시간 결정 ---
    # hold_s 음수면 GUI 모드는 무한, headless 는 0초
    is_headless = bool(getattr(args_cli, "headless", False))
    if args_cli.hold_s < 0:
        hold_s = 0.0 if is_headless else float("inf")
    else:
        hold_s = float(args_cli.hold_s)

    if hold_s == float("inf"):
        print("[motion1] holding final pose. Close window to exit.")
        while simulation_app.is_running():
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
    elif hold_s > 0:
        print(f"[motion1] holding final pose for {hold_s:.1f}s then exit.")
        n_hold = duration_to_steps(hold_s)
        for _ in range(n_hold):
            if not simulation_app.is_running():
                break
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
    else:
        print("[motion1] no hold — exiting immediately.")


# -------------------- main --------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.4, 0.4, 0.9], [0.3, -0.25, 0.10])

    scene_cfg = MotionSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[motion1] Sim ready.")
    run_pipeline(sim, scene)


if __name__ == "__main__":
    import os
    try:
        main()
    finally:
        simulation_app.close()
    # PhysX 자원 해제 hang 방지 — 강제 종료
    os._exit(0)
