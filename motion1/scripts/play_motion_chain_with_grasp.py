"""motion1 — Motion chain + 학습된 Grasp RL 정책 통합 (6단계).

play_motion_chain.py 와 동일 구조이지만:
 - 박스 spawn: xy ±10cm, yaw ±80° random (학습 cfg 와 동일 분포)
 - 매 run reset 마다 박스 새 random + ee 시작점 박스 + 3~5cm random offset
 - 단계 2 (grasp 미세조정): **학습된 grasp 정책 (PPO + VecNormalize) 으로 inference**
 - 단계 1, 3, 4, 5, 6: motion (그대로)

사용 모델:
    checkpoints/motion1_grasp.zip
    checkpoints/motion1_grasp_vecnorm.pkl

실행:
    ./isaaclab.sh -p source/motion1/scripts/play_motion_chain_with_grasp.py --hold_s 30
    ./isaaclab.sh -p source/motion1/scripts/play_motion_chain_with_grasp.py --repeat 5 --hold_s 5
"""
from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

# -------------------- argparse / app launcher --------------------
parser = argparse.ArgumentParser(description="motion1 chain + 학습된 grasp 정책")
parser.add_argument("--gripper_close", type=float, default=0.8)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--hold_s", type=float, default=5.0)
parser.add_argument("--checkpoint", type=str, default="checkpoints/motion1_grasp.zip")
parser.add_argument("--vecnorm",    type=str, default="checkpoints/motion1_grasp_vecnorm.pkl")
parser.add_argument("--rl_max_steps", type=int, default=300,
                    help="단계 2 RL inference 최대 step (기본 300=5초@60Hz)")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------- imports (after app start) --------------------
import pickle
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_from_angle_axis, quat_mul, quat_slerp, subtract_frame_transforms,
)

from stable_baselines3 import PPO

from source.omy.omy_robot_cfg import OMY_OFF_SELF_COLLISION_CFG

# -------------------- constants (env-relative coords) --------------------
BOX_SPAWN = (0.30, -0.10, 0.07)
BOX_SIZE = (0.139, 0.044, 0.118)
BOX_MASS = 0.3

CELL_CENTER_X = 0.30
CELL_CENTER_Y = -0.30
CELL_INNER_X = 0.16
CELL_INNER_Y = 0.065
WALL_THICKNESS = 0.008
WALL_HEIGHT = 0.12

# z 좌표
# BOX_SPAWN[2] = 0.07
PRE_GRASP_Z = BOX_SPAWN[2] + 0.10        # 0.17 — grasp RL 학습의 ee_fixed_z 와 동일
GRASP_Z     = BOX_SPAWN[2] + 0.045        # 0.13
LIFT_Z      = 0.26
TRANSPORT_Z = 0.26
PLACE_Z     = BOX_SPAWN[2] + 0.095       # 0.165
# PLACE_Z = 0.26
RETRACT_Z   = 0.26

STAGE_DURATION_S: dict[str, float] = {
    "move_above_box": 2.5,    # 1. home → ee 시작점 (박스 + offset)
    # 2. RL inference — duration 은 args_cli.rl_max_steps 로
    "descend":        1.0,    # 3a
    "close":          1.5,    # 3b
    "lift":           2.0,    # 3c
    "transport":      3.0,    # 3d
    "align_insert":   0.6,    # 4
    "insert":         1.5,    # 5a
    "release":        0.7,    # 5b
    "retract_up":     1.5,    # 6a
    "retract_home":   3.0,    # 6b
}
SETTLE_S = 0.8
GRIPPER_OPEN = 0.0

# 박스 random spawn 범위 (학습 cfg 와 동일)
BOX_SPAWN_XY_NOISE = 0.10        # ±10cm
BOX_SPAWN_YAW_MAX  = 1.396       # ±80°

# ee 시작점 noise (학습 cfg 와 동일: 박스로부터 3~5cm random 거리 + random 방향)
EE_OFFSET_MIN_M = 0.03
EE_OFFSET_MAX_M = 0.05

# 단계 4 정렬 placeholder offset (학습 안 했으니 1cm 만)
ALIGN_OFFSET_INSERT_M = 0.01

# RL action scale (학습 cfg 와 동일)
RL_ACTION_SCALE_XY  = 0.01     # 10mm/step
RL_ACTION_SCALE_YAW = 0.05     # ~2.86°/step

# RL ee yaw clip (학습 cfg 와 동일)
RL_EE_YAW_MIN = -1.5708
RL_EE_YAW_MAX =  1.5708

# RL success 판정 (학습 cfg 와 동일)
RL_ALIGN_XY_THRESHOLD = 0.005
RL_ALIGN_YAW_THRESHOLD = 0.05
RL_SUCCESS_HOLD_STEPS = 30

# 셀 벽 사이즈 / 위치
_V_WALL_SIZE = (WALL_THICKNESS, CELL_INNER_Y + 2 * WALL_THICKNESS, WALL_HEIGHT)
_H_WALL_SIZE = (CELL_INNER_X + 2 * WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT)
_WALL_Z = WALL_HEIGHT / 2


# -------------------- Scene cfg --------------------
@configclass
class MotionSceneCfg(InteractiveSceneCfg):
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
                rigid_body_enabled=True, disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.7, 0.7, 0.72), metallic=0.5, roughness=0.4),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                static_friction=3.0, dynamic_friction=3.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=BOX_SPAWN),
    )

    wall_v_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_VL",
        spawn=sim_utils.CuboidCfg(
            size=_V_WALL_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CELL_CENTER_X - CELL_INNER_X / 2 - WALL_THICKNESS / 2,
                 CELL_CENTER_Y, _WALL_Z)),
    )
    wall_v_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_VR",
        spawn=sim_utils.CuboidCfg(
            size=_V_WALL_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CELL_CENTER_X + CELL_INNER_X / 2 + WALL_THICKNESS / 2,
                 CELL_CENTER_Y, _WALL_Z)),
    )
    wall_h_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_HF",
        spawn=sim_utils.CuboidCfg(
            size=_H_WALL_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CELL_CENTER_X,
                 CELL_CENTER_Y - CELL_INNER_Y / 2 - WALL_THICKNESS / 2, _WALL_Z)),
    )
    wall_h_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_HB",
        spawn=sim_utils.CuboidCfg(
            size=_H_WALL_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(CELL_CENTER_X,
                 CELL_CENTER_Y + CELL_INNER_Y / 2 + WALL_THICKNESS / 2, _WALL_Z)),
    )


# -------------------- helpers --------------------
def grip_center_pos(robot, l_id, r_id):
    return 0.5 * (robot.data.body_pos_w[:, l_id] + robot.data.body_pos_w[:, r_id])

def grip_center_quat(robot, l_id):
    return robot.data.body_quat_w[:, l_id]

def grip_center_jacobian(robot, l_jac_idx, r_jac_idx, joint_ids):
    J = robot.root_physx_view.get_jacobians()
    j_l = J[:, l_jac_idx, :, :][:, :, joint_ids]
    j_r = J[:, r_jac_idx, :, :][:, :, joint_ids]
    return 0.5 * (j_l + j_r)

def grip_center_lin_vel(robot, l_id, r_id):
    return 0.5 * (robot.data.body_lin_vel_w[:, l_id] + robot.data.body_lin_vel_w[:, r_id])

def cartesian_lerp(start, end, num_steps):
    alphas = torch.linspace(0, 1, num_steps, device=start.device).unsqueeze(-1)
    return start.unsqueeze(0) * (1 - alphas) + end.unsqueeze(0) * alphas

def quat_z_yaw(q_wxyz: torch.Tensor) -> torch.Tensor:
    """quat (N,4) wxyz → yaw (N,)."""
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


# -------------------- pipeline --------------------
def run_pipeline(sim, scene):
    robot = scene["robot"]
    box   = scene["box"]
    device = sim.device
    dt = sim.get_physics_dt()
    duration_to_steps = lambda s: max(1, int(s / dt))

    if args_cli.seed is not None:
        torch.manual_seed(args_cli.seed)

    # ---- joint / body indices ----
    arm_names = [f"joint{i}" for i in range(1, 7)]
    gripper_names = ["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"]
    arm_ids = [robot.find_joints(n)[0][0] for n in arm_names]
    gripper_ids = [robot.find_joints(n)[0][0] for n in gripper_names]
    all_joint_ids = arm_ids + gripper_ids
    left_id = robot.find_bodies("rh_p12_rn_l2")[0][0]
    right_id = robot.find_bodies("rh_p12_rn_r2")[0][0]
    if robot.is_fixed_base:
        l_jac, r_jac = left_id - 1, right_id - 1
    else:
        l_jac, r_jac = left_id, right_id

    # ---- IK controller ----
    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=device)

    # ---- Grasp RL 정책 + VecNormalize 통계 로드 ----
    print(f"[chain+rl] loading PPO model: {args_cli.checkpoint}")
    model = PPO.load(args_cli.checkpoint, device=device)
    print(f"[chain+rl] loading VecNormalize: {args_cli.vecnorm}")
    with open(args_cli.vecnorm, "rb") as f:
        venv_norm = pickle.load(f)
    obs_mean = venv_norm.obs_rms.mean
    obs_var  = venv_norm.obs_rms.var
    obs_eps  = float(getattr(venv_norm, "epsilon", 1e-8))
    obs_clip = float(getattr(venv_norm, "clip_obs", 10.0))
    print(f"[chain+rl] obs_rms.mean shape={obs_mean.shape}, clip={obs_clip}")

    def normalize_obs(obs_np: np.ndarray) -> np.ndarray:
        norm = (obs_np - obs_mean) / np.sqrt(obs_var + obs_eps)
        return np.clip(norm, -obs_clip, obs_clip).astype(np.float32)

    # ---- Home 자세 reset ----
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
    robot.set_joint_position_target(home_q)
    robot.reset()
    box.reset()
    for _ in range(60):
        scene.write_data_to_sim(); sim.step(); scene.update(dt)

    env_origin = scene.env_origins[0]
    home_grip_w = grip_center_pos(robot, left_id, right_id)[0]
    home_grip_quat_w = grip_center_quat(robot, left_id)[0]
    home_grip_env = home_grip_w - env_origin
    print(f"[chain+rl] home grip env-rel: {home_grip_env.tolist()}")

    # base ee quat (수직 아래) — motion-only 와 동일
    base_ee_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)

    gripper_close = float(args_cli.gripper_close)

    # ---- one IK step ----
    def control_step(target_pos_env, gripper_value, target_quat_w=None):
        if target_quat_w is None:
            target_quat_w = base_ee_quat
        ee_pos_w = grip_center_pos(robot, left_id, right_id)
        ee_quat_w = grip_center_quat(robot, left_id)
        cur_arm_q = robot.data.joint_pos[:, arm_ids]
        jac = grip_center_jacobian(robot, l_jac, r_jac, arm_ids)

        target_pos_w = target_pos_env.unsqueeze(0) + env_origin.unsqueeze(0)
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        tgt_pos_b, tgt_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, target_pos_w, target_quat_w)
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

        ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))
        arm_target = ik.compute(ee_pos_b, ee_quat_b, jac, cur_arm_q)

        tip_ratio = 2.3
        gripper_target = torch.tensor(
            [[gripper_value, gripper_value * tip_ratio, gripper_value, gripper_value * tip_ratio]],
            device=device).expand(scene.num_envs, -1)
        full_target = torch.cat([arm_target, gripper_target], dim=-1)
        robot.set_joint_position_target(full_target, joint_ids=all_joint_ids)
        scene.write_data_to_sim(); sim.step(); scene.update(dt)

    def report(tag):
        ee = (grip_center_pos(robot, left_id, right_id)[0] - env_origin).tolist()
        bx_w = box.data.root_pos_w[0] - env_origin
        d = (grip_center_pos(robot, left_id, right_id)[0] - box.data.root_pos_w[0]).norm().item()
        print(f"  [status @ {tag:24s}] ee=({ee[0]:+.3f},{ee[1]:+.3f},{ee[2]:+.3f}) "
              f"box=({bx_w[0]:+.3f},{bx_w[1]:+.3f},{bx_w[2]:+.3f}) ee↔box={d*100:.1f}cm")

    def stage_move(start_pos_env, end_pos_env, dur_s, gripper_val, label,
                   start_quat_w=None, end_quat_w=None):
        n = duration_to_steps(dur_s); s = duration_to_steps(SETTLE_S)
        traj = cartesian_lerp(start_pos_env, end_pos_env, n)
        do_slerp = (start_quat_w is not None) and (end_quat_w is not None)
        slerp_tag = "+slerp" if do_slerp else ""
        print(f"[stage] {label:36s} | move={n} settle={s} grip={gripper_val:.2f}{slerp_tag}")
        for i in range(n):
            if do_slerp:
                tau = (i + 1) / n
                q_i = quat_slerp(start_quat_w[0], end_quat_w[0], tau).unsqueeze(0)
            else:
                q_i = end_quat_w
            control_step(traj[i], gripper_val, target_quat_w=q_i)
        end_q = end_quat_w if (end_quat_w is not None) else None
        for _ in range(s):
            control_step(end_pos_env, gripper_val, target_quat_w=end_q)
        report(label)

    def stage_hold(at_pos_env, dur_s, gripper_val, label, hold_quat_w=None):
        n = duration_to_steps(dur_s)
        print(f"[stage] {label:36s} | HOLD steps={n} grip={gripper_val:.2f}")
        for _ in range(n):
            control_step(at_pos_env, gripper_val, target_quat_w=hold_quat_w)
        report(label)

    # ---- Stage 2: 학습된 grasp RL 정책 inference ----
    def stage_rl_grasp(max_steps: int) -> tuple[float, bool]:
        """학습된 PPO 정책으로 박스 위 미세 정렬. ee_target_yaw 누적, IK 매 step.

        Returns (final_ee_target_yaw, success_bool).
        """
        # internal state
        ee_target_yaw = 0.0
        prev_ee_target_yaw = 0.0
        aligned_count = 0
        success = False

        print(f"[stage] 2. RL grasp align (PPO inference)  | max_steps={max_steps}")
        sim_dt_ctrl = dt   # standalone 에서는 dt × decimation 인데 우리 standalone 은 decim 없이 dt 그대로

        for step_i in range(max_steps):
            # ---- state 계산 (grasp_env._get_observations 와 동일 식) ----
            box_pos_w = box.data.root_pos_w
            ee_pos_w = grip_center_pos(robot, left_id, right_id)
            box_xy_env = box_pos_w[:, :2] - scene.env_origins[:, :2]
            ee_xy_env  = ee_pos_w[:, :2] - scene.env_origins[:, :2]
            obj_rel_x = (box_xy_env[:, 0] - ee_xy_env[:, 0])[0].item()
            obj_rel_y = (box_xy_env[:, 1] - ee_xy_env[:, 1])[0].item()

            box_yaw = quat_z_yaw(box.data.root_quat_w)[0].item()
            obj_yaw_err = float(wrap_to_pi(torch.tensor([box_yaw - ee_target_yaw])).item())

            ee_vel = grip_center_lin_vel(robot, left_id, right_id)
            ee_vel_x = ee_vel[0, 0].item()
            ee_vel_y = ee_vel[0, 1].item()

            yaw_vel = (ee_target_yaw - prev_ee_target_yaw) / max(sim_dt_ctrl, 1e-6)

            obs_np = np.array(
                [obj_rel_x, obj_rel_y, obj_yaw_err, ee_vel_x, ee_vel_y, yaw_vel],
                dtype=np.float32,
            )
            obs_norm = normalize_obs(obs_np)

            action, _ = model.predict(obs_norm, deterministic=True)
            action = np.clip(action, -1.0, 1.0)

            # ---- ee_target xy / yaw 업데이트 (relative) ----
            delta_xy = action[:2] * RL_ACTION_SCALE_XY
            ee_target_xy_w = (ee_pos_w[0, :2] + torch.tensor(delta_xy, device=device, dtype=torch.float))

            prev_ee_target_yaw = ee_target_yaw
            delta_yaw = float(action[2]) * RL_ACTION_SCALE_YAW
            ee_target_yaw = max(RL_EE_YAW_MIN, min(RL_EE_YAW_MAX, ee_target_yaw + delta_yaw))

            # target pose 구성
            target_pos_env = torch.tensor(
                [ee_target_xy_w[0].item() - env_origin[0].item(),
                 ee_target_xy_w[1].item() - env_origin[1].item(),
                 PRE_GRASP_Z], device=device, dtype=torch.float)

            yaw_q = quat_from_angle_axis(
                torch.tensor([ee_target_yaw], device=device, dtype=torch.float),
                torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float))
            ee_quat_now = quat_mul(yaw_q, base_ee_quat)

            control_step(target_pos_env, GRIPPER_OPEN, target_quat_w=ee_quat_now)

            # ---- aligned 판정 ----
            aligned = (
                (abs(obj_rel_x) < RL_ALIGN_XY_THRESHOLD) and
                (abs(obj_rel_y) < RL_ALIGN_XY_THRESHOLD) and
                (abs(obj_yaw_err) < RL_ALIGN_YAW_THRESHOLD)
            )
            if aligned:
                aligned_count += 1
                if aligned_count >= RL_SUCCESS_HOLD_STEPS:
                    success = True
                    print(f"  [stage 2] SUCCESS @ step {step_i+1} "
                          f"(aligned {aligned_count} steps)")
                    break
            else:
                aligned_count = 0

        if not success:
            print(f"  [stage 2] timeout {max_steps} steps — final aligned_count={aligned_count}")
        report("2. RL grasp align")
        return ee_target_yaw, success

    # ---- 매 repeat 시 박스 random spawn ----
    def random_box_spawn():
        # xy noise ±10cm
        nx = float(torch.empty(1).uniform_(-BOX_SPAWN_XY_NOISE, BOX_SPAWN_XY_NOISE).item())
        ny = float(torch.empty(1).uniform_(-BOX_SPAWN_XY_NOISE, BOX_SPAWN_XY_NOISE).item())
        bx = BOX_SPAWN[0] + nx
        by = BOX_SPAWN[1] + ny
        # yaw ±80°
        yaw = float(torch.empty(1).uniform_(-BOX_SPAWN_YAW_MAX, BOX_SPAWN_YAW_MAX).item())

        # write to sim
        box_pos_w = torch.tensor(
            [[bx + env_origin[0].item(),
              by + env_origin[1].item(),
              BOX_SPAWN[2] + env_origin[2].item()]],
            device=device, dtype=torch.float)
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float)
        box_quat = quat_from_angle_axis(
            torch.tensor([yaw], device=device, dtype=torch.float), z_axis)
        pose = torch.cat([box_pos_w, box_quat], dim=-1)  # (1,7)
        vel = torch.zeros((1, 6), device=device, dtype=torch.float)
        box.write_root_pose_to_sim(pose)
        box.write_root_velocity_to_sim(vel)
        return bx, by, yaw

    def random_ee_offset():
        ang = float(torch.empty(1).uniform_(0, 2 * math.pi).item())
        dist = float(torch.empty(1).uniform_(EE_OFFSET_MIN_M, EE_OFFSET_MAX_M).item())
        return dist * math.cos(ang), dist * math.sin(ang)

    def random_insert_offset():
        ang = float(torch.empty(1).uniform_(0, 2 * math.pi).item())
        return ALIGN_OFFSET_INSERT_M * math.cos(ang), ALIGN_OFFSET_INSERT_M * math.sin(ang)

    # ============= 6 단계 실행 (매 repeat 새 random spawn) =============
    n_repeat = max(1, int(args_cli.repeat))
    cx, cy = CELL_CENTER_X, CELL_CENTER_Y
    success_count = 0

    for rep in range(n_repeat):
        print(f"\n========== run {rep+1}/{n_repeat} START ==========")

        # ---- robot home reset ----
        robot.write_joint_state_to_sim(home_q, joint_vel)
        robot.set_joint_position_target(home_q)
        robot.reset()

        # ---- 박스 random spawn ----
        bx, by, byaw = random_box_spawn()
        print(f"[run {rep+1}] box spawn: xy=({bx:+.3f},{by:+.3f}) yaw={math.degrees(byaw):+.1f}°")

        for _ in range(30):
            scene.write_data_to_sim(); sim.step(); scene.update(dt)

        # ---- ee 시작점 noise ----
        off_grasp = random_ee_offset()
        print(f"[run {rep+1}] ee offset (grasp): dxy=({off_grasp[0]*100:+.2f}, {off_grasp[1]*100:+.2f}) cm")
        off_insert = random_insert_offset()

        # ---- waypoints (박스/셀 actual xy 기반) ----
        pre_grasp_offset = torch.tensor([bx + off_grasp[0], by + off_grasp[1], PRE_GRASP_Z], device=device, dtype=torch.float)
        grasp_pos        = torch.tensor([bx, by, GRASP_Z], device=device, dtype=torch.float)
        lift_pos         = torch.tensor([bx, by, LIFT_Z], device=device, dtype=torch.float)
        transport_offset = torch.tensor([cx + off_insert[0], cy + off_insert[1], TRANSPORT_Z], device=device, dtype=torch.float)
        transport_pos    = torch.tensor([cx, cy, TRANSPORT_Z], device=device, dtype=torch.float)
        place_pos        = torch.tensor([cx, cy, PLACE_Z], device=device, dtype=torch.float)
        retract_pos      = torch.tensor([cx, cy, RETRACT_Z], device=device, dtype=torch.float)

        # ---- 1. 물체 위 이동 (motion + quat slerp home → base_ee_quat) ----
        stage_move(home_grip_env, pre_grasp_offset,
                   STAGE_DURATION_S["move_above_box"], GRIPPER_OPEN,
                   "1. Move above box (3-5cm off)",
                   start_quat_w=home_grip_quat_w.unsqueeze(0),
                   end_quat_w=base_ee_quat)

        # ---- 2. RL grasp align ----
        final_yaw, rl_success = stage_rl_grasp(args_cli.rl_max_steps)
        if rl_success:
            success_count += 1

        # 이후 stage 의 ee orientation = R_z(final_yaw) ⊗ base_ee_quat
        yaw_q_final = quat_from_angle_axis(
            torch.tensor([final_yaw], device=device, dtype=torch.float),
            torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float))
        ee_quat_after_align = quat_mul(yaw_q_final, base_ee_quat)

        # ---- 3a. Descend (PRE_GRASP_Z → GRASP_Z) ----
        # 시작점 = ee 현재 xy (RL 정렬 끝점). 끝점은 박스 정확 xy + GRASP_Z
        cur_ee = (grip_center_pos(robot, left_id, right_id)[0] - env_origin)
        descend_start = torch.tensor([cur_ee[0].item(), cur_ee[1].item(), PRE_GRASP_Z],
                                     device=device, dtype=torch.float)
        # 박스 actual xy 위로 descend (RL 이 정확히 정렬했으면 ~동일)
        descend_end = grasp_pos
        stage_move(descend_start, descend_end,
                   STAGE_DURATION_S["descend"], GRIPPER_OPEN,
                   "3a. Descend to grasp depth",
                   start_quat_w=ee_quat_after_align, end_quat_w=ee_quat_after_align)

        # ---- 3b. close + hold ----
        stage_hold(grasp_pos, STAGE_DURATION_S["close"], gripper_close,
                   "3b. Grasp close+hold", hold_quat_w=ee_quat_after_align)

        # ---- 3c. Lift ----
        stage_move(grasp_pos, lift_pos,
                   STAGE_DURATION_S["lift"], gripper_close,
                   "3c. Lift",
                   start_quat_w=ee_quat_after_align, end_quat_w=ee_quat_after_align)

        # ---- 3d. Transport over cell + yaw 0 정렬 (rule-based) ----
        # transport 도중 ee yaw 를 박스 yaw → 0 (셀 yaw) 으로 점차 회전.
        # 박스도 같이 회전 → 셀 yaw 와 정렬됨 → 셀 안 잘 들어감.
        stage_move(lift_pos, transport_offset,
                   STAGE_DURATION_S["transport"], gripper_close,
                   "3d. Transport + yaw align (→ cell yaw=0)",
                   start_quat_w=ee_quat_after_align,   # 박스 yaw 따라가던 자세
                   end_quat_w=base_ee_quat)            # 셀 yaw=0 정렬

        # ---- 이후 단계 모두 base_ee_quat (yaw=0) 으로 ----
        # ---- 4. Insert align (motion placeholder) ----
        stage_move(transport_offset, transport_pos,
                   STAGE_DURATION_S["align_insert"], gripper_close,
                   "4. Fine-align insert [motion]",
                   start_quat_w=base_ee_quat, end_quat_w=base_ee_quat)

        # ---- 5a. Insert descend ----
        stage_move(transport_pos, place_pos,
                   STAGE_DURATION_S["insert"], gripper_close,
                   "5a. Insert descend",
                   start_quat_w=base_ee_quat, end_quat_w=base_ee_quat)
        # ---- 5b. Release ----
        stage_hold(place_pos, STAGE_DURATION_S["release"], GRIPPER_OPEN,
                   "5b. Release", hold_quat_w=base_ee_quat)

        # ---- 6a. Retract up ----
        stage_move(place_pos, retract_pos,
                   STAGE_DURATION_S["retract_up"], GRIPPER_OPEN,
                   "6a. Retract up",
                   start_quat_w=base_ee_quat, end_quat_w=base_ee_quat)
        # ---- 6b. Retract home ----
        stage_move(retract_pos, home_grip_env,
                   STAGE_DURATION_S["retract_home"], GRIPPER_OPEN,
                   "6b. Retract home",
                   start_quat_w=base_ee_quat, end_quat_w=home_grip_quat_w.unsqueeze(0))

        obj_pos_w = box.data.root_pos_w[0] - env_origin
        cell_xy_dist = ((obj_pos_w[0] - cx) ** 2 + (obj_pos_w[1] - cy) ** 2).sqrt().item()
        print(f"\n[run {rep+1}] FINAL box (env-rel): "
              f"({obj_pos_w[0]:+.3f},{obj_pos_w[1]:+.3f},{obj_pos_w[2]:+.3f})")
        print(f"[run {rep+1}] FINAL box xy-dist to cell: {cell_xy_dist*100:.2f} cm")
        print(f"[run {rep+1}] FINAL box z: {obj_pos_w[2].item()*100:.2f} cm")
        print(f"[run {rep+1}] RL grasp success: {rl_success}")
        print(f"========== run {rep+1}/{n_repeat} DONE ==========\n")

    print(f"\n[chain+rl] RL grasp success rate: {success_count}/{n_repeat}")

    # ---- hold ----
    is_headless = bool(getattr(args_cli, "headless", False))
    if args_cli.hold_s < 0:
        hold_s = 0.0 if is_headless else float("inf")
    else:
        hold_s = float(args_cli.hold_s)

    if hold_s == float("inf"):
        print("[chain+rl] holding final pose. Close window to exit.")
        while simulation_app.is_running():
            scene.write_data_to_sim(); sim.step(); scene.update(dt)
    elif hold_s > 0:
        print(f"[chain+rl] holding for {hold_s:.1f}s.")
        n_hold = duration_to_steps(hold_s)
        for _ in range(n_hold):
            if not simulation_app.is_running(): break
            scene.write_data_to_sim(); sim.step(); scene.update(dt)


# -------------------- main --------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.4, 0.4, 0.9], [0.3, -0.25, 0.10])
    scene_cfg = MotionSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[chain+rl] Sim ready.")
    run_pipeline(sim, scene)


if __name__ == "__main__":
    import os
    try:
        main()
    finally:
        simulation_app.close()
    os._exit(0)
