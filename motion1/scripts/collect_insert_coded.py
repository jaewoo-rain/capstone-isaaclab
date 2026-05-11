"""motion1 — Insert RL 학습용 handoff dataset 수집 (코드 자세 직접 매핑 버전).

기존 collect_insert_handoff.py 와 다르게 motion stage (1-3: grasp PPO + descend + close + lift + transport)
**모두 제거**. 대신 직접:
 1. robot home reset
 2. random cell pose (xy ±5cm + yaw ±80°) + random ee_dxy (3~5cm)
 3. **IK 로 ee 를 cell + ee_dxy 위 (z=0.26) 로 이동** (cell yaw 회전 적용)
 4. **박스를 grip center 에 강제 매핑** (xy = ee_xy, z = ee_z - box_z_offset, yaw = cell_yaw)
 5. gripper close + 짧은 settle

장점:
 - dataset 손상 sample 0 (motion 없으니 박스 흔들림 없음)
 - 한 cycle ~3초 → 1000 sample ~50분
 - 박스가 손에 정확히 매핑됨 → is_grasping rate 보장

저장 파일: checkpoints/insert_handoff_states.npz (insert_env 기존 형식 그대로)
"""
from __future__ import annotations

import argparse
import math
import os

from isaaclab.app import AppLauncher

# -------------------- argparse --------------------
parser = argparse.ArgumentParser(description="Collect handoff dataset for Insert RL.")
parser.add_argument("--gripper_close", type=float, default=0.8)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--target", type=int, default=5000,
                    help="수집할 handoff state 개수")
parser.add_argument("--hold_s", type=float, default=0.0)
parser.add_argument("--checkpoint", type=str, default="checkpoints/motion1_grasp.zip")
parser.add_argument("--vecnorm",    type=str, default="checkpoints/motion1_grasp_vecnorm.pkl")
parser.add_argument("--rl_max_steps", type=int, default=300)
parser.add_argument("--out_path", type=str, default="checkpoints/insert_handoff_states.npz")
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

# -------------------- constants (env-relative) --------------------
BOX_SPAWN = (0.30, -0.10, 0.07)
BOX_SIZE = (0.139, 0.044, 0.118)
BOX_MASS = 0.3

CELL_CENTER_X = 0.30
CELL_CENTER_Y = -0.30
CELL_INNER_X = 0.16
CELL_INNER_Y = 0.065
WALL_THICKNESS = 0.008
WALL_HEIGHT = 0.12

# z (motion-only / chain 과 동일)
PRE_GRASP_Z = BOX_SPAWN[2] + 0.10
GRASP_Z     = BOX_SPAWN[2] + 0.045
LIFT_Z      = 0.26
TRANSPORT_Z = 0.26
PLACE_Z     = BOX_SPAWN[2] + 0.095
RETRACT_Z   = 0.26

# stage 시간
STAGE_DURATION_S: dict[str, float] = {
    "move_above_box": 2.5,
    "descend":        1.0,
    "close":          1.5,
    "lift":           2.0,
    "transport":      3.0,
}
SETTLE_S = 0.5
GRIPPER_OPEN = 0.0

# 박스 random spawn (학습 grasp cfg 와 동일)
BOX_SPAWN_XY_NOISE = 0.10
BOX_SPAWN_YAW_MAX  = 1.396

# 셀 random spawn — collect 시점에만 random
CELL_SPAWN_YAW_MAX = 1.396     # ±80°
CELL_SPAWN_XY_NOISE = 0.05     # ±5cm (robot reach 안전 범위)

# ee 시작 offset (grasp 와 동일)
EE_OFFSET_MIN_M = 0.03
EE_OFFSET_MAX_M = 0.05

# RL action scale
RL_ACTION_SCALE_XY  = 0.01
RL_ACTION_SCALE_YAW = 0.05
RL_EE_YAW_MIN = -1.5708
RL_EE_YAW_MAX =  1.5708
RL_ALIGN_XY_THRESHOLD = 0.005
RL_ALIGN_YAW_THRESHOLD = 0.05
RL_SUCCESS_HOLD_STEPS = 30

# 셀 벽 사이즈 / 위치 (cell yaw 고정 0 으로 spawn → 추후 collect loop 에서 cell pose 만 변수로 사용)
_V_WALL_SIZE = (WALL_THICKNESS, CELL_INNER_Y + 2 * WALL_THICKNESS, WALL_HEIGHT)
_H_WALL_SIZE = (CELL_INNER_X + 2 * WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT)
_WALL_Z = WALL_HEIGHT / 2


# -------------------- Scene cfg (cell wall 없음 — collect 단계는 cell pose 만 변수로) --------------------
@configclass
class CollectSceneCfg(InteractiveSceneCfg):
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
                rigid_body_enabled=True, disable_gravity=True,  # collect 시만 — 박스 free-fall 방지
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

def quat_z_yaw(q_wxyz):
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def wrap_to_pi(angle):
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


# -------------------- run pipeline --------------------
def run_collect(sim, scene):
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

    # ---- IK ----
    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=device)

    # ---- Grasp PPO 정책 + VecNormalize ----
    print(f"[collect] loading PPO model: {args_cli.checkpoint}")
    model = PPO.load(args_cli.checkpoint, device=device)
    print(f"[collect] loading VecNormalize: {args_cli.vecnorm}")
    with open(args_cli.vecnorm, "rb") as f:
        venv_norm = pickle.load(f)
    obs_mean = venv_norm.obs_rms.mean
    obs_var  = venv_norm.obs_rms.var
    obs_eps  = float(getattr(venv_norm, "epsilon", 1e-8))
    obs_clip = float(getattr(venv_norm, "clip_obs", 10.0))

    def normalize_obs(obs_np):
        norm = (obs_np - obs_mean) / np.sqrt(obs_var + obs_eps)
        return np.clip(norm, -obs_clip, obs_clip).astype(np.float32)

    # ---- Home reset ----
    # gripper close 자세로 시작 (close cmd 0.8 × tip_ratio 2.3 — 닫힌 상태)
    _g_close = float(args_cli.gripper_close)
    _tip_ratio = 2.3
    HOME_JOINT_POS = {
        "joint1": 0.0, "joint2": -1.55, "joint3": 2.66,
        "joint4": -1.1, "joint5": 1.6, "joint6": 0.0,
        "rh_r1_joint": _g_close, "rh_r2": _g_close * _tip_ratio,
        "rh_l1": _g_close, "rh_l2": _g_close * _tip_ratio,
    }
    home_q = torch.zeros((scene.num_envs, robot.num_joints), device=device)
    for n, v in HOME_JOINT_POS.items():
        jid = robot.find_joints(n)[0][0]
        home_q[:, jid] = v
    joint_vel = torch.zeros_like(home_q)
    robot.write_joint_state_to_sim(home_q, joint_vel)
    robot.set_joint_position_target(home_q)
    robot.reset(); box.reset()
    for _ in range(60):
        scene.write_data_to_sim(); sim.step(); scene.update(dt)

    env_origin = scene.env_origins[0]
    home_grip_w = grip_center_pos(robot, left_id, right_id)[0]
    home_grip_quat_w = grip_center_quat(robot, left_id)[0]
    home_grip_env = home_grip_w - env_origin
    base_ee_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)

    gripper_close = float(args_cli.gripper_close)

    # ---- 한 번 IK control step ----
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

    def stage_move(start_pos_env, end_pos_env, dur_s, gripper_val,
                   start_quat_w=None, end_quat_w=None):
        n = duration_to_steps(dur_s)
        s = duration_to_steps(SETTLE_S)
        traj = cartesian_lerp(start_pos_env, end_pos_env, n)
        do_slerp = (start_quat_w is not None) and (end_quat_w is not None)
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

    def stage_hold(at_pos_env, dur_s, gripper_val, hold_quat_w=None):
        n = duration_to_steps(dur_s)
        for _ in range(n):
            control_step(at_pos_env, gripper_val, target_quat_w=hold_quat_w)

    def stage_rl_grasp(max_steps):
        ee_target_yaw = 0.0
        prev_ee_target_yaw = 0.0
        aligned_count = 0
        success = False
        sim_dt_ctrl = dt
        for _ in range(max_steps):
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
            obs_np = np.array([obj_rel_x, obj_rel_y, obj_yaw_err,
                               ee_vel_x, ee_vel_y, yaw_vel], dtype=np.float32)
            obs_norm = normalize_obs(obs_np)
            action, _ = model.predict(obs_norm, deterministic=True)
            action = np.clip(action, -1.0, 1.0)
            delta_xy = action[:2] * RL_ACTION_SCALE_XY
            ee_target_xy_w = ee_pos_w[0, :2] + torch.tensor(delta_xy, device=device, dtype=torch.float)
            prev_ee_target_yaw = ee_target_yaw
            delta_yaw = float(action[2]) * RL_ACTION_SCALE_YAW
            ee_target_yaw = max(RL_EE_YAW_MIN, min(RL_EE_YAW_MAX, ee_target_yaw + delta_yaw))
            target_pos_env = torch.tensor(
                [ee_target_xy_w[0].item() - env_origin[0].item(),
                 ee_target_xy_w[1].item() - env_origin[1].item(),
                 PRE_GRASP_Z], device=device, dtype=torch.float)
            yaw_q = quat_from_angle_axis(
                torch.tensor([ee_target_yaw], device=device, dtype=torch.float),
                torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float))
            ee_quat_now = quat_mul(yaw_q, base_ee_quat)
            control_step(target_pos_env, GRIPPER_OPEN, target_quat_w=ee_quat_now)
            aligned = (
                (abs(obj_rel_x) < RL_ALIGN_XY_THRESHOLD) and
                (abs(obj_rel_y) < RL_ALIGN_XY_THRESHOLD) and
                (abs(obj_yaw_err) < RL_ALIGN_YAW_THRESHOLD)
            )
            if aligned:
                aligned_count += 1
                if aligned_count >= RL_SUCCESS_HOLD_STEPS:
                    success = True
                    break
            else:
                aligned_count = 0
        return ee_target_yaw, success

    # ---- random spawn helpers ----
    def random_box_spawn():
        nx = float(torch.empty(1).uniform_(-BOX_SPAWN_XY_NOISE, BOX_SPAWN_XY_NOISE).item())
        ny = float(torch.empty(1).uniform_(-BOX_SPAWN_XY_NOISE, BOX_SPAWN_XY_NOISE).item())
        bx_ = BOX_SPAWN[0] + nx
        by_ = BOX_SPAWN[1] + ny
        yaw = float(torch.empty(1).uniform_(-BOX_SPAWN_YAW_MAX, BOX_SPAWN_YAW_MAX).item())
        box_pos_w = torch.tensor(
            [[bx_ + env_origin[0].item(),
              by_ + env_origin[1].item(),
              BOX_SPAWN[2] + env_origin[2].item()]],
            device=device, dtype=torch.float)
        z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float)
        box_quat = quat_from_angle_axis(
            torch.tensor([yaw], device=device, dtype=torch.float), z_axis)
        pose = torch.cat([box_pos_w, box_quat], dim=-1)
        vel = torch.zeros((1, 6), device=device, dtype=torch.float)
        box.write_root_pose_to_sim(pose)
        box.write_root_velocity_to_sim(vel)
        return bx_, by_, yaw

    def random_cell_pose():
        """cell xy ±2cm + yaw ±80° random."""
        nx = float(torch.empty(1).uniform_(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE).item())
        ny = float(torch.empty(1).uniform_(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE).item())
        cx_ = CELL_CENTER_X + nx
        cy_ = CELL_CENTER_Y + ny
        cyaw_ = float(torch.empty(1).uniform_(-CELL_SPAWN_YAW_MAX, CELL_SPAWN_YAW_MAX).item())
        return cx_, cy_, cyaw_

    def random_ee_offset():
        ang = float(torch.empty(1).uniform_(0, 2 * math.pi).item())
        dist = float(torch.empty(1).uniform_(EE_OFFSET_MIN_M, EE_OFFSET_MAX_M).item())
        return dist * math.cos(ang), dist * math.sin(ang)

    # ---- collect loop ----
    out_path = args_cli.out_path
    target = int(args_cli.target)
    print(f"[collect] target = {target}, out = {out_path}")

    out_joint_pos = []
    out_box_pos_env = []
    out_box_quat = []
    out_cell_xy = []
    out_cell_yaw = []
    out_ee_target_yaw = []

    # 박스 멀리 두기 (motion 단계 동안 ee 와 안 부딪히게)
    FAR_BOX_POS = (10.0, 10.0, 0.5)
    BOX_Z_OFFSET = 0.06   # box center z = ee z - 0.06 (chain runner 의 transport 끝 자세 와 일치)

    z_axis_t = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float)

    def park_box_far():
        far_pos = torch.tensor([[FAR_BOX_POS[0] + env_origin[0].item(),
                                 FAR_BOX_POS[1] + env_origin[1].item(),
                                 FAR_BOX_POS[2] + env_origin[2].item()]],
                               device=device, dtype=torch.float)
        far_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float)
        box.write_root_pose_to_sim(torch.cat([far_pos, far_quat], dim=-1))
        box.write_root_velocity_to_sim(torch.zeros((1, 6), device=device, dtype=torch.float))

    collected = 0
    attempted = 0
    while collected < target:
        attempted += 1

        # 1. robot home reset + 박스 멀리
        robot.write_joint_state_to_sim(home_q, joint_vel)
        robot.set_joint_position_target(home_q)
        robot.reset()
        park_box_far()
        for _ in range(20):
            scene.write_data_to_sim(); sim.step(); scene.update(dt)

        # 2. random cell + ee_dxy
        cx, cy, cyaw = random_cell_pose()
        ox, oy = random_ee_offset()

        # 3. IK 로 ee 이동 (gripper 이미 close 자세 — finger 닫힌 채 이동)
        target_pos_env = torch.tensor(
            [cx + ox, cy + oy, TRANSPORT_Z],
            device=device, dtype=torch.float)
        stage_move(home_grip_env, target_pos_env,
                   2.0, gripper_close,
                   start_quat_w=home_grip_quat_w.unsqueeze(0),
                   end_quat_w=base_ee_quat)

        # 4. 박스 grip center 에 강제 매핑
        # box quat = ee 의 actual quat (IK 정확도 한계로 cell_target_quat 와 다를 수 있음)
        # 매핑 직전 측정 → close 시 ee/박스 같은 자세 유지
        ee_pos_w = grip_center_pos(robot, left_id, right_id)[0]
        ee_quat_w_actual = grip_center_quat(robot, left_id)[0:1]  # (1, 4)
        box_pos_target_w = torch.tensor(
            [[ee_pos_w[0].item(),
              ee_pos_w[1].item(),
              ee_pos_w[2].item() - BOX_Z_OFFSET]],
            device=device, dtype=torch.float)
        box.write_root_pose_to_sim(torch.cat([box_pos_target_w, ee_quat_w_actual], dim=-1))
        box.write_root_velocity_to_sim(torch.zeros((1, 6), device=device, dtype=torch.float))

        # 5. 매핑 후 짧은 settle (gripper close 유지) — 박스가 finger 사이 자세 잡음
        stage_hold(target_pos_env, 0.3, gripper_close, hold_quat_w=base_ee_quat)

        # 6. 박스 잡힘 검증 — finger ↔ box 거리 < 7cm
        ee_pos_after = grip_center_pos(robot, left_id, right_id)[0]
        box_pos_after = box.data.root_pos_w[0]
        finger_box_dist = (ee_pos_after - box_pos_after).norm().item()
        if finger_box_dist > 0.07:
            if attempted % 20 == 0:
                print(f"  [skip] grasp not stable (dist={finger_box_dist*100:.1f}cm, "
                      f"attempts={attempted}, collected={collected})")
            continue

        # 7. 자세 저장
        joint_pos_now = robot.data.joint_pos[0].cpu().numpy().astype(np.float32)
        box_pos_w_now = box.data.root_pos_w[0].cpu().numpy()
        box_pos_env_now = (box_pos_w_now - env_origin.cpu().numpy()).astype(np.float32)
        box_quat_now = box.data.root_quat_w[0].cpu().numpy().astype(np.float32)
        cell_xy_arr = np.array([cx, cy], dtype=np.float32)
        cell_yaw_arr = np.array([cyaw], dtype=np.float32)
        ee_target_yaw_arr = np.array([cyaw], dtype=np.float32)

        out_joint_pos.append(joint_pos_now)
        out_box_pos_env.append(box_pos_env_now)
        out_box_quat.append(box_quat_now)
        out_cell_xy.append(cell_xy_arr)
        out_cell_yaw.append(cell_yaw_arr)
        out_ee_target_yaw.append(ee_target_yaw_arr)
        collected += 1

        if collected % 10 == 0 or collected == target:
            print(f"  [collect] {collected}/{target} (attempts={attempted}, "
                  f"success_rate={collected/attempted:.2%})")

        # 50개마다 부분 저장
        if collected % 50 == 0 and collected > 0 and collected < target:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez(
                out_path,
                joint_pos=np.stack(out_joint_pos),
                box_pos_env=np.stack(out_box_pos_env),
                box_quat=np.stack(out_box_quat),
                cell_xy=np.stack(out_cell_xy),
                cell_yaw=np.stack(out_cell_yaw),
                ee_target_yaw=np.stack(out_ee_target_yaw),
            )
            print(f"  [partial save] {collected} samples → {out_path}", flush=True)

    # ---- save ----
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(
        out_path,
        joint_pos=np.stack(out_joint_pos),
        box_pos_env=np.stack(out_box_pos_env),
        box_quat=np.stack(out_box_quat),
        cell_xy=np.stack(out_cell_xy),
        cell_yaw=np.stack(out_cell_yaw),
        ee_target_yaw=np.stack(out_ee_target_yaw),
    )
    print(f"\n[collect] saved {collected} handoff states to {out_path}")
    print(f"[collect] grasp success rate = {collected/attempted:.2%} ({collected}/{attempted})")


# -------------------- main --------------------
def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.4, 0.4, 0.9], [0.3, -0.25, 0.10])
    scene_cfg = CollectSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    print("[collect] Sim ready.")
    run_collect(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    os._exit(0)
