"""motion3 — Insert RL 학습용 handoff dataset 수집.

play_motion_chain_with_grasp.py 와 같은 chain 시뮬을 N 회 반복하면서,
**셀 yaw 도 random ±80°** 으로 spawn → stage 3d 끝 (박스 잡힌 채 셀 위 + yaw 정렬) 의 자세 저장.

저장 항목 (per episode):
 - joint_pos (10): arm 6 + gripper 4
 - box_pos_env (3) + box_quat_w (4)
 - cell_xy (2) + cell_yaw (1)
 - ee_target_yaw_at_handoff (1) — stage 3d 끝의 ee target yaw (= cell_yaw + 작은 chain 결과)

저장 파일: checkpoints/insert_handoff_states_v15.npz

실행:
    ./isaaclab.sh -p source/motion3/scripts/collect_insert_handoff.py \\
        --headless --num_envs 32 --target 5000 --hold_s 0
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
parser.add_argument("--out_path", type=str, default="checkpoints/insert_handoff_states_v15.npz")
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

from source.motion3.robot_cfg import OMY_TABLE_MOUNTED_CFG
from source.motion3 import layout, scene_helpers

# -------------------- constants (env-relative, layout 단일출처) --------------------
BOX_SPAWN = layout.BOX_SPAWN          # 책상 위
BOX_SIZE = layout.BOX_SIZE
BOX_MASS = 0.3

CELL_CENTER_X = layout.CELL_GRID_CENTER[0]   # 뒤쪽 grid 중심
CELL_CENTER_Y = layout.CELL_GRID_CENTER[1]
WALL_HEIGHT = layout.WALL_HEIGHT

# z (앞쪽 grasp/lift = 책상평면 / 뒤쪽 insert = ground, layout)
PRE_GRASP_Z = layout.PRE_GRASP_Z
GRASP_Z     = layout.GRASP_Z
LIFT_Z      = layout.LIFT_Z
TRANSPORT_Z = layout.INSERT_HOVER_Z   # 뒤쪽 저고도 hover (handoff 기록 높이)
PLACE_Z     = layout.PLACE_Z
RETRACT_Z   = layout.INSERT_HOVER_Z

# 뒤를 향한 turn-around seed (joint1 ~-170°)
BACK_HOME_JOINTS = {"joint1": -2.90, "joint2": 0.73, "joint3": 0.64,
                    "joint4": 0.17, "joint5": 1.571, "joint6": 0.0}

# stage 시간
STAGE_DURATION_S: dict[str, float] = {
    "move_above_box": 2.5,
    "descend":        1.0,
    "close":          1.5,
    "lift":           2.0,
    "turn_around":    3.0,
    "down_align":     0.6,
    "transport":      3.0,
}
SETTLE_S = 0.5
GRIPPER_OPEN = 0.0

# 박스 random spawn (layout)
BOX_SPAWN_XY_NOISE = layout.BOX_SPAWN_XY_NOISE
BOX_SPAWN_YAW_MAX  = layout.BOX_SPAWN_YAW_MAX

# 셀 random spawn — "거의 일자" yaw ±10°, 위치 ±3cm (layout)
CELL_SPAWN_YAW_MAX = layout.CELL_SPAWN_YAW_MAX
CELL_SPAWN_XY_NOISE = layout.CELL_SPAWN_XY_NOISE

# ee 시작 offset (grasp 와 동일, layout)
EE_OFFSET_MIN_M = layout.EE_OFFSET_MIN_M
EE_OFFSET_MAX_M = layout.EE_OFFSET_MAX_M

# RL action scale (grasp)
RL_ACTION_SCALE_XY  = 0.01
RL_ACTION_SCALE_YAW = 0.05
RL_EE_YAW_MIN = -1.5708
RL_EE_YAW_MAX =  1.5708
RL_ALIGN_XY_THRESHOLD = 0.005
RL_ALIGN_YAW_THRESHOLD = 0.05
RL_SUCCESS_HOLD_STEPS = 30


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
    robot = OMY_TABLE_MOUNTED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 책상 플랫폼 (앞쪽, 블록이 놓임). 상판 top=TABLE_HEIGHT.
    desk: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=sim_utils.CuboidCfg(
            size=(layout.DESK_SIZE_XY[0], layout.DESK_SIZE_XY[1], layout.DESK_TOP_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.35, 0.25))),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(layout.DESK_CENTER[0], layout.DESK_CENTER[1], layout.DESK_TOP_Z / 2)),
    )

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
    robot.reset(); box.reset()
    for _ in range(60):
        scene.write_data_to_sim(); sim.step(); scene.update(dt)

    env_origin = scene.env_origins[0]
    home_grip_w = grip_center_pos(robot, left_id, right_id)[0]
    home_grip_quat_w = grip_center_quat(robot, left_id)[0]
    home_grip_env = home_grip_w - env_origin
    base_ee_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)

    gripper_close = float(args_cli.gripper_close)

    # grasp 동안 joint1 클램프 (None=제한없음). 앞쪽 단계에서만 켠다.
    _J1_CLAMP = [None]

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
        if _J1_CLAMP[0] is not None:  # grasp 단계: joint1 큰 swing 방지 (yaw는 손목이 처리)
            arm_target[:, 0] = arm_target[:, 0].clamp(_J1_CLAMP[0][0], _J1_CLAMP[0][1])
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

    def stage_joint_move(target_joint_dict, dur_s, gripper_val):
        """joint-space 보간(IK 미사용) — joint1 turn-around용. ease-in/out, arm+gripper만."""
        n = duration_to_steps(dur_s)
        start_arm = robot.data.joint_pos[:, arm_ids].clone()
        end_arm = torch.tensor([[target_joint_dict[f"joint{i+1}"] for i in range(6)]],
                               device=device, dtype=torch.float)
        grip = torch.tensor([[gripper_val, gripper_val * 2.3, gripper_val, gripper_val * 2.3]],
                            device=device, dtype=torch.float)
        for i in range(n):
            tau = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / n)
            arm_i = start_arm * (1 - tau) + end_arm * tau
            robot.set_joint_position_target(torch.cat([arm_i, grip], dim=-1), joint_ids=all_joint_ids)
            scene.write_data_to_sim(); sim.step(); scene.update(dt)

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

    n_cells = layout.GRID_NUM_X * layout.GRID_NUM_Y

    def random_cell_pose():
        """그리드 전체 jitter(±3cm/±10°) + N칸 중 타깃 셀 1개 선택. (cx,cy)=타깃 셀 world xy."""
        gx = CELL_CENTER_X + float(torch.empty(1).uniform_(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE).item())
        gy = CELL_CENTER_Y + float(torch.empty(1).uniform_(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE).item())
        cyaw_ = float(torch.empty(1).uniform_(-CELL_SPAWN_YAW_MAX, CELL_SPAWN_YAW_MAX).item())
        tgt = int(torch.randint(0, n_cells, (1,)).item())
        _walls, cells = scene_helpers.grid_world_poses(gx, gy, cyaw_)
        cx_, cy_ = cells[tgt]
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

    collected = 0
    attempted = 0
    while collected < target:
        attempted += 1

        # robot home reset
        robot.write_joint_state_to_sim(home_q, joint_vel)
        robot.set_joint_position_target(home_q)
        robot.reset()
        # box random spawn
        bx_, by_, byaw = random_box_spawn()
        # cell pose random (xy ±2cm + yaw ±80°). 셀 wall 은 spawn 안 함.
        cx, cy, cyaw = random_cell_pose()

        for _ in range(30):
            scene.write_data_to_sim(); sim.step(); scene.update(dt)

        # ee offset
        ox, oy = random_ee_offset()
        pre_grasp_offset = torch.tensor([bx_ + ox, by_ + oy, PRE_GRASP_Z],
                                        device=device, dtype=torch.float)
        grasp_pos = torch.tensor([bx_, by_, GRASP_Z], device=device, dtype=torch.float)
        lift_pos  = torch.tensor([bx_, by_, LIFT_Z], device=device, dtype=torch.float)

        # 앞쪽(grasp) 단계 동안 joint1 큰 swing 제한 ON
        _J1_CLAMP[0] = layout.GRASP_JOINT1_CLAMP

        # Stage 1: home → pre_grasp_offset (slerp home_quat → base)
        stage_move(home_grip_env, pre_grasp_offset,
                   STAGE_DURATION_S["move_above_box"], GRIPPER_OPEN,
                   start_quat_w=home_grip_quat_w.unsqueeze(0),
                   end_quat_w=base_ee_quat)

        # Stage 2: grasp RL
        final_yaw, success = stage_rl_grasp(args_cli.rl_max_steps)
        if not success:
            if attempted % 20 == 0:
                print(f"  [skip] grasp fail (attempts={attempted}, collected={collected})")
            continue

        # Stage 3a: descend
        yaw_q_final = quat_from_angle_axis(
            torch.tensor([final_yaw], device=device, dtype=torch.float),
            torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float))
        ee_quat_after_align = quat_mul(yaw_q_final, base_ee_quat)
        cur_ee = (grip_center_pos(robot, left_id, right_id)[0] - env_origin)
        descend_start = torch.tensor([cur_ee[0].item(), cur_ee[1].item(), PRE_GRASP_Z],
                                     device=device, dtype=torch.float)
        stage_move(descend_start, grasp_pos,
                   STAGE_DURATION_S["descend"], GRIPPER_OPEN,
                   start_quat_w=ee_quat_after_align, end_quat_w=ee_quat_after_align)

        # Stage 3b: close
        stage_hold(grasp_pos, STAGE_DURATION_S["close"], gripper_close,
                   hold_quat_w=ee_quat_after_align)

        # Stage 3c: lift (책상 위로)
        stage_move(grasp_pos, lift_pos,
                   STAGE_DURATION_S["lift"], gripper_close,
                   start_quat_w=ee_quat_after_align, end_quat_w=ee_quat_after_align)

        # 앞쪽 끝 → joint1 클램프 해제 (turn-around은 의도적으로 joint1 크게 회전)
        _J1_CLAMP[0] = None

        # Stage 3c2: turn-around — joint1만 회전(joint2~6은 lift 자세 유지).
        # 수직 아래는 Z축(joint1) 회전에 불변 → 그리퍼가 계속 수직 유지 → down-align 불필요, joint5 안 돎.
        cur_arm = robot.data.joint_pos[0, arm_ids].tolist()
        turn_target = {"joint1": BACK_HOME_JOINTS["joint1"]}
        for k in range(2, 7):
            turn_target[f"joint{k}"] = cur_arm[k - 1]
        stage_joint_move(turn_target, STAGE_DURATION_S["turn_around"], gripper_close)

        # cell_yaw 정렬 quat — 박스 180° 대칭: cyaw / cyaw+π 중 현재(돌아간) 그리퍼에 가까운 쪽
        # 선택 → joint6가 180° 안 뒤집고 작은 yaw만 맞춤.
        turned_quat = grip_center_quat(robot, left_id)[0].unsqueeze(0)  # 현재(수직, 돌아간 yaw)
        _zaxis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float)

        def _cell_quat(yaw_val):
            return quat_mul(quat_from_angle_axis(
                torch.tensor([yaw_val], device=device, dtype=torch.float), _zaxis), base_ee_quat)

        _cand0, _cand1 = _cell_quat(cyaw), _cell_quat(cyaw + math.pi)
        _d0 = (turned_quat * _cand0).sum(-1).abs()
        _d1 = (turned_quat * _cand1).sum(-1).abs()
        cell_target_quat = _cand0 if bool((_d0 >= _d1).item()) else _cand1

        tx_off, ty_off = random_ee_offset()
        cur_after = (grip_center_pos(robot, left_id, right_id)[0] - env_origin)
        high_z = float(cur_after[2].item())   # turn 후 뒤-높음 z (≈ LIFT)
        align_start = torch.tensor([cur_after[0].item(), cur_after[1].item(), high_z],
                                   device=device, dtype=torch.float)

        # Stage 3d-1: 높은 데서 타깃 셀 위로 수평 이동 + yaw 정렬 (하강 전 yaw 먼저)
        above_high = torch.tensor([cx + tx_off, cy + ty_off, high_z], device=device, dtype=torch.float)
        stage_move(align_start, above_high,
                   STAGE_DURATION_S["transport"], gripper_close,
                   start_quat_w=turned_quat, end_quat_w=cell_target_quat)

        # Stage 3d-2: 수직 하강 (high → hover), yaw 고정
        transport_pos = torch.tensor([cx + tx_off, cy + ty_off, TRANSPORT_Z], device=device, dtype=torch.float)
        stage_move(above_high, transport_pos,
                   STAGE_DURATION_S["lift"], gripper_close,
                   start_quat_w=cell_target_quat, end_quat_w=cell_target_quat)

        # ---- handoff state 저장 ----
        joint_pos_now = robot.data.joint_pos[0].cpu().numpy().astype(np.float32)
        box_pos_w_now = box.data.root_pos_w[0].cpu().numpy()
        box_pos_env_now = (box_pos_w_now - env_origin.cpu().numpy()).astype(np.float32)
        box_quat_now = box.data.root_quat_w[0].cpu().numpy().astype(np.float32)
        cell_xy_arr = np.array([cx, cy], dtype=np.float32)
        cell_yaw_arr = np.array([cyaw], dtype=np.float32)
        ee_target_yaw_arr = np.array([cyaw], dtype=np.float32)  # stage 3d 끝에서 ee_target_yaw 가 cyaw 로 정렬됨

        out_joint_pos.append(joint_pos_now)
        out_box_pos_env.append(box_pos_env_now)
        out_box_quat.append(box_quat_now)
        out_cell_xy.append(cell_xy_arr)
        out_cell_yaw.append(cell_yaw_arr)
        out_ee_target_yaw.append(ee_target_yaw_arr)
        collected += 1

        if collected % 10 == 0 or collected == target:
            print(f"  [collect] {collected}/{target} (attempts={attempted}, "
                  f"grasp_success_rate={collected/attempted:.2%})")

        # 50개마다 부분 저장 (stuck / crash 시 손실 최소화)
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
