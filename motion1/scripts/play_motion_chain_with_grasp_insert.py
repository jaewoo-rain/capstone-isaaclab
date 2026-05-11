"""motion1 — Motion chain + Grasp RL + Insert RL 통합 (6단계).

play_motion_chain_with_grasp.py 의 확장:
 - 박스 spawn: xy ±10cm, yaw ±80° random (grasp 학습 cfg 와 동일)
 - **셀 spawn: xy ±10cm, yaw ±80° random** (cell wall 4개 random pose)
 - 단계 1: home → 박스 + 3~5cm offset (motion + slerp)
 - 단계 2: **Grasp RL inference** (정렬+잡기)
 - 단계 3a-c: descend → close → lift (motion)
 - 단계 3d: transport → cell + 3~5cm offset (motion + ee yaw → cell yaw)
 - 단계 4: **Insert RL inference** (정렬)
 - 단계 5a-b: insert descend → release (motion)
 - 단계 6a-b: retract up → home (motion)

사용 모델:
    checkpoints/motion1_grasp.zip   + _vecnorm.pkl
    checkpoints/motion1_insert.zip  + _vecnorm.pkl

실행:
    ./isaaclab.sh -p source/motion1/scripts/play_motion_chain_with_grasp_insert.py --hold_s 30
    ./isaaclab.sh -p source/motion1/scripts/play_motion_chain_with_grasp_insert.py --repeat 5 --hold_s 5
"""
from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

# -------------------- argparse / app launcher --------------------
parser = argparse.ArgumentParser(description="motion1 chain + grasp/insert RL 정책")
parser.add_argument("--gripper_close", type=float, default=0.8)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--hold_s", type=float, default=5.0)
parser.add_argument("--grasp_checkpoint", type=str, default="checkpoints/motion1_grasp.zip")
parser.add_argument("--grasp_vecnorm",    type=str, default="checkpoints/motion1_grasp_vecnorm.pkl")
parser.add_argument("--insert_checkpoint", type=str, default="checkpoints/motion1_insert.zip")
parser.add_argument("--insert_vecnorm",    type=str, default="checkpoints/motion1_insert_vecnorm.pkl")
parser.add_argument("--rl_max_steps", type=int, default=300,
                    help="단계 2/4 RL inference 최대 step (기본 300=5초@60Hz)")
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

# 박스 random spawn 범위 (grasp 학습 cfg 와 동일)
BOX_SPAWN_XY_NOISE = 0.00        # 0.10 = ±10cm
BOX_SPAWN_YAW_MAX  = 1.396       # ±80°

# 셀 random spawn 범위 (insert dataset 재수집 분포 와 동일)
CELL_SPAWN_XY_NOISE = 0.05       # ±5cm (robot reach 안전, dataset 과 일치)
CELL_SPAWN_YAW_MAX  = 1.396      # ±80°

# ee 시작점 noise (3~5cm random 거리 + random 방향)
EE_OFFSET_MIN_M = 0.03
EE_OFFSET_MAX_M = 0.05

# 단계 3d 끝 transport noise (~1cm) — RL 정책이 fine 정렬만 하도록
INSERT_OFFSET_MIN_M = 0.005
INSERT_OFFSET_MAX_M = 0.01

# RL action scale — grasp 와 insert 학습 cfg 가 다름 분리
RL_GRASP_ACTION_SCALE_XY = 0.01    # grasp_env_cfg: 10mm/step
RL_GRASP_ACTION_SCALE_YAW = 0.05   # grasp_env_cfg: ~2.86°/step
RL_INSERT_ACTION_SCALE_XY = 0.005  # insert_env_cfg: 5mm/step (변경됨)
RL_INSERT_ACTION_SCALE_YAW = 0.05  # insert_env_cfg: ~2.86°/step

# RL ee yaw clip (학습 cfg 와 동일)
RL_EE_YAW_MIN = -1.5708
RL_EE_YAW_MAX =  1.5708

# RL success 판정 (학습 cfg 와 동일)
# Grasp RL success 판정 (grasp_env_cfg 와 일치)
RL_GRASP_ALIGN_XY = 0.005       # 5mm
RL_GRASP_ALIGN_YAW = 0.05       # ~2.86°
RL_GRASP_HOLD_STEPS = 30        # 30 step (0.5초)

# Insert RL success 판정 (완화 — insert_env_cfg 와 일치)
RL_INSERT_ALIGN_XY = 0.010      # 10mm
RL_INSERT_ALIGN_YAW = 0.087     # ~5°
RL_INSERT_HOLD_STEPS = 15       # 15 step (0.25초)

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

    # Cell wall: kinematic rigid body — 매 run reset 시 random pose 로 강제 이동.
    # kinematic_enabled=True 면 외부 충돌에 안 밀리고 write_root_pose_to_sim 으로 이동 가능.
    wall_v_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_VL",
        spawn=sim_utils.CuboidCfg(
            size=_V_WALL_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(CELL_CENTER_X - CELL_INNER_X / 2 - WALL_THICKNESS / 2,
                 CELL_CENTER_Y, _WALL_Z)),
    )
    wall_v_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_VR",
        spawn=sim_utils.CuboidCfg(
            size=_V_WALL_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(CELL_CENTER_X + CELL_INNER_X / 2 + WALL_THICKNESS / 2,
                 CELL_CENTER_Y, _WALL_Z)),
    )
    wall_h_front: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_HF",
        spawn=sim_utils.CuboidCfg(
            size=_H_WALL_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(CELL_CENTER_X,
                 CELL_CENTER_Y - CELL_INNER_Y / 2 - WALL_THICKNESS / 2, _WALL_Z)),
    )
    wall_h_back: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CellWall_HB",
        spawn=sim_utils.CuboidCfg(
            size=_H_WALL_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=RigidObjectCfg.InitialStateCfg(
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
    dt = sim.get_physics_dt()  # 1/60 (sim physics dt = control rate)
    DECIMATION = 1              # 1 sim step per control step (insert_env_cfg 와 일치)
    control_dt = dt * DECIMATION  # 1/60
    duration_to_steps = lambda s: max(1, int(s / control_dt))

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

    # ---- RL 정책 + VecNormalize 통계 로드 helper ----
    import os
    import time
    def load_policy(ckpt_path: str, vec_path: str, tag: str):
        # ckpt 파일 정보 출력
        ckpt_abs = os.path.abspath(ckpt_path)
        vec_abs = os.path.abspath(vec_path)
        ckpt_size_kb = os.path.getsize(ckpt_path) / 1024 if os.path.exists(ckpt_path) else -1
        vec_size_kb = os.path.getsize(vec_path) / 1024 if os.path.exists(vec_path) else -1
        ckpt_mtime = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(ckpt_path)))
                      if os.path.exists(ckpt_path) else "N/A")
        vec_mtime = (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(vec_path)))
                     if os.path.exists(vec_path) else "N/A")
        print(f"[{tag}] === Checkpoint Loading ===")
        print(f"[{tag}]   ckpt   : {ckpt_abs}")
        print(f"[{tag}]            size={ckpt_size_kb:.1f} KB, mtime={ckpt_mtime}")
        print(f"[{tag}]   vecnorm: {vec_abs}")
        print(f"[{tag}]            size={vec_size_kb:.1f} KB, mtime={vec_mtime}")

        m = PPO.load(ckpt_path, device=device)
        print(f"[{tag}]   PPO loaded — num_timesteps={getattr(m, 'num_timesteps', 'N/A')}, "
              f"policy={type(m.policy).__name__}, device={m.device}")

        with open(vec_path, "rb") as f:
            v = pickle.load(f)
        v.training = False
        v.norm_reward = False
        mean_shape = v.obs_rms.mean.shape if hasattr(v.obs_rms, "mean") else "?"
        clip_ = float(getattr(v, "clip_obs", 10.0))
        eps_ = float(getattr(v, "epsilon", 1e-8))
        print(f"[{tag}]   VecNormalize loaded — obs_rms.mean shape={mean_shape}, clip={clip_}, eps={eps_}")
        if hasattr(v.obs_rms, "mean"):
            print(f"[{tag}]   obs_rms.mean = {np.asarray(v.obs_rms.mean).round(4).tolist()}")
            print(f"[{tag}]   obs_rms.var  = {np.asarray(v.obs_rms.var).round(4).tolist()}")
        print(f"[{tag}] === Loaded OK ===")

        def normalize(obs_np: np.ndarray) -> np.ndarray:
            obs_batch = obs_np[None, :]
            norm = v.normalize_obs(obs_batch)
            return np.asarray(norm[0], dtype=np.float32)
        return m, normalize

    grasp_model, grasp_normalize_obs = load_policy(
        args_cli.grasp_checkpoint, args_cli.grasp_vecnorm, "chain+grasp")
    insert_model, insert_normalize_obs = load_policy(
        args_cli.insert_checkpoint, args_cli.insert_vecnorm, "chain+insert")

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

    # ---- one control step (= decimation sub-step + IK 매번 재계산, RL 학습과 동일) ----
    def control_step(target_pos_env, gripper_value, target_quat_w=None):
        if target_quat_w is None:
            target_quat_w = base_ee_quat
        target_pos_w = target_pos_env.unsqueeze(0) + env_origin.unsqueeze(0)
        tip_ratio = 2.3
        gripper_target = torch.tensor(
            [[gripper_value, gripper_value * tip_ratio, gripper_value, gripper_value * tip_ratio]],
            device=device).expand(scene.num_envs, -1)
        # decimation sub-step (RL 학습 _apply_action 매 sub-step 호출 mimic)
        for _ in range(DECIMATION):
            ee_pos_w = grip_center_pos(robot, left_id, right_id)
            ee_quat_w = grip_center_quat(robot, left_id)
            cur_arm_q = robot.data.joint_pos[:, arm_ids]
            jac = grip_center_jacobian(robot, l_jac, r_jac, arm_ids)

            root_pos_w = robot.data.root_pos_w
            root_quat_w = robot.data.root_quat_w
            tgt_pos_b, tgt_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, target_pos_w, target_quat_w)
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)

            ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))
            arm_target = ik.compute(ee_pos_b, ee_quat_b, jac, cur_arm_q)

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
            obs_norm = grasp_normalize_obs(obs_np)

            action, _ = grasp_model.predict(obs_norm, deterministic=True)
            action = np.clip(action, -1.0, 1.0)

            # ---- ee_target xy / yaw 업데이트 (relative) ----
            delta_xy = action[:2] * RL_GRASP_ACTION_SCALE_XY
            ee_target_xy_w = (ee_pos_w[0, :2] + torch.tensor(delta_xy, device=device, dtype=torch.float))

            prev_ee_target_yaw = ee_target_yaw
            delta_yaw = float(action[2]) * RL_GRASP_ACTION_SCALE_YAW
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
                (abs(obj_rel_x) < RL_GRASP_ALIGN_XY) and
                (abs(obj_rel_y) < RL_GRASP_ALIGN_XY) and
                (abs(obj_yaw_err) < RL_GRASP_ALIGN_YAW)
            )
            if aligned:
                aligned_count += 1
                if aligned_count >= RL_GRASP_HOLD_STEPS:
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

    # ---- Stage 4: 학습된 insert RL 정책 inference ----
    def stage_rl_insert(max_steps: int, cell_xy_v, cell_yaw_v: float) -> bool:
        """학습된 PPO 정책으로 cell xy/yaw 미세 정렬. yaw 비누적, ee_z=TRANSPORT_Z 고정.

        cell_xy_v: (cell_x, cell_y) env-rel float tuple
        cell_yaw_v: float
        """
        aligned_count = 0
        success = False

        print(f"[stage] 4. RL insert align (PPO inference) | max_steps={max_steps}")
        z_axis_t = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float)

        for step_i in range(max_steps):
            # ---- state 계산 (insert_env._get_observations 와 동일 식) ----
            ee_pos_w = grip_center_pos(robot, left_id, right_id)
            ee_xy_env = ee_pos_w[:, :2] - scene.env_origins[:, :2]
            slot_rel_x = float(cell_xy_v[0] - ee_xy_env[0, 0].item())
            slot_rel_y = float(cell_xy_v[1] - ee_xy_env[0, 1].item())

            # actual ee yaw (insert env 와 동일 공식)
            ee_quat_w = grip_center_quat(robot, left_id)
            cur_ee_yaw = quat_z_yaw(ee_quat_w)[0].item()
            slot_yaw_err = float(wrap_to_pi(
                torch.tensor([cell_yaw_v - cur_ee_yaw])).item())

            # is_grasping = 1 (chain runner 에서 박스 잡혀 있다고 가정)
            is_grasping = 1.0

            ee_vel = grip_center_lin_vel(robot, left_id, right_id)
            ee_vel_x = ee_vel[0, 0].item()
            ee_vel_y = ee_vel[0, 1].item()

            # actual ang vel z (yaw rate)
            ang_l = robot.data.body_ang_vel_w[0, left_id, 2].item()
            ang_r = robot.data.body_ang_vel_w[0, right_id, 2].item()
            yaw_vel = 0.5 * (ang_l + ang_r)

            obs_np = np.array(
                [slot_rel_x, slot_rel_y, slot_yaw_err, is_grasping,
                 ee_vel_x, ee_vel_y, yaw_vel],
                dtype=np.float32,
            )
            obs_norm = insert_normalize_obs(obs_np)

            action, _ = insert_model.predict(obs_norm, deterministic=True)
            action = np.clip(action, -1.0, 1.0)

            # ---- action 적용 (insert_env 와 동일 — 비누적 xy + 비누적 yaw) ----
            delta_xy = action[:2] * RL_INSERT_ACTION_SCALE_XY
            ee_target_xy_w_now = (
                ee_pos_w[0, :2] + torch.tensor(delta_xy, device=device, dtype=torch.float))

            delta_yaw = float(action[2]) * RL_INSERT_ACTION_SCALE_YAW
            new_yaw = max(RL_EE_YAW_MIN, min(RL_EE_YAW_MAX, cur_ee_yaw + delta_yaw))

            target_pos_env = torch.tensor(
                [ee_target_xy_w_now[0].item() - env_origin[0].item(),
                 ee_target_xy_w_now[1].item() - env_origin[1].item(),
                 TRANSPORT_Z], device=device, dtype=torch.float)

            yaw_q = quat_from_angle_axis(
                torch.tensor([new_yaw], device=device, dtype=torch.float),
                z_axis_t)
            ee_quat_now = quat_mul(yaw_q, base_ee_quat)

            control_step(target_pos_env, gripper_close, target_quat_w=ee_quat_now)

            # ---- aligned 판정 ----
            aligned = (
                abs(slot_rel_x) < RL_INSERT_ALIGN_XY and
                abs(slot_rel_y) < RL_INSERT_ALIGN_XY and
                abs(slot_yaw_err) < RL_INSERT_ALIGN_YAW
            )
            if aligned:
                aligned_count += 1
                if aligned_count >= RL_INSERT_HOLD_STEPS:
                    success = True
                    print(f"  [stage 4] SUCCESS @ step {step_i+1} "
                          f"(aligned {aligned_count} steps)")
                    break
            else:
                aligned_count = 0

        if not success:
            print(f"  [stage 4] timeout {max_steps} steps — final aligned_count={aligned_count}")
        report("4. RL insert align")
        return success

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
        dist = float(torch.empty(1).uniform_(INSERT_OFFSET_MIN_M, INSERT_OFFSET_MAX_M).item())
        return dist * math.cos(ang), dist * math.sin(ang)

    def random_cell_pose():
        nx = float(torch.empty(1).uniform_(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE).item())
        ny = float(torch.empty(1).uniform_(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE).item())
        cx_ = CELL_CENTER_X + nx
        cy_ = CELL_CENTER_Y + ny
        cyaw_ = float(torch.empty(1).uniform_(-CELL_SPAWN_YAW_MAX, CELL_SPAWN_YAW_MAX).item())
        return cx_, cy_, cyaw_

    # Cell wall 4개의 cell-local offset (xy)
    _CELL_WALL_OFFSETS = [
        ("wall_v_left",  -(CELL_INNER_X / 2 + WALL_THICKNESS / 2), 0.0),
        ("wall_v_right", +(CELL_INNER_X / 2 + WALL_THICKNESS / 2), 0.0),
        ("wall_h_front", 0.0, -(CELL_INNER_Y / 2 + WALL_THICKNESS / 2)),
        ("wall_h_back",  0.0, +(CELL_INNER_Y / 2 + WALL_THICKNESS / 2)),
    ]

    def update_cell_walls(cx_: float, cy_: float, cyaw_: float):
        """Cell 4 walls 의 root pose 를 cell xy/yaw 에 맞게 강제 이동 (kinematic)."""
        cos_y = math.cos(cyaw_)
        sin_y = math.sin(cyaw_)
        half = cyaw_ / 2.0
        cell_quat = torch.tensor(
            [[math.cos(half), 0.0, 0.0, math.sin(half)]],
            device=device, dtype=torch.float)
        for name, dx_loc, dy_loc in _CELL_WALL_OFFSETS:
            wx_local = cos_y * dx_loc - sin_y * dy_loc
            wy_local = sin_y * dx_loc + cos_y * dy_loc
            wall_pos_w = torch.tensor(
                [[cx_ + wx_local + env_origin[0].item(),
                  cy_ + wy_local + env_origin[1].item(),
                  _WALL_Z + env_origin[2].item()]],
                device=device, dtype=torch.float)
            pose = torch.cat([wall_pos_w, cell_quat], dim=-1)
            scene[name].write_root_pose_to_sim(pose)
            scene[name].write_root_velocity_to_sim(
                torch.zeros((1, 6), device=device, dtype=torch.float))

    # ============= 6 단계 실행 (매 repeat 새 random spawn) =============
    n_repeat = max(1, int(args_cli.repeat))
    grasp_success_count = 0
    insert_success_count = 0

    for rep in range(n_repeat):
        print(f"\n========== run {rep+1}/{n_repeat} START ==========")

        # ---- robot home reset ----
        robot.write_joint_state_to_sim(home_q, joint_vel)
        robot.set_joint_position_target(home_q)
        robot.reset()

        # ---- 박스 + 셀 random spawn ----
        bx, by, byaw = random_box_spawn()
        cx, cy, cyaw = random_cell_pose()
        update_cell_walls(cx, cy, cyaw)
        print(f"[run {rep+1}] box xy=({bx:+.3f},{by:+.3f}) yaw={math.degrees(byaw):+.1f}° "
              f"| cell xy=({cx:+.3f},{cy:+.3f}) yaw={math.degrees(cyaw):+.1f}°")

        for _ in range(30):
            scene.write_data_to_sim(); sim.step(); scene.update(dt)

        # ---- ee 시작점 noise ----
        off_grasp = random_ee_offset()
        off_insert = random_insert_offset()
        print(f"[run {rep+1}] ee offsets: grasp dxy=({off_grasp[0]*100:+.2f},{off_grasp[1]*100:+.2f})cm "
              f"insert dxy=({off_insert[0]*100:+.2f},{off_insert[1]*100:+.2f})cm")

        # ---- waypoints (박스/셀 actual xy 기반) ----
        pre_grasp_offset = torch.tensor([bx + off_grasp[0], by + off_grasp[1], PRE_GRASP_Z],
                                        device=device, dtype=torch.float)
        grasp_pos        = torch.tensor([bx, by, GRASP_Z], device=device, dtype=torch.float)
        lift_pos         = torch.tensor([bx, by, LIFT_Z], device=device, dtype=torch.float)
        transport_offset = torch.tensor([cx + off_insert[0], cy + off_insert[1], TRANSPORT_Z],
                                        device=device, dtype=torch.float)

        # cell yaw 회전 적용한 ee 자세 (transport 끝 ~ insert/place 까지 유지)
        cell_yaw_q = quat_from_angle_axis(
            torch.tensor([cyaw], device=device, dtype=torch.float),
            torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float))
        cell_ee_quat = quat_mul(cell_yaw_q, base_ee_quat)

        # ---- 1. 물체 위 이동 (motion + quat slerp home → base_ee_quat) ----
        stage_move(home_grip_env, pre_grasp_offset,
                   STAGE_DURATION_S["move_above_box"], GRIPPER_OPEN,
                   "1. Move above box (3-5cm off)",
                   start_quat_w=home_grip_quat_w.unsqueeze(0),
                   end_quat_w=base_ee_quat)

        # ---- 2. RL grasp align ----
        final_yaw, grasp_success = stage_rl_grasp(args_cli.rl_max_steps)
        if grasp_success:
            grasp_success_count += 1

        # 이후 stage 의 ee orientation = R_z(final_yaw) ⊗ base_ee_quat
        yaw_q_final = quat_from_angle_axis(
            torch.tensor([final_yaw], device=device, dtype=torch.float),
            torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float))
        ee_quat_after_align = quat_mul(yaw_q_final, base_ee_quat)

        # ---- 3a. Descend (PRE_GRASP_Z → GRASP_Z) ----
        cur_ee = (grip_center_pos(robot, left_id, right_id)[0] - env_origin)
        descend_start = torch.tensor([cur_ee[0].item(), cur_ee[1].item(), PRE_GRASP_Z],
                                     device=device, dtype=torch.float)
        stage_move(descend_start, grasp_pos,
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

        # ---- 3d. Transport → cell + insert offset, yaw → cell_yaw ----
        # transport 도중 ee yaw (박스 yaw 따라가던) → cell_yaw 점진적 회전.
        # 박스도 같이 회전 → cell yaw 정렬됨.
        stage_move(lift_pos, transport_offset,
                   STAGE_DURATION_S["transport"], gripper_close,
                   "3d. Transport + yaw align (→ cell yaw)",
                   start_quat_w=ee_quat_after_align,
                   end_quat_w=cell_ee_quat)

        # ---- 4. Insert RL align (PPO inference) ----
        insert_success = stage_rl_insert(args_cli.rl_max_steps, (cx, cy), cyaw)
        if insert_success:
            insert_success_count += 1

        # ---- 5a. Insert descend (TRANSPORT_Z → PLACE_Z, ee xy 는 RL 정렬 끝점 그대로) ----
        cur_ee_2 = (grip_center_pos(robot, left_id, right_id)[0] - env_origin)
        descend_start_2 = torch.tensor(
            [cur_ee_2[0].item(), cur_ee_2[1].item(), TRANSPORT_Z],
            device=device, dtype=torch.float)
        descend_end_2 = torch.tensor(
            [cur_ee_2[0].item(), cur_ee_2[1].item(), PLACE_Z],
            device=device, dtype=torch.float)
        stage_move(descend_start_2, descend_end_2,
                   STAGE_DURATION_S["insert"], gripper_close,
                   "5a. Insert descend",
                   start_quat_w=cell_ee_quat, end_quat_w=cell_ee_quat)

        # ---- 5b. Release ----
        stage_hold(descend_end_2, STAGE_DURATION_S["release"], GRIPPER_OPEN,
                   "5b. Release", hold_quat_w=cell_ee_quat)

        # ---- 6a. Retract up (cell yaw → 0 풀기) ----
        retract_pos_actual = torch.tensor(
            [cur_ee_2[0].item(), cur_ee_2[1].item(), RETRACT_Z],
            device=device, dtype=torch.float)
        stage_move(descend_end_2, retract_pos_actual,
                   STAGE_DURATION_S["retract_up"], GRIPPER_OPEN,
                   "6a. Retract up",
                   start_quat_w=cell_ee_quat, end_quat_w=base_ee_quat)

        # ---- 6b. Retract home ----
        stage_move(retract_pos_actual, home_grip_env,
                   STAGE_DURATION_S["retract_home"], GRIPPER_OPEN,
                   "6b. Retract home",
                   start_quat_w=base_ee_quat, end_quat_w=home_grip_quat_w.unsqueeze(0))

        obj_pos_w = box.data.root_pos_w[0] - env_origin
        cell_xy_dist = ((obj_pos_w[0] - cx) ** 2 + (obj_pos_w[1] - cy) ** 2).sqrt().item()
        print(f"\n[run {rep+1}] FINAL box (env-rel): "
              f"({obj_pos_w[0]:+.3f},{obj_pos_w[1]:+.3f},{obj_pos_w[2]:+.3f})")
        print(f"[run {rep+1}] FINAL box xy-dist to cell: {cell_xy_dist*100:.2f} cm")
        print(f"[run {rep+1}] FINAL box z: {obj_pos_w[2].item()*100:.2f} cm")
        print(f"[run {rep+1}] grasp success: {grasp_success}, insert success: {insert_success}")
        print(f"========== run {rep+1}/{n_repeat} DONE ==========\n")

    print(f"\n[chain+rl] grasp  success rate: {grasp_success_count}/{n_repeat}")
    print(f"[chain+rl] insert success rate: {insert_success_count}/{n_repeat}")

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
