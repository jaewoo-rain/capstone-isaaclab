"""motion3 — Motion chain + Grasp RL + Insert RL 통합 (6단계).

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
    checkpoints/motion3_insert.zip  + _vecnorm.pkl

실행:
    ./isaaclab.sh -p source/motion3/scripts/play_motion_chain_with_grasp_insert.py --hold_s 30
    ./isaaclab.sh -p source/motion3/scripts/play_motion_chain_with_grasp_insert.py --repeat 5 --hold_s 5
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
parser.add_argument("--insert_checkpoint", type=str, default="checkpoints/motion3_insert.zip")
parser.add_argument("--insert_vecnorm",    type=str, default="checkpoints/motion3_insert_vecnorm.pkl")
parser.add_argument("--rl_max_steps", type=int, default=300,
                    help="단계 2/4 RL inference 최대 step (기본 300=5초@60Hz)")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--demo_grid", action="store_true",
                    help="5x2 그리드 순차 채우기 데모 (그리드 고정 + 셀 0→9 순차 + 놓은 박스 누적)")
parser.add_argument("--grid_yaw", type=float, default=0.0, help="고정 그리드 yaw(도) (demo_grid)")
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

# -------------------- constants (env-relative coords, layout 단일출처) --------------------
# 좌표계: ground(z=0)=뒤쪽 셀 바닥, robot/책상=z=TABLE_HEIGHT. 앞쪽 grasp/lift는 +H, 뒤쪽 insert/place는 ground.
BOX_SPAWN = layout.BOX_SPAWN              # 책상 위 (z=0.07+H)
BOX_SIZE = layout.BOX_SIZE
BOX_MASS = 0.3

# 그리드 중심(뒤쪽 -x). 단일 셀 CELL_CENTER 대신 9셀 grid — 타깃 셀은 run마다 선택.
CELL_CENTER_X = layout.CELL_GRID_CENTER[0]
CELL_CENTER_Y = layout.CELL_GRID_CENTER[1]
WALL_HEIGHT = layout.WALL_HEIGHT

# z 좌표 (앞쪽=책상평면 / 뒤쪽=ground)
PRE_GRASP_Z = layout.PRE_GRASP_Z         # 책상 위 호버 (grasp RL ee_fixed_z)
GRASP_Z     = layout.GRASP_Z             # 책상 위 파지
LIFT_Z      = layout.LIFT_Z              # 책상 위로 들어 뒤로 운반
# ★ RL 정렬 hover 를 벽 바로 위로 낮춤 (사용자 요청: "하강 후 정렬").
#   기존 layout.INSERT_HOVER_Z=0.56 은 너무 높아 정렬 후 49cm 하강 → IK lag 로 grip xy 가
#   -x 로 ~10cm 드리프트(적재 실패 주범). 0.32 = 박스 밑면 ~0.17 (벽 0.12 위 ~5cm) 로,
#   RL 이 "벽 위에서" 정렬(셀 안 갇히기 전) + 최종 하강이 0.32→place 25cm 로 절반.
#   insert obs 는 z-무관(slot_rel_xy/yaw/vel) → v25 정책 재학습 없이 전이. env/collect/RETRACT 는 0.56 유지.
RL_HOVER_Z  = 0.32                       # sweet spot. 박스 밑면 ~0.17 (벽top 0.12 위 5cm).
                                         #   0.28 로 더 내려보니 박스가 벽 모서리에 걸려 텀블(run4 33cm/yaw58° 카타스트로피).
                                         #   0.32 는 4 run 무사고(최종 1~5cm). 벽이 하드 제약이라 더 못 내림.
                                         #   ↑높이면 정렬 후 free 하강 길어져 드리프트↑(0.56 원본은 0.2~18cm 들쭉날쭉).
TRANSPORT_Z = RL_HOVER_Z                  # 3d-2 가 여기까지 하강 → 그 높이에서 stage_rl_insert
PLACE_Z     = layout.PLACE_Z             # 셀 바닥 안착
RETRACT_Z   = layout.INSERT_HOVER_Z      # retract 는 높이 복귀(0.56) — 벽 클리어 후 turn

# 뒤를 향한 turn-around seed (joint1 ~-170°). transport에서 joint1을 먼저 돌려야 자연스럽게 reach.
BACK_HOME_JOINTS = {"joint1": -2.90, "joint2": 0.73, "joint3": 0.64,
                    "joint4": 0.17, "joint5": 1.571, "joint6": 0.0}

STAGE_DURATION_S: dict[str, float] = {
    "move_above_box": 2.5,    # 1. home → ee 시작점 (박스 + offset)
    # 2. RL inference — duration 은 args_cli.rl_max_steps 로
    "descend":        1.0,    # 3a
    "close":          1.5,    # 3b
    "lift":           2.0,    # 3c
    "transport":      3.0,    # 3d
    "align_insert":   0.6,    # 4
    "insert":         2.5,    # 5a (1.5→2.5: 최종 하강 느리게 → step당 z이동↓ → DLS lag 드리프트↓)
    "release":        0.7,    # 5b
    "retract_up":     1.5,    # 6a
    "retract_home":   3.0,    # 6b
}
SETTLE_S = 0.8
GRIPPER_OPEN = 0.0

# 박스 random spawn 범위 (layout 단일출처)
BOX_SPAWN_XY_NOISE = layout.BOX_SPAWN_XY_NOISE   # ±10cm
BOX_SPAWN_YAW_MAX  = layout.BOX_SPAWN_YAW_MAX    # ±80°

# 셀 random spawn 범위 — 적재는 "거의 일자" → yaw ±10°, 그리드 위치 ±3cm (layout)
CELL_SPAWN_XY_NOISE = layout.CELL_SPAWN_XY_NOISE  # ±3cm
CELL_SPAWN_YAW_MAX  = layout.CELL_SPAWN_YAW_MAX   # ±10°

# ee 시작점 noise (3~5cm random 거리 + random 방향, layout)
EE_OFFSET_MIN_M = layout.EE_OFFSET_MIN_M
EE_OFFSET_MAX_M = layout.EE_OFFSET_MAX_M

# 단계 3d 끝 transport noise (~1cm) — RL 정책이 fine 정렬만 하도록
INSERT_OFFSET_MIN_M = 0.005
INSERT_OFFSET_MAX_M = 0.01

# RL action scale — grasp 와 insert 학습 cfg 가 다름 분리
RL_GRASP_ACTION_SCALE_XY = 0.01    # grasp_env_cfg: 10mm/step
RL_GRASP_ACTION_SCALE_YAW = 0.05   # grasp_env_cfg: ~2.86°/step
RL_INSERT_ACTION_SCALE_XY = 0.005  # insert_env_cfg: 5mm/step (변경됨)
RL_INSERT_ACTION_SCALE_YAW = 0.05  # insert_env_cfg: ~2.86°/step

# RL ee yaw clip (grasp 학습 cfg 와 동일 — grasp 만 사용)
RL_EE_YAW_MIN = -1.5708
RL_EE_YAW_MAX =  1.5708
# ★ insert v25: reference-anchored unwrap. 절대 ±π clip 폐기 → stage 진입 시 측정 grip yaw 를
#   기준(yaw_ref)으로 ±margin 안에서만 누적. insert_env_cfg.yaw_margin 과 반드시 일치.
RL_INSERT_YAW_MARGIN = 1.75

# RL success 판정 (학습 cfg 와 동일)
# Grasp RL success 판정 (grasp_env_cfg 와 일치)
RL_GRASP_ALIGN_XY = 0.005       # 5mm
RL_GRASP_ALIGN_YAW = 0.05       # ~2.86°
RL_GRASP_HOLD_STEPS = 30        # 30 step (0.5초)

# Insert RL success 판정 (완화 — insert_env_cfg 와 일치)
RL_INSERT_ALIGN_XY = 0.010      # 10mm
RL_INSERT_ALIGN_YAW = 0.087     # ~5°
RL_INSERT_HOLD_STEPS = 15       # 15 step (0.25초)

# grid 격벽 spec (scene_helpers 단일출처). 개수=(nx+1)+(ny+1) — grid 크기 바뀌면 자동 반영.
_WALL_SPECS = scene_helpers.wall_specs()   # [(name, size(x,y,z), local_xy)]
_WALL_Z = WALL_HEIGHT / 2
# 격벽 이름 리스트 (update_grid_walls / 검증에서 사용)
GRID_WALL_NAMES = [s[0] for s in _WALL_SPECS]


def _grid_wall_cfg(name, size, local_xy):
    """grid 중심 기준 local_xy 위치에 kinematic 격벽 cfg 생성 (run마다 grid pose로 재이동)."""
    gx, gy = layout.CELL_GRID_CENTER
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(gx + local_xy[0], gy + local_xy[1], _WALL_Z)),
    )


# 데모(--demo_grid)용: 셀에 안착되어 누적 표시되는 박스. kinematic, 초기엔 바닥 아래 숨김.
N_CELL = layout.GRID_NUM_X * layout.GRID_NUM_Y
PLACED_BOX_NAMES = [f"PlacedBox{i}" for i in range(N_CELL)]


def _placed_box_cfg(i):
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/PlacedBox{i}",
        spawn=sim_utils.CuboidCfg(
            size=BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.85, 0.55, 0.25), metallic=0.3, roughness=0.5)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -1.0)),
    )


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
    # 로봇: 책상 위(z=TABLE_HEIGHT)에 마운트
    robot = OMY_TABLE_MOUNTED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # 책상 플랫폼: 앞쪽 작업영역(블록이 놓임). 상판 top=TABLE_HEIGHT, kinematic 콜리전.
    desk: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=sim_utils.CuboidCfg(
            size=(layout.DESK_SIZE_XY[0], layout.DESK_SIZE_XY[1], layout.DESK_TOP_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.35, 0.25)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max", static_friction=1.0, dynamic_friction=1.0)),
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

    # grid 격벽 (뒤쪽, kinematic — run마다 grid pose로 재이동). 개수 동적: (nx+1)+(ny+1).
    # __post_init__에서 setattr → InteractiveScene가 cfg.__dict__ 순회 시 자동 등록.
    def __post_init__(self):
        for _name, _size, _lxy in _WALL_SPECS:
            setattr(self, _name, _grid_wall_cfg(_name, _size, _lxy))
        for _i in range(N_CELL):   # 데모용 누적 박스 (--demo_grid). 평소엔 바닥 아래 숨김.
            setattr(self, f"PlacedBox{_i}", _placed_box_cfg(_i))


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

    # grasp 동안 joint1 클램프 (None=제한없음). 앞쪽 단계에서만 켠다.
    _J1_CLAMP = [None]

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
            if _J1_CLAMP[0] is not None:  # grasp 단계: joint1 큰 swing 방지
                arm_target[:, 0] = arm_target[:, 0].clamp(_J1_CLAMP[0][0], _J1_CLAMP[0][1])

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

    def stage_joint_move(target_joint_dict, dur_s, gripper_val, label):
        """joint-space 보간 이동 (IK 미사용). joint1 turn-around 처럼 base 회전이 필요한
        큰 운동에 사용 — cartesian 직선은 base를 관통/특이점에 걸리므로.
        ease-in/out(코사인)으로 각가속 피크를 낮춰 박스 slip 방지. arm+gripper만 명령(joint_ids)."""
        n = duration_to_steps(dur_s)
        start_arm = robot.data.joint_pos[:, arm_ids].clone()
        end_arm = torch.tensor(
            [[target_joint_dict[f"joint{i+1}"] for i in range(6)]],
            device=device, dtype=torch.float)
        grip = torch.tensor(
            [[gripper_val, gripper_val * 2.3, gripper_val, gripper_val * 2.3]],
            device=device, dtype=torch.float)
        print(f"[stage] {label:36s} | JOINT move={n} grip={gripper_val:.2f} (ease)")
        for i in range(n):
            # ease-in/out: 0→1 부드럽게 (각가속 피크 ↓ → 박스 원심력 slip 방지)
            tau = 0.5 - 0.5 * math.cos(math.pi * (i + 1) / n)
            arm_i = start_arm * (1 - tau) + end_arm * tau
            target = torch.cat([arm_i, grip], dim=-1)  # (1, 10) arm6+grip4
            robot.set_joint_position_target(target, joint_ids=all_joint_ids)
            scene.write_data_to_sim(); sim.step(); scene.update(dt)
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
    def stage_rl_insert(max_steps: int, cell_xy_v, cell_yaw_v: float) -> tuple[float, bool]:
        """학습된 PPO 정책으로 cell yaw 미세 정렬(xy=cell 고정 holding). yaw 누적 setpoint anchor.

        cell_xy_v: (cell_x, cell_y) env-rel float tuple
        cell_yaw_v: float
        Returns (final_ee_target_yaw, success_bool). final yaw 는 downstream 자세에 사용.
        """
        aligned_count = 0
        success = False

        print(f"[stage] 4. RL insert align (PPO inference) | max_steps={max_steps}")
        z_axis_t = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float)

        # ★ wrap-safe yaw parity (insert_env v25 reference-anchored unwrap):
        #   stage 진입 시 측정 grip yaw 를 기준(yaw_ref)으로 고정 → 누적 setpoint 를 ±margin 안에서만.
        #   env reset: _yaw_ref = _ee_target_yaw = (측정)grip yaw 와 동일 convention.
        yaw_ref = quat_z_yaw(grip_center_quat(robot, left_id))[0].item()
        ee_target_yaw_acc = yaw_ref

        for step_i in range(max_steps):
            # ---- state 계산 (insert_env._get_observations 와 동일 식) ----
            ee_pos_w = grip_center_pos(robot, left_id, right_id)
            ee_xy_env = ee_pos_w[:, :2] - scene.env_origins[:, :2]
            slot_rel_x = float(cell_xy_v[0] - ee_xy_env[0, 0].item())
            slot_rel_y = float(cell_xy_v[1] - ee_xy_env[0, 1].item())

            # actual ee yaw (insert env 와 동일 공식)
            ee_quat_w = grip_center_quat(robot, left_id)
            cur_ee_yaw = quat_z_yaw(ee_quat_w)[0].item()
            # 박스 180° 대칭 → yaw 오차 ±90° fold (insert_env.fold_yaw_sym 와 동일 — 정책 parity)
            _raw_yaw = cell_yaw_v - cur_ee_yaw
            slot_yaw_err = float((((_raw_yaw + math.pi / 2.0) % math.pi) - math.pi / 2.0))

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

            # ---- action 적용 (insert_env v25 parity) ----
            # 버그B 수정: env 는 yaw_only → 정책 xy action 무시, xy 는 cell 에 고정 holding.
            #   (이전: 정책 xy 적용 → 드리프트. env 와 불일치였음.)
            # 버그A 수정: yaw 는 누적 setpoint(yaw_ref ± margin). 비누적(측정yaw+Δ)+±90°clip 폐기.
            delta_yaw = float(action[2]) * RL_INSERT_ACTION_SCALE_YAW
            ee_target_yaw_acc = min(yaw_ref + RL_INSERT_YAW_MARGIN,
                                    max(yaw_ref - RL_INSERT_YAW_MARGIN,
                                        ee_target_yaw_acc + delta_yaw))

            # xy 는 cell 에 고정 (정책 xy 버림 — env yaw_only 와 일치)
            target_pos_env = torch.tensor(
                [cell_xy_v[0], cell_xy_v[1], TRANSPORT_Z],
                device=device, dtype=torch.float)

            yaw_q = quat_from_angle_axis(
                torch.tensor([ee_target_yaw_acc], device=device, dtype=torch.float),
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
        return ee_target_yaw_acc, success

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

    n_cells = layout.GRID_NUM_X * layout.GRID_NUM_Y

    def random_grid_target():
        """그리드 전체 jitter(±3cm/±10°) + 9셀 중 타깃 1개 선택.
        반환: (grid_yaw, 타깃셀idx, 타깃셀 world xy, 8벽 world pose 목록)."""
        gx = CELL_CENTER_X + float(torch.empty(1).uniform_(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE).item())
        gy = CELL_CENTER_Y + float(torch.empty(1).uniform_(-CELL_SPAWN_XY_NOISE, CELL_SPAWN_XY_NOISE).item())
        gyaw = float(torch.empty(1).uniform_(-CELL_SPAWN_YAW_MAX, CELL_SPAWN_YAW_MAX).item())
        tgt = int(torch.randint(0, n_cells, (1,)).item())
        walls, cells = scene_helpers.grid_world_poses(gx, gy, gyaw)
        tx, ty = cells[tgt]
        return gyaw, tgt, tx, ty, walls

    def update_grid_walls(walls):
        """8개 격벽을 grid pose 로 강제 이동 (kinematic)."""
        for name, _size, (wx, wy, wz), wyaw in walls:
            half = wyaw / 2.0
            q = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]],
                             device=device, dtype=torch.float)
            pos = torch.tensor([[wx + env_origin[0].item(),
                                 wy + env_origin[1].item(),
                                 wz + env_origin[2].item()]],
                               device=device, dtype=torch.float)
            scene[name].write_root_pose_to_sim(torch.cat([pos, q], dim=-1))
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

        # ---- 박스(앞,책상 위) + 그리드(뒤) spawn, 타깃 셀 선택 ----
        bx, by, byaw = random_box_spawn()
        if args_cli.demo_grid:
            # 데모: 그리드 고정(jitter 없음) + 셀 0→9 순차
            cyaw = math.radians(args_cli.grid_yaw)
            tgt_cell = rep % n_cells
            walls, _cells = scene_helpers.grid_world_poses(CELL_CENTER_X, CELL_CENTER_Y, cyaw)
            cx, cy = _cells[tgt_cell]
        else:
            cyaw, tgt_cell, cx, cy, walls = random_grid_target()
        update_grid_walls(walls)
        print(f"[run {rep+1}] box xy=({bx:+.3f},{by:+.3f}) yaw={math.degrees(byaw):+.1f}° "
              f"| target cell #{tgt_cell} xy=({cx:+.3f},{cy:+.3f}) grid_yaw={math.degrees(cyaw):+.1f}°")

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

        # 앞쪽(grasp) 단계 joint1 큰 swing 제한 ON
        _J1_CLAMP[0] = layout.GRASP_JOINT1_CLAMP

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

        # ---- 3c. Lift (책상 위 높이) ----
        stage_move(grasp_pos, lift_pos,
                   STAGE_DURATION_S["lift"], gripper_close,
                   "3c. Lift",
                   start_quat_w=ee_quat_after_align, end_quat_w=ee_quat_after_align)

        # 앞쪽 끝 → joint1 클램프 해제 (turn-around은 의도적으로 joint1 크게 회전)
        _J1_CLAMP[0] = None

        # ---- 3c2. Turn-around: joint1만 회전(joint2~6 lift 유지) → 그리퍼 수직 유지, joint5 안 돎 ----
        cur_arm = robot.data.joint_pos[0, arm_ids].tolist()
        turn_target = {"joint1": BACK_HOME_JOINTS["joint1"]}
        for k in range(2, 7):
            turn_target[f"joint{k}"] = cur_arm[k - 1]
        stage_joint_move(turn_target, STAGE_DURATION_S["transport"], gripper_close,
                         "3c2. Turn around (joint1 only)")

        # turn 후 자연 그리퍼 yaw — cell 로 pre-align 하지 않음(RL 이 yaw 보정 담당, sim2real 충실).
        #   (이전: 여기서 정확한 cyaw 로 slerp 해 RL 전에 yaw 를 다 맞춰 RL 을 우회했음 → 제거.
        #    박스 180° 대칭 + RL obs 의 fold_yaw_sym 이 branch 를 자동 처리하므로 분기 선택 불필요.)
        turned_quat = grip_center_quat(robot, left_id)[0].unsqueeze(0)
        _zaxis = torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float)
        cell_ee_quat = turned_quat   # placeholder — stage 4 RL 후 RL 최종 yaw 로 갱신.

        # ---- 3d-1. 높은 데서 타깃 셀 위로 (xy 만, 자연 yaw 유지 — yaw 는 RL 이 보정) ----
        cur_after = (grip_center_pos(robot, left_id, right_id)[0] - env_origin)
        high_z = float(cur_after[2].item())
        align_start = torch.tensor([cur_after[0].item(), cur_after[1].item(), high_z],
                                   device=device, dtype=torch.float)
        above_high = torch.tensor([transport_offset[0].item(), transport_offset[1].item(), high_z],
                                  device=device, dtype=torch.float)
        stage_move(align_start, above_high,
                   STAGE_DURATION_S["transport"], gripper_close,
                   "3d-1. Above cell (xy only, hold natural yaw)",
                   start_quat_w=turned_quat, end_quat_w=turned_quat)

        # ---- 3d-2. 수직 하강 (high → hover), 자연 yaw 고정 ----
        stage_move(above_high, transport_offset,
                   STAGE_DURATION_S["lift"], gripper_close,
                   "3d-2. Descend to hover (hold natural yaw)",
                   start_quat_w=turned_quat, end_quat_w=turned_quat)

        # ---- 4. Insert RL align (PPO inference) — yaw 를 여기서 처음 보정 ----
        insert_final_yaw, insert_success = stage_rl_insert(args_cli.rl_max_steps, (cx, cy), cyaw)
        if insert_success:
            insert_success_count += 1
        # downstream(5a/5b/6a) 자세 = RL 최종 yaw (motion 이 다시 cyaw 로 snap 하지 않도록).
        cell_ee_quat = quat_mul(
            quat_from_angle_axis(
                torch.tensor([insert_final_yaw], device=device, dtype=torch.float), _zaxis),
            base_ee_quat)

        # ---- ★ xy 진단 로그 (RL 직후): grip(=IK holding 대상)과 box(매달림)가 cell xy 에 얼마나 ----
        _ee_xy = (grip_center_pos(robot, left_id, right_id)[0] - env_origin)
        _bx_xy = (box.data.root_pos_w[0] - env_origin)
        _ee_err = ((_ee_xy[0].item() - cx) ** 2 + (_ee_xy[1].item() - cy) ** 2) ** 0.5
        _bx_err = ((_bx_xy[0].item() - cx) ** 2 + (_bx_xy[1].item() - cy) ** 2) ** 0.5
        _fold = lambda a: abs((((a) + math.pi / 2.0) % math.pi) - math.pi / 2.0)  # 박스 180° 대칭
        _eyaw1 = math.degrees(_fold(cyaw - quat_z_yaw(grip_center_quat(robot, left_id))[0].item()))
        _byaw1 = math.degrees(_fold(cyaw - quat_z_yaw(box.data.root_quat_w)[0].item()))
        print(f"  [xy@RL-end ] cell=({cx:+.3f},{cy:+.3f}) | "
              f"grip=({_ee_xy[0].item():+.3f},{_ee_xy[1].item():+.3f}) Δgrip={_ee_err*100:5.2f}cm | "
              f"box=({_bx_xy[0].item():+.3f},{_bx_xy[1].item():+.3f}) Δbox={_bx_err*100:5.2f}cm | "
              f"Δyaw ee={_eyaw1:4.1f}° box={_byaw1:4.1f}°")

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

        # ---- ★ xy 진단 로그 (5a 하강완료): box 가 cell xy 에 안착됐나 (하강 중 스윙/벽접촉 드리프트 확인) ----
        _ee_xy2 = (grip_center_pos(robot, left_id, right_id)[0] - env_origin)
        _bx_xy2 = (box.data.root_pos_w[0] - env_origin)
        _ee_err2 = ((_ee_xy2[0].item() - cx) ** 2 + (_ee_xy2[1].item() - cy) ** 2) ** 0.5
        _bx_err2 = ((_bx_xy2[0].item() - cx) ** 2 + (_bx_xy2[1].item() - cy) ** 2) ** 0.5
        _eyaw2 = math.degrees(_fold(cyaw - quat_z_yaw(grip_center_quat(robot, left_id))[0].item()))
        _byaw2 = math.degrees(_fold(cyaw - quat_z_yaw(box.data.root_quat_w)[0].item()))
        print(f"  [xy@descend] cell=({cx:+.3f},{cy:+.3f}) | "
              f"grip=({_ee_xy2[0].item():+.3f},{_ee_xy2[1].item():+.3f}) Δgrip={_ee_err2*100:5.2f}cm | "
              f"box=({_bx_xy2[0].item():+.3f},{_bx_xy2[1].item():+.3f}) Δbox={_bx_err2*100:5.2f}cm | "
              f"Δyaw ee={_eyaw2:4.1f}° box={_byaw2:4.1f}°")

        # ---- 5b. Release ----
        stage_hold(descend_end_2, STAGE_DURATION_S["release"], GRIPPER_OPEN,
                   "5b. Release", hold_quat_w=cell_ee_quat)

        # ---- 데모: 방금 안착한 박스를 누적 표시박스로 복사(셀에 남김) → 다음 run 에서 active box reset 돼도 유지 ----
        if args_cli.demo_grid:
            for _ in range(20):   # 박스 안착 안정화
                scene.write_data_to_sim(); sim.step(); scene.update(dt)
            _bp = box.data.root_pos_w[0:1].clone()
            _bq = box.data.root_quat_w[0:1].clone()
            scene[f"PlacedBox{tgt_cell}"].write_root_pose_to_sim(torch.cat([_bp, _bq], dim=-1))
            scene[f"PlacedBox{tgt_cell}"].write_root_velocity_to_sim(
                torch.zeros((1, 6), device=device, dtype=torch.float))
            print(f"[demo] cell #{tgt_cell} 채움 ({tgt_cell+1}/{n_cells})")

        # ---- 6a. Retract up (수직만 — cell yaw 유지, 손목 안 돎) ----
        #   ★ 이전엔 여기서 cell_ee_quat→base_ee_quat slerp 으로 yaw 를 풀었는데, 팔이 아직
        #   turn-around(joint1≈-2.9) 상태라 IK 가 world-yaw-0 자세를 만들려고 joint6 를 ~180° 크랭크
        #   = "올라올 때 joint6 flip". world yaw 는 joint1 이 만든 것이므로 joint1 으로 풀어야 함(아래 6b).
        #   여기선 cell_ee_quat 유지 → 빈 그리퍼 순수 수직 상승(셀 벽 클리어).
        retract_pos_actual = torch.tensor(
            [cur_ee_2[0].item(), cur_ee_2[1].item(), RETRACT_Z],
            device=device, dtype=torch.float)
        stage_move(descend_end_2, retract_pos_actual,
                   STAGE_DURATION_S["retract_up"], GRIPPER_OPEN,
                   "6a. Retract up (vertical, hold yaw)",
                   start_quat_w=cell_ee_quat, end_quat_w=cell_ee_quat)

        # ---- 6b. Retract home (joint-space 역-turn) ----
        #   forward 3c2(joint1-only turn)의 역순. joint1 -2.9→0 untwist 하면 world yaw 가 자연히
        #   앞으로 풀리고 joint6 는 home 값 유지 → flip 없음 + base 관통 없음(§1 turn-around=joint1만 준수).
        stage_joint_move(HOME_JOINT_POS, STAGE_DURATION_S["retract_home"], GRIPPER_OPEN,
                         "6b. Retract home (joint untwist)")

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
