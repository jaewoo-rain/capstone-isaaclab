"""example8_2 — Multi-box chain inference

MultiBoxEnv (3개 박스 동시 spawn) 위에서:
- Active box i 처리: example5 grasp+lift → example7 place
- 박스 i 완료 → active box i+1로 전환
- robot reset (다음 박스 위치로)
- 3 박스 모두 완료 시 종료

example5/example7 obs는 MultiBoxEnv state로부터 외부 변환.

Usage:
  python source/example8_2/scripts/chain_multi.py --headless
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--grasp_ckpt", type=str, default="checkpoints/example5.zip")
parser.add_argument("--grasp_vecnorm", type=str, default="checkpoints/example5_vecnorm.pkl")
parser.add_argument("--place_ckpt", type=str, default="checkpoints/example7.zip")
parser.add_argument("--num_boxes", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--lift_threshold", type=float, default=0.15)
parser.add_argument("--max_steps_per_box", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper

from source.example8_2.tasks.multi_box import MultiBoxEnv, MultiBoxEnvCfg


# ======================================================================
# example5 obs 변환: MultiBoxEnv state → 34-dim LiftEnv obs format
# ======================================================================
def compute_grasp_obs(raw_env: MultiBoxEnv, active_box_idx: int, target_override_w: torch.Tensor = None) -> np.ndarray:
    """MultiBoxEnv state로부터 example5용 34-dim obs.

    target_override_w: None이면 박스 잡기용 (기본). 제공되면 그 위치를 grasp_target으로 사용
                       → example5가 그 위치로 grip을 가져가려고 함 (transport trick)
    """
    n = raw_env.num_envs
    device = raw_env.device

    # arm + gripper joint pos/vel (10 = 6 + 4)
    all_ids = raw_env.arm_joint_ids + raw_env.gripper_joint_ids
    joint_pos = raw_env._robot.data.joint_pos[:, all_ids]
    joint_vel = raw_env._robot.data.joint_vel[:, all_ids]
    lower = raw_env.robot_dof_lower_limits[all_ids]
    upper = raw_env.robot_dof_upper_limits[all_ids]
    pos_scaled = 2.0 * (joint_pos - lower) / (upper - lower + 1e-8) - 1.0
    vel_scaled = joint_vel * 1.0  # cfg.dof_velocity_scale = 1.0

    # active box pos
    active_obj = raw_env._objects[active_box_idx]
    obj_pos_w = active_obj.data.root_pos_w
    env_origins = raw_env.scene.env_origins
    obj_pos_rel = obj_pos_w - env_origins

    # Body pos
    l_pos = raw_env._robot.data.body_pos_w[:, raw_env.left_finger_body_id, :]
    r_pos = raw_env._robot.data.body_pos_w[:, raw_env.right_finger_body_id, :]
    grip_center = 0.5 * (l_pos + r_pos)

    # grasp target — 기본 박스 위, override 시 target 위치
    if target_override_w is not None:
        # Transport trick: target_cell을 grasp_target으로 → example5가 그쪽으로 이동
        grasp_target = target_override_w.clone()
    else:
        grasp_target = obj_pos_w.clone()
        grasp_target[:, 2] += 0.04  # grasp_target_z_offset
    obj_to_grip = grasp_target - grip_center

    # 손가락 → "타겟" 벡터 (transport 시 박스 → 타겟이 아니라 박스 그대로)
    if target_override_w is not None:
        # 박스는 이미 잡혀있으니, 손가락→박스 거리를 그대로 사용
        l_to_obj = obj_pos_w - l_pos
        r_to_obj = obj_pos_w - r_pos
    else:
        l_to_obj = obj_pos_w - l_pos
        r_to_obj = obj_pos_w - r_pos

    # gripper close
    grip_pos = raw_env._robot.data.joint_pos[:, raw_env.main_gripper_joint_id]
    g_lower = raw_env.robot_dof_lower_limits[raw_env.main_gripper_joint_id]
    g_upper = raw_env.robot_dof_upper_limits[raw_env.main_gripper_joint_id]
    gripper_close = (grip_pos - g_lower) / (g_upper - g_lower + 1e-8)

    # to_lift_target — transport 시는 target z, grasp 시는 lift threshold
    if target_override_w is not None:
        to_lift_target = target_override_w[:, 2] - obj_pos_w[:, 2]
    else:
        to_lift_target = 0.20 - obj_pos_w[:, 2]

    obs = torch.cat([
        pos_scaled,                         # 10
        vel_scaled,                         # 10
        obj_pos_rel,                        # 3
        obj_to_grip,                        # 3
        l_to_obj,                           # 3
        r_to_obj,                           # 3
        gripper_close.unsqueeze(-1),        # 1
        to_lift_target.unsqueeze(-1),       # 1
    ], dim=-1)  # = 34
    return obs.cpu().numpy().astype(np.float32)


# ======================================================================
# example7 obs 변환: MultiBoxEnv state → 31-dim PlaceEnv obs (Dict)
# ======================================================================
def compute_place_obs(raw_env: MultiBoxEnv, active_box_idx: int, target_cell_w: torch.Tensor):
    """MultiBoxEnv state → PlaceEnv 31-dim obs Dict."""
    arm_pos = raw_env._robot.data.joint_pos[:, raw_env.arm_joint_ids]
    arm_vel = raw_env._robot.data.joint_vel[:, raw_env.arm_joint_ids]
    arm_lower = raw_env.robot_dof_lower_limits[raw_env.arm_joint_ids]
    arm_upper = raw_env.robot_dof_upper_limits[raw_env.arm_joint_ids]
    arm_pos_scaled = 2.0 * (arm_pos - arm_lower) / (arm_upper - arm_lower + 1e-8) - 1.0
    arm_vel_scaled = arm_vel * 1.0

    grip_pos = raw_env._robot.data.joint_pos[:, raw_env.main_gripper_joint_id].unsqueeze(-1)
    g_lower = raw_env.robot_dof_lower_limits[raw_env.main_gripper_joint_id].view(1, 1)
    g_upper = raw_env.robot_dof_upper_limits[raw_env.main_gripper_joint_id].view(1, 1)
    gripper_close = (grip_pos - g_lower) / (g_upper - g_lower + 1e-8)

    env_origins = raw_env.scene.env_origins
    active_obj = raw_env._objects[active_box_idx]
    obj_pos_w = active_obj.data.root_pos_w
    obj_pos_rel = obj_pos_w - env_origins
    obj_vel = active_obj.data.root_lin_vel_w

    # ee_pos
    l_pos = raw_env._robot.data.body_pos_w[:, raw_env.left_finger_body_id, :]
    r_pos = raw_env._robot.data.body_pos_w[:, raw_env.right_finger_body_id, :]
    ee_pos_w = 0.5 * (l_pos + r_pos)
    ee_pos_rel = ee_pos_w - env_origins

    target_pos_rel = target_cell_w - env_origins
    obj_to_target = target_pos_rel - obj_pos_rel

    core = torch.cat([
        arm_pos_scaled,       # 6
        arm_vel_scaled,       # 6
        gripper_close,        # 1
        ee_pos_rel,           # 3
        obj_pos_rel,          # 3
        obj_vel,              # 3
        obj_to_target,        # 3
    ], dim=-1)
    core = torch.clamp(core, -5.0, 5.0)

    return {
        "observation": core.cpu().numpy().astype(np.float32),
        "achieved_goal": obj_pos_w.cpu().numpy().astype(np.float32),
        "desired_goal": target_cell_w.cpu().numpy().astype(np.float32),
    }


def main():
    import sys as _sys
    def _p(msg):
        print(msg, flush=True)
        _sys.stdout.flush()

    _p("[diag] main() start")
    _p("=" * 70)
    _p("example8_2 Multi-Box Chain Inference")
    _p("=" * 70)

    if not os.path.exists(args_cli.grasp_ckpt):
        _p(f"❌ {args_cli.grasp_ckpt} 없음")
        return
    if not os.path.exists(args_cli.place_ckpt):
        _p(f"❌ {args_cli.place_ckpt} 없음")
        return
    _p("[diag] checkpoints exist")

    # MultiBoxEnv 생성
    _p("[diag] before MultiBoxEnvCfg()")
    cfg = MultiBoxEnvCfg()
    _p("[diag] MultiBoxEnvCfg() done")
    cfg.scene.num_envs = args_cli.num_envs
    _p("[diag] before MultiBoxEnv(cfg)")
    raw_env = MultiBoxEnv(cfg=cfg)
    _p(f"✅ MultiBoxEnv 생성 (3 박스, 3 셀)")

    # 정책 로드
    # 1) grasp policy (example5 PPO + VecNormalize)
    _p(f"[diag] loading PPO from {args_cli.grasp_ckpt}")
    grasp_policy = PPO.load(args_cli.grasp_ckpt, device="cuda")
    # VecNormalize 통계 로드 (raw obs 정규화에 사용)
    grasp_vecnorm = None
    if os.path.exists(args_cli.grasp_vecnorm):
        # VecNormalize는 env 없이도 stats만 사용 가능
        import pickle
        with open(args_cli.grasp_vecnorm, "rb") as f:
            grasp_vecnorm = pickle.load(f)
        print(f"✅ grasp VecNormalize: {args_cli.grasp_vecnorm}")
    print(f"✅ grasp policy: {args_cli.grasp_ckpt}")

    # 2) place policy (example7 SAC + HER) — dummy dict env
    class DummyDictEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Dict({
                "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32),
                "achieved_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "desired_goal": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            })
            self.action_space = gym.spaces.Box(low=-100.0, high=100.0, shape=(7,), dtype=np.float32)
        def reset(self, *, seed=None, options=None):
            return self.observation_space.sample(), {}
        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}
        def compute_reward(self, achieved_goal, desired_goal, info):
            return -np.linalg.norm(achieved_goal - desired_goal, axis=-1)

    _p(f"[diag] loading SAC from {args_cli.place_ckpt}")
    place_policy = SAC.load(args_cli.place_ckpt, env=DummyDictEnv(), device="cuda")
    _p(f"✅ place policy: {args_cli.place_ckpt}")
    _p("")

    # ===== 실행 =====
    _p("[diag] raw_env.reset() ...")
    raw_env.reset()  # 3개 박스 spawn
    _p("[diag] reset done, starting sim loop")

    box_results = []
    import time
    sim_start = time.time()
    for box_idx in range(args_cli.num_boxes):
        print(f"━━━ Box {box_idx} (active) → Cell {cfg.box_to_cell[box_idx]} ━━━", flush=True)
        raw_env.set_active_box(0, box_idx)

        cell_idx = cfg.box_to_cell[box_idx]
        target_cell_local = raw_env.cell_centers_local[cell_idx]
        target_cell_w = raw_env.scene.env_origins + target_cell_local.unsqueeze(0)

        phase = "GRASP_LIFT"
        success = False
        max_z = 0.0
        release_steps = 0
        box_start = time.time()

        for step in range(args_cli.max_steps_per_box):
            if not simulation_app.is_running():
                break

            obj_z = raw_env._objects[box_idx].data.root_pos_w[0, 2].item()
            env_z = raw_env.scene.env_origins[0, 2].item()
            obj_z_rel = obj_z - env_z
            max_z = max(max_z, obj_z_rel)

            obj_xy = raw_env._objects[box_idx].data.root_pos_w[0, :2]
            cell_xy = target_cell_w[0, :2]
            xy_dist = torch.norm(obj_xy - cell_xy).item()

            # Progress every 100 steps
            if step % 100 == 0:
                el = time.time() - box_start
                print(f"  [step {step}] phase={phase} z={obj_z_rel:.3f} xy_dist={xy_dist:.3f} max_z={max_z:.3f} elapsed={el:.1f}s", flush=True)

            # Phase 전환
            if phase == "GRASP_LIFT" and obj_z_rel >= args_cli.lift_threshold:
                phase = "TRANSPORT"
                print(f"  🔄 → TRANSPORT (step={step}, z={obj_z_rel:.3f})", flush=True)
            elif phase == "TRANSPORT" and xy_dist < 0.05 and obj_z_rel >= 0.20:
                phase = "RELEASE"
                print(f"  🔄 → RELEASE (step={step}, xy_dist={xy_dist:.3f})", flush=True)

            # Action 계산
            if phase == "GRASP_LIFT":
                obs = compute_grasp_obs(raw_env, box_idx)
                if grasp_vecnorm is not None:
                    obs_norm = (obs - grasp_vecnorm.obs_rms.mean) / np.sqrt(grasp_vecnorm.obs_rms.var + grasp_vecnorm.epsilon)
                    obs_norm = np.clip(obs_norm, -grasp_vecnorm.clip_obs, grasp_vecnorm.clip_obs).astype(np.float32)
                    action, _ = grasp_policy.predict(obs_norm, deterministic=True)
                else:
                    action, _ = grasp_policy.predict(obs, deterministic=True)
            elif phase == "TRANSPORT":
                # example5 trick: target = cell xy at z=0.30
                target_above_cell = target_cell_w.clone()
                target_above_cell[:, 2] = 0.30
                obs = compute_grasp_obs(raw_env, box_idx, target_override_w=target_above_cell)
                if grasp_vecnorm is not None:
                    obs_norm = (obs - grasp_vecnorm.obs_rms.mean) / np.sqrt(grasp_vecnorm.obs_rms.var + grasp_vecnorm.epsilon)
                    obs_norm = np.clip(obs_norm, -grasp_vecnorm.clip_obs, grasp_vecnorm.clip_obs).astype(np.float32)
                    action, _ = grasp_policy.predict(obs_norm, deterministic=True)
                else:
                    action, _ = grasp_policy.predict(obs, deterministic=True)
            else:  # RELEASE
                # 모든 arm 정지 + gripper open
                action = np.zeros((1, 7), dtype=np.float32)
                action[0, 6] = -1.0  # gripper open
                release_steps += 1

            # step
            action_torch = torch.tensor(action, device=raw_env.device, dtype=torch.float32)
            if action_torch.dim() == 1:
                action_torch = action_torch.unsqueeze(0)
            raw_env.step(action_torch)

            # Success 판정
            obj_bottom = obj_z_rel - 0.059
            if phase == "RELEASE" and release_steps > 30:
                # release 후 30 step 지나면 결과 평가
                if xy_dist < 0.05 and obj_bottom < 0.05:
                    success = True
                    print(f"  ✅ PLACE 성공 (step={step}, xy_dist={xy_dist:.3f}, bottom={obj_bottom:.3f})", flush=True)
                else:
                    print(f"  ⚠️ Release done but missed (xy={xy_dist:.3f}, bot={obj_bottom:.3f})", flush=True)
                break

        if not success:
            print(f"  ❌ Box {box_idx} 실패 (max_z={max_z:.3f})", flush=True)

        box_results.append({"box": box_idx, "success": success, "max_z": max_z})

    # 결과
    print()
    print("=" * 70)
    print("📊 결과 요약")
    print("=" * 70)
    success_count = sum(1 for r in box_results if r["success"])
    print(f"  Total: {success_count}/{args_cli.num_boxes}")
    for r in box_results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} Box {r['box']}: max_z={r['max_z']:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()
