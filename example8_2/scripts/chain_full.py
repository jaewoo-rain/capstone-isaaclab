"""example8_2 Full Chain — example5 (grasp+lift) + example7 (place) chained.

LiftEnv 위에서 두 정책 chain:
- Phase 0 (GRASP_LIFT): example5 policy + LiftEnv obs (34-dim)
- Phase 1 (PLACE): example7 policy + 변환된 PlaceEnv obs (31-dim)
- 전환: box_z >= lift_phase_threshold (0.15)

3 박스 sequential (한 박스 완료 후 다음).

Usage:
  python source/example8_2/scripts/chain_full.py
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
parser.add_argument("--max_steps", type=int, default=720)
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
from stable_baselines3.common.vec_env import VecNormalize, VecEnvWrapper

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper

from source.example5.tasks.lift.lift_env_cfg import LiftEnvCfg
from source.example5.tasks.lift.lift_env import LiftEnv


# ======================================================================
# Place obs converter (LiftEnv state → PlaceEnv 31-dim format)
# ======================================================================
def compute_place_obs(raw_env: LiftEnv, target_cell_pos_w: torch.Tensor) -> np.ndarray:
    """LiftEnv state로부터 example7용 31-dim obs 생성.

    PlaceEnv obs 구조:
    - arm_joint_pos×6 (정규화 -1~1)
    - arm_joint_vel×6 (정규화)
    - gripper_close×1 (0~1)
    - obj_pos×3 (env-rel)
    - obj_vel×3
    - ee_pos×3 (env-rel)
    - obj_to_target×3
    - achieved_goal×3 (obj pos)
    - desired_goal×3 (target cell pos)
    """
    n = raw_env.num_envs
    device = raw_env.device

    arm_ids = raw_env.arm_joint_ids
    arm_pos = raw_env._robot.data.joint_pos[:, arm_ids]
    arm_vel = raw_env._robot.data.joint_vel[:, arm_ids]
    arm_lower = raw_env.robot_dof_lower_limits[arm_ids]
    arm_upper = raw_env.robot_dof_upper_limits[arm_ids]
    # PlaceEnv 정확한 형식 매칭
    arm_pos_scaled = 2.0 * (arm_pos - arm_lower) / (arm_upper - arm_lower + 1e-8) - 1.0
    arm_vel_scaled = arm_vel * 1.0  # cfg.dof_velocity_scale = 1.0

    grip_pos = raw_env._robot.data.joint_pos[:, raw_env.main_gripper_joint_id].unsqueeze(-1)
    grip_lower = raw_env.robot_dof_lower_limits[raw_env.main_gripper_joint_id].view(1, 1)
    grip_upper = raw_env.robot_dof_upper_limits[raw_env.main_gripper_joint_id].view(1, 1)
    gripper_close = (grip_pos - grip_lower) / (grip_upper - grip_lower + 1e-8)

    env_origins = raw_env.scene.env_origins
    obj_pos_w = raw_env._object.data.root_pos_w
    obj_pos_rel = obj_pos_w - env_origins
    obj_vel = raw_env._object.data.root_lin_vel_w

    ee_pos_rel = raw_env.grip_center_pos - env_origins
    target_pos_rel = target_cell_pos_w - env_origins
    obj_to_target = target_pos_rel - obj_pos_rel

    # PlaceEnv 순서: arm_pos, arm_vel, gripper, ee_pos, obj_pos, obj_vel, obj_to_target
    core = torch.cat([
        arm_pos_scaled,       # 6
        arm_vel_scaled,       # 6
        gripper_close,        # 1
        ee_pos_rel,           # 3 ← obj_pos보다 앞
        obj_pos_rel,          # 3
        obj_vel,              # 3
        obj_to_target,        # 3
    ], dim=-1)
    # PlaceEnv는 [-5, 5] clamp
    core = torch.clamp(core, -5.0, 5.0)

    # achieved/desired 모두 world coord
    achieved = obj_pos_w.clone()
    desired = target_cell_pos_w.clone()

    # Dict for HER
    obs = {
        "observation": core.cpu().numpy().astype(np.float32),
        "achieved_goal": achieved.cpu().numpy().astype(np.float32),
        "desired_goal": desired.cpu().numpy().astype(np.float32),
    }
    return obs


def main():
    print("=" * 70)
    print("example8_2 Full Chain Inference (lift + place)")
    print("=" * 70)

    # 정책 로드
    if not os.path.exists(args_cli.grasp_ckpt):
        print(f"❌ {args_cli.grasp_ckpt} 없음")
        return
    if not os.path.exists(args_cli.place_ckpt):
        print(f"❌ {args_cli.place_ckpt} 없음")
        return

    # ===== LiftEnv 생성 (chain의 base env) =====
    cfg = LiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.episode_length_s = 30.0  # 충분한 시간 (chain은 30초 max)
    # threshold는 0.20 그대로 (obs 일관성), chain script에서 done monkey-patch
    raw_env = LiftEnv(cfg=cfg)

    # Monkey-patch: LiftEnv done에서 lift_success 제거 (chain mode에서 PLACE까지 진행)
    original_dones = raw_env._get_dones
    def patched_dones():
        terminated, truncated = original_dones()
        # lift_success(obj_z > threshold)로 인한 종료 무력화
        from isaaclab.utils.math import quat_apply
        obj_height = raw_env.obj_pos_w[:, 2]
        # 원래: terminated = (height > threshold) | (height < -0.1) | fallen_on_ground
        # 변경: 첫 조건 제거, 나머지만
        obj_quat = raw_env._object.data.root_quat_w
        local_up = torch.zeros((obj_quat.shape[0], 3), device=raw_env.device)
        local_up[:, 2] = 1.0
        world_up = quat_apply(obj_quat, local_up)
        upright = world_up[:, 2]
        on_ground = obj_height < 0.07
        fallen_on_ground = (upright < 0.1) & on_ground
        terminated_new = (obj_height < -0.1) | fallen_on_ground
        return terminated_new, truncated
    raw_env._get_dones = patched_dones

    grasp_env = Sb3VecEnvWrapper(raw_env)
    if os.path.exists(args_cli.grasp_vecnorm):
        grasp_env = VecNormalize.load(args_cli.grasp_vecnorm, grasp_env)
        grasp_env.training = False
        grasp_env.norm_reward = False
    grasp_policy = PPO.load(args_cli.grasp_ckpt, env=grasp_env, device="cuda")
    # SAC 로드 — dummy dict env로 wrap (predict만 사용, step 안 함)
    import gymnasium as gym

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

    dummy_env = DummyDictEnv()
    place_policy = SAC.load(args_cli.place_ckpt, env=dummy_env, device="cuda")

    print(f"✅ grasp policy: {args_cli.grasp_ckpt}")
    print(f"✅ place policy: {args_cli.place_ckpt}")
    print()

    # ===== Cell 위치 정의 (3개 셀) =====
    # example7은 1x1 grid (cell 1개) 학습 → 모두 동일 cell로 (테스트)
    trained_cell = torch.tensor([0.25, -0.45, 0.06], device=raw_env.device)
    cell_positions = [trained_cell, trained_cell, trained_cell]
    box_positions = [
        (0.45, -0.20, 0.06),
        (0.45, -0.10, 0.06),
        (0.45,  0.00, 0.06),
    ]

    # ===== 3-box sequential =====
    box_results = []
    for box_idx in range(args_cli.num_boxes):
        target_cell_pos = cell_positions[box_idx]
        box_pos = box_positions[box_idx]
        target_cell_pos_w = raw_env.scene.env_origins + target_cell_pos.unsqueeze(0)

        print(f"━━━ Box {box_idx} (start {box_pos[0]:.2f},{box_pos[1]:.2f}) → Cell {box_idx} ━━━")

        # 박스 위치 변경 후 reset
        cfg.object.init_state.pos = box_pos
        obs_grasp = grasp_env.reset()

        phase = "GRASP_LIFT"
        success = False
        max_z = 0.0
        switch_step = -1

        for step in range(args_cli.max_steps):
            if not simulation_app.is_running():
                break

            # 박스 z 측정
            obj_z = raw_env._object.data.root_pos_w[0, 2].item()
            env_z = raw_env.scene.env_origins[0, 2].item()
            obj_z_rel = obj_z - env_z
            max_z = max(max_z, obj_z_rel)

            # Phase 전환: box_z >= lift_threshold
            if phase == "GRASP_LIFT" and obj_z_rel >= args_cli.lift_threshold:
                phase = "PLACE"
                switch_step = step
                print(f"  🔄 phase 전환 → PLACE (step={step}, z={obj_z_rel:.3f})")

            # Action 계산
            if phase == "GRASP_LIFT":
                action, _ = grasp_policy.predict(obs_grasp, deterministic=True)
            else:  # PLACE
                place_obs = compute_place_obs(raw_env, target_cell_pos_w)
                action, _ = place_policy.predict(place_obs, deterministic=True)

            # Step
            obs_grasp, reward, done, info = grasp_env.step(action)

            # PLACE 성공 판정 (박스가 cell xy 근처 + 바닥)
            if phase == "PLACE":
                obj_xy = raw_env._object.data.root_pos_w[0, :2]
                cell_xy = (raw_env.scene.env_origins[0, :2] + target_cell_pos[:2])
                xy_dist = torch.norm(obj_xy - cell_xy).item()
                obj_bottom = obj_z_rel - 0.059  # box height/2
                if xy_dist < 0.03 and obj_bottom < 0.01:
                    success = True
                    print(f"  ✅ PLACE 성공 (step={step}, xy_dist={xy_dist:.3f}, bottom={obj_bottom:.3f})")
                    break

            if done[0]:
                print(f"  ⚠️ episode 종료 step={step}, phase={phase}")
                break

        if not success:
            print(f"  ❌ Box {box_idx} 실패 (max_z={max_z:.3f}, switch_step={switch_step})")

        box_results.append({"box": box_idx, "success": success, "max_z": max_z, "switch_step": switch_step})

    # 결과
    print()
    print("=" * 70)
    print("📊 결과 요약")
    print("=" * 70)
    success_count = sum(1 for r in box_results if r["success"])
    print(f"  Total: {success_count}/{args_cli.num_boxes}")
    for r in box_results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} Box {r['box']}: max_z={r['max_z']:.3f}, switch_step={r['switch_step']}")
    print("=" * 70)

    grasp_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
