"""example8_2 MVP — example5 정책으로 3개 박스 lift 시나리오 시뮬.

각 박스에 대해:
1. env reset (LiftEnv, 박스 1개)
2. example5 정책으로 grasp + lift
3. 박스 z >= 0.18 도달 또는 episode 종료까지 실행
4. 결과 기록 (lift 성공 여부, 도달 z, 시간)
5. 다음 박스로 (env reset, cell index 변경)
6. 3번 반복

Phase 1 MVP — example7 (place) 통합은 다음 단계.
True multi-box env (3개 동시 spawn)는 Phase 2에서.

Usage:
  python source/example8_2/scripts/chain_mvp.py --num_envs 1
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="example8_2 MVP: example5 chain for 3 boxes")
parser.add_argument("--grasp_ckpt", type=str, default="checkpoints/example5.zip")
parser.add_argument("--grasp_vecnorm", type=str, default="checkpoints/example5_vecnorm.pkl")
parser.add_argument("--num_boxes", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps_per_box", type=int, default=720)  # 12s at 60Hz
parser.add_argument("--lift_threshold", type=float, default=0.15)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper

from source.example5.tasks.lift.lift_env_cfg import LiftEnvCfg
from source.example5.tasks.lift.lift_env import LiftEnv


def main():
    print("=" * 70)
    print("example8_2 MVP: 3-box Chain (example5 단독, place 미적용)")
    print("=" * 70)

    if not os.path.exists(args_cli.grasp_ckpt):
        print(f"❌ {args_cli.grasp_ckpt} 없음")
        return

    # ----- env 생성 -----
    cfg = LiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    raw_env = LiftEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)
    if os.path.exists(args_cli.grasp_vecnorm):
        env = VecNormalize.load(args_cli.grasp_vecnorm, env)
        env.training = False
        env.norm_reward = False
        print(f"✅ VecNormalize: {args_cli.grasp_vecnorm}")
    model = PPO.load(args_cli.grasp_ckpt, env=env, device="cuda")
    print(f"✅ example5 정책: {args_cli.grasp_ckpt}")

    # ----- 박스 위치 매핑 (3박스, 고정 순서) -----
    # 현재 LiftEnv는 단일 박스만 spawn. 다음 iteration에서 위치 변경.
    box_positions = [
        (0.45, -0.20, 0.06),  # Box 0
        (0.45, -0.10, 0.06),  # Box 1 (default)
        (0.45,  0.00, 0.06),  # Box 2
    ]
    cell_targets = [
        "Cell (0,0)",  # Box 0 → 셀 0
        "Cell (0,1)",  # Box 1 → 셀 1
        "Cell (0,2)",  # Box 2 → 셀 2
    ]

    # ----- 3박스 sequential 처리 -----
    print()
    print(f"📦 {args_cli.num_boxes}개 박스 sequential 처리 시작")
    print()

    obs = env.reset()
    box_results = []

    for box_idx in range(args_cli.num_boxes):
        target_pos = box_positions[box_idx]
        target_cell = cell_targets[box_idx]

        # 박스 위치 변경 (raw_env 직접 조작)
        cfg.object.init_state.pos = target_pos
        # Note: 실제로 box position을 변경하려면 env reset 필요. 매 iteration 첫 reset에서 적용.

        print(f"━━━ Box {box_idx} ({target_pos[0]:.2f}, {target_pos[1]:.2f}) → {target_cell} ━━━")
        max_z_reached = 0.0
        lift_success = False
        step = 0

        # 매 iteration env reset
        obs = env.reset()

        while step < args_cli.max_steps_per_box and simulation_app.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            step += 1

            obj_z = raw_env._object.data.root_pos_w[0, 2].item()
            env_z = raw_env.scene.env_origins[0, 2].item()
            obj_z_rel = obj_z - env_z
            max_z_reached = max(max_z_reached, obj_z_rel)

            # lift 성공 (z >= threshold) → 다음 phase 갈 차례 (TODO: example7 호출)
            if obj_z_rel >= args_cli.lift_threshold and not lift_success:
                lift_success = True
                print(f"  ✅ lift 도달 z={obj_z_rel:.3f} step={step}")
                print(f"     [TODO] example7 정책으로 place 실행 → {target_cell}")
                # MVP에서는 여기서 종료, 다음 박스로
                break

            if done[0]:
                if not lift_success:
                    print(f"  ⚠️ episode 종료 (max_z={max_z_reached:.3f}, step={step})")
                break

        box_results.append({
            "box": box_idx,
            "target_cell": target_cell,
            "lift_success": lift_success,
            "max_z": max_z_reached,
            "steps": step,
        })

        if not lift_success:
            print(f"  ❌ Box {box_idx} lift 실패")

    # ----- 결과 요약 -----
    print()
    print("=" * 70)
    print("📊 결과 요약")
    print("=" * 70)
    success_count = sum(1 for r in box_results if r["lift_success"])
    print(f"  Lift 성공: {success_count}/{args_cli.num_boxes}")
    for r in box_results:
        status = "✅" if r["lift_success"] else "❌"
        print(f"  {status} Box {r['box']} → {r['target_cell']}: max_z={r['max_z']:.3f}m, steps={r['steps']}")
    print()
    print("⏳ 다음 단계 (Phase 2):")
    print("  1. example7 정책 통합 (place + insert)")
    print("  2. True 3-box 동시 spawn env")
    print("  3. 박스 i 잡고 셀 i에 넣은 후 박스 i+1 (현재는 매번 reset)")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
