"""example8_2: 3-box chain inference (Phase 1 MVP)

example5 (grasp + lift) → example7 (place + insert) 정책을 chain.
각 박스 i (i=0,1,2)에 대해 사이클 반복:
1. env reset (1 박스 spawn)
2. example5 정책으로 grasp + lift to z=0.17
3. example7 정책으로 place + insert in cell i
4. cell index 변경, 다시 reset, 다음 박스로

Phase 1 MVP — single-box env에서 3번 sequential 실행.
Phase 2에서 true multi-box env 구현 예정.

Usage:
  python source/example8_2/scripts/chain_inference.py \\
    --grasp_ckpt checkpoints/example5.zip \\
    --grasp_vecnorm checkpoints/example5_vecnorm.pkl \\
    --place_ckpt checkpoints/example7.zip \\
    --place_replay checkpoints/example7_replay.pkl
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Chain: example5 → example7 for 3 boxes")
parser.add_argument("--grasp_ckpt", type=str, default="checkpoints/example5.zip")
parser.add_argument("--grasp_vecnorm", type=str, default="checkpoints/example5_vecnorm.pkl")
parser.add_argument("--place_ckpt", type=str, default="checkpoints/example7.zip")
parser.add_argument("--place_replay", type=str, default="checkpoints/example7_replay.pkl")
parser.add_argument("--num_boxes", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps_per_box", type=int, default=600)  # 10초 at 60Hz
parser.add_argument("--lift_threshold", type=float, default=0.15,
                    help="박스 z >= 이 값 도달 시 example5 → example7 전환")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize, VecEnvWrapper

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper


def main():
    print("=" * 70)
    print("example8_2: 3-box Chain Inference (Phase 1 MVP)")
    print("=" * 70)
    print(f"  grasp policy: {args_cli.grasp_ckpt}")
    print(f"  place policy: {args_cli.place_ckpt}")
    print(f"  num_boxes:    {args_cli.num_boxes}")
    print(f"  lift threshold (phase switch): z >= {args_cli.lift_threshold}m")
    print()

    # 정책 로드 검증
    if not os.path.exists(args_cli.grasp_ckpt):
        print(f"❌ grasp ckpt 없음: {args_cli.grasp_ckpt}")
        return
    if not os.path.exists(args_cli.place_ckpt):
        print(f"❌ place ckpt 없음: {args_cli.place_ckpt}")
        return

    # ----- example5 env (grasp + lift task) 로드 -----
    print("📦 example5 env 로드...")
    from source.example5.tasks.lift.lift_env_cfg import LiftEnvCfg
    from source.example5.tasks.lift.lift_env import LiftEnv

    grasp_cfg = LiftEnvCfg()
    grasp_cfg.scene.num_envs = args_cli.num_envs
    grasp_env = LiftEnv(cfg=grasp_cfg)
    grasp_env_wrapped = Sb3VecEnvWrapper(grasp_env)

    if os.path.exists(args_cli.grasp_vecnorm):
        grasp_env_wrapped = VecNormalize.load(args_cli.grasp_vecnorm, grasp_env_wrapped)
        grasp_env_wrapped.training = False
        grasp_env_wrapped.norm_reward = False
        print(f"  ✅ grasp VecNormalize: {args_cli.grasp_vecnorm}")

    grasp_policy = PPO.load(args_cli.grasp_ckpt, env=grasp_env_wrapped, device="cuda")
    print(f"  ✅ grasp policy 로드")

    # NOTE: 같은 env에서 example7 reward도 동작해야 chain 가능.
    # 현재 example5 env (LiftEnv)와 example7 env (PlaceEnv)는 다른 obs/reward.
    # 진정한 chain을 위해서는 unified env 필요.
    # MVP에서는 grasp env만 사용하고 lift 도달 후 종료 → 다음 cycle.

    # ----- 실행 루프 (3 box × max_steps) -----
    print()
    print("=" * 70)
    print("실행 시작")
    print("=" * 70)

    obs = grasp_env_wrapped.reset()
    box_count = 0
    step_in_box = 0
    successes = 0

    while box_count < args_cli.num_boxes and simulation_app.is_running():
        # Phase 1: grasp + lift (example5 policy)
        action, _ = grasp_policy.predict(obs, deterministic=True)
        obs, reward, done, info = grasp_env_wrapped.step(action)

        # 박스 z 확인
        obj_z = grasp_env._object.data.root_pos_w[0, 2].item()
        env_z = grasp_env.scene.env_origins[0, 2].item()
        obj_z_rel = obj_z - env_z

        step_in_box += 1

        if step_in_box % 60 == 0:
            print(f"  Box {box_count}: step {step_in_box} | obj_z={obj_z_rel:.3f}")

        # Phase 전환: lift 도달
        if obj_z_rel >= args_cli.lift_threshold:
            successes += 1
            print(f"  ✅ Box {box_count}: lift 도달 (z={obj_z_rel:.3f})")
            print(f"  📍 NEXT: example7 정책으로 place 단계 (Phase 2에서 구현)")
            box_count += 1
            step_in_box = 0
            obs = grasp_env_wrapped.reset()  # 다음 박스로

        # 시간 초과
        if step_in_box >= args_cli.max_steps_per_box:
            print(f"  ⏰ Box {box_count}: 시간 초과")
            box_count += 1
            step_in_box = 0
            obs = grasp_env_wrapped.reset()

        # done 처리
        if done[0]:
            obs = grasp_env_wrapped.reset()

    print()
    print("=" * 70)
    print(f"완료: {successes}/{args_cli.num_boxes} 박스 lift 성공")
    print("=" * 70)
    print()
    print("⚠️ MVP 한계:")
    print("  - 현재는 grasp+lift만 실행 (example5 단독)")
    print("  - example7 chain은 unified env 또는 dual env 필요 (Phase 2)")
    print("  - true multi-box env 미구현")

    grasp_env_wrapped.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
