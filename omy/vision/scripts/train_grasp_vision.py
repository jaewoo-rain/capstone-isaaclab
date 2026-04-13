from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# -----------------------------
# CLI 인자 정의
# -----------------------------
parser = argparse.ArgumentParser(description="Train OMY grasp vision PPO")

# IsaacLab 기본 실행 인자 추가
AppLauncher.add_app_launcher_args(parser)

# 학습 관련 사용자 인자
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="이어서 학습할 PPO 체크포인트 경로 (.zip)",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=16,
    help="병렬 환경 개수",
)
parser.add_argument(
    "--timesteps",
    type=int,
    default=2_000_000,
    help="총 학습 스텝 수",
)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

# -----------------------------
# Isaac Sim / Isaac Lab 실행
# -----------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from source.omy.vision.tasks.common.omy_vision_env_cfg import OmyVisionEnvCfg
from source.omy.vision.tasks.grasp.omy_grasp_vision_env import OmyGraspVisionEnv


def main():
    # -----------------------------
    # 환경 설정
    # -----------------------------
    cfg = OmyVisionEnvCfg()

    # CLI에서 env 개수 조절
    cfg.scene.num_envs = args_cli.num_envs

    # 처음에는 GT 기반으로 학습
    cfg.enable_yolo = False

    # base env 생성
    base_env = OmyGraspVisionEnv(cfg, render_mode=None)

    # SB3 wrapper 적용
    env = Sb3VecEnvWrapper(base_env)

    # 저장 폴더 생성
    os.makedirs("checkpoints/rl", exist_ok=True)
    os.makedirs("./logs/grasp_vision", exist_ok=True)

    # -----------------------------
    # 모델 생성 또는 이어서 학습
    # -----------------------------
    if args_cli.resume is not None:
        print(f"🔄 resume from checkpoint: {args_cli.resume}")

        model = PPO.load(
            args_cli.resume,
            env=env,
            device="cuda",
        )

        # logger 다시 연결
        model.tensorboard_log = "./logs/grasp_vision"

    else:
        print("🆕 train from scratch")

        model = PPO(
            "MlpPolicy",
            env,
            n_steps=256,
            batch_size=8192,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log="./logs/grasp_vision",
            device="cuda",
        )

    # -----------------------------
    # 학습 시작
    # -----------------------------
    print("🚀 training start")
    print(f"   num_envs   = {args_cli.num_envs}")
    print(f"   timesteps  = {args_cli.timesteps}")
    print(f"   resume     = {args_cli.resume}")

    model.learn(
        total_timesteps=args_cli.timesteps,
        reset_num_timesteps=False if args_cli.resume is not None else True,
    )

    # -----------------------------
    # 저장
    # -----------------------------
    save_path = "checkpoints/rl/grasp_vision.zip"
    model.save(save_path)
    print(f"✅ training done, saved to: {save_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()