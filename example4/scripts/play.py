"""학습된 Lift PPO 정책 재생 — VecNormalize 포함"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained PPO example4 Lift")
parser.add_argument("--checkpoint", type=str, default="checkpoints/example4.zip")
parser.add_argument("--vecnorm",    type=str, default="checkpoints/example4_vecnorm.pkl",
                    help="VecNormalize pkl 경로 (학습 시 저장된 것)")
parser.add_argument("--num_envs",   type=int, default=1)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper

from source.example4.tasks.lift.lift_env import LiftEnv
from source.example4.tasks.lift.lift_env_cfg import LiftEnvCfg


def main():
    cfg = LiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    # ------------------------------------------------------------------
    # Env 생성
    # ------------------------------------------------------------------
    env = LiftEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(env)

    # ------------------------------------------------------------------
    # VecNormalize 로드 — 학습 시와 동일한 정규화 적용
    # ------------------------------------------------------------------
    if os.path.exists(args_cli.vecnorm):
        env = VecNormalize.load(args_cli.vecnorm, env)
        env.training    = False   # 통계 업데이트 중지
        env.norm_reward = False   # 추론 시 reward 정규화 불필요
        print(f"✅ VecNormalize 로드: {args_cli.vecnorm}")
    else:
        print(f"⚠️  VecNormalize 파일 없음 ({args_cli.vecnorm}). 정규화 없이 실행합니다.")
        print("    성능이 학습 때보다 크게 떨어질 수 있습니다!")

    # ------------------------------------------------------------------
    # 모델 로드
    # ------------------------------------------------------------------
    model = PPO.load(args_cli.checkpoint, env=env, device="cuda")
    print(f"✅ 모델 로드: {args_cli.checkpoint}")

    # ------------------------------------------------------------------
    # 재생 루프
    # ------------------------------------------------------------------
    obs = env.reset()

    episode_count   = 0
    success_count   = 0
    episode_rewards = []
    current_reward  = 0.0

    while simulation_app.is_running():
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)
        current_reward += float(rewards[0])

        if dones[0]:
            episode_count += 1
            episode_rewards.append(current_reward)

            # 성공 여부 출력
            # VecNormalize를 거쳤으므로 info에서 height 확인
            print(
                f"Episode {episode_count:4d} | "
                f"reward={current_reward:8.2f} | "
                f"avg_last10={sum(episode_rewards[-10:])/min(10, len(episode_rewards)):.2f}"
            )
            current_reward = 0.0

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()