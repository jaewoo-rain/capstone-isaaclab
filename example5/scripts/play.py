"""학습된 Lift PPO 정책 재생 — VecNormalize 포함"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained PPO example5 Lift")
parser.add_argument("--checkpoint", type=str, default="checkpoints/example5.zip")
parser.add_argument("--vecnorm",    type=str, default="checkpoints/example5_vecnorm.pkl",
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

from source.example5.tasks.lift.lift_env import LiftEnv
from source.example5.tasks.lift.lift_env_cfg import LiftEnvCfg


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
    # 재생 루프 (종료 원인 출력)
    # ------------------------------------------------------------------
    from isaaclab.utils.math import quat_apply
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    while not hasattr(raw_env, "_object"):
        raw_env = getattr(raw_env, "venv", None) or getattr(raw_env, "env", None)
        if raw_env is None:
            break

    obs = env.reset()

    episode_count = 0
    success_count = 0
    fallen_count = 0
    tilted_count = 0
    timeout_count = 0
    episode_rewards = []
    current_reward = 0.0
    step_in_ep = 0
    max_z_in_ep = 0.0

    LIFT_THRESHOLD = float(cfg.lift_height_threshold)
    MAX_STEPS = int(cfg.episode_length_s / (cfg.sim.dt * cfg.decimation))

    while simulation_app.is_running():
        # ★ step 전 (=현재 frame) 상태 미리 캡쳐 — 자동 reset 직전 값 보존
        env_z = float(raw_env.scene.env_origins[0, 2].item())
        obj_z_rel = float(raw_env._object.data.root_pos_w[0, 2].item()) - env_z
        max_z_in_ep = max(max_z_in_ep, obj_z_rel)

        obj_quat = raw_env._object.data.root_quat_w[0:1]
        local_up = torch.zeros((1, 3), device=raw_env.device); local_up[:, 2] = 1.0
        upright = float(quat_apply(obj_quat, local_up)[0, 2].item())

        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)
        current_reward += float(rewards[0])
        step_in_ep += 1

        if dones[0]:
            episode_count += 1
            episode_rewards.append(current_reward)

            # 종료 원인 추정 — max_z 가 threshold 의 0.99 이상이면 success
            if max_z_in_ep >= LIFT_THRESHOLD * 0.99:
                tag, mark = "SUCCESS", "✅"
                success_count += 1
            elif obj_z_rel < -0.05:
                tag, mark = "FALLEN ", "⬇️"
                fallen_count += 1
            elif upright < 0.4:
                tag, mark = "TILTED ", "🔄"
                tilted_count += 1
            elif step_in_ep >= MAX_STEPS - 5:
                tag, mark = "TIMEOUT", "⏰"
                timeout_count += 1
            else:
                tag, mark = "?      ", "❓"

            sr = success_count / episode_count
            print(
                f"Ep {episode_count:3d} {mark} {tag} | "
                f"max_z={max_z_in_ep:.3f} (target={LIFT_THRESHOLD:.2f}) | "
                f"steps={step_in_ep:3d} | "
                f"upright={upright:+.2f} | "
                f"reward={current_reward:7.1f} | "
                f"SR={sr:.2%} ({success_count}/{episode_count}) "
                f"[F={fallen_count} T={tilted_count} O={timeout_count}]"
            )
            current_reward = 0.0
            step_in_ep = 0
            max_z_in_ep = 0.0

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()