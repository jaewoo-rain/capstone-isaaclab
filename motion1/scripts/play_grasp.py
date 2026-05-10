"""motion1 — 학습된 Grasp 정책 시각화.

실행:
    ./isaaclab.sh -p source/motion1/scripts/play_grasp.py \
        --checkpoint checkpoints/motion1_grasp.zip \
        --vecnorm checkpoints/motion1_grasp_vecnorm.pkl
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="motion1 Grasp RL — play")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default="checkpoints/motion1_grasp.zip")
parser.add_argument("--vecnorm", type=str, default="checkpoints/motion1_grasp_vecnorm.pkl")
parser.add_argument("--episodes", type=int, default=10, help="시각화할 에피소드 수")
parser.add_argument("--deterministic", action="store_true", default=True)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper

from source.motion1.tasks.grasp.grasp_env import GraspEnv
from source.motion1.tasks.grasp.grasp_env_cfg import GraspEnvCfg


def main():
    cfg = GraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    raw_env = GraspEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)

    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(f"checkpoint 없음: {args_cli.checkpoint}")
    if not os.path.exists(args_cli.vecnorm):
        raise FileNotFoundError(f"vecnorm 없음: {args_cli.vecnorm}")

    print(f"🔄 checkpoint: {args_cli.checkpoint}")
    print(f"🔄 vecnorm   : {args_cli.vecnorm}")

    vec_env = VecNormalize.load(args_cli.vecnorm, env)
    vec_env.training = False    # 통계 업데이트 안 함
    vec_env.norm_reward = False  # play 시 reward 정규화 X
    model = PPO.load(args_cli.checkpoint, env=vec_env, device="auto")

    obs = vec_env.reset()
    ep_count = 0
    ep_reward = 0.0
    ep_aligned = 0.0
    ep_steps = 0
    success_count = 0
    print("\n========== Play 시작 ==========")
    while simulation_app.is_running() and ep_count < args_cli.episodes:
        action, _ = model.predict(obs, deterministic=args_cli.deterministic)
        obs, reward, dones, infos = vec_env.step(action)
        ep_reward += float(np.asarray(reward).mean())
        ep_aligned = max(ep_aligned, float(raw_env.reward_log.get("aligned_rate", 0.0)))
        ep_steps += 1

        if bool(np.asarray(dones).any()):
            ep_count += 1
            log = raw_env.reward_log
            success = log.get("aligned_rate", 0.0) > 0.0
            if success:
                success_count += 1
            print(
                f"  ep {ep_count:>3} | steps={ep_steps:>3} | reward={ep_reward:>7.2f} | "
                f"aligned={ep_aligned:.2f} | "
                f"r_xy={log.get('r_xy_align', 0):.3f} r_yaw={log.get('r_yaw_align', 0):.3f}"
            )
            ep_reward = 0.0
            ep_aligned = 0.0
            ep_steps = 0

    print(f"\n========== Play 끝: {success_count}/{ep_count} 성공 ==========")

    # 자원 정리 — env.close() 안 하면 hang 가능
    vec_env.close()


if __name__ == "__main__":
    import os
    try:
        main()
    finally:
        simulation_app.close()
    os._exit(0)
