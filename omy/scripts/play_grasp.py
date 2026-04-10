from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained OMY grasp PPO")

parser.add_argument("--checkpoint", type=str, default="checkpoints/omy_grasp_ppo.zip")

# ✅ 추가
parser.add_argument(
    "--disable_camera",
    action="store_true",
    help="카메라 센서를 생성하지 않음",
)

AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from source.omy.tasks.common.omy_env_cfg import OmyLiftEnvCfg
from source.omy.tasks.grasp.omy_grasp_env import OmyGraspEnv


def main():
    cfg = OmyLiftEnvCfg()
    cfg.scene.num_envs = 1

    # ✅ 여기 핵심
    cfg.use_camera = not args_cli.disable_camera

    base_env = OmyGraspEnv(cfg, render_mode=None)
    env = Sb3VecEnvWrapper(base_env)

    model = PPO.load(args_cli.checkpoint, env=env, device="cuda")

    obs = env.reset()
    print("play start")

    while simulation_app.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done.any():
            obs = env.reset()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()