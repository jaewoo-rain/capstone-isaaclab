from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained OMY place PPO")
parser.add_argument("--checkpoint", type=str, default="checkpoints/omy_place_ppo.zip")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from source.omy.omy_env_cfg import OmyLiftEnvCfg
from source.omy.omy_place_env import OmyPlaceEnv


def main():
    cfg = OmyLiftEnvCfg()
    cfg.scene.num_envs = 1

    base_env = OmyPlaceEnv(cfg, render_mode=None)
    env = Sb3VecEnvWrapper(base_env)

    model = PPO.load(args_cli.checkpoint, env=env, device="cuda")

    obs = env.reset()
    print("place play start")

    while simulation_app.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(action)

        if done.any():
            obs = env.reset()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()