from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='Play OMY vision policy')
parser.add_argument('--checkpoint', type=str, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
from source.omy.vision.tasks.common.omy_vision_env_cfg import OmyVisionEnvCfg
from source.omy.vision.tasks.grasp.omy_grasp_vision_env import OmyGraspVisionEnv as EnvCls

def main():
    cfg = OmyVisionEnvCfg()
    cfg.scene.num_envs = 1
    env = Sb3VecEnvWrapper(EnvCls(cfg, render_mode=None))
    model = PPO.load(args_cli.checkpoint, env=env, device='cuda')
    obs = env.reset()
    while simulation_app.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)


if __name__ == '__main__':
    main()
