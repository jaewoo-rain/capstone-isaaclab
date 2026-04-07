"""
env 자체 동작 확인
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test OMY grasp env")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from source.omy.omy_env_cfg import OmyLiftEnvCfg
from source.omy.omy_grasp_env import OmyGraspEnv


def main():
    cfg = OmyLiftEnvCfg()
    cfg.scene.num_envs = 4

    env = OmyGraspEnv(cfg, render_mode=None)

    obs, _ = env.reset()
    print("env reset ok")
    print("obs shape:", obs["policy"].shape)

    for _ in range(20):
        actions = torch.zeros((env.num_envs, cfg.action_space), device=env.device)
        obs, rew, terminated, truncated, info = env.step(actions)
        print("reward mean:", rew.mean().item())

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()