import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained PPO for Example1 tasks")
parser.add_argument("--task", type=str, choices=["reach", "grasp", "lift"], default="reach")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from stable_baselines3 import PPO

from isaaclab_rl.sb3 import Sb3VecEnvWrapper


def build_env_and_cfg(task_name: str):
    if task_name == "reach":
        import source.example1.tasks.reach
        from source.example1.tasks.reach.reach_env_cfg import ReachEnvCfg
        return "Example1-Reach-Franka-v0", ReachEnvCfg()
    elif task_name == "grasp":
        import source.example1.tasks.grasp
        from source.example1.tasks.grasp.grasp_env_cfg import GraspEnvCfg
        return "Example1-Grasp-Franka-v0", GraspEnvCfg()
    else:
        import source.example1.tasks.lift
        from source.example1.tasks.lift.lift_env_cfg import LiftEnvCfg
        return "Example1-Lift-Franka-v0", LiftEnvCfg()


def main():
    env_id, cfg = build_env_and_cfg(args_cli.task)
    cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(env_id, cfg=cfg)
    env = Sb3VecEnvWrapper(env)

    default_checkpoint = f"checkpoints/example1_{args_cli.task}_ppo.zip"
    checkpoint_path = args_cli.checkpoint if args_cli.checkpoint is not None else default_checkpoint

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    model = PPO.load(checkpoint_path, env=env)

    obs = env.reset()

    while simulation_app.is_running():
        action, _ = model.predict(obs, deterministic=True)
        obs, _, _, _ = env.step(action)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()