from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play OMY full pipeline")
parser.add_argument("--grasp_checkpoint", type=str, default="checkpoints/omy_grasp_ppo.zip")
parser.add_argument("--lift_checkpoint", type=str, default="checkpoints/omy_lift_ppo.zip")
parser.add_argument("--place_checkpoint", type=str, default="checkpoints/omy_place_ppo.zip")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper

from source.omy.omy_env_cfg import OmyLiftEnvCfg
from source.omy.omy_grasp_env import OmyGraspEnv


def main():
    cfg = OmyLiftEnvCfg()
    cfg.scene.num_envs = 1

    base_env = OmyGraspEnv(cfg, render_mode=None)
    env = Sb3VecEnvWrapper(base_env)

    grasp_model = PPO.load(args_cli.grasp_checkpoint, env=env, device="cuda")
    lift_model = PPO.load(args_cli.lift_checkpoint, env=env, device="cuda")
    place_model = PPO.load(args_cli.place_checkpoint, env=env, device="cuda")

    obs = env.reset()
    mode = "grasp"
    print("mode:", mode)

    while simulation_app.is_running():
        if mode == "grasp":
            action, _ = grasp_model.predict(obs, deterministic=True)
        elif mode == "lift":
            action, _ = lift_model.predict(obs, deterministic=True)
        else:
            action, _ = place_model.predict(obs, deterministic=True)

        obs, rew, done, info = env.step(action)

        terms = base_env._get_common_terms()
        obj_height = base_env._get_obj_height()[0].item()
        obj_pos = base_env._get_obj_pos()[0]

        grasp_success = terms["is_grasping"][0].item() > 0.5
        lifted_enough = obj_height > cfg.lift_height_threshold

        env_origin = base_env.scene.env_origins[0]
        target_x = env_origin[0].item() + 0.60
        target_y = env_origin[1].item() + 0.20
        target_z = env_origin[2].item() + 0.08

        placed_near = (
            abs(obj_pos[0].item() - target_x) < 0.04
            and abs(obj_pos[1].item() - target_y) < 0.04
            and abs(obj_pos[2].item() - target_z) < 0.05
        )

        if mode == "grasp" and grasp_success:
            mode = "lift"
            print("switch to lift")

        elif mode == "lift" and lifted_enough:
            mode = "place"
            print("switch to place")

        elif mode == "place" and placed_near:
            obs = env.reset()
            mode = "grasp"
            print("placed -> reset -> mode: grasp")

        if done.any():
            obs = env.reset()
            mode = "grasp"
            print("reset -> mode:", mode)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()