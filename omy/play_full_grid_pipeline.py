from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play OMY full 3x3 grid pipeline")
parser.add_argument("--grasp_checkpoint", type=str, default="checkpoints/omy_grasp_ppo.zip")
parser.add_argument("--lift_checkpoint", type=str, default="checkpoints/omy_lift_ppo.zip")
parser.add_argument("--place_checkpoint", type=str, default="checkpoints/omy_place_grid_ppo.zip")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from stable_baselines3 import PPO
from isaaclab_rl.sb3 import Sb3VecEnvWrapper
import torch

from source.omy.omy_env_cfg import OmyLiftEnvCfg
from source.omy.omy_place_grid_env import OmyPlaceGridEnv


def main():
    cfg = OmyLiftEnvCfg()
    cfg.scene.num_envs = 1

    base_env = OmyPlaceGridEnv(cfg, render_mode=None)
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

        current_slot = int(base_env.current_slot[0].item())
        target_pos = base_env._get_current_target_pos()[0]

        grasp_success = bool(terms["is_grasping"][0].item() > 0.5)
        lifted_enough = bool(obj_height > cfg.lift_height_threshold)

        target_dist = torch.norm(target_pos - obj_pos).item()
        placed_near = bool((target_dist < 0.035) and (obj_height > 0.02) and (not grasp_success))

        if mode == "grasp" and grasp_success:
            mode = "lift"
            print(f"[slot {current_slot}] switch to lift")

        elif mode == "lift" and lifted_enough:
            mode = "place"
            print(f"[slot {current_slot}] switch to place")

        elif mode == "place" and placed_near:
            next_slot = int(base_env.current_slot[0].item())
            last_placed = bool(base_env.last_placed[0].item())

            if next_slot >= 8 and last_placed:
                print("[grid] all slots finished -> reset")
                obs = env.reset()
                mode = "grasp"
                print("mode:", mode)
            else:
                print(f"[slot {current_slot}] placed -> next slot {next_slot} -> mode grasp")
                mode = "grasp"

        if done.any():
            obs = env.reset()
            mode = "grasp"
            print("reset -> mode:", mode)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()