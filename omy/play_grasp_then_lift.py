from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play OMY grasp then lift")
parser.add_argument("--grasp_checkpoint", type=str, default="checkpoints/omy_grasp_ppo.zip")
parser.add_argument("--lift_checkpoint", type=str, default="checkpoints/omy_lift_ppo.zip")
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

    # 공통 env
    base_env = OmyGraspEnv(cfg, render_mode=None)
    env = Sb3VecEnvWrapper(base_env)

    grasp_model = PPO.load(args_cli.grasp_checkpoint, env=env, device="cuda")
    lift_model = PPO.load(args_cli.lift_checkpoint, env=env, device="cuda")

    obs = env.reset()

    mode = "grasp"
    print("mode:", mode)

    while simulation_app.is_running():
        if mode == "grasp":
            action, _ = grasp_model.predict(obs, deterministic=True)
        else:
            action, _ = lift_model.predict(obs, deterministic=True)

        obs, rew, done, info = env.step(action)

        # 내부 상태 확인은 wrapper 안쪽 실제 env(base_env)에서 읽기
        terms = base_env._get_common_terms()
        obj_height = base_env._get_obj_height()

        grasp_success = terms["is_grasping"][0].item() > 0.5
        lifted_enough = obj_height[0].item() > cfg.lift_height_threshold

        # grasp 성공 -> lift 정책 전환
        if mode == "grasp" and grasp_success:
            mode = "lift"
            print("switch to lift")

        # 충분히 들었거나 에피소드 끝나면 reset
        if lifted_enough or done.any():
            obs = env.reset()
            mode = "grasp"
            print("reset -> mode:", mode)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()