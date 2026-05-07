"""학습된 Place SAC 정책 재생 — VecNormalize 옵션 포함"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play trained SAC example7 Place+Insert")
parser.add_argument("--checkpoint", type=str, default="checkpoints/example7.zip")
parser.add_argument(
    "--vecnorm", type=str, default="checkpoints/example7_vecnorm.pkl",
    help="VecNormalize pkl 경로 (--vecnorm_on일 때만 사용)",
)
parser.add_argument("--vecnorm_on", action="store_true", help="VecNormalize 활성화")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, VecEnvWrapper

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper

from source.example7.tasks.place.place_env import PlaceEnv
from source.example7.tasks.place.place_env_cfg import PlaceEnvCfg


# train.py의 GoalEnvVecWrapper와 동일 (중복 정의 — 모듈 의존 최소화)
class GoalEnvVecWrapper(VecEnvWrapper):
    def __init__(self, venv, core_dim: int, goal_dim: int, compute_reward_fn):
        super().__init__(venv)
        self._core_dim = core_dim
        self._goal_dim = goal_dim
        self._compute_reward_fn = compute_reward_fn

        low = np.full(core_dim, -np.inf, dtype=np.float32)
        high = np.full(core_dim, np.inf, dtype=np.float32)
        goal_low = np.full(goal_dim, -np.inf, dtype=np.float32)
        goal_high = np.full(goal_dim, np.inf, dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=low, high=high, dtype=np.float32),
            "achieved_goal": gym.spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
            "desired_goal": gym.spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
        })

    def _split(self, obs: np.ndarray) -> dict:
        core = obs[..., : self._core_dim]
        achieved = obs[..., self._core_dim : self._core_dim + self._goal_dim]
        desired = obs[..., self._core_dim + self._goal_dim :]
        return {
            "observation": core.astype(np.float32),
            "achieved_goal": achieved.astype(np.float32),
            "desired_goal": desired.astype(np.float32),
        }

    def reset(self):
        obs = self.venv.reset()
        return self._split(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._split(obs), rewards, dones, infos

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._compute_reward_fn(achieved_goal, desired_goal, info)


def main():
    cfg = PlaceEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    raw_env = PlaceEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)
    env = GoalEnvVecWrapper(
        env,
        core_dim=PlaceEnv.OBS_CORE_DIM,
        goal_dim=PlaceEnv.GOAL_DIM,
        compute_reward_fn=raw_env.compute_reward,
    )

    if args_cli.vecnorm_on:
        if os.path.exists(args_cli.vecnorm):
            env = VecNormalize.load(args_cli.vecnorm, env)
            env.training = False
            env.norm_reward = False
            print(f"✅ VecNormalize 로드: {args_cli.vecnorm}")
        else:
            print(f"⚠️ VecNormalize 파일 없음 ({args_cli.vecnorm}). 정규화 없이 실행.")

    model = SAC.load(args_cli.checkpoint, env=env, device="cuda")
    print(f"✅ 모델 로드: {args_cli.checkpoint}")

    obs = env.reset()
    episode_count = 0
    success_count = 0
    tilted_count = 0
    timeout_count = 0
    other_count = 0
    current_reward = 0.0
    step_in_ep = 0

    import torch as _t
    from isaaclab.utils.math import quat_apply
    while simulation_app.is_running():
        # step 전 상태 캡쳐 (auto-reset 직전 값 보존)
        env_z = float(raw_env.scene.env_origins[0, 2].item())
        obj_pos_w = raw_env._object.data.root_pos_w[0]
        target_w = raw_env.target_cell_pos_w[0]
        xy_dist = float(_t.norm(obj_pos_w[:2] - target_w[:2]).item())
        obj_z_rel = float(obj_pos_w[2].item()) - env_z

        grip_to_obj = float(_t.norm(raw_env.grip_center_pos[0] - obj_pos_w).item())
        gp = raw_env._robot.data.joint_pos[0, raw_env.main_gripper_joint_id]
        gl = raw_env.robot_dof_lower_limits[raw_env.main_gripper_joint_id]
        gu = raw_env.robot_dof_upper_limits[raw_env.main_gripper_joint_id]
        grip_close = float(((gp - gl) / (gu - gl + 1e-8)).item())

        oq = raw_env._object.data.root_quat_w[0:1]
        lup = _t.zeros((1, 3), device=raw_env.device); lup[:, 2] = 1.0
        upright = float(quat_apply(oq, lup)[0, 2].item())

        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)
        current_reward += float(rewards[0])
        step_in_ep += 1

        if dones[0]:
            episode_count += 1
            xy_aligned = xy_dist < 0.025
            high_enough = obj_z_rel >= 0.28
            grip_holding = (grip_to_obj < 0.08) and (grip_close > 0.5)
            success = xy_aligned and high_enough and grip_holding
            on_floor = (obj_z_rel - 0.059) < 0.005
            tilted = upright < 0.85

            if success:
                tag, mark = "SUCCESS", "✅"
                success_count += 1
            elif tilted and on_floor:
                tag, mark = "TILT_FLOOR", "🔄"
                tilted_count += 1
            elif step_in_ep >= 590:
                tag, mark = "TIMEOUT  ", "⏰"
                timeout_count += 1
            else:
                tag, mark = "?         ", "❓"
                other_count += 1

            sr = success_count / episode_count
            print(
                f"Ep {episode_count:3d} {mark} {tag} | "
                f"xy={xy_dist:.3f} z={obj_z_rel:+.3f} grip={grip_to_obj:.3f}/c={grip_close:.2f} | "
                f"up={upright:+.2f} | steps={step_in_ep:3d} | "
                f"R={current_reward:6.0f} | "
                f"SR={sr:.1%} ({success_count}/{episode_count}) "
                f"[T={tilted_count} O={timeout_count} ?={other_count}]"
            )
            current_reward = 0.0
            step_in_ep = 0

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
