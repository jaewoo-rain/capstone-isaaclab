"""motion1 — Insert RL v2 학습 스크립트 (SAC + HER).

example6/scripts/train.py 의 SAC+HER 패턴 + InsertEnvV2 사용.

실행:
    ./isaaclab.sh -p source/motion1/scripts/train_insert_v2.py --headless --num_envs 64 --timesteps 2000000
    ./isaaclab.sh -p source/motion1/scripts/train_insert_v2.py --headless --resume --timesteps 1000000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

from isaaclab.app import AppLauncher

# -------------------- argparse --------------------
parser = argparse.ArgumentParser(description="motion1 Insert RL v2 — SAC+HER 학습")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timesteps", type=int, default=2_000_000)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--no_save", action="store_true")
parser.add_argument("--name", type=str, default="motion1_insert_v2")
parser.add_argument("--checkpoint", type=str, default=None)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------- imports (after app start) --------------------
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.her import HerReplayBuffer

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper

from source.motion1.tasks.insert_v2.insert_env_v2 import InsertEnvV2
from source.motion1.tasks.insert_v2.insert_env_v2_cfg import InsertEnvV2Cfg


# ============================================================
# GoalEnvVecWrapper — flat obs → Dict(observation, achieved_goal, desired_goal)
# ============================================================
class GoalEnvVecWrapper(VecEnvWrapper):
    """InsertEnvV2 가 flat 13dim obs 를 반환 → HER 용 Dict 로 split.

    구성: core(5) + achieved_goal(4) + desired_goal(4) = 13
    """

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

    def _split_single(self, flat_obs: np.ndarray) -> dict:
        core = flat_obs[: self._core_dim]
        achieved = flat_obs[self._core_dim : self._core_dim + self._goal_dim]
        desired = flat_obs[self._core_dim + self._goal_dim :]
        return {
            "observation": core.astype(np.float32),
            "achieved_goal": achieved.astype(np.float32),
            "desired_goal": desired.astype(np.float32),
        }

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        for i in range(len(infos)):
            t_obs = infos[i].get("terminal_observation", None)
            if t_obs is not None and not isinstance(t_obs, dict):
                infos[i]["terminal_observation"] = self._split_single(np.asarray(t_obs))
        return self._split(obs), rewards, dones, infos

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._compute_reward_fn(achieved_goal, desired_goal, info)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        if method_name == "compute_reward":
            result = self._compute_reward_fn(*method_args, **method_kwargs)
            return [result]
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)


# ============================================================
class TrainCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        checkpoint_path: str,
        env_ref: InsertEnvV2,
        save_interval: int = 200_000,
        print_interval: int = 20_000,
    ):
        super().__init__(verbose=0)
        self.total_timesteps_target = total_timesteps
        self.checkpoint_path = checkpoint_path
        self.env_ref = env_ref
        self.save_interval = save_interval
        self.print_interval = print_interval
        self.last_print = 0
        self.last_save = 0
        self.start_time: float | None = None
        self.start_step = 0
        self._reward_log_buf: dict[str, list] = {}
        self._best_mean_reward = -float("inf")

    def _on_training_start(self):
        self.start_time = time.time()
        self.start_step = self.num_timesteps
        self.last_print = 0
        self.last_save = 0
        print("=" * 70)
        print("🚀 Insert RL v2 학습 시작 (SAC + HER)")
        print(f"   목표 step      : {self.total_timesteps_target:,}")
        print(f"   시작 누적 step : {self.start_step:,}")
        print("=" * 70)

    def _on_step(self) -> bool:
        for k, v in self.env_ref.reward_log.items():
            self._reward_log_buf.setdefault(k, []).append(v)

        total_step = self.num_timesteps
        session_step = total_step - self.start_step

        if session_step - self.last_print >= self.print_interval:
            self.last_print = session_step
            assert self.start_time is not None
            elapsed = time.time() - self.start_time
            sps = session_step / elapsed if elapsed > 0 else 0.0
            remain_steps = max(self.total_timesteps_target - session_step, 0)
            remain_s = remain_steps / sps if sps > 0 else 0.0
            pct = (session_step / self.total_timesteps_target * 100.0
                   if self.total_timesteps_target > 0 else 0.0)
            mean_reward = 0.0
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = sum(ep["r"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
            print(
                f"[{pct:5.1f}%] session={session_step:>9,} | total={total_step:>9,} | "
                f"reward={mean_reward:7.2f} | SPS={sps:,.0f} | 남은={remain_s/60:.1f}min"
            )
            parts = []
            for key, vals in self._reward_log_buf.items():
                if len(vals) > 0:
                    parts.append(f"{key}={sum(vals)/len(vals):.4f}")
            print("  └─ " + " | ".join(parts))
            self._reward_log_buf.clear()

            if mean_reward > self._best_mean_reward and len(self.model.ep_info_buffer) > 30:
                self._best_mean_reward = mean_reward
                if not args_cli.no_save:
                    best_ckpt = f"{self.checkpoint_path}_best"
                    self.model.save(best_ckpt)
                    print(f"  🏆 best 갱신: mean_reward={mean_reward:.2f}")

        if not args_cli.no_save and session_step - self.last_save >= self.save_interval:
            self.last_save = session_step
            ckpt = f"{self.checkpoint_path}_step{total_step}"
            self.model.save(ckpt)
            self.model.save(self.checkpoint_path)
            print(f"  💾 저장: {ckpt}")

        return True


# ============================================================
def main():
    cfg = InsertEnvV2Cfg()
    cfg.scene.num_envs = args_cli.num_envs
    torch.manual_seed(args_cli.seed)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    default_checkpoint_path = os.path.join(checkpoint_dir, args_cli.name)

    log_dir = "source/motion1/logs/insert_v2"
    os.makedirs(log_dir, exist_ok=True)

    # ----- env -----
    raw_env = InsertEnvV2(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)

    env = GoalEnvVecWrapper(
        env,
        core_dim=InsertEnvV2.OBS_CORE_DIM,
        goal_dim=InsertEnvV2.GOAL_DIM,
        compute_reward_fn=raw_env.compute_reward,
    )

    # ----- SAC hyperparams -----
    sac_kwargs = dict(
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        learning_starts=cfg.learning_starts,
        ent_coef=cfg.ent_coef,
        verbose=1,
        seed=args_cli.seed,
        tensorboard_log=log_dir,
    )

    replay_buffer_kwargs = dict(
        n_sampled_goal=cfg.n_sampled_goal,
        goal_selection_strategy=cfg.goal_selection_strategy,
    )

    # ----- load or new -----
    load_checkpoint_path = None
    if args_cli.checkpoint is not None:
        load_checkpoint_path = args_cli.checkpoint
    elif args_cli.resume:
        load_checkpoint_path = default_checkpoint_path + ".zip"

    checkpoint_path = default_checkpoint_path

    if load_checkpoint_path is not None and os.path.exists(load_checkpoint_path):
        print(f"🔄 체크포인트 로드: {load_checkpoint_path}")
        model = SAC.load(
            load_checkpoint_path,
            env=env,
            device="cuda",
            **sac_kwargs,
        )
        # replay buffer 로드 시도
        load_base_path, _ = os.path.splitext(load_checkpoint_path)
        rb_path = f"{load_base_path}_replay.pkl"
        if os.path.exists(rb_path):
            model.load_replay_buffer(rb_path)
            print(f"🔄 Replay buffer 로드: {rb_path} ({model.replay_buffer.size()} transitions)")
        else:
            original_ls = cfg.learning_starts
            model.learning_starts = model.num_timesteps + original_ls
            print(f"⚠️ Replay buffer 없음. learning_starts → {model.learning_starts}")

        load_base, ext = os.path.splitext(load_checkpoint_path)
        checkpoint_path = load_base if ext == ".zip" else load_checkpoint_path
        print("✅ 이어서 학습")
    else:
        if load_checkpoint_path is not None:
            print(f"⚠️ checkpoint 못 찾음: {load_checkpoint_path} → 신규 학습")
        print("🆕 신규 학습")
        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            device="cuda",
            **sac_kwargs,
        )

    callback = TrainCallback(
        total_timesteps=args_cli.timesteps,
        checkpoint_path=checkpoint_path,
        env_ref=raw_env,
        save_interval=200_000,
        print_interval=20_000,
    )

    model.learn(
        total_timesteps=args_cli.timesteps,
        callback=callback,
        reset_num_timesteps=not (args_cli.resume or args_cli.checkpoint is not None),
    )

    if not args_cli.no_save:
        model.save(checkpoint_path)
        try:
            model.save_replay_buffer(f"{checkpoint_path}_replay.pkl")
            print(f"✅ Replay buffer 저장: {checkpoint_path}_replay.pkl")
        except Exception as e:
            print(f"⚠️ Replay buffer 저장 실패: {e}")
        print(f"\n💾 최종 저장: {checkpoint_path}.zip")

    print("\n✅ 학습 완료.")
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    os._exit(0)
