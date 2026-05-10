"""motion1 — Grasp RL 학습 스크립트 (PPO + VecNormalize + early stop).

example5/scripts/train.py 패턴 기반.
- session_step / total_step 분리 (resume 시 진행률 정확)
- 100k step 마다 best 갱신 (mean episode reward)
- early stop: success rate >= 0.9 + 100 episodes 연속 유지
- TensorBoard log: source/motion1/logs/grasp/

실행:
    ./isaaclab.sh -p source/motion1/scripts/train_grasp.py --headless --num_envs 128 --timesteps 1000000
    # resume:
    ./isaaclab.sh -p source/motion1/scripts/train_grasp.py --headless --resume --timesteps 500000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

from isaaclab.app import AppLauncher

# -------------------- argparse --------------------
parser = argparse.ArgumentParser(description="motion1 Grasp RL — PPO 학습")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timesteps", type=int, default=1_000_000)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--no_save", action="store_true")
parser.add_argument("--name", type=str, default="motion1_grasp")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="불러올 PPO checkpoint(.zip) 경로")
parser.add_argument("--vecnorm", type=str, default=None,
                    help="불러올 VecNormalize pkl 경로 (미지정 시 자동 추론)")
parser.add_argument("--early_stop_success_rate", type=float, default=0.9,
                    help="이 success rate 이상 + N episodes 연속 시 학습 종료")
parser.add_argument("--early_stop_episodes", type=int, default=100,
                    help="early stop 조건의 연속 episodes 수")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------- imports (after app start) --------------------
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper

from source.motion1.tasks.grasp.grasp_env import GraspEnv
from source.motion1.tasks.grasp.grasp_env_cfg import GraspEnvCfg


# ============================================================
# Callback
# ============================================================
class TrainCallback(BaseCallback):
    """학습 진행 출력 + 중간 저장 + best 갱신 + early stop."""

    def __init__(
        self,
        total_timesteps: int,
        checkpoint_path: str,
        vecnorm_path: str,
        env_ref: GraspEnv,
        save_interval: int = 100_000,
        print_interval: int = 20_000,
        early_stop_success_rate: float = 0.9,
        early_stop_episodes: int = 100,
    ):
        super().__init__(verbose=0)
        self.total_timesteps_target = total_timesteps
        self.checkpoint_path = checkpoint_path
        self.vecnorm_path = vecnorm_path
        self.env_ref = env_ref

        self.save_interval = save_interval
        self.print_interval = print_interval
        self.last_print = 0
        self.last_save = 0

        self.start_time: float | None = None
        self.start_step = 0

        self._reward_log_buf: dict[str, list] = {}
        self._best_mean_reward = -float("inf")

        # early stop — recent N episode 의 aligned (success) 여부 추적
        self._aligned_buf: deque = deque(maxlen=early_stop_episodes)
        self._early_stop_thresh = early_stop_success_rate
        self._early_stop_n = early_stop_episodes

    def _on_training_start(self):
        self.start_time = time.time()
        self.start_step = self.num_timesteps
        self.last_print = 0
        self.last_save = 0

        print("=" * 70)
        print("🚀 Grasp RL 학습 시작")
        print(f"   목표 step      : {self.total_timesteps_target:,}")
        print(f"   시작 누적 step : {self.start_step:,}")
        print(f"   early stop     : success_rate >= {self._early_stop_thresh} "
              f"+ 최근 {self._early_stop_n} episode 평균")
        print("=" * 70)

    def _on_step(self) -> bool:
        # reward log 누적
        for k, v in self.env_ref.reward_log.items():
            self._reward_log_buf.setdefault(k, []).append(v)

        total_step = self.num_timesteps
        session_step = total_step - self.start_step

        # ---- 출력 ----
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

            recent_aligned_rate = (
                sum(self._aligned_buf) / len(self._aligned_buf)
                if len(self._aligned_buf) > 0 else 0.0
            )

            print(
                f"[{pct:5.1f}%] session={session_step:>9,} | total={total_step:>9,} | "
                f"reward={mean_reward:7.2f} | success_recent={recent_aligned_rate:.2f} "
                f"({len(self._aligned_buf)}/{self._early_stop_n}) | "
                f"SPS={sps:,.0f} | 남은={remain_s/60:.1f}min"
            )
            parts = []
            for key, vals in self._reward_log_buf.items():
                if len(vals) > 0:
                    parts.append(f"{key}={sum(vals)/len(vals):.4f}")
            print("  └─ " + " | ".join(parts))
            self._reward_log_buf.clear()

            # best 갱신
            if mean_reward > self._best_mean_reward and len(self.model.ep_info_buffer) > 30:
                self._best_mean_reward = mean_reward
                if not args_cli.no_save:
                    best_ckpt = f"{self.checkpoint_path}_best"
                    self.model.save(best_ckpt)
                    if isinstance(self.training_env, VecNormalize):
                        self.training_env.save(f"{best_ckpt}_vecnorm.pkl")
                    print(f"  🏆 best 갱신: mean_reward={mean_reward:.2f} → {best_ckpt}.zip")

        # ---- 중간 저장 ----
        if not args_cli.no_save and session_step - self.last_save >= self.save_interval:
            self.last_save = session_step
            ckpt = f"{self.checkpoint_path}_step{total_step}"
            self.model.save(ckpt)
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(f"{ckpt}_vecnorm.pkl")
            # 또 default 경로도 같이 (resume 용)
            self.model.save(self.checkpoint_path)
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(self.vecnorm_path)
            print(f"  💾 저장: {ckpt}")

        # ---- early stop 체크 ----
        # env.reward_log 의 aligned_rate 사용 (per-step 평균이라 그대로 buf 에)
        # episode 끝나는 시점에만 push 하는 게 더 정확하지만 simplicity 로 매 step
        aligned = float(self.env_ref.reward_log.get("aligned_rate", 0.0))
        self._aligned_buf.append(aligned)
        if len(self._aligned_buf) >= self._early_stop_n:
            recent = sum(self._aligned_buf) / len(self._aligned_buf)
            if recent >= self._early_stop_thresh:
                print(f"\n✅ early stop: 최근 {self._early_stop_n} step 의 aligned_rate "
                      f"= {recent:.3f} >= {self._early_stop_thresh}")
                return False  # 학습 중단

        return True


# ============================================================
# Helpers
# ============================================================
def infer_vecnorm_path_from_checkpoint(ckpt_path: str) -> str:
    base, ext = os.path.splitext(ckpt_path)
    if ext == ".zip":
        return f"{base}_vecnorm.pkl"
    return f"{ckpt_path}_vecnorm.pkl"


# ============================================================
# Main
# ============================================================
def main():
    cfg = GraspEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    torch.manual_seed(args_cli.seed)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    default_checkpoint_path = os.path.join(checkpoint_dir, args_cli.name)
    default_vecnorm_path = os.path.join(checkpoint_dir, f"{args_cli.name}_vecnorm.pkl")

    log_dir = "source/motion1/logs/grasp"
    os.makedirs(log_dir, exist_ok=True)

    # env
    raw_env = GraspEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)

    # PPO kwargs
    ppo_kwargs = dict(
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=1.0,
        verbose=1,
        seed=args_cli.seed,
        tensorboard_log=log_dir,
    )

    # 로드 경로 결정
    load_checkpoint_path = None
    load_vecnorm_path = None
    if args_cli.checkpoint is not None:
        load_checkpoint_path = args_cli.checkpoint
        load_vecnorm_path = (args_cli.vecnorm if args_cli.vecnorm is not None
                             else infer_vecnorm_path_from_checkpoint(args_cli.checkpoint))
    elif args_cli.resume:
        load_checkpoint_path = default_checkpoint_path + ".zip"
        load_vecnorm_path = default_vecnorm_path

    checkpoint_path = default_checkpoint_path
    vecnorm_path = default_vecnorm_path

    if load_checkpoint_path is not None and os.path.exists(load_checkpoint_path) and os.path.exists(load_vecnorm_path):
        print(f"🔄 체크포인트 로드: {load_checkpoint_path}")
        print(f"🔄 VecNormalize 로드: {load_vecnorm_path}")
        vec_env = VecNormalize.load(load_vecnorm_path, env)
        vec_env.training = True
        model = PPO.load(load_checkpoint_path, env=vec_env, device="auto", **ppo_kwargs)
    else:
        if load_checkpoint_path is not None:
            print(f"⚠️  체크포인트 / vecnorm 못 찾음. 신규 학습 시작.")
        vec_env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True,
                               clip_obs=10.0, clip_reward=10.0, gamma=cfg.gamma)
        model = PPO("MlpPolicy", vec_env, device="auto", **ppo_kwargs)

    callback = TrainCallback(
        total_timesteps=args_cli.timesteps,
        checkpoint_path=checkpoint_path,
        vecnorm_path=vecnorm_path,
        env_ref=raw_env,
        save_interval=100_000,
        print_interval=20_000,
        early_stop_success_rate=args_cli.early_stop_success_rate,
        early_stop_episodes=args_cli.early_stop_episodes,
    )

    model.learn(total_timesteps=args_cli.timesteps, callback=callback, reset_num_timesteps=False)

    # 최종 저장
    if not args_cli.no_save:
        model.save(checkpoint_path)
        vec_env.save(vecnorm_path)
        print(f"\n💾 최종 저장: {checkpoint_path}.zip + {vecnorm_path}")

    print("\n✅ 학습 완료.")

    # 자원 정리 — env.close() 안 하면 hang 가능
    vec_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    # PhysX 자원 해제 hang 방지 — 강제 종료
    os._exit(0)
