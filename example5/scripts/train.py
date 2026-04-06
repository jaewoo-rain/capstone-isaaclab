
"""SB3 PPO 학습 스크립트 — Franka Lift"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train PPO for Franka Lift Task")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timesteps", type=int, default=5_000_000)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--no_save", action="store_true")
parser.add_argument("--name", type=str, default="example5")

# 추가: 특정 체크포인트 불러오기
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="불러올 PPO checkpoint(.zip) 경로. 예: checkpoints/example5_step500000.zip",
)
parser.add_argument(
    "--vecnorm",
    type=str,
    default=None,
    help="불러올 VecNormalize pkl 경로. 미지정 시 checkpoint 이름 기준으로 자동 추론",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper

from source.example5.tasks.lift.lift_env_cfg import LiftEnvCfg
from source.example5.tasks.lift.lift_env import LiftEnv


# ------------------------------------------------------------------
# Callback
# ------------------------------------------------------------------
class TrainCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        checkpoint_path: str,
        vecnorm_path: str,
        env_ref: LiftEnv,
        save_interval: int = 500_000,
        print_interval: int = 100_000,
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
        self.start_time = None
        self._reward_log_buf = {}

    def _on_training_start(self):
        self.start_time = time.time()
        print("=" * 60)
        print(f"🚀 학습 시작: 총 {self.total_timesteps_target:,} steps")
        print("=" * 60)

    def _on_step(self) -> bool:
        for k, v in self.env_ref.reward_log.items():
            self._reward_log_buf.setdefault(k, []).append(v)

        step = self.num_timesteps

        if step - self.last_print >= self.print_interval:
            self.last_print = step
            elapsed = time.time() - self.start_time
            sps = step / elapsed if elapsed > 0 else 0
            remain_s = (self.total_timesteps_target - step) / sps if sps > 0 else 0
            pct = step / self.total_timesteps_target * 100

            mean_reward = 0.0
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = (
                    sum(ep["r"] for ep in self.model.ep_info_buffer)
                    / len(self.model.ep_info_buffer)
                )

            print(
                f"[{pct:5.1f}%] step={step:>9,} | "
                f"reward={mean_reward:7.2f} | "
                f"SPS={sps:,.0f} | "
                f"남은시간={remain_s/60:.1f}min"
            )

            reward_keys = [
                "xy_align_reward",
                "z_align_reward",
                "approach_reward",
                "close_reward",
                "grasp_bonus",
                "lift_reward",
                "success_rate",
            ]
            parts = []
            for key in reward_keys:
                vals = self._reward_log_buf.get(key, [])
                parts.append(
                    f"{key}={sum(vals)/len(vals):.4f}" if vals else f"{key}=n/a"
                )
            print("  └─ " + " | ".join(parts))

            self._reward_log_buf.clear()

        if step - self.last_save >= self.save_interval:
            self.last_save = step
            ckpt = f"{self.checkpoint_path}_step{step}"
            self.model.save(ckpt)
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(f"{ckpt}_vecnorm.pkl")
            print(f"  💾 중간 저장: {ckpt}")

        return True


# ------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------
def infer_vecnorm_path_from_checkpoint(checkpoint_path: str) -> str:
    """
    checkpoints/a.zip -> checkpoints/a_vecnorm.pkl
    checkpoints/a     -> checkpoints/a_vecnorm.pkl
    """
    base, ext = os.path.splitext(checkpoint_path)
    if ext == ".zip":
        return f"{base}_vecnorm.pkl"
    return f"{checkpoint_path}_vecnorm.pkl"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    cfg = LiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    default_checkpoint_path = os.path.join(checkpoint_dir, args_cli.name)
    default_vecnorm_path = os.path.join(checkpoint_dir, f"{args_cli.name}_vecnorm.pkl")

    # ------------------------------------------------------------------
    # Env 생성
    # ------------------------------------------------------------------
    raw_env = LiftEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)

    # ------------------------------------------------------------------
    # PPO 하이퍼파라미터
    # ------------------------------------------------------------------
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
    )

    # ------------------------------------------------------------------
    # 로드 경로 결정
    # ------------------------------------------------------------------
    load_checkpoint_path = None
    load_vecnorm_path = None

    if args_cli.checkpoint is not None:
        load_checkpoint_path = args_cli.checkpoint
        load_vecnorm_path = (
            args_cli.vecnorm
            if args_cli.vecnorm is not None
            else infer_vecnorm_path_from_checkpoint(args_cli.checkpoint)
        )
    elif args_cli.resume:
        load_checkpoint_path = default_checkpoint_path + ".zip"
        load_vecnorm_path = default_vecnorm_path

    # 최종 저장 기준 이름
    checkpoint_path = default_checkpoint_path
    vecnorm_path = default_vecnorm_path

    # ------------------------------------------------------------------
    # 체크포인트 로드 or 신규 학습
    # ------------------------------------------------------------------
    if load_checkpoint_path is not None:
        has_ckpt = os.path.exists(load_checkpoint_path)
        has_vecnorm = os.path.exists(load_vecnorm_path)

        if has_ckpt and has_vecnorm:
            print(f"🔄 체크포인트 로드: {load_checkpoint_path}")
            print(f"🔄 VecNormalize 로드: {load_vecnorm_path}")

            env = VecNormalize.load(load_vecnorm_path, env)
            env.training = True
            env.norm_reward = True

            model = PPO.load(
                load_checkpoint_path,
                env=env,
                device="cuda",
                **ppo_kwargs,
            )

            # 저장 이름도 로드한 checkpoint 기준으로 맞추고 싶으면 base 사용
            load_base, ext = os.path.splitext(load_checkpoint_path)
            if ext == ".zip":
                checkpoint_path = load_base
            else:
                checkpoint_path = load_checkpoint_path
            vecnorm_path = infer_vecnorm_path_from_checkpoint(checkpoint_path)

        else:
            print("⚠️ 지정한 checkpoint 또는 vecnorm 파일이 없습니다. 새로 시작합니다.")
            print(f"   checkpoint: {load_checkpoint_path}")
            print(f"   vecnorm   : {load_vecnorm_path}")

            env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            model = PPO(
                "MlpPolicy",
                env,
                tensorboard_log="./logs/sb3/lift",
                device="cuda",
                **ppo_kwargs,
            )
    else:
        print("🆕 새 학습 시작")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        model = PPO(
            "MlpPolicy",
            env,
            tensorboard_log="./logs/sb3/lift",
            device="cuda",
            **ppo_kwargs,
        )

    # ------------------------------------------------------------------
    # 학습
    # ------------------------------------------------------------------
    callback = TrainCallback(
        total_timesteps=args_cli.timesteps,
        checkpoint_path=checkpoint_path,
        vecnorm_path=vecnorm_path,
        env_ref=raw_env,
        save_interval=500_000,
        print_interval=100_000,
    )

    model.learn(
        total_timesteps=args_cli.timesteps,
        callback=callback,
        reset_num_timesteps=not (args_cli.resume or args_cli.checkpoint is not None),
    )

    # ------------------------------------------------------------------
    # 최종 저장
    # ------------------------------------------------------------------
    if not args_cli.no_save:
        model.save(checkpoint_path)
        env.save(vecnorm_path)
        print(f"✅ 최종 저장 완료: {checkpoint_path}")
        print(f"✅ VecNormalize 저장 완료: {vecnorm_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()