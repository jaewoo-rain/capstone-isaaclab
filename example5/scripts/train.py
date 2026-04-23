"""SB3 PPO 학습 스크립트 — Franka Lift

수정 내용:
1. --resume 시에도 진행률/남은시간이 "이번 실행 기준"으로 계산되도록 수정
2. session_step(이번 실행에서 추가된 step)와 total_step(누적 step)을 분리
3. 중간 저장 주기는 session_step 기준으로 동작
4. 저장 파일명은 total_step 기준으로 유지해서 체크포인트 누적 상태를 쉽게 파악 가능
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

# ------------------------------------------------------------------
# CLI 인자
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train PPO for Franka Lift Task")

parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--seed", type=int, default=42)

# 이번 실행에서 추가로 학습할 step 수
parser.add_argument("--timesteps", type=int, default=5_000_000)

# 기존 학습 이어서 할지
parser.add_argument("--resume", action="store_true")

# 저장 안 할지
parser.add_argument("--no_save", action="store_true")

# 저장 이름
parser.add_argument("--name", type=str, default="example5")

# 특정 체크포인트 직접 불러오기
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="불러올 PPO checkpoint(.zip) 경로. 예: checkpoints/example5_step500000.zip",
)

# 특정 VecNormalize 직접 불러오기
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

# ------------------------------------------------------------------
# import
# ------------------------------------------------------------------
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
    """
    학습 중간 출력 / 중간 저장 담당 callback

    중요:
    - total_step: SB3 내부 누적 step
    - session_step: 이번 실행에서 추가로 진행한 step

    --resume 시에도 남은시간/퍼센트가 정상적으로 보이도록
    모든 진행률 계산은 session_step 기준으로 한다.
    """

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

        # 이번 실행에서 목표로 하는 step 수
        self.total_timesteps_target = total_timesteps

        # 최종 저장 기준 경로
        self.checkpoint_path = checkpoint_path
        self.vecnorm_path = vecnorm_path

        # reward_log 읽기용 env 참조
        self.env_ref = env_ref

        # 몇 step마다 저장/출력할지
        self.save_interval = save_interval
        self.print_interval = print_interval

        # "이번 실행 기준" 마지막 출력/저장 시점
        self.last_print = 0
        self.last_save = 0

        # 시간 측정
        self.start_time = None

        # resume 대응용:
        # 학습 시작 시점의 누적 step
        self.start_step = 0

        # 최근 reward component 평균 출력용 버퍼
        self._reward_log_buf = {}

    def _on_training_start(self):
        """
        학습 시작 시 호출
        resume 여부와 상관없이 현재 누적 step을 start_step으로 저장한다.
        """
        self.start_time = time.time()

        # 현재 모델의 누적 step을 저장
        # resume 시 0이 아닐 수 있음
        self.start_step = self.num_timesteps

        # session 기준 카운터 초기화
        self.last_print = 0
        self.last_save = 0

        print("=" * 70)
        print(f"🚀 학습 시작")
        print(f"   이번 실행 목표 step : {self.total_timesteps_target:,}")
        print(f"   시작 누적 step      : {self.start_step:,}")
        print("=" * 70)

    def _on_step(self) -> bool:
        """
        rollout 진행 중 매 step 호출

        여기서 total_step(누적)와 session_step(이번 실행)을 분리해서 계산한다.
        """
        # env에서 모아둔 reward 로그들을 버퍼에 누적
        for k, v in self.env_ref.reward_log.items():
            self._reward_log_buf.setdefault(k, []).append(v)

        # SB3 누적 step
        total_step = self.num_timesteps

        # 이번 실행에서 실제로 추가된 step
        session_step = total_step - self.start_step

        # --------------------------------------------------------------
        # 중간 출력
        # --------------------------------------------------------------
        if session_step - self.last_print >= self.print_interval:
            self.last_print = session_step

            elapsed = time.time() - self.start_time

            # SPS도 이번 실행 기준으로 계산해야 정확함
            sps = session_step / elapsed if elapsed > 0 else 0.0

            # 이번 실행 목표까지 남은 시간
            remain_steps = max(self.total_timesteps_target - session_step, 0)
            remain_s = remain_steps / sps if sps > 0 else 0.0

            # 진행률도 이번 실행 기준
            pct = (
                session_step / self.total_timesteps_target * 100.0
                if self.total_timesteps_target > 0
                else 0.0
            )

            # episode reward 평균
            mean_reward = 0.0
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = (
                    sum(ep["r"] for ep in self.model.ep_info_buffer)
                    / len(self.model.ep_info_buffer)
                )

            print(
                f"[{pct:5.1f}%] "
                f"session_step={session_step:>10,} | "
                f"total_step={total_step:>10,} | "
                f"reward={mean_reward:8.2f} | "
                f"SPS={sps:,.0f} | "
                f"남은시간={remain_s / 60:.1f}min"
            )

            parts = []
            for key, vals in self._reward_log_buf.items():
                if len(vals) > 0:
                    parts.append(f"{key}={sum(vals) / len(vals):.4f}")
                else:
                    parts.append(f"{key}=n/a")

            print("  └─ " + " | ".join(parts))

            # 출력 후 버퍼 초기화
            self._reward_log_buf.clear()

        # --------------------------------------------------------------
        # 중간 저장
        # 저장 주기는 이번 실행(session_step) 기준으로 판단
        # 저장 파일명은 누적 total_step 기준으로 생성
        # --------------------------------------------------------------
        if session_step - self.last_save >= self.save_interval:
            self.last_save = session_step

            # 파일명은 누적 step 기준으로 저장
            ckpt = f"{self.checkpoint_path}_step{total_step}"

            self.model.save(ckpt)

            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(f"{ckpt}_vecnorm.pkl")

            print(f"  💾 중간 저장 완료: {ckpt}")

        return True


# ------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------
def infer_vecnorm_path_from_checkpoint(checkpoint_path: str) -> str:
    """
    checkpoint 경로로부터 vecnorm pkl 경로를 추론

    예:
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
    # --------------------------------------------------------------
    # 환경 설정
    # --------------------------------------------------------------
    cfg = LiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    # seed 고정
    torch.manual_seed(args_cli.seed)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 기본 저장 경로
    default_checkpoint_path = os.path.join(checkpoint_dir, args_cli.name)
    default_vecnorm_path = os.path.join(checkpoint_dir, f"{args_cli.name}_vecnorm.pkl")

    # --------------------------------------------------------------
    # Env 생성
    # --------------------------------------------------------------
    raw_env = LiftEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)

    # --------------------------------------------------------------
    # PPO 하이퍼파라미터
    # --------------------------------------------------------------
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
    )

    # --------------------------------------------------------------
    # 로드 경로 결정
    # 우선순위:
    # 1) --checkpoint 직접 지정
    # 2) --resume 이면 기본 경로 로드
    # --------------------------------------------------------------
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

    # 최종 저장 기준 경로
    checkpoint_path = default_checkpoint_path
    vecnorm_path = default_vecnorm_path

    # --------------------------------------------------------------
    # 체크포인트 로드 or 신규 학습
    # --------------------------------------------------------------
    if load_checkpoint_path is not None:
        has_ckpt = os.path.exists(load_checkpoint_path)
        has_vecnorm = os.path.exists(load_vecnorm_path)

        if has_ckpt and has_vecnorm:
            print(f"🔄 체크포인트 로드: {load_checkpoint_path}")
            print(f"🔄 VecNormalize 로드: {load_vecnorm_path}")

            # VecNormalize 상태 복원
            env = VecNormalize.load(load_vecnorm_path, env)
            env.training = True
            env.norm_reward = True

            # PPO 모델 복원
            model = PPO.load(
                load_checkpoint_path,
                env=env,
                device="cuda",
                **ppo_kwargs,
            )

            # 저장 기준 이름을 현재 불러온 checkpoint 기반으로 맞추고 싶으면 갱신
            load_base, ext = os.path.splitext(load_checkpoint_path)
            if ext == ".zip":
                checkpoint_path = load_base
            else:
                checkpoint_path = load_checkpoint_path

            vecnorm_path = infer_vecnorm_path_from_checkpoint(checkpoint_path)

            print("✅ 이어서 학습합니다.")

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

            print("🆕 새 학습 시작")
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

    # --------------------------------------------------------------
    # Callback 생성
    # total_timesteps=args_cli.timesteps 는
    # "이번 실행에서 추가로 돌릴 목표량" 의미로 사용
    # --------------------------------------------------------------
    callback = TrainCallback(
        total_timesteps=args_cli.timesteps,
        checkpoint_path=checkpoint_path,
        vecnorm_path=vecnorm_path,
        env_ref=raw_env,
        save_interval=500_000,
        print_interval=100_000,
    )

    # --------------------------------------------------------------
    # 학습 시작
    #
    # reset_num_timesteps:
    # - 새 학습이면 True
    # - resume/checkpoint면 False
    #
    # 이렇게 해야 SB3 내부 누적 step이 이어진다.
    # 대신 callback 내부에서 session_step을 따로 계산한다.
    # --------------------------------------------------------------
    model.learn(
        total_timesteps=args_cli.timesteps,
        callback=callback,
        reset_num_timesteps=not (args_cli.resume or args_cli.checkpoint is not None),
    )

    # --------------------------------------------------------------
    # 최종 저장
    # --------------------------------------------------------------
    if not args_cli.no_save:
        model.save(checkpoint_path)
        env.save(vecnorm_path)

        print(f"✅ 최종 저장 완료: {checkpoint_path}")
        print(f"✅ VecNormalize 저장 완료: {vecnorm_path}")

    # --------------------------------------------------------------
    # 종료
    # --------------------------------------------------------------
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()