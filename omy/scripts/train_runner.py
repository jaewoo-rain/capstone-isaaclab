from __future__ import annotations

import argparse
import importlib
import os
import sys

from isaaclab.app import AppLauncher
from stable_baselines3.common.callbacks import BaseCallback


from stable_baselines3.common.callbacks import BaseCallback


class StepLoggerCallback(BaseCallback):
    def __init__(self, base_env, log_freq: int = 100000):
        super().__init__()
        self.base_env = base_env
        self.log_freq = log_freq
        self._last_logged_step = -1

    def _on_step(self) -> bool:
        # 정확히 100000 배수가 안 찍힐 수도 있으니
        # "구간이 넘어갔는지" 기준으로 한 번만 출력
        current_bucket = self.num_timesteps // self.log_freq
        last_bucket = self._last_logged_step // self.log_freq if self._last_logged_step >= 0 else -1

        if current_bucket > last_bucket:
            self._last_logged_step = self.num_timesteps

            print("=" * 60)
            print(f"[STEP {self.num_timesteps}]")

            # 1) 네 env reward_log 출력
            if hasattr(self.base_env, "reward_log"):
                print("[ENV reward_log]")
                for k, v in self.base_env.reward_log.items():
                    if isinstance(v, float):
                        print(f"{k}: {v:.4f}")
                    else:
                        print(f"{k}: {v}")

            # 2) SB3 logger 출력
            print("[SB3 logger]")
            for k, v in self.logger.name_to_value.items():
                if isinstance(v, float):
                    print(f"{k}: {v:.4f}")
                else:
                    print(f"{k}: {v}")

            print("=" * 60)

        return True


def choose_batch_size(n_steps: int, num_envs: int, preferred: int) -> int:
    rollout_size = n_steps * num_envs

    if preferred <= 0:
        return rollout_size

    if rollout_size % preferred == 0:
        return preferred

    for candidate in range(min(preferred, rollout_size), 0, -1):
        if rollout_size % candidate == 0:
            return candidate

    return rollout_size


def build_train_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--resume", action="store_true", help="기본 체크포인트에서 이어 학습")
    parser.add_argument("--checkpoint", type=str, default=None, help="불러올 체크포인트 경로(.zip)")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="이번 실행에서 추가 학습할 timesteps")
    parser.add_argument("--num_envs", type=int, default=16, help="병렬 env 개수")
    parser.add_argument("--disable_camera", action="store_true", help="카메라 센서를 생성하지 않음")
    parser.add_argument("--save_path", type=str, default=None, help="저장 경로 prefix (.zip 제외 권장)")
    parser.add_argument("--log_dir", type=str, default=None, help="tensorboard 로그 경로")
    parser.add_argument(
        "--ppo_device",
        type=str,
        default=None,
        help='SB3 PPO device. 예: "cpu", "cuda". 미지정 시 IsaacLab device 사용',
    )

    AppLauncher.add_app_launcher_args(parser)
    return parser


def import_class(import_path: str):
    """
    예: "source.omy.omy_grasp_env.OmyGraspEnv"
    """
    module_path, class_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def run_ppo_train(
    *,
    env_cls_path: str,
    cfg_cls_path: str,
    description: str,
    default_save_path: str,
    default_resume_path: str,
    default_log_dir: str,
) -> None:
    parser = build_train_parser(description)
    args_cli, hydra_args = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # AppLauncher 이후에 import
    from stable_baselines3 import PPO
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper

    # Isaac 관련 env/cfg도 여기서 import
    env_cls = import_class(env_cls_path)
    cfg_cls = import_class(cfg_cls_path)

    cfg = cfg_cls()
    cfg.scene.num_envs = args_cli.num_envs

    if hasattr(cfg, "use_camera"):
        cfg.use_camera = not args_cli.disable_camera

    save_path = args_cli.save_path or default_save_path
    log_dir = args_cli.log_dir or default_log_dir

    n_steps = cfg.n_steps
    batch_size = choose_batch_size(cfg.n_steps, cfg.scene.num_envs, cfg.batch_size)
    ppo_device = args_cli.ppo_device if args_cli.ppo_device is not None else args_cli.device

    print("=" * 80)
    print("[Train Config]")
    print(f"env_cls_path  : {env_cls_path}")
    print(f"cfg_cls_path  : {cfg_cls_path}")
    print(f"num_envs      : {cfg.scene.num_envs}")
    print(f"use_camera    : {getattr(cfg, 'use_camera', 'N/A')}")
    print(f"timesteps     : {args_cli.timesteps}")
    print(f"n_steps       : {n_steps}")
    print(f"batch_size    : {batch_size}")
    print(f"isaac_device  : {args_cli.device}")
    print(f"ppo_device    : {ppo_device}")
    print(f"resume        : {args_cli.resume}")
    print(f"checkpoint    : {args_cli.checkpoint}")
    print(f"save_path     : {save_path}")
    print(f"log_dir       : {log_dir}")
    print("=" * 80)

    base_env = env_cls(cfg, render_mode=None)
    env = Sb3VecEnvWrapper(base_env)

    load_path = None
    if args_cli.checkpoint is not None:
        load_path = args_cli.checkpoint
    elif args_cli.resume:
        load_path = default_resume_path

    if load_path is not None:
        if not os.path.exists(load_path):
            env.close()
            simulation_app.close()
            raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {load_path}")

        print(f"[INFO] 기존 모델 불러오는 중: {load_path}")
        model = PPO.load(load_path, env=env, device=ppo_device)
        reset_num_timesteps = False
    else:
        print("[INFO] 새 모델로 학습 시작")
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            n_epochs=cfg.n_epochs,
            verbose=1,
            tensorboard_log=log_dir,
            device=ppo_device,
        )
        reset_num_timesteps = True

    print("[INFO] 학습 시작")
    callback = StepLoggerCallback(base_env=base_env, log_freq=100000)

    model.learn(
        total_timesteps=args_cli.timesteps,
        reset_num_timesteps=reset_num_timesteps,
        callback=callback,
    )

    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    model.save(save_path)
    print(f"[INFO] 모델 저장 완료: {save_path}.zip")

    env.close()
    simulation_app.close()