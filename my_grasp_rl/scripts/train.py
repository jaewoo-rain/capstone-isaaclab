"""SB3 PPO 공용 학습 스크립트 (grasp / stacking / good 공용).

기능:
1. task 선택 가능
2. num_envs, seed, timesteps 조절 가능
3. --resume 으로 기존 체크포인트 이어서 학습 가능
4. --no 옵션으로 학습 후 모델/VecNormalize 저장 안 하도록 설정 가능
5. --name 옵션으로 checkpoints에 저장할 이름을 직접 지정 가능
6. 2000 step마다 현재 진행률과 남은 steps 출력
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------
# CLI 인자
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train PPO for IsaacLab task")

# 학습할 task 종류 선택
parser.add_argument(
    "--task",
    type=str,
    default="stacking",
    choices=["grasp", "stacking", "good"],
    help="학습할 task 이름 선택",
)

# 병렬 환경 개수
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="병렬로 돌릴 환경 개수 (설정 안 하면 cfg 기본값 사용)",
)

# 랜덤 시드
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="랜덤 시드 값",
)

# 총 학습 step 수
parser.add_argument(
    "--timesteps",
    type=int,
    default=1_000_000,
    help="총 학습 timesteps",
)

# 기존 모델 이어서 학습할지 여부
parser.add_argument(
    "--resume",
    action="store_true",
    help="기존 체크포인트를 불러와 이어서 학습",
)

# 학습 후 저장 안 할지 여부
# action='store_true' 이므로:
# --no 를 안 쓰면 False
# --no 를 쓰면 True
parser.add_argument(
    "--no",
    action="store_true",
    help="학습 종료 후 모델과 VecNormalize를 저장하지 않음",
)

# 체크포인트 이름 커스텀
parser.add_argument(
    "--name",
    type=str,
    default=None,
    help="체크포인트 이름 커스텀 (예: exp1, testA 등)",
)

# Isaac Sim 관련 추가 인자
AppLauncher.add_app_launcher_args(parser)

# parse_known_args():
# Isaac 관련 hydra 인자와 우리가 직접 정의한 인자를 분리해서 받음
args_cli, hydra_args = parser.parse_known_args()

# hydra 인자를 sys.argv 로 다시 세팅
sys.argv = [sys.argv[0]] + hydra_args

# ---------------------------------------------------------------------
# Isaac Sim 실행
# ---------------------------------------------------------------------
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------
# 필요한 라이브러리 import
# ---------------------------------------------------------------------
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from my_grasp_rl.tasks.sb3_ppo_cfg import SB3_PPO_CFG

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except Exception as e:
    raise RuntimeError(
        "isaaclab_rl의 SB3 wrapper를 불러오지 못했습니다. "
        "Isaac Lab 버전에 맞는 wrapper 경로를 확인하세요."
    ) from e


class ProgressCallback(BaseCallback):
    """학습 진행률을 일정 step마다 출력하는 callback."""

    def __init__(self, total_timesteps: int, print_interval: int = 2000, verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps_target = total_timesteps
        self.print_interval = print_interval

        self.last_print_step = 0
        self.start_time = None

        # 이번 학습 세션이 시작될 때의 누적 step 저장
        self.start_step = 0

    def _on_training_start(self) -> None:
        self.start_time = time.time()

        # resume 여부와 상관없이, 학습 시작 시점의 누적 step을 저장
        self.start_step = self.num_timesteps
        self.last_print_step = 0

        print("📈 학습 진행 상황 출력 시작")
        print(f"   total timesteps : {self.total_timesteps_target}")
        print(f"   print interval  : {self.print_interval}")
        print(f"   start step      : {self.start_step}")

    def _on_step(self) -> bool:
        # 전체 누적 step
        global_step = self.num_timesteps

        # 이번 세션에서 진행한 step
        current_step = global_step - self.start_step

        if current_step - self.last_print_step >= self.print_interval:
            self.last_print_step = current_step

            remaining_steps = max(self.total_timesteps_target - current_step, 0)
            progress_percent = (current_step / self.total_timesteps_target) * 100.0
            progress_percent = min(progress_percent, 100.0)

            elapsed = time.time() - self.start_time if self.start_time is not None else 0.0

            print(
                f"📊 Step: {current_step}/{self.total_timesteps_target} "
                f"({progress_percent:.2f}%) | 남은 steps: {remaining_steps} "
                f"| 누적 global step: {global_step} "
                f"| 경과 시간: {elapsed:.1f}s"
            )

        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time if self.start_time is not None else 0.0
        print("✅ 학습 종료")
        print(f"   총 경과 시간: {elapsed:.1f}s")


def build_task(task_name: str):
    """task 이름에 따라 cfg, env class, 저장 prefix를 반환한다."""

    if task_name == "grasp":
        from my_grasp_rl.tasks.grasp.grasp_franka_env_cfg import GraspFrankaEnvCfg
        from my_grasp_rl.tasks.grasp.grasp_franka_env import GraspFrankaEnv

        return {
            "cfg": GraspFrankaEnvCfg(),
            "env_class": GraspFrankaEnv,
            "prefix": "grasp_franka",
        }

    elif task_name == "stacking":
        from my_grasp_rl.tasks.stacking.stacking_franka_env_cfg import StackingFrankaEnvCfg
        from my_grasp_rl.tasks.stacking.stacking_franka_env import StackingFrankaEnv

        return {
            "cfg": StackingFrankaEnvCfg(),
            "env_class": StackingFrankaEnv,
            "prefix": "stacking_franka",
        }

    elif task_name == "good":
        from my_grasp_rl.tasks.good_robot.good_robot_franka_env_cfg import GoodRobotFrankaEnvCfg
        from my_grasp_rl.tasks.good_robot.good_robot_franka_env import GoodRobotFrankaEnv

        return {
            "cfg": GoodRobotFrankaEnvCfg(),
            "env_class": GoodRobotFrankaEnv,
            "prefix": "good_franka",
        }

    else:
        raise ValueError(f"Unknown task: {task_name}")


def main():
    # -------------------------------------------------------------
    # task에 맞는 cfg / env class / prefix 가져오기
    # -------------------------------------------------------------
    task_info = build_task(args_cli.task)

    cfg = task_info["cfg"]
    env_class = task_info["env_class"]
    prefix = task_info["prefix"]

    # num_envs를 CLI로 따로 주면 cfg 값 덮어쓰기
    if args_cli.num_envs is not None:
        cfg.scene.num_envs = args_cli.num_envs

    # -------------------------------------------------------------
    # 저장 이름 결정
    # -------------------------------------------------------------
    if args_cli.name is not None:
        save_name = args_cli.name
    else:
        save_name = prefix

    # 체크포인트 폴더 생성
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 저장 경로
    checkpoint_path = os.path.join(checkpoint_dir, save_name)
    vecnorm_path = os.path.join(checkpoint_dir, f"{save_name}_vecnormalize.pkl")

    # -------------------------------------------------------------
    # env 생성
    # -------------------------------------------------------------
    env = env_class(cfg=cfg, render_mode=None)
    env = Sb3VecEnvWrapper(env)

    # -------------------------------------------------------------
    # 진행 상황 callback 생성
    # -------------------------------------------------------------
    progress_callback = ProgressCallback(
        total_timesteps=args_cli.timesteps,
        print_interval=20000,
    )

    # -------------------------------------------------------------
    # resume 여부에 따라 이어서 학습 / 새 학습 분기
    # -------------------------------------------------------------
    has_checkpoint = os.path.exists(checkpoint_path + ".zip")
    has_vecnorm = os.path.exists(vecnorm_path)

    if args_cli.resume and has_checkpoint and has_vecnorm:
        print(f"🔄 기존 {args_cli.task} 모델과 VecNormalize를 불러와 이어서 학습합니다.")
        print(f"   save name    : {save_name}")
        print(f"   model path   : {checkpoint_path}.zip")
        print(f"   vecnorm path : {vecnorm_path}")

        # 기존 VecNormalize 통계 불러오기
        env = VecNormalize.load(vecnorm_path, env)

        # 학습 모드 활성화
        env.training = True
        env.norm_reward = True

        # 기존 모델 불러오기
        model = PPO.load(checkpoint_path, env=env)

        # 이어서 학습
        model.learn(
            total_timesteps=args_cli.timesteps,
            reset_num_timesteps=False,
            callback=progress_callback,
        )

    else:
        if args_cli.resume and (not has_checkpoint or not has_vecnorm):
            print("⚠️ --resume 옵션이 들어왔지만 기존 체크포인트 또는 VecNormalize 파일이 없습니다.")
            print("   새 모델로 학습을 시작합니다.")
        else:
            print(f"🆕 새 {args_cli.task} 모델로 학습을 시작합니다.")

        print(f"   save name    : {save_name}")
        print(f"   model path   : {checkpoint_path}.zip")
        print(f"   vecnorm path : {vecnorm_path}")

        # 새 학습에서는 VecNormalize 새로 생성
        env = VecNormalize(
            env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )

        # PPO 모델 생성
        model = PPO(
            env=env,
            seed=args_cli.seed,
            tensorboard_log=f"./logs/sb3/{args_cli.task}",
            **SB3_PPO_CFG,
        )

        # 처음부터 학습
        model.learn(
            total_timesteps=args_cli.timesteps,
            reset_num_timesteps=True,
            callback=progress_callback,
        )

    # -------------------------------------------------------------
    # 저장 여부 분기
    # -------------------------------------------------------------
    if args_cli.no:
        print("💾 --no 옵션이 설정되어 저장을 건너뜁니다.")
        print("   model.save(...) / env.save(...) 를 수행하지 않습니다.")
    else:
        print("💾 학습 결과를 저장합니다.")
        print(f"   save name    : {save_name}")
        print(f"   model path   : {checkpoint_path}.zip")
        print(f"   vecnorm path : {vecnorm_path}")

        model.save(checkpoint_path)
        env.save(vecnorm_path)

    # env 종료
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()