"""SB3 PPO 공용 재생 스크립트 (grasp / stacking / good 공용).

기능:
1. task 선택 가능
2. --checkpoint 로 직접 체크포인트 경로 지정 가능
3. --name 으로 train 때 저장한 이름을 간단히 불러올 수 있음
4. --num_envs 로 재생 환경 개수 조절 가능
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------
# CLI 인자
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play trained PPO policy for IsaacLab task")

# 재생할 task 선택
parser.add_argument(
    "--task",
    type=str,
    default="stacking",
    choices=["grasp", "stacking", "good"],
    help="재생할 task 이름 선택",
)

# 체크포인트 직접 경로 지정
# 예:
#   --checkpoint checkpoints/stacking_franka_exp1_ppo
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="직접 체크포인트 경로 지정 (예: checkpoints/stacking_franka_exp1_ppo)",
)

# train에서 저장할 때 썼던 name 재사용
# 예:
#   train.py --task stacking --name exp1
#   play.py  --task stacking --name exp1
# 그러면 자동으로 checkpoints/stacking_franka_exp1_ppo 를 찾음
parser.add_argument(
    "--name",
    type=str,
    default=None,
    help="train 때 저장한 이름 사용 (예: exp1, testA 등)",
)

# 재생 env 개수
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="재생할 환경 개수 (보통 play는 1 추천)",
)

# Isaac Sim 관련 추가 인자
AppLauncher.add_app_launcher_args(parser)

# hydra 인자 분리
args_cli, hydra_args = parser.parse_known_args()
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

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except Exception as e:
    raise RuntimeError(
        "isaaclab_rl의 SB3 wrapper를 불러오지 못했습니다. "
        "Isaac Lab 버전에 맞는 wrapper 경로를 확인하세요."
    ) from e


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

    # play는 보통 1개 env로 보는 게 편하지만,
    # 사용자가 따로 주면 그 값 사용
    if args_cli.num_envs is not None:
        cfg.scene.num_envs = args_cli.num_envs

    checkpoint_dir = "checkpoints"

    # -------------------------------------------------------------
    # 체크포인트 이름 결정
    # 우선순위:
    # 1) --checkpoint 직접 지정
    # 2) --name 사용
    # 3) 기본 prefix 사용
    # -------------------------------------------------------------
    if args_cli.checkpoint is not None:
        # 사용자가 경로를 직접 줬으면 그걸 그대로 사용
        checkpoint_path = args_cli.checkpoint

        # vecnorm 경로는 checkpoint 이름 규칙에 맞춰 자동 추정
        # 예:
        #   checkpoints/stacking_franka_exp1_ppo
        # -> checkpoints/stacking_franka_exp1_vecnormalize.pkl
        if checkpoint_path.endswith("_ppo"):
            vecnorm_path = checkpoint_path[:-4] + "_vecnormalize.pkl"
        else:
            # 혹시 _ppo 형식이 아니면 기본 규칙대로 붙여서 추정
            vecnorm_path = checkpoint_path + "_vecnormalize.pkl"

    elif args_cli.name is not None:
        # train 때 저장한 name을 기반으로 자동 생성
        # 예:
        # prefix = stacking_franka, name = exp1
        # -> checkpoints/stacking_franka_exp1_ppo
        save_name = args_cli.name
        checkpoint_path = os.path.join(checkpoint_dir, save_name)
        vecnorm_path = os.path.join(checkpoint_dir, f"{save_name}_vecnormalize.pkl")

    else:
        # 기본 prefix 사용
        checkpoint_path = os.path.join(checkpoint_dir, prefix)
        vecnorm_path = os.path.join(checkpoint_dir, f"{prefix}_vecnormalize.pkl")

    # stable-baselines3는 .zip 없이 넣어도 load 가능하지만,
    # 파일 존재 여부 확인은 .zip 기준으로 체크
    if not os.path.exists(checkpoint_path + ".zip"):
        raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {checkpoint_path}.zip")

    if not os.path.exists(vecnorm_path):
        raise FileNotFoundError(f"VecNormalize 파일을 찾을 수 없습니다: {vecnorm_path}")

    print(f"▶ task           : {args_cli.task}")
    print(f"▶ checkpoint     : {checkpoint_path}.zip")
    print(f"▶ vecnormalize   : {vecnorm_path}")
    print(f"▶ num_envs       : {cfg.scene.num_envs}")

    # -------------------------------------------------------------
    # env 생성
    # -------------------------------------------------------------
    env = env_class(cfg=cfg, render_mode=None)
    env = Sb3VecEnvWrapper(env)

    # 학습 때 저장한 VecNormalize 불러오기
    env = VecNormalize.load(vecnorm_path, env)

    # 평가 모드로 전환
    env.training = False
    env.norm_reward = False

    # 모델 로드
    model = PPO.load(checkpoint_path, env=env)

    # 초기 observation
    obs = env.reset()

    # -------------------------------------------------------------
    # 재생 루프
    # -------------------------------------------------------------
    while simulation_app.is_running():
        # deterministic=True 이면 평가 시 항상 같은 정책 행동 선택
        action, _states = model.predict(obs, deterministic=True)

        # 환경 한 step 진행
        obs, rewards, dones, infos = env.step(action)

        # VecEnv에서는 dones가 배열 형태
        if dones is not None and dones.any():
            obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()