"""example5 PPO 정책으로 expert demo trajectory 수집.

저장 형식: pickle 파일 하나에 list[dict] — 각 dict가 하나의 에피소드.
    {
        "obs":       (T, obs_dim)  float32   # VecNormalize 적용된 정규화 obs
        "next_obs":  (T, obs_dim)  float32   # 정규화 obs (다음 스텝)
        "actions":   (T, act_dim)  float32
        "rewards":   (T,)          float32   # 정규화 X (raw)
        "dones":     (T,)          bool
        "success":   bool
        "length":    int
    }

obs는 VecNormalize 적용된 값으로 저장 → BC, SAC 모두 같은 정규화 통계로 학습.
VecNormalize pkl은 별도 경로(--vecnorm_out)로 한 번 더 복사 저장.

수집 종료 조건: 성공 에피소드 수가 --num_episodes 도달.
"""

import argparse
import os
import pickle
import shutil
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect expert demos from example5 PPO policy")
parser.add_argument("--checkpoint", type=str, default="checkpoints/omy_lift.zip")
parser.add_argument("--vecnorm", type=str, default="checkpoints/omy_lift_vecnorm.pkl")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_episodes", type=int, default=300,
                    help="수집할 성공 에피소드 수")
parser.add_argument("--max_total_episodes", type=int, default=2000,
                    help="안전장치: 이 수 도달 시 부족해도 종료")
parser.add_argument("--output", type=str, default="source/example7/demos/example5_demos.pkl")
parser.add_argument("--vecnorm_out", type=str, default="checkpoints/example7_vecnorm.pkl",
                    help="example7 학습이 사용할 VecNormalize pkl 복사본")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper

from source.example7.tasks.lift.lift_env_cfg import LiftEnvCfg
from source.example7.tasks.lift.lift_env import LiftEnv


def main():
    cfg = LiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    raw_env = LiftEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)

    if not os.path.exists(args_cli.vecnorm):
        raise FileNotFoundError(f"VecNormalize 파일 없음: {args_cli.vecnorm}")
    env = VecNormalize.load(args_cli.vecnorm, env)
    env.training = False
    env.norm_reward = False
    print(f"✅ VecNormalize 로드: {args_cli.vecnorm}")

    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(f"PPO checkpoint 없음: {args_cli.checkpoint}")
    model = PPO.load(args_cli.checkpoint, env=env, device="cuda")
    print(f"✅ PPO 모델 로드: {args_cli.checkpoint}")

    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    os.makedirs(os.path.dirname(args_cli.vecnorm_out) or ".", exist_ok=True)

    # 학습 루프와 동일한 정규화 통계를 example7이 그대로 사용하도록 복사
    shutil.copy(args_cli.vecnorm, args_cli.vecnorm_out)
    print(f"💾 VecNormalize 복사: {args_cli.vecnorm_out}")

    # 환경별 진행 중 trajectory 버퍼
    cur_obs = [[] for _ in range(args_cli.num_envs)]
    cur_next = [[] for _ in range(args_cli.num_envs)]
    cur_act = [[] for _ in range(args_cli.num_envs)]
    cur_rew = [[] for _ in range(args_cli.num_envs)]
    cur_done = [[] for _ in range(args_cli.num_envs)]

    # 에피소드 내 최대 obj_z 추적 (성공 판정용 — raw_env에서 직접 읽음)
    max_obj_z = np.zeros(args_cli.num_envs, dtype=np.float32)

    completed: list[dict] = []
    success_count = 0
    total_episodes = 0
    step = 0

    obs = env.reset()
    lift_threshold = cfg.lift_height_threshold

    # 성공 판정 임계값 — done 직전 마지막으로 읽은 obj_z는 reset으로 사라지므로
    # 한 스텝 전(=현재 iteration 시작 시점) 값이 임계 직전값이라 살짝 낮춤
    success_z_margin = 0.02  # 2cm 여유
    success_threshold = lift_threshold - success_z_margin

    print("=" * 70)
    print(f"🎬 demo 수집 시작 — 목표 성공 에피소드: {args_cli.num_episodes}")
    print(f"   lift_threshold={lift_threshold:.3f}m, success_threshold={success_threshold:.3f}m (max_obj_z 기준)")
    print("=" * 70)

    while success_count < args_cli.num_episodes and total_episodes < args_cli.max_total_episodes:
        # 이번 step 직전의 raw obj_z를 기록 (reset 전 마지막 값)
        cur_obj_z = raw_env._object.data.root_pos_w[:, 2].detach().cpu().numpy()
        max_obj_z = np.maximum(max_obj_z, cur_obj_z)

        actions, _ = model.predict(obs, deterministic=True)
        next_obs, rewards, dones, infos = env.step(actions)

        for i in range(args_cli.num_envs):
            cur_obs[i].append(obs[i].copy())
            cur_next[i].append(next_obs[i].copy())
            cur_act[i].append(actions[i].copy())
            cur_rew[i].append(float(rewards[i]))
            cur_done[i].append(bool(dones[i]))

        if dones.any():
            for i in range(args_cli.num_envs):
                if dones[i]:
                    total_episodes += 1

                    # max_obj_z 기준 + 마지막 step에서도 obj가 안 떨어졌는지 확인
                    is_success = max_obj_z[i] >= success_threshold

                    length = len(cur_obs[i])
                    if is_success and length > 5:
                        traj = {
                            "obs": np.asarray(cur_obs[i], dtype=np.float32),
                            "next_obs": np.asarray(cur_next[i], dtype=np.float32),
                            "actions": np.asarray(cur_act[i], dtype=np.float32),
                            "rewards": np.asarray(cur_rew[i], dtype=np.float32),
                            "dones": np.asarray(cur_done[i], dtype=bool),
                            "success": True,
                            "length": length,
                            "max_obj_z": float(max_obj_z[i]),
                        }
                        completed.append(traj)
                        success_count += 1

                    cur_obs[i].clear()
                    cur_next[i].clear()
                    cur_act[i].clear()
                    cur_rew[i].clear()
                    cur_done[i].clear()
                    max_obj_z[i] = 0.0

        obs = next_obs
        step += 1

        if step % 200 == 0:
            print(f"  step={step:>6,} | success={success_count:>4d}/{args_cli.num_episodes} "
                  f"| total_ep={total_episodes:>4d} | success_rate={success_count/max(total_episodes,1):.2%} "
                  f"| max_z_seen={float(max_obj_z.max()):.3f}", flush=True)

    print("=" * 70)
    print(f"✅ 수집 완료: success={success_count}, total_episodes={total_episodes}")
    if len(completed) > 0:
        print(f"   평균 trajectory 길이: {np.mean([t['length'] for t in completed]):.1f} steps")
        print(f"   평균 max_obj_z: {np.mean([t['max_obj_z'] for t in completed]):.3f}m")
    else:
        print("   ⚠️ 성공 에피소드 0개. checkpoint 품질 또는 success_threshold 확인 필요.")

    with open(args_cli.output, "wb") as f:
        pickle.dump(completed, f)
    print(f"💾 저장: {args_cli.output} ({len(completed)} trajectories)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
