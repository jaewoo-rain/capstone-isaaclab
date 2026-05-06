"""SAC + HER 학습 스크립트 — OMY Place

example5의 train.py와 동일한 기능 제공:
- session_step / total_step 분리 (resume 시 진행률 정확하게 표시)
- 중간 저장 (save_interval)
- 중간 출력 + reward_log
- 체크포인트 / VecNormalize 로드 및 이어서 학습
- --checkpoint 직접 지정 지원

차이점:
- PPO → SAC + HerReplayBuffer + MultiInputPolicy
- flat 31dim obs → GoalEnvVecWrapper로 Dict로 split
- VecNormalize는 옵션 플래그(--vecnorm_on)로만 활성화, 기본 off
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train SAC+HER for OMY Place Task")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timesteps", type=int, default=3_000_000)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--no_save", action="store_true")
parser.add_argument("--name", type=str, default="example6")
parser.add_argument(
    "--checkpoint", type=str, default=None,
    help="불러올 SAC checkpoint(.zip) 경로",
)
parser.add_argument(
    "--vecnorm", type=str, default=None,
    help="불러올 VecNormalize pkl 경로 (--vecnorm_on일 때만 사용)",
)
parser.add_argument(
    "--vecnorm_on", action="store_true",
    help="VecNormalize 활성화 (기본 off)",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------
# import
# ------------------------------------------------------------------
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.her import HerReplayBuffer

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper

from source.example6.tasks.place.place_env_cfg import PlaceEnvCfg
from source.example6.tasks.place.place_env import PlaceEnv


# ------------------------------------------------------------------
# GoalEnv VecWrapper
# PlaceEnv가 31dim flat tensor를 반환 → Dict(observation, achieved_goal, desired_goal)로 split
# SB3 HER + MultiInputPolicy는 Dict 형식을 요구함
# ------------------------------------------------------------------
class GoalEnvVecWrapper(VecEnvWrapper):
    """flat 31dim obs를 HER용 Dict로 변환.

    마지막 6차원은 [achieved_goal(3), desired_goal(3)] 로 인코딩되어 있음.
    compute_reward는 래핑된 PlaceEnv에 위임.
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
        """단일 env 관측값(1D)을 Dict로 split."""
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
        # infos의 terminal_observation도 Dict로 변환
        for i in range(len(infos)):
            t_obs = infos[i].get("terminal_observation", None)
            if t_obs is not None and not isinstance(t_obs, dict):
                infos[i]["terminal_observation"] = self._split_single(np.asarray(t_obs))
        return self._split(obs), rewards, dones, infos

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._compute_reward_fn(achieved_goal, desired_goal, info)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """IsaacLab의 Sb3VecEnvWrapper는 단일 결과를 반환하지만,
        SB3 HER은 list를 기대(`rewards[0]`). 여기서 list로 감싼다."""
        if method_name == "compute_reward":
            result = self._compute_reward_fn(*method_args, **method_kwargs)
            return [result]
        # fallback: 하위 venv로 위임
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)


# ------------------------------------------------------------------
# Callback (example5 train.py 구조 그대로)
# ------------------------------------------------------------------
class TrainCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        checkpoint_path: str,
        vecnorm_path: str,
        env_ref: PlaceEnv,
        save_interval: int = 500_000,
        print_interval: int = 50_000,
        save_vecnorm: bool = False,
    ):
        super().__init__(verbose=0)
        self.total_timesteps_target = total_timesteps
        self.checkpoint_path = checkpoint_path
        self.vecnorm_path = vecnorm_path
        self.env_ref = env_ref
        self.save_interval = save_interval
        self.print_interval = print_interval
        self.save_vecnorm = save_vecnorm

        self.last_print = 0
        self.last_save = 0
        self.start_time = None
        self.start_step = 0
        self._reward_log_buf = {}

    def _on_training_start(self):
        self.start_time = time.time()
        self.start_step = self.num_timesteps
        self.last_print = 0
        self.last_save = 0

        print("=" * 70)
        print("🚀 학습 시작 (SAC + HER)")
        print(f"   이번 실행 목표 step : {self.total_timesteps_target:,}")
        print(f"   시작 누적 step      : {self.start_step:,}")
        print("=" * 70)

    def _on_step(self) -> bool:
        # env reward_log 누적
        for k, v in self.env_ref.reward_log.items():
            self._reward_log_buf.setdefault(k, []).append(v)

        total_step = self.num_timesteps
        session_step = total_step - self.start_step

        # 출력
        if session_step - self.last_print >= self.print_interval:
            self.last_print = session_step
            elapsed = time.time() - self.start_time
            sps = session_step / elapsed if elapsed > 0 else 0.0
            remain_steps = max(self.total_timesteps_target - session_step, 0)
            remain_s = remain_steps / sps if sps > 0 else 0.0
            pct = (
                session_step / self.total_timesteps_target * 100.0
                if self.total_timesteps_target > 0 else 0.0
            )

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

            # prefix로 그룹화해서 섹션별 출력
            # env0_ / dist_ / rate_ / rew_ / (그 외)
            groups = {"env0": {}, "dist": {}, "rate": {}, "rew": {}, "other": {}}
            for key, vals in self._reward_log_buf.items():
                if not vals:
                    continue
                avg = sum(vals) / len(vals)
                if key.startswith("env0_"):
                    groups["env0"][key[5:]] = avg
                elif key.startswith("dist_"):
                    groups["dist"][key[5:]] = avg
                elif key.startswith("rate_"):
                    groups["rate"][key[5:]] = avg
                elif key.startswith("rew_"):
                    groups["rew"][key[4:]] = avg
                else:
                    groups["other"][key] = avg

            # env0: 좌표를 (x,y,z) 형태로 깔끔하게
            if groups["env0"]:
                e = groups["env0"]
                print(
                    f"  ├─ [env0] obj=({e.get('obj_x', 0):.3f},{e.get('obj_y', 0):.3f},{e.get('obj_z', 0):.3f}) "
                    f"| grip=({e.get('grip_x', 0):.3f},{e.get('grip_y', 0):.3f},{e.get('grip_z', 0):.3f}) "
                    f"| tgt=({e.get('tgt_x', 0):.3f},{e.get('tgt_y', 0):.3f},{e.get('tgt_z', 0):.3f}) "
                    f"| upright={e.get('upright', 0):.3f} | grip_close={e.get('grip_close', 0):.3f}"
                )
                print(
                    f"  ├─ [env0_joints] "
                    f"j1={e.get('j1', 0):+.3f} j2={e.get('j2', 0):+.3f} "
                    f"j3={e.get('j3', 0):+.3f} j4={e.get('j4', 0):+.3f} "
                    f"j5={e.get('j5', 0):+.3f} j6={e.get('j6', 0):+.3f}"
                )

            for gname in ("dist", "rate", "rew"):
                if groups[gname]:
                    parts = [f"{k}={v:.4f}" for k, v in groups[gname].items()]
                    print(f"  ├─ [{gname}] " + " | ".join(parts))

            if groups["other"]:
                parts = [f"{k}={v:.4f}" for k, v in groups["other"].items()]
                print(f"  └─ [other] " + " | ".join(parts))

            self._reward_log_buf.clear()

        # 저장
        if session_step - self.last_save >= self.save_interval:
            self.last_save = session_step
            ckpt = f"{self.checkpoint_path}_step{total_step}"
            self.model.save(ckpt)
            if self.save_vecnorm and isinstance(self.training_env, VecNormalize):
                self.training_env.save(f"{ckpt}_vecnorm.pkl")
            print(f"  💾 중간 저장 완료: {ckpt}")

        return True


# ------------------------------------------------------------------
# 유틸
# ------------------------------------------------------------------
def infer_vecnorm_path_from_checkpoint(checkpoint_path: str) -> str:
    base, ext = os.path.splitext(checkpoint_path)
    if ext == ".zip":
        return f"{base}_vecnorm.pkl"
    return f"{checkpoint_path}_vecnorm.pkl"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    cfg = PlaceEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    torch.manual_seed(args_cli.seed)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    default_checkpoint_path = os.path.join(checkpoint_dir, args_cli.name)
    default_vecnorm_path = os.path.join(checkpoint_dir, f"{args_cli.name}_vecnorm.pkl")

    # ----- env 생성 -----
    raw_env = PlaceEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)

    # HER용 Dict observation으로 변환
    env = GoalEnvVecWrapper(
        env,
        core_dim=PlaceEnv.OBS_CORE_DIM,
        goal_dim=PlaceEnv.GOAL_DIM,
        compute_reward_fn=raw_env.compute_reward,
    )

    # VecNormalize 옵션
    if args_cli.vecnorm_on:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

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
    )

    # HerReplayBuffer는 env.compute_reward를 호출
    replay_buffer_kwargs = dict(
        n_sampled_goal=cfg.n_sampled_goal,
        goal_selection_strategy=cfg.goal_selection_strategy,
    )

    # ----- 로드 경로 결정 -----
    load_checkpoint_path = None
    load_vecnorm_path = None
    if args_cli.checkpoint is not None:
        load_checkpoint_path = args_cli.checkpoint
        load_vecnorm_path = (
            args_cli.vecnorm if args_cli.vecnorm is not None
            else infer_vecnorm_path_from_checkpoint(args_cli.checkpoint)
        )
    elif args_cli.resume:
        load_checkpoint_path = default_checkpoint_path + ".zip"
        load_vecnorm_path = default_vecnorm_path

    checkpoint_path = default_checkpoint_path
    vecnorm_path = default_vecnorm_path

    # ----- 로드 or 신규 -----
    if load_checkpoint_path is not None and os.path.exists(load_checkpoint_path):
        has_vecnorm = os.path.exists(load_vecnorm_path)

        if args_cli.vecnorm_on and has_vecnorm:
            print(f"🔄 VecNormalize 로드: {load_vecnorm_path}")
            env = VecNormalize.load(load_vecnorm_path, env)
            env.training = True
            env.norm_reward = True

        print(f"🔄 체크포인트 로드: {load_checkpoint_path}")
        model = SAC.load(
            load_checkpoint_path,
            env=env,
            device="cuda",
            **sac_kwargs,
        )

        # replay buffer 로드 (SAC는 checkpoint에 buffer 포함 안됨)
        load_base_path, _ = os.path.splitext(load_checkpoint_path)
        rb_path = f"{load_base_path}_replay.pkl"
        if os.path.exists(rb_path):
            model.load_replay_buffer(rb_path)
            print(f"🔄 Replay buffer 로드: {rb_path} ({model.replay_buffer.size()} transitions)")
        else:
            # replay buffer 없음 → learning_starts를 현재 num_timesteps + 원래 값으로 밀어서
            # 재수집할 시간 확보
            original_ls = cfg.learning_starts
            model.learning_starts = model.num_timesteps + original_ls
            print(f"⚠️ Replay buffer 없음. learning_starts 를 {model.learning_starts} 으로 조정 (재수집)")

        load_base, ext = os.path.splitext(load_checkpoint_path)
        checkpoint_path = load_base if ext == ".zip" else load_checkpoint_path
        vecnorm_path = infer_vecnorm_path_from_checkpoint(checkpoint_path)
        print("✅ 이어서 학습합니다.")
    else:
        if load_checkpoint_path is not None:
            print(f"⚠️ 지정한 checkpoint 없음: {load_checkpoint_path} → 새로 시작")
        print("🆕 새 학습 시작")

        model = SAC(
            policy="MultiInputPolicy",
            env=env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            tensorboard_log="./logs/sb3/place",
            device="cuda",
            **sac_kwargs,
        )

    # ----- Callback -----
    callback = TrainCallback(
        total_timesteps=args_cli.timesteps,
        checkpoint_path=checkpoint_path,
        vecnorm_path=vecnorm_path,
        env_ref=raw_env,
        save_interval=500_000,
        print_interval=25_000,
        save_vecnorm=args_cli.vecnorm_on,
    )

    # ----- 학습 -----
    model.learn(
        total_timesteps=args_cli.timesteps,
        callback=callback,
        reset_num_timesteps=not (args_cli.resume or args_cli.checkpoint is not None),
    )

    # ----- 최종 저장 -----
    if not args_cli.no_save:
        model.save(checkpoint_path)
        # replay buffer 저장 (resume 시 HER 학습 연속성 유지)
        try:
            model.save_replay_buffer(f"{checkpoint_path}_replay.pkl")
            print(f"✅ Replay buffer 저장: {checkpoint_path}_replay.pkl")
        except Exception as e:
            print(f"⚠️ Replay buffer 저장 실패: {e}")
        if args_cli.vecnorm_on and isinstance(env, VecNormalize):
            env.save(vecnorm_path)
            print(f"✅ VecNormalize 저장: {vecnorm_path}")
        print(f"✅ 최종 저장 완료: {checkpoint_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
