"""SAC fine-tune 학습 스크립트 — BC pretrain된 SAC를 실제 환경에서 추가 학습.

기본 시나리오:
    python source/example7/scripts/train.py --num_envs 64 --timesteps 3000000

체크포인트 우선순위:
    1) --checkpoint 직접 지정
    2) --resume 시 checkpoints/example7.zip
    3) 둘 다 없으면 --bc_checkpoint(=checkpoints/example7_bc.zip)에서 시작
       (BC pretrain 결과를 warm-start로 사용)

VecNormalize는 collect_demos에서 복사된 checkpoints/example7_vecnorm.pkl을 사용.
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train SAC for example7 (Lift, BC warm-start)")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--timesteps", type=int, default=3_000_000)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--no_save", action="store_true")
parser.add_argument("--name", type=str, default="example7")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="이어서 학습할 SAC zip 경로")
parser.add_argument("--vecnorm", type=str, default="checkpoints/example7_vecnorm.pkl")
parser.add_argument("--bc_checkpoint", type=str, default="checkpoints/example7_bc.zip",
                    help="BC pretrain된 SAC zip — 기본 warm-start 출처")
parser.add_argument("--demos", type=str, default="source/example7/demos/example5_demos.pkl",
                    help="demo pkl — replay buffer에 prefill (BC warm-start와 함께 사용 시)")
parser.add_argument("--no_prefill_demos", action="store_true",
                    help="demos prefill 비활성화 (resume 시 자동으로 비활성)")
parser.add_argument("--learning_rate", type=float, default=1e-4,
                    help="BC warm-start를 보존하기 위해 기본 SAC lr보다 낮춤 (3e-4 → 1e-4)")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--buffer_size", type=int, default=1_000_000)
parser.add_argument("--gradient_steps", type=int, default=1,
                    help="env step당 gradient step 수. -1이면 train_freq만큼")
parser.add_argument("--learning_starts", type=int, default=0,
                    help="replay 채워진 상태로 시작하므로 0이 합리적")
parser.add_argument("--ent_coef", type=str, default="auto_0.05",
                    help="SAC entropy 계수. 'auto_0.05' = auto 모드인데 초기값 0.05로 시작 (탐색 약하게)")
parser.add_argument("--target_entropy", type=str, default="-3.0",
                    help="auto ent_coef target. 기본 SB3는 -action_dim(=-7) → -3 으로 완화하면 탐색 압력 ↓")
parser.add_argument("--bc_coef", type=float, default=2.5,
                    help="actor loss에 추가되는 BC regularization 초기 가중치 (Tier 3)")
parser.add_argument("--bc_decay_steps", type=int, default=500_000,
                    help="bc_coef를 0으로 선형 감소시킬 gradient step 수")
parser.add_argument("--no_bc_reg", action="store_true",
                    help="BC regularization 끄기 (순수 SAC fine-tune)")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import pickle
import types

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import polyak_update
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


# ------------------------------------------------------------------
# Demo prefill — 모델의 replay buffer에 expert transition을 채움
# n_envs 차원을 자동으로 맞춰 chunk 단위로 add 호출
# ------------------------------------------------------------------
def prefill_replay_from_demos(model: SAC, env, demos_path: str) -> None:
    print(f"📦 demos 로드 → replay buffer prefill: {demos_path}")
    with open(demos_path, "rb") as f:
        demos = pickle.load(f)

    all_obs = np.concatenate([t["obs"] for t in demos], axis=0)
    all_next = np.concatenate([t["next_obs"] for t in demos], axis=0)
    all_act = np.concatenate([t["actions"] for t in demos], axis=0)
    all_rew = np.concatenate([t["rewards"] for t in demos], axis=0)
    all_done = np.concatenate([t["dones"] for t in demos], axis=0)
    n = all_obs.shape[0]

    rb = model.replay_buffer
    n_envs = rb.n_envs

    # chunk 단위로 batched add
    n_full = n // n_envs
    for c in range(n_full):
        s = c * n_envs
        e = s + n_envs
        rb.add(
            obs=all_obs[s:e],
            next_obs=all_next[s:e],
            action=all_act[s:e],
            reward=all_rew[s:e],
            done=all_done[s:e],
            infos=[{} for _ in range(n_envs)],
        )
    # 남는 transition은 버려도 무방 (보통 n_envs 단위로 정렬됨)
    leftover = n - n_full * n_envs
    print(f"✅ prefill 완료: {n_full * n_envs:,} transitions (leftover={leftover}) "
          f"| rb.size={rb.size():,} | rb.n_envs={n_envs}")


# ------------------------------------------------------------------
# Tier 3: BC regularization patched into SAC.train
# actor loss = SAC actor loss + bc_coef * MSE(tanh(actor.mean(s)), expert_a)
# bc_coef는 bc_decay_steps에 걸쳐 0으로 선형 감소
# ------------------------------------------------------------------
def patch_sac_train_with_bc(model: SAC, demo_obs_t: torch.Tensor,
                            demo_act_t: torch.Tensor, bc_coef_init: float,
                            bc_decay_steps: int) -> None:
    # state는 model 인스턴스에 저장해서 save/restore 사이에 보존
    if not hasattr(model, "_bc_state"):
        model._bc_state = {"step": 0}
    state = model._bc_state
    state["n_demos"] = int(demo_obs_t.shape[0])

    def custom_train(self, gradient_steps: int, batch_size: int = 64):
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers.append(self.ent_coef_optimizer)
        self._update_learning_rate(optimizers)

        actor_losses, critic_losses, bc_losses = [], [], []
        ent_coefs, ent_coef_losses = [], []
        last_bc_coef = 0.0

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env)

            # 현재 ent_coef 결정 (auto이면 log_ent_coef로부터)
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = torch.exp(self.log_ent_coef.detach())
            else:
                ent_coef = self.ent_coef_tensor

            # critic update (standard SAC)
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(
                    replay_data.next_observations)
                next_q = torch.cat(
                    self.critic_target(replay_data.next_observations, next_actions),
                    dim=1)
                next_q, _ = torch.min(next_q, dim=1, keepdim=True)
                next_q = next_q - ent_coef * next_log_prob.reshape(-1, 1)
                target_q = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q

            current_q = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(q, target_q) for q in current_q)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(critic_loss.item())

            # actor update with BC reg
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef_loss = -(self.log_ent_coef *
                                  (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())

            q_pi = torch.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_q_pi, _ = torch.min(q_pi, dim=1, keepdim=True)
            sac_actor_loss = (ent_coef * log_prob - min_q_pi).mean()

            # BC regularization (TD3+BC 스타일, Q 크기에 맞춰 BC 가중치 동적 조정)
            n_demos = state["n_demos"]
            demo_idx = torch.randint(0, n_demos, (batch_size,), device=self.device)
            d_obs = demo_obs_t[demo_idx]
            d_act = demo_act_t[demo_idx]
            mean_actions, _, _ = self.actor.get_action_dist_params(d_obs)
            pred_act = torch.tanh(mean_actions)
            bc_loss = F.mse_loss(pred_act, d_act)

            with torch.no_grad():
                # Q값의 평균 크기에 BC 가중치를 비례시켜 두 loss의 영향력 균형
                q_normalizer = min_q_pi.abs().mean().clamp(min=1.0)

            decay_progress = min(state["step"] / max(bc_decay_steps, 1), 1.0)
            bc_coef = bc_coef_init * (1.0 - decay_progress)
            bc_weight = bc_coef * q_normalizer
            last_bc_coef = bc_coef

            actor_loss = sac_actor_loss + bc_weight * bc_loss

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            actor_losses.append(sac_actor_loss.item())
            bc_losses.append(bc_loss.item())

            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            ent_coefs.append(float(ent_coef))

            if state["step"] % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(),
                              self.critic_target.parameters(), self.tau)

            state["step"] += 1

        self._n_updates += gradient_steps
        if hasattr(self, "logger"):
            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
            self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
            self.logger.record("train/bc_loss", float(np.mean(bc_losses)))
            self.logger.record("train/bc_coef", float(last_bc_coef))
            self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
            if ent_coef_losses:
                self.logger.record("train/ent_coef_loss", float(np.mean(ent_coef_losses)))

        # 디버깅용 stdout 출력 (logger 우회)
        if state["step"] % (gradient_steps * 200) < gradient_steps:
            print(f"  🩹 grad_step={state['step']:>7} | bc_coef={last_bc_coef:.3f} "
                  f"| bc_loss={float(np.mean(bc_losses)):.5f} "
                  f"| q_avg={float(np.mean([abs(c) for c in critic_losses])):.1f}", flush=True)

    model.train = types.MethodType(custom_train, model)
    print(f"🩹 SAC.train patched with BC regularization "
          f"(init={bc_coef_init}, decay over {bc_decay_steps:,} grad steps)")


def _safe_save(model: SAC, ckpt_path: str) -> None:
    """BC patch가 적용된 model을 저장.

    monkey-patched train method가 cloudpickle 불가능한 closure를 갖고 있어
    저장 시점엔 원본 SAC.train으로 잠시 복원 + custom 속성 제거 → 저장 → 다시 패치.
    BC step 카운터(_bc_state)는 보존해서 decay 진행률 유지.
    """
    patched = getattr(model, "_bc_patcher", None)
    saved_attrs = {}
    if patched is not None:
        # instance dict에서 patched train 제거 (클래스 메서드로 폴백)
        if "train" in model.__dict__:
            del model.__dict__["train"]
        # cloudpickle 불가능한 custom 속성을 임시 분리
        for attr in ("_bc_patcher", "_bc_state"):
            if hasattr(model, attr):
                saved_attrs[attr] = getattr(model, attr)
                delattr(model, attr)
    try:
        model.save(ckpt_path)
    except TypeError as e:
        # 어떤 attribute가 unpicklable인지 진단
        print(f"⚠️ save 실패: {e}")
        import cloudpickle
        for k, v in list(model.__dict__.items()):
            try:
                cloudpickle.dumps(v)
            except Exception as ex:
                print(f"   ❌ unpicklable attr: {k!r} ({type(v).__name__}) — {ex}")
        raise
    finally:
        # 속성 복원 후 패치 재적용 (train 메서드 다시 monkey-patch)
        for attr, val in saved_attrs.items():
            setattr(model, attr, val)
        if patched is not None:
            patched()


def _maybe_patch_bc(model: SAC, args_cli) -> None:
    """args_cli 옵션과 demos 파일 존재 여부에 따라 BC reg patch 적용."""
    if args_cli.no_bc_reg:
        print("ℹ️ BC regularization 비활성화 (--no_bc_reg)")
        model._bc_patcher = None
        return
    if not os.path.exists(args_cli.demos):
        print(f"⚠️ demos 파일 없어 BC reg 비활성: {args_cli.demos}")
        model._bc_patcher = None
        return
    with open(args_cli.demos, "rb") as f:
        demos = pickle.load(f)
    obs = np.concatenate([t["obs"] for t in demos], axis=0)
    act = np.concatenate([t["actions"] for t in demos], axis=0).clip(-0.999, 0.999)
    demo_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model.device)
    demo_act_t = torch.as_tensor(act, dtype=torch.float32, device=model.device)

    def _apply():
        patch_sac_train_with_bc(
            model, demo_obs_t, demo_act_t,
            bc_coef_init=args_cli.bc_coef,
            bc_decay_steps=args_cli.bc_decay_steps,
        )
    _apply()
    model._bc_patcher = _apply


# ------------------------------------------------------------------
# Callback — 출력 및 중간 저장
# ------------------------------------------------------------------
class TrainCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        checkpoint_path: str,
        vecnorm_path: str,
        env_ref: LiftEnv,
        save_interval: int = 200_000,
        print_interval: int = 50_000,
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
        self.start_step = 0
        self._reward_log_buf: dict[str, list[float]] = {}

    def _on_training_start(self):
        self.start_time = time.time()
        self.start_step = self.num_timesteps
        self.last_print = 0
        self.last_save = 0
        print("=" * 70)
        print(f"🚀 SAC fine-tune 시작")
        print(f"   이번 실행 목표 step : {self.total_timesteps_target:,}")
        print(f"   시작 누적 step      : {self.start_step:,}")
        print("=" * 70)

    def _on_step(self) -> bool:
        for k, v in self.env_ref.reward_log.items():
            self._reward_log_buf.setdefault(k, []).append(v)

        total_step = self.num_timesteps
        session_step = total_step - self.start_step

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
                f"남은시간={remain_s/60:.1f}min"
            )
            parts = []
            for key, vals in self._reward_log_buf.items():
                if len(vals) > 0:
                    parts.append(f"{key}={sum(vals)/len(vals):.4f}")
            print("  └─ " + " | ".join(parts))
            self._reward_log_buf.clear()

        if session_step - self.last_save >= self.save_interval:
            self.last_save = session_step
            ckpt = f"{self.checkpoint_path}_step{total_step}"
            _safe_save(self.model, ckpt)
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(f"{ckpt}_vecnorm.pkl")
            print(f"  💾 중간 저장: {ckpt}")

        return True


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    cfg = LiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    torch.manual_seed(args_cli.seed)

    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    default_checkpoint_path = os.path.join(checkpoint_dir, args_cli.name)
    default_vecnorm_path = os.path.join(checkpoint_dir, f"{args_cli.name}_vecnorm.pkl")

    # ----------------------------------------------------------
    # Env + VecNormalize
    # ----------------------------------------------------------
    raw_env = LiftEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)

    if not os.path.exists(args_cli.vecnorm):
        raise FileNotFoundError(
            f"VecNormalize 없음: {args_cli.vecnorm}\n"
            f"먼저 collect_demos.py를 실행해 vecnorm을 복사하세요."
        )
    env = VecNormalize.load(args_cli.vecnorm, env)
    # SAC fine-tune 동안 obs 통계 업데이트는 끔(BC와 분포 일치 유지)
    env.training = False
    env.norm_reward = False
    print(f"✅ VecNormalize 로드(고정): {args_cli.vecnorm}")

    # ----------------------------------------------------------
    # 로드 경로 결정
    # ----------------------------------------------------------
    if args_cli.checkpoint is not None:
        load_path = args_cli.checkpoint
    elif args_cli.resume and os.path.exists(default_checkpoint_path + ".zip"):
        load_path = default_checkpoint_path + ".zip"
    elif os.path.exists(args_cli.bc_checkpoint):
        load_path = args_cli.bc_checkpoint
        print(f"🌱 BC pretrain warm-start: {load_path}")
    else:
        load_path = None

    # target_entropy: float 변환 시도
    try:
        target_entropy_val = float(args_cli.target_entropy)
    except ValueError:
        target_entropy_val = args_cli.target_entropy  # "auto" 등

    sac_kwargs = dict(
        learning_rate=args_cli.learning_rate,
        buffer_size=args_cli.buffer_size,
        batch_size=args_cli.batch_size,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=args_cli.gradient_steps,
        learning_starts=args_cli.learning_starts,
        ent_coef=args_cli.ent_coef,
        target_entropy=target_entropy_val,
        verbose=0,
        seed=args_cli.seed,
    )

    if load_path is not None and os.path.exists(load_path):
        print(f"🔄 SAC 체크포인트 로드: {load_path}")
        model = SAC.load(load_path, env=env, device="cuda", **sac_kwargs)
        _maybe_patch_bc(model, args_cli)

        # 기존 replay buffer 로드 시도 (이전 SAC 학습본만)
        loaded_replay = False
        load_base, ext = os.path.splitext(load_path)
        if ext == ".zip":
            rp = load_base + "_replay.pkl"
            if os.path.exists(rp):
                model.load_replay_buffer(rp)
                print(f"📦 replay buffer 로드: {rp} (size={model.replay_buffer.size():,})")
                loaded_replay = True

        # demos prefill — replay buffer가 비어있고 BC warm-start인 경우만
        # (resume 시 이미 가득 차있으면 skip)
        is_resuming = args_cli.resume or args_cli.checkpoint is not None
        if (not loaded_replay) and (not is_resuming) and (not args_cli.no_prefill_demos) \
                and os.path.exists(args_cli.demos):
            prefill_replay_from_demos(model, env, args_cli.demos)
    else:
        print("🆕 신규 SAC 학습 시작 (BC warm-start 없음)")
        model = SAC(
            "MlpPolicy",
            env,
            tensorboard_log="./logs/sb3/example7",
            device="cuda",
            **sac_kwargs,
        )

        if (not args_cli.no_prefill_demos) and os.path.exists(args_cli.demos):
            prefill_replay_from_demos(model, env, args_cli.demos)
        _maybe_patch_bc(model, args_cli)

    # ----------------------------------------------------------
    # Callback + 학습
    # ----------------------------------------------------------
    callback = TrainCallback(
        total_timesteps=args_cli.timesteps,
        checkpoint_path=default_checkpoint_path,
        vecnorm_path=default_vecnorm_path,
        env_ref=raw_env,
        save_interval=200_000,
        print_interval=50_000,
    )

    model.learn(
        total_timesteps=args_cli.timesteps,
        callback=callback,
        reset_num_timesteps=not (args_cli.resume or args_cli.checkpoint is not None),
    )

    if not args_cli.no_save:
        _safe_save(model, default_checkpoint_path)
        env.save(default_vecnorm_path)
        model.save_replay_buffer(default_checkpoint_path + "_replay.pkl")
        print(f"✅ 최종 저장: {default_checkpoint_path}.zip")
        print(f"✅ VecNormalize: {default_vecnorm_path}")
        print(f"✅ replay buffer: {default_checkpoint_path}_replay.pkl")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
