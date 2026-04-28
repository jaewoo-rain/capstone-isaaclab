"""SAC actor를 BC(Behavior Cloning)로 pretrain.

흐름:
1. SAC 모델을 새로 만든다 (env wrap + VecNormalize 로드).
2. demos.pkl에서 (obs, action) 페어를 PyTorch DataLoader로 만든다.
3. SAC actor의 squashed mean(=tanh(mean_actions))를 expert action에 MSE로 fit.
4. demo trajectory를 SAC replay buffer에 prefill.
5. 결과를 SAC zip + replay pkl로 저장 → train.py가 이를 로드해 fine-tune.

BC만 하면 distribution shift로 분포 밖 상태에서 망가지지만, 이후 SAC fine-tune
단계에서 실제 환경 rollout으로 교정된다.
"""

import argparse
import os
import pickle
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="BC pretrain SAC actor from expert demos")
parser.add_argument("--demos", type=str, default="source/example7/demos/example5_demos.pkl")
parser.add_argument("--vecnorm", type=str, default="checkpoints/example7_vecnorm.pkl")
parser.add_argument("--num_envs", type=int, default=4,
                    help="BC만 할 때는 env가 step되지 않으므로 작게 둠")
parser.add_argument("--epochs", type=int, default=30, help="BC actor pretrain epochs")
parser.add_argument("--critic_epochs", type=int, default=20,
                    help="Critic pretrain epochs (Tier 2 — actor BC 후 critic도 워밍업)")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--alpha", type=float, default=0.05,
                    help="Critic pretrain 시 entropy 가중치 (작게 유지해서 BC actor 영향 보존)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default="checkpoints/example7_bc.zip",
                    help="BC pretrain 결과 SAC zip 경로")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn.functional as F

from stable_baselines3 import SAC
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
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    # ----------------------------------------------------------
    # 환경 + VecNormalize (학습 시 동일한 obs 정규화 사용)
    # ----------------------------------------------------------
    cfg = LiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    raw_env = LiftEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)

    if not os.path.exists(args_cli.vecnorm):
        raise FileNotFoundError(f"VecNormalize 없음: {args_cli.vecnorm}")
    env = VecNormalize.load(args_cli.vecnorm, env)
    env.training = False
    env.norm_reward = False
    print(f"✅ VecNormalize 로드: {args_cli.vecnorm}")

    # ----------------------------------------------------------
    # SAC 모델 생성 (BC 후 fine-tune에서도 동일 구조 유지)
    # ----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # train.py와 일관된 하이퍼파라미터로 생성 (저장된 zip을 그대로 로드해 fine-tune)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=1e-4,           # train.py default와 일치 (BC warm-start 보존 위해 낮춤)
        buffer_size=1_000_000,
        batch_size=args_cli.batch_size,
        tau=args_cli.tau,
        gamma=args_cli.gamma,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto_0.05",         # 초기 alpha 작게 → 탐색 압력 ↓
        target_entropy=-3.0,          # 기본 -7 → -3, 탐색 약화
        learning_starts=0,
        verbose=0,
        device=device,
        seed=args_cli.seed,
        tensorboard_log="./logs/sb3/example7",
    )
    print("✅ SAC 모델 초기화 완료")

    # ----------------------------------------------------------
    # demos 로드 → 텐서로 변환
    # ----------------------------------------------------------
    if not os.path.exists(args_cli.demos):
        raise FileNotFoundError(f"demos 파일 없음: {args_cli.demos}")
    with open(args_cli.demos, "rb") as f:
        demos = pickle.load(f)
    print(f"✅ demos 로드: {len(demos)} trajectories")

    all_obs = np.concatenate([t["obs"] for t in demos], axis=0)
    all_act = np.concatenate([t["actions"] for t in demos], axis=0)
    all_next = np.concatenate([t["next_obs"] for t in demos], axis=0)
    all_rew = np.concatenate([t["rewards"] for t in demos], axis=0)
    all_done = np.concatenate([t["dones"] for t in demos], axis=0)
    n_transitions = all_obs.shape[0]
    print(f"   총 transition: {n_transitions:,}")
    print(f"   obs shape: {all_obs.shape}, action shape: {all_act.shape}")

    obs_t = torch.as_tensor(all_obs, dtype=torch.float32, device=device)
    next_obs_t = torch.as_tensor(all_next, dtype=torch.float32, device=device)
    act_t = torch.as_tensor(all_act, dtype=torch.float32, device=device).clamp(-0.999, 0.999)
    rew_t = torch.as_tensor(all_rew, dtype=torch.float32, device=device).unsqueeze(-1)
    done_t = torch.as_tensor(all_done.astype(np.float32), dtype=torch.float32, device=device).unsqueeze(-1)
    # 주의: tanh의 atanh가 폭발하지 않도록 clamp

    # ----------------------------------------------------------
    # BC 학습 — actor의 squashed mean을 expert action에 fit
    # ----------------------------------------------------------
    actor = model.policy.actor
    optim = torch.optim.Adam(actor.parameters(), lr=args_cli.lr)

    n_batches_per_epoch = max(1, n_transitions // args_cli.batch_size)

    print("=" * 70)
    print(f"🎓 BC 학습 시작 — epochs={args_cli.epochs}, batch={args_cli.batch_size}")
    print("=" * 70)

    for epoch in range(args_cli.epochs):
        perm = torch.randperm(n_transitions, device=device)
        epoch_loss = 0.0

        for b in range(n_batches_per_epoch):
            idx = perm[b * args_cli.batch_size : (b + 1) * args_cli.batch_size]
            o = obs_t[idx]
            a = act_t[idx]

            # SB3 SAC actor: get_action_dist_params(obs) -> (mean_actions, log_std, kwargs)
            mean_actions, _, _ = actor.get_action_dist_params(o)
            pred = torch.tanh(mean_actions)

            loss = F.mse_loss(pred, a)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches_per_epoch
        print(f"  epoch {epoch+1:3d}/{args_cli.epochs} | bc_loss={avg_loss:.5f}")

    # ----------------------------------------------------------
    # Critic pretrain (Tier 2) — actor BC 직후 동일 demos로 critic 워밍업
    # 목적: SAC fine-tune 시작 시 Q값이 무작위가 아니라 BC actor 행동에 일관되도록
    # ----------------------------------------------------------
    print("=" * 70)
    print(f"🎓 Critic pretrain 시작 — epochs={args_cli.critic_epochs}, alpha={args_cli.alpha}")
    print("=" * 70)

    critic = model.critic
    critic_target = model.critic_target
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args_cli.lr)
    gamma = args_cli.gamma
    tau = args_cli.tau
    alpha = args_cli.alpha

    for epoch in range(args_cli.critic_epochs):
        perm = torch.randperm(n_transitions, device=device)
        epoch_loss = 0.0

        for b in range(n_batches_per_epoch):
            idx = perm[b * args_cli.batch_size : (b + 1) * args_cli.batch_size]
            s = obs_t[idx]
            s2 = next_obs_t[idx]
            a = act_t[idx]
            r = rew_t[idx]
            d = done_t[idx]

            with torch.no_grad():
                # next action / log_prob from BC-trained actor
                next_a, next_log_prob = actor.action_log_prob(s2)
                # min over twin Q from target critic
                next_q_list = critic_target(s2, next_a)
                next_q = torch.min(torch.cat(next_q_list, dim=1), dim=1, keepdim=True)[0]
                # subtract entropy bonus (small alpha → BC actor 우선)
                next_q = next_q - alpha * next_log_prob.reshape(-1, 1)
                target_q = r + gamma * (1.0 - d) * next_q

            # current Q from twin critics
            current_q_list = critic(s, a)
            critic_loss = sum(F.mse_loss(q, target_q) for q in current_q_list) / len(current_q_list)

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            # soft target update per batch (SAC style)
            with torch.no_grad():
                for p, p_target in zip(critic.parameters(), critic_target.parameters()):
                    p_target.data.mul_(1 - tau)
                    p_target.data.add_(tau * p.data)

            epoch_loss += critic_loss.item()

        avg_loss = epoch_loss / n_batches_per_epoch
        print(f"  critic epoch {epoch+1:3d}/{args_cli.critic_epochs} | q_loss={avg_loss:.4f}")

    # ----------------------------------------------------------
    # 저장 — replay buffer prefill은 train.py에서 (n_envs 일치 위해)
    # ----------------------------------------------------------
    os.makedirs(os.path.dirname(args_cli.output) or ".", exist_ok=True)
    model.save(args_cli.output)
    print(f"💾 SAC(BC pretrain) 저장: {args_cli.output}")
    print("ℹ️  replay buffer prefill은 train.py가 동일 demos.pkl을 로드해서 처리함")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
