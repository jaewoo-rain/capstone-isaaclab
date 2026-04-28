# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 개요

OMY 로봇 팔이 박스를 집어 들어올리는 **Lift 태스크 (example5와 동일)** 를 두 단계로 학습:

1. **BC (Behavior Cloning)** — example5의 학습된 PPO 정책을 expert로 사용, 데모 수집 → SAC actor를 supervised learning으로 pretrain
2. **SAC fine-tune** — pretrain된 SAC를 실제 환경에서 RL로 미세 조정 (replay buffer에 데모 prefill)

목적: example5 PPO 결과를 SAC로 옮겨 sample-efficient + 부드러운 정책으로 만들고, 향후 sim2real / 후속 태스크에 재활용.

## 파이프라인

```
example5 PPO checkpoint (omy_lift.zip)
       │
       ▼  collect_demos.py
demos/example5_demos.pkl  +  checkpoints/example7_vecnorm.pkl(=omy_lift_vecnorm 복사)
       │
       ▼  train_bc.py
checkpoints/example7_bc.zip  +  checkpoints/example7_bc_replay.pkl
       │
       ▼  train.py (SAC fine-tune)
checkpoints/example7.zip  +  checkpoints/example7_vecnorm.pkl  +  checkpoints/example7_replay.pkl
       │
       ▼  play.py
시각화
```

## 실행 커맨드

```bash
# 1) demo 수집 (성공 에피소드 300개 목표, 부족하면 --num_episodes 500)
python source/example7/scripts/collect_demos.py \
    --checkpoint checkpoints/omy_lift.zip \
    --vecnorm checkpoints/omy_lift_vecnorm.pkl \
    --num_envs 64 --num_episodes 300

# 2) BC pretrain (replay buffer prefill 포함)
python source/example7/scripts/train_bc.py --epochs 30

# 3) SAC fine-tune (BC warm-start)
python source/example7/scripts/train.py --num_envs 64 --timesteps 3000000

# 4) 이어서 학습
python source/example7/scripts/train.py --resume --timesteps 1000000

# 5) 재생
python source/example7/scripts/play.py
```

## 알고리즘 결정 사항

| 항목 | 선택 | 이유 |
|------|------|------|
| IL | **BC** | expert(PPO)가 sim에 항상 있음 → DAgger/GAIL 불필요. 단순/빠름. distribution shift는 SAC fine-tune에서 자연 교정. |
| RL | **SAC** | off-policy → 데모를 replay buffer에 그대로 활용 가능 (PPO는 on-policy라 BC 효과 빨리 희석). entropy term으로 부드러운 정책 → sim2real 유리. |
| obs 정규화 | **PPO의 VecNormalize 통계 고정** | BC와 fine-tune이 같은 obs 분포를 보도록. `env.training=False`. |
| reward 정규화 | **꺼둠** | SAC는 entropy bonus와 결합돼 raw reward가 더 안정. (norm_reward=False) |

## 파일 구조

```
source/example7/
  CLAUDE.md                       # 이 파일
  demos/
    example5_demos.pkl            # 수집된 expert 에피소드들 (collect_demos가 생성)
  scripts/
    collect_demos.py              # PPO rollout → trajectory pkl
    train_bc.py                   # BC pretrain → SAC actor + replay prefill
    train.py                      # SAC fine-tune (BC warm-start)
    play.py                       # 학습된 SAC 정책 재생
  tasks/lift/
    lift_env.py                   # LiftEnv (example5 복사 — 동일 환경 유지)
    lift_env_cfg.py               # LiftEnvCfg (example5 복사)
```

env는 example5와 완전히 동일하다. obs/action 형식이 같아야 BC가 의미 있음.

## demos 형식

`demos/example5_demos.pkl` = `list[dict]`. 각 dict가 한 에피소드:

| 키 | shape / 타입 | 설명 |
|----|--------------|------|
| `obs` | (T, 34) float32 | **VecNormalize 적용된** 정규화 obs |
| `next_obs` | (T, 34) float32 | 다음 스텝 정규화 obs |
| `actions` | (T, 7) float32 | PPO deterministic action (-1~1) |
| `rewards` | (T,) float32 | raw reward (norm_reward=False) |
| `dones` | (T,) bool | episode 종료 플래그 |
| `success` | bool | 성공 여부 (lift_height_threshold 통과) |
| `length` | int | 에피소드 길이 |

## BC 학습 손실

SB3 SAC actor에서 `get_action_dist_params(obs)` → `mean_actions, log_std`. squashed mean = `tanh(mean_actions)`을 expert action에 MSE.

```python
mean_actions, _, _ = actor.get_action_dist_params(obs)
loss = F.mse_loss(torch.tanh(mean_actions), expert_action)
```

log_std는 BC에서 학습 안 함 (SAC fine-tune이 entropy term으로 자동 조정).

## SAC 하이퍼파라미터 (기본값)

| 파라미터 | 값 |
|----------|-----|
| learning_rate | 3e-4 |
| buffer_size | 1,000,000 |
| batch_size | 512 |
| tau | 0.005 |
| gamma | 0.99 |
| train_freq | 1 |
| gradient_steps | 1 |
| ent_coef | "auto" |
| learning_starts | 0 (replay에 demo prefill됐으므로 즉시 학습 시작) |

## 체크포인트 우선순위 (train.py)

1. `--checkpoint <path>` 직접 지정
2. `--resume` 시 `checkpoints/example7.zip`
3. 둘 다 없으면 `checkpoints/example7_bc.zip` (BC warm-start)
4. 그것도 없으면 신규 SAC

replay buffer는 같은 이름의 `_replay.pkl` → 없으면 `bc_replay` 순서로 시도.

## 주의사항

- `collect_demos.py`는 example5의 환경/체크포인트(`omy_lift.zip`, `omy_lift_vecnorm.pkl`)에 의존. 이름이 다르면 `--checkpoint`/`--vecnorm`으로 직접 지정.
- env 코드는 example5와 동기화 유지가 중요. example5 lift_env.py를 수정하면 example7도 같이 갱신해야 BC obs 분포가 깨지지 않음.
- BC prefill된 transition은 PPO가 만든 것이라 SAC entropy 분포와 약간 다름. fine-tune 초반 몇십만 step은 ent_coef 자동 조정으로 알아서 안정됨.
