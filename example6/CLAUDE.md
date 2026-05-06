# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 태스크 개요

**Pick-and-Place to Grid Cell**

OMY 로봇 팔이 물체를 집어 들어올린 뒤(example5 담당), **3×3 그리드의 특정 셀 안에 수직으로 내려놓는** 태스크.

- example5(Lift)와 계층 구조로 동작 — example5가 물체를 20cm 높이로 들어올리면, example6(Place) policy가 이어받아 타겟 셀로 이동 후 내려놓음
- 추론 시 두 policy를 순서대로 실행: `object_height > threshold → example5→example6 전환`
- 향후 실제 로봇(sim2real) 배포 계획 있음

---

## 알고리즘: SB3 + SAC + HER

### 선택 이유

| 선택 | 이유 |
|------|------|
| **SAC** (Soft Actor-Critic) | off-policy → sample efficient, 정밀 manipulation에 강함, sim2real 시 부드러운 policy 생성 |
| **HER** (Hindsight Experience Replay) | 타겟 셀 미스 실패 에피소드를 "다른 목표 성공"으로 재활용 → goal-conditioned 태스크 필수 |
| **SB3** | SAC+HER 조합이 manipulation 태스크에서 가장 검증됨, example5 인프라 재사용 가능 |

### HER 사용 시 observation 형식

SB3의 HER은 `GoalEnv` 형식을 요구함:

```python
{
    "observation": np.array(...),      # 실제 obs 벡터
    "achieved_goal": np.array(...),    # 현재 물체 위치 (x, y)
    "desired_goal": np.array(...),     # 타겟 셀 중심 위치 (x, y)
}
```

reward 함수도 `(obs, achieved_goal, desired_goal, info)` 시그니처로 구현해야 함.

---

## example5와의 관계 (계층 구조)

```
에피소드 시작
    └─ example5 policy 실행 (Lift)
           └─ 물체 높이 > 20cm 달성
                  └─ example6 policy 실행 (Place)
                         └─ 타겟 셀 위에서 물체 내려놓기
                                └─ gripper open → 성공 판정
```

### example6 초기 상태 (Handoff State)

- 물체가 그리퍼에 잡혀 있는 상태, 높이 **20cm** (example5의 `lift_height_threshold`와 동일)
- **Handoff randomization 적용** (sim2real 핵심): 초기 물체 위치, 관절 자세에 노이즈 추가
  - 물체 그리퍼 내 위치 노이즈: ±2cm
  - 팔 관절 노이즈: ±0.05 rad
  - 이유: 실제 로봇은 매 grasp마다 자세가 달라지므로, example6이 다양한 handoff 자세에서도 동작해야 함

---

## 그리드 설계

### 물체 크기 (example5 기준)

- 박스 크기: **0.139m × 0.044m × 0.118m** (x × y × z)
- 그리퍼가 y축 방향으로 물체를 잡음 → 내려놓을 때 footprint = x × y = 13.9cm × 4.4cm

### 셀 크기 (살짝 여유)

- 셀 내부 x: **0.17m** (13.9 + 3.1cm 여유)
- 셀 내부 y: **0.06m** (4.4 + 1.6cm 여유)
- 격벽 두께: **0.008m** (8mm)
- 격벽 높이: **0.12m** (물체 높이 11.8cm와 거의 동일 → 마지막에 살짝 떨어뜨려야 함)

### 그리드 구조 (3×3)

- 셀 피치(x): 0.17 + 0.008 = 0.178m
- 셀 피치(y): 0.06 + 0.008 = 0.068m
- 전체 크기: x ≈ 0.55m, y ≈ 0.21m
- 격벽은 Isaac Lab `CuboidCfg`로 프로그래밍 방식으로 생성 (별도 USD 없음)

### 그리드 위치

- OMY 팔을 **오른쪽으로 최대 뻗었을 때** 마지막 셀에 닿는 위치에 배치
- 그리드 중심 (환경 원점 기준, 시뮬레이션 검증 후 조정 필요):
  - `grid_center_x ≈ 0.25m` (로봇 약간 앞)
  - `grid_center_y ≈ -0.45m` (오른쪽, 최대 reach 근처)
- 향후 19×19로 확장 시: 그리드가 로봇 오른쪽 넓게 펼쳐지는 구조

### 셀 채우는 순서

고정 순서: **row-major (왼쪽→오른쪽, 앞→뒤)**

```
[0,0] [0,1] [0,2]
[1,0] [1,1] [1,2]
[2,0] [2,1] [2,2]
```

---

## 내려놓기 방법

- **수직으로만 내려놓음** (수평 이동 후 수직 하강)
- 격벽 높이 = 물체 높이이므로 **격벽 바로 위까지 이동 후 살짝 떨어뜨리는 방식**
- 물체가 셀 XY 범위 내에 위치 + gripper open 상태에서 물체가 안정적으로 정지하면 성공
- 성공 판정: 물체 중심이 셀 내부에 있고 + 속도 ≈ 0 + gripper open

---

## Observation 설계 (GoalEnv 형식)

### observation 벡터

| 항목 | 차원 | 설명 |
|------|------|------|
| arm joint pos (정규화) | 6 | joint1~6 위치 (-1~1) |
| arm joint vel | 6 | joint1~6 속도 |
| gripper close state | 1 | 0=열림, 1=닫힘 |
| object pos (env 기준) | 3 | 물체 현재 위치 |
| object vel | 3 | 물체 속도 |
| end_effector pos | 3 | 엔드이펙터(그리퍼 중심) 위치 |
| object to target | 3 | 물체 → 타겟 셀 벡터 |

### achieved_goal / desired_goal

- `achieved_goal`: 물체 현재 XY 위치 `[x, y]`
- `desired_goal`: 타겟 셀 중심 XY 위치 `[x, y]`
- HER이 이 둘을 이용해 실패 에피소드를 재활용

---

## Reward 설계

HER과 함께 **sparse reward** 기반 (HER relabeling이 sparse에서 가장 효과적):

```python
def compute_reward(achieved_goal, desired_goal, info):
    dist_xy = norm(achieved_goal[:2] - desired_goal[:2])
    in_cell = dist_xy < cell_tolerance  # 셀 내부 판정
    return 0.0 if in_cell else -1.0     # sparse: 성공 0, 실패 -1
```

추가 dense shaping (선택적):
- **XY approach**: 물체가 타겟 셀 XY에 가까울수록 보상
- **height reward**: 셀 위에서 물체를 충분히 낮추면 보상
- **gripper open penalty**: 셀 밖에서 gripper 열면 패널티
- **action smoothness**: 급격한 action 변화 패널티

---

## 종료 조건

| 조건 | 타입 |
|------|------|
| 물체가 타겟 셀 내부에 안정적으로 안착 (10 스텝 유지) | terminated (성공) |
| 물체 낙하 (z < -0.05m) | terminated (실패) |
| 그리드 경계 밖으로 이탈 | terminated (실패) |
| 물체 기울어짐 (`upright_score < 0.3`) | terminated (실패, reset만) |
| 그리퍼가 물체와 0.15m 이상 멀어짐 | terminated (실패, reset만) |
| 에피소드 시간 초과 (10초) | truncated |

패널티 없이 reset만 합니다 (사용자 요청).

---

## Curriculum Stage

학습 난이도 단계적 분리 (`PlaceEnvCfg.curriculum_stage`):

### Stage 1 — 상자 위에 호버링
- 성공 조건: `xy_dist < 0.03m` AND `0.15m < obj_z < 0.30m`
- gripper는 **닫힘 유지**가 기본 (열면 패널티)
- 목적: 타겟 셀 위치로 팔을 정확하게 이동하는 법 먼저 학습

### Stage 2 — 셀에 실제로 넣기
- 성공 조건: `in_cell_xy & stable & gripper_open & (z_dist < 0.05)`
- 현재 설계된 full placement

**전환 기준**: stage1_success_rate > 0.5 연속 2 iteration 이상 유지 시 stage 2로 전환.

---

## Autonomous Loop 구조

파일들:
- `source/example6/autonomous_loop_prompt.md` — /loop 프롬프트 (실행 프로토콜)
- `source/example6/loop_state.json` — 상태 (iteration, stage, best, history)
- `source/example6/scripts/parse_log.py` — 학습 로그 → JSON metrics 파싱
- `source/example6/logs/iter_XXX.log` — iteration별 학습 stdout
- `checkpoints/example6.zip` — 최신 체크포인트
- `checkpoints/example6_best.zip` — 최고 성능 체크포인트

매 iteration에서 Claude는:
1. `loop_state.json` 읽고 현재 상태/iter/stage 파악
2. 최근 metrics 보고 변경사항 결정 (reward 가중치, lr, etc.)
3. 코드 수정 → 학습 실행(timeout 180~300s) → 로그 파싱
4. state 업데이트, best 체크포인트 유지
5. 다음 iteration

종료:
- `iteration >= max_iterations` (30)
- `best_success_rate > 0.5` + 최근 3 iter 증가 추세 (80% 확신)

---

## 학습 및 실행 커맨드

```bash
# 신규 학습
python source/example6/scripts/train.py --num_envs 64 --timesteps 3000000

# 이어서 학습
python source/example6/scripts/train.py --resume --timesteps 2000000

# 재생 (시각화)
python source/example6/scripts/play.py --checkpoint checkpoints/example6.zip --vecnorm checkpoints/example6_vecnorm.pkl

./isaaclab.sh -p source/example6/scripts/play.py --checkpoint checkpoints/example6_best.zip --vecnorm checkpoints/example6_best_replay.pkl 
```

SAC는 PPO보다 적은 병렬 환경(64~128)으로도 잘 동작함 (off-policy라 replay buffer 사용).

---

## 파일 구조

```
source/example6/
  CLAUDE.md                           # 이 파일
  scripts/
    train.py                          # SAC+HER 학습 스크립트
    play.py                           # 학습된 policy 재생
  tasks/place/
    place_env.py                      # PlaceEnv (DirectRLEnv + GoalEnv 믹스인)
    place_env_cfg.py                  # PlaceEnvCfg (설정 dataclass)
    grid_builder.py                   # 3×3 그리드 USD 프로그래밍 생성 유틸
```

---

## 미래 확장 계획

| 단계 | 내용 |
|------|------|
| **현재** | 3×3 그리드, 물체 잡은 방향 그대로 내려놓기, obs로 타겟 위치 입력 |
| **단계 2** | 물체를 뒤집어서 놓기 (바닥면이 위로 가게 → 관절 회전 추가) |
| **단계 3** | 외부 카메라로 타겟 셀 위치 수신 (obs의 desired_goal 대체) |
| **단계 4** | 19×19 그리드로 확장 |
| **단계 5** | 실제 로봇 배포 (sim2real, handoff randomization으로 준비됨) |

---

## 핵심 설계 결정 이유 (Why)

- **HER 사용**: 타겟 셀 위치가 매 에피소드마다 달라지는 goal-conditioned 태스크 → HER 없이는 sparse reward 학습 거의 불가
- **SAC (PPO 아님)**: 정밀 placement는 on-policy PPO보다 off-policy SAC가 sample efficient, sim2real 시 부드러운 동작
- **격벽 높이 = 물체 높이**: 실제 박스 보관 구조 모사, 셀에 넣으면 격벽이 쓰러짐 방지
- **Handoff randomization**: example5의 실제 grasp 자세가 매번 다를 수 있으므로, example6 훈련 시 다양한 초기 자세에서 시작해야 real robot에서도 동작
- **수직 하강만 허용**: 격벽 충돌 방지, 실제 물류 로봇 동작 방식 모사
- **row-major 순서**: 가장 단순하고 예측 가능, 나중에 외부 명령으로 교체 용이

---

## OMY 로봇 설정

example5의 `OMY_OFF_SELF_COLLISION_CFG` 동일하게 사용:
- USD: `/home/jaewoo/IsaacLab/source/omy_f3m_urdf/OMY.usd`
- Arm: joint1-6, Gripper: rh_r1_joint, rh_r2, rh_l1, rh_l2
- 자기충돌 비활성화
