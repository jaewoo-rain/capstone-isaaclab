# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 태스크 개요

**Pick-and-Place + Insert to Grid Cell** (example7)

example7(Place)에서 보상함수만 새로 설계한 버전. 코드 구조/알고리즘/파일명은 동일.

OMY 로봇 팔이 물체를 집어 들어올린 뒤(example5 담당), **3×3 그리드의 특정 셀 안에 정확히 정렬해서 8cm 깊이로 삽입한 후 release 하는** 태스크.

- example5(Lift)와 계층 구조로 동작 — example5가 물체를 20cm 높이로 들어올리면, example7(Place+Insert) policy가 이어받아 타겟 셀에 정렬 후 삽입
- 추론 시 두 policy를 순서대로 실행: `object_height > threshold → example5→example7 전환`
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
                  └─ example7 policy 실행 (Place)
                         └─ 타겟 셀 위에서 물체 내려놓기
                                └─ gripper open → 성공 판정
```

### example7 초기 상태 (Handoff State)

- 물체가 그리퍼에 잡혀 있는 상태, 높이 **20cm** (example5의 `lift_height_threshold`와 동일)
- **Handoff randomization 적용** (sim2real 핵심): 초기 물체 위치, 관절 자세에 노이즈 추가
  - 물체 그리퍼 내 위치 노이즈: ±2cm
  - 팔 관절 노이즈: ±0.05 rad
  - 이유: 실제 로봇은 매 grasp마다 자세가 달라지므로, example7이 다양한 handoff 자세에서도 동작해야 함

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

## ⚠️ 현재 Reward 설계 (2026-05-04 기준, 활발히 튜닝 중)

아래 "원래 3-phase 설계"는 초기 의도였고, 학습 과정에서 여러 차례 재구성되었습니다. 현재 활성 reward는 [tasks/place/place_env.py:425-490](tasks/place/place_env.py#L425) 참고. 핵심 항목:

| 항목 | 가중치 | 역할 |
|---|---|---|
| T1 `r_xy_close = exp(-25*xy²)` | **300** | XY 정렬 (가우시안, 곱셈 게이팅 제거) |
| T1b `r_z_close = exp(-30*(z-0.30)²)` | 50 | Z 목표 높이 0.30m |
| T3 `endpoint_aligned * upright * gated` | 80 | 끝점 정렬 (T1×upright≥0.8 일 때만) |
| T4 `success` (xy_aligned & on_floor) | 500 | 성공 |
| T5 `grip_near_obj = exp(-50*dist²)` | 10 | 잡기 유지 |
| T6 `keep_closed` (near_obj & closed) | 5 | 그리퍼 닫혀있기 |
| T7 `r_upright = clamp(upright_score, 0, 1)` | 150 | 수직 유지 (continuous) |
| T8 `not_tilted` (≤10°, upright>0.985) | +100 | 거의 완벽 수직 보너스 |
| T9 `severe_tilt` (>30°, upright<0.85) | **-100** | 30° 넘게 기울면 페널티 |
| T10 `safe_height` (~aligned & 0.20≤z≤0.40) | +30 | 위에서 접근 보상 |
| T11 `action_penalty = sum(action²)` | -0.5 | 부드러운 움직임 |

**Reset 조건** ([_get_dones](tasks/place/place_env.py#L596-L630)):
- `success_stable` — 성공 + 1 step 유지
- `tilted_on_floor` — `upright<0.85` AND on_floor (공중 회전은 허용)
- `severely_tilted` — `upright<0` (180° 가까이 뒤집힘)
- `abandoned_active` — `grip_to_obj_dist > 0.20m` AND NOT (on_floor AND xy_dist<5cm) (셀 안 정확히 떨어뜨리면 grip 떨어져도 OK = release)
- `out_of_bounds`, `fallen_below`, `truncated`

**핵심 cfg 파라미터**:
- `cell_tolerance: 0.025` (success 판정 xy 임계, 2.5cm)
- `tilt_upright_threshold: 0.85` (~30°, T9 페널티/reset 임계)
- `bonus_upright_threshold: 0.985` (~10°, T8 보너스 임계)
- `abandoned_dist_threshold: 0.20` (그리퍼-박스 20cm 임계)

## 학습 진행 history (v1~v10)

| ver | 변경 | 결과 |
|---|---|---|
| v1~v4 | upright reward 도입, T7 가중치 튜닝 | upright 0.55→0.86 |
| v5 | T7+T8 콤보, abandoned 게이팅 | upright 큰 개선, 0.71→0.86 |
| v6 | clean handoff (upright>0.95) 100개 | abandoned 30%↓, upright 약간 ↓ |
| v7 | T1 곱셈 게이팅(×upright), k=50 | xy_close 0.42→0.05 (역효과) |
| v8 | T1 게이팅 제거, k=25, abandoned 강화 | **첫 success 16번** (cumulative) |
| v9 | T10 페널티 도입 | xy_aligned 후퇴 |
| v10 | T10 페널티→보상 형태, T11 추가, z [0.20, 0.40] 범위 | on_floor 7배↑, success 회귀 (16→0) |

**근본 원인 발견**: USD에서 gripper mimic 정보 손실 → 4관절(r1, r2, l1, l2) 독립 작동 → 박스 비대칭 grasp → 학습 발목.

**해결 진행 중**: example5_2 에서 코드 레벨 mimic 패치 → handoff 재수집 → example7 재학습 예정.

---

## (참고) 원래 3-phase 설계 (현재는 미사용)

example7과 다른 핵심: **curriculum stage 사용 안 함**. 모든 phase가 한 reward 함수 안에서 phase gating으로 활성화됨.

### 3 Phase 정의

**Phase 1 — 적재 위치 위로 들어올림**
- 조건: `xy_dist < phase1_xy_tolerance(0.05m)` AND `obj_z >= cell_top + obj_height + 5cm`
- 즉 셀 격벽 위에 충분히 높이 들어올려서, 격벽에 부딪히지 않고 진입할 준비
- 물체 z 기준값 = `wall_height(0.12) + obj_h(0.118) + lift_clearance(0.05) = 0.288m`

**Phase 2 — yaw 정렬**
- 두 점 매칭: 물체 중심 + 카메라쪽 끝점 ↔ 셀 중심 + 대응 끝점
- 카메라쪽 끝점 = 그리퍼가 잡고 있는 물체 끝(local +x, `cfg.camera_side_sign`로 flip 가능)
- 셀의 대응 끝점 = `cell_center + (cell_inner_x/2, 0, 0)` (`cfg.target_endpoint_sign`로 flip 가능)
- `endpoint_xy_dist < yaw_tolerance(0.02m)` 이면 yaw 정렬 완료
- Phase 1 done & xy_aligned_tight & yaw_aligned → Phase 2 done

**Phase 3 — 삽입 + release**
- z 좌표 하강 → `insertion_depth = cell_top - obj_bottom` 가 8cm 이상 도달
- 8cm 도달 후 그리퍼 open → 중력으로 나머지 깊이 낙하 ("살짝 놓기")
- 그리퍼가 끝까지 못 넣으니 release로 마무리하는 효과

### sparse compute_reward (HER relabeling)

```python
def compute_reward(achieved_goal, desired_goal, info):
    # 3D 거리 기반 — 셀 안 안착 위치(z=obj_h/2)와 obj 위치
    dist = norm(achieved_goal - desired_goal)
    return 0.0 if dist < cell_tolerance*2 else -1.0
```

### dense shaping 항목

- **xy_approach**: 항상 활성, `exp(-30 * xy_dist²)`
- **xy_linear**: 항상 활성, `-xy_dist`
- **height_ramp**: phase1 유도, z=0.20→0.288 사이 ramp
- **phase1_bonus**: phase1 완료 step 보상
- **endpoint_align**: phase1 done 후 활성, `exp(-50 * endpoint_xy_dist²)`
- **phase2_bonus**: phase2 완료 step 보상
- **descent_progress**: phase2 정렬 후 활성, depth/8cm ramp
- **release_reward**: phase2+8cm 도달 후 gripper open 시 큰 보상
- **premature_open_penalty**: 정렬 안 됐는데 그리퍼 열면 패널티
- **grip_near_obj**: 깊이 도달 전까지 그리퍼와 물체 거리 가까이 유지
- **upright_reward**: 기울지 않게 유지
- **action_penalty**: 급격한 action 패널티

### 주요 Phase Gating 변수

| 변수 | 의미 |
|------|------|
| `above_threshold` | obj_z ≥ 0.288m |
| `xy_aligned_loose` | xy_dist < 0.05m (phase1 기준) |
| `xy_aligned_tight` | xy_dist < cell_tolerance(0.015m) |
| `yaw_aligned` | endpoint_xy_dist < 0.02m |
| `phase1_done` | above_threshold & xy_aligned_loose |
| `phase2_done` | above_threshold & xy_aligned_tight & yaw_aligned |
| `deep_enough` | insertion_depth ≥ 8cm |
| `success` | descent_aligned & deep_enough & gripper_open & stable |

---

## 종료 조건

| 조건 | 타입 |
|------|------|
| 물체가 셀 안에 정렬+삽입+release+stable로 50스텝 유지 | terminated (성공) |
| 물체 낙하 (z < -0.05m) | terminated (실패) |
| 그리드 경계 밖으로 이탈 | terminated (실패) |
| 물체 기울어짐 (`upright_score < 0.3`) | terminated (실패, reset만) |
| 그리퍼가 물체와 0.15m 이상 멀어짐 **AND 8cm 미삽입** | terminated (실패, reset만) |
| 에피소드 시간 초과 (10초) | truncated |

**중요 변경**: abandoned 조건은 `~deep_enough`로 게이팅 — 8cm 삽입 후에는 그리퍼가 멀어져도 (release 후) 종료 안 됨.

---

## Curriculum 미사용 (example7)

example7에서는 stage 1→2 curriculum을 사용했지만, example7은 **단일 reward 함수에 phase gating**으로 통합:
- 모든 phase가 동시에 reward에 포함되며, 이전 phase가 완료될 때만 다음 phase의 shaping 항목이 활성화됨
- `cfg.curriculum_stage` / `stage1_*` 필드는 legacy로 남겨두었으나 reward 계산에는 사용되지 않음
- 자율 loop은 phase1/phase2/deep_enough rate 메트릭으로 진행도 추적

---

## Autonomous Loop 구조

파일들:
- `source/example7/autonomous_loop_prompt.md` — /loop 프롬프트 (실행 프로토콜)
- `source/example7/loop_state.json` — 상태 (iteration, stage, best, history)
- `source/example7/scripts/parse_log.py` — 학습 로그 → JSON metrics 파싱
- `source/example7/logs/iter_XXX.log` — iteration별 학습 stdout
- `checkpoints/example7.zip` — 최신 체크포인트
- `checkpoints/example7_best.zip` — 최고 성능 체크포인트

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
python source/example7/scripts/train.py --num_envs 64 --timesteps 3000000

# 이어서 학습
python source/example7/scripts/train.py --resume --timesteps 2000000

# 재생 (시각화)
python source/example7/scripts/play.py --checkpoint checkpoints/example7.zip --vecnorm checkpoints/example7_vecnorm.pkl

./isaaclab.sh -p source/example7/scripts/play.py --checkpoint checkpoints/example7_best.zip --vecnorm checkpoints/example7_best_replay.pkl 

./isaaclab.sh -p source/example7/scripts/train.py ----timesteps 2400000 --headless


```

SAC는 PPO보다 적은 병렬 환경(64~128)으로도 잘 동작함 (off-policy라 replay buffer 사용).

---

## 파일 구조

```
source/example7/
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
- **Handoff randomization**: example5의 실제 grasp 자세가 매번 다를 수 있으므로, example7 훈련 시 다양한 초기 자세에서 시작해야 real robot에서도 동작
- **수직 하강만 허용**: 격벽 충돌 방지, 실제 물류 로봇 동작 방식 모사
- **row-major 순서**: 가장 단순하고 예측 가능, 나중에 외부 명령으로 교체 용이

---

## OMY 로봇 설정

example5의 `OMY_OFF_SELF_COLLISION_CFG` 동일하게 사용:
- USD: `/home/jaewoo/IsaacLab/source/omy_f3m_urdf/OMY.usd`
- Arm: joint1-6, Gripper: rh_r1_joint, rh_r2, rh_l1, rh_l2
- 자기충돌 비활성화
