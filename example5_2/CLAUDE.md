# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 개요

OMY 로봇 팔(6DOF + gripper)이 박스를 집어 들어올리는 Lift 태스크를 Isaac Lab + Stable Baselines3 PPO로 학습하는 프로젝트.

**example5와의 차이 (example5_2)**: OMY USD가 gripper mimic 정보를 잃어버린 채 변환되어 있어, 4관절(rh_r1, rh_r2, rh_l1, rh_l2)이 독립적으로 움직이는 문제 발견 (example7 학습 중 박스가 비대칭으로 잡히는 원인). example5_2 에서는 코드 레벨 mimic 강제 패치 적용:

1. **Limit 강제** ([lift_env.py:67-68, 132-138](tasks/lift/lift_env.py#L132-L138)): URDF 사양 `[0.0, 1.135]`로 4관절 모두 clamp (USD 일부 관절 limit이 1.37까지 허용되는 문제 해결)
2. **Mimic target + write_joint_state 동기화** ([_apply_action](tasks/lift/lift_env.py#L296-L310)): 매 step에 master(rh_r1_joint)의 actual joint position을 r2/l1/l2에 강제 복사 (set_target만으로는 PD가 박스 반발력에 의해 어긋남 → write_joint_state 직접 동기화)
3. **초기 상태 동기화** ([_reset_idx](tasks/lift/lift_env.py#L728-L731)): reset 시 4관절 모두 master 값으로 통일

→ 박스가 더 대칭적/안정적으로 잡히도록 (handoff 데이터셋 품질 ↑, example7 학습 안정성 ↑).

## ⚠️ USD/URDF 진단 (2026-05-04)

**URDF**: 정확함. mimic 명시(`multiplier=1`), limit `[0.0, 1.135]`, axis 명시.
**USD**: ROBOTIS 공식 OMY.usd가 mimic 구조 손실됨. GitHub URL과 현재 파일 MD5 hash 동일 (즉 공식 = 깨진 상태).

**시도한 해결책들**:
1. ✅ **코드 mimic 패치** (현재 적용): `_apply_action`에서 매 step `write_joint_state_to_sim`로 r2/l1/l2를 master에 강제. 약 95% 효과 — fingertip 거의 평행. 완벽한 4-bar linkage는 아님.
2. ❌ **URDF→USD 재변환** (`omy_f3m_abs.urdf` 생성): mesh 경로를 절대경로로 변환. Isaac Sim 4.5 importer는 mimic은 인식했지만 link 강체 결합이 깨져 gripper 부품이 분리되어 floating. 사용 불가.
3. ❌ **Stiffness 폭증** (1500 → 10000): PD가 박스 반발력 압도하려 했으나 부족. `write_joint_state` 강제가 더 효과적.

**결론**: ROBOTIS 공식 OMY.usd + 코드 mimic 패치가 현실적 최선. USD 재변환은 더 안 좋아짐.

## 적용된 물리/태스크 파라미터

[omy_robot_cfg.py](../omy/omy_robot_cfg.py):
- gripper actuator: stiffness 1500, effort_limit 400, damping 60 (기본 500/150/30 → 강화)

[lift_env_cfg.py](tasks/lift/lift_env_cfg.py):
- friction (static/dynamic): 3.0/3.0 (기본 1.0)
- grasp_target_z_offset: 0.02 (기본 0.04 → 박스 정중앙 잡기 유도, 단 바닥 충돌 회피 위해 살짝 위)
- lift_height_threshold: 0.2 (변경 없음, success 판정)

[lift_env.py:678-686](tasks/lift/lift_env.py#L678) reset 조건:
- fallen 임계 0.3 → **0.1** (84°까지 허용)
- fallen reset은 박스가 **바닥 근처(`obj_height < 0.07`)일 때만** 발동 (공중 회전은 허용 — 들어올리는 도중 박스 회전 안 죽음)

## Reward 튜닝 history (2026-05-04)

| ver | 변경 | 결과 |
|---|---|---|
| v1 | 원본 example5 + mimic 패치만 | success 0.003 (가끔 lift) |
| v2~v4 | center_grasp_reward, gripper 강화 등 추가 | lift 줄어듦 (center_grasp이 lift 방해) |
| v5 | center_grasp 제거 | fingers_near 0.51 |
| v6 | threshold 0.4, lift_reward 150 | 변화 없음 |
| v7 | 닫기 보상 강화 | "박스 옆서 닫기만" exploit |
| v8 | near_object 5cm + fingers_near reward | fingers_near 0.80, lift 시작 |
| v9 | horizontal_reward 20 | "박스 멀리서 손가락 수평" exploit |
| v10 | horizontal 제거, box_upright × grasp 추가 | best까지 (lift 0.012, 76% 잡고 똑바로) |
| v11~v14 | inner_xy_align 추가, tight_grip 시도 | 다양한 local optimum |
| v15 | tight_grip = xy×inner_xy×z × closed_smooth × exp(-50*gap²) | 박스 가까이 가지만 닫기 안 함 |
| v16 | box_inside × fingers_near 곱셈 | 학습 신호 0 (실패) |
| v17 | v15 복원 + lift_reward 200 | 변화 없음 (lift 0이라 가중치 무효) |

**현재 활성 reward** ([lift_env.py:639-651](tasks/lift/lift_env.py#L639) 부근):
- xy_align (×15): outer fingertip(r2,l2) 중점 xy 정렬
- inner_xy_align (×5): inner joint(r1,l1) 중점 xy 정렬 — 그리퍼 수직 자세
- z_align (×12): grip_center z 정렬
- horizontal_reward (×0.5): 두 손가락 z 차이 (보조)
- tight_grip (×100): `(xy_align × inner_xy_align × z_align) × closed_smooth × exp(-50*finger_gap²)` — 정렬+닫음+squeeze 통합
- lift_reward (×200): 박스 들어올린 높이
- success_reward (×10000): obj_height > 0.2m
- premature_close_penalty (×-5)
- action_penalty (×-0.001)

## v22~v44 학습 history (2026-05-05/06)

총 **60M+ step PPO**, **23개 reward 변형** 시도. lift task는 끝내 안 풀림 (success 0%).

### 결정적 발견 (v40-v41)

사용자 제안으로 **lift reward 제거** + 잡기 task만 학습 → **그립 학습 성공** (v41 30M):

| 지표 | v22 best | v41 30M |
|---|---|---|
| aligned (xy<2cm & z<2cm) | 0.001 | **0.66** (660배!) |
| in_grasp (continuous) | 0.25 | **0.70** |
| pushing_penalty | 0.62 | **0.0001** (사실상 0) |
| fingers_near | 0.11 | **1.00** |
| finger_gap | 0.024 (push) | 0.086 (real grip) |
| **success** | 0 | 0 |

### v41 reward 핵심 ([lift_env.py 참조](tasks/lift/lift_env.py))

```python
upright_smooth = (upright_score + 1.0) * 0.5
centered_grip = in_grasp * closed_smooth * upright_smooth
close_inside_bonus = closed_smooth * in_grasp
gap_when_grasping = clamp(finger_gap - 0.04, min=0) * in_grasp

reward = (
    + 30.0 * finger_proximity         # 손가락 박스 근처
    + 50.0 * in_grasp                 # 박스가 손가락 중점에
    + 50.0 * centered_grip            # 박스 중심 닫기 + 정자세
    + 100.0 * close_inside_bonus      # 박스 안에서 닫기 강화
    + 10.0 * upright_smooth           # 정자세 유지
    + 10.0 * opened_when_far          # 멀 때 그리퍼 열기
    - 100.0 * gap_when_grasping       # 잡고도 안 닫으면 페널티
    - 0.001 * action_penalty
)
# lift, success 보상 없음 (사용자 제안)
```

### lift 못 푼 이유 (확정)

v42-v44에서 grip 정책에 lift_reward 추가 시도 모두 실패:
- v42 (lift weight 30): aligned 0.66 → 0.29 (regression)
- v43 (lift weight 200): box_lifted 0.81로 잠시 증가했으나 pushing_penalty 130배↑ (push 회귀)
- v44 (lift if in_grasp>0.7 게이트): aligned 0.09로 drop, lift 변화 없음

→ **PPO + 긴 박스(13.9cm) + OMY 그리퍼**로 lift 자연 학습 불가능 결론.

## v45-v46: entropy boost로 첫 success 발견 (2026-05-06)

50M+ step 정체 후 ent_coef 0.005 → 0.05 → 0.02로 조정해서 **첫 success 발화**:

| 시점 | 설정 | 결과 |
|---|---|---|
| v44 40M | ent 0.005 | success 0 |
| v45 40M | ent 0.05 (10x) | **success 0.0001** 🎯 첫 발화 |
| v45 50M | ent 0.05 continued | regression (over-explore) |
| **v46 50M** | ent 0.02 (4x) | **success 0.0001 안정, box_lifted 0.24** ⭐ |
| v46 65M | ent 0.02 continued | regression |

**Pattern**: continued training 5M+은 안정적, 그 이후 drift 발생. v46 50M이 stable peak.

### v46 reward (현재)
v41 grip reward + lift 추가 (continuous in_grasp 게이트):
```python
solid_lift_reward = lift_progress * in_grasp * closed_smooth_for_lift
# in_grasp 곱: 진짜 잡기 중일 때만 lift 보상 (binary 게이트는 학습 끊김)

reward = (
    + 30.0 * finger_proximity
    + 50.0 * in_grasp
    + 50.0 * centered_grip
    + 50.0 * close_inside_bonus
    + 10.0 * upright_smooth
    + 10.0 * opened_when_far
    + 100.0 * solid_lift_reward       # NEW: continuous gate
    + 1000.0 * success_reward
    - 50.0 * gap_when_grasping
    - 0.001 * action_penalty
)
# ent_coef: 0.02 (cfg에서 변경)
```

## handoff 수집 (2026-05-06)

### v41 30M (grip-only)
```bash
python source/example7/scripts/collect_handoff.py \
  --checkpoint checkpoints/example5_2_v41_30M.zip \
  --vecnorm checkpoints/example5_2_v41_30M_vecnorm.pkl \
  --num_samples 100 --capture_height 0.06 --upright_threshold 0.95 \
  --num_envs 32 --output checkpoints/handoff_states_v41.npz --headless
```
→ 100 samples, obj_z=0.060m (모두 바닥 근처), upright 0.998

### v46 50M (grip + occasional lift) ⭐
```bash
python source/example7/scripts/collect_handoff.py \
  --checkpoint checkpoints/example5_2_v46_50M.zip \
  --vecnorm checkpoints/example5_2_v46_50M_vecnorm.pkl \
  --num_samples 100 --capture_height 0.06 --upright_threshold 0.95 \
  --num_envs 32 --output checkpoints/handoff_states_v46.npz --headless
```
→ 100 samples, obj_z 0.060~0.100m (**일부 4cm 들기!**), upright 0.989

## 다음에 할 일

**현실적 path: example7로 lift+place 통합 학습**:
1. example7에서 `handoff_states_v46.npz` 로드 (다양한 시작 상태)
2. example7 reward에 lift task 추가 (현재는 place 위주)
3. SAC+HER로 학습 (PPO보다 sparse reward 강함)
4. handoff에 다양한 lift 단계 포함되어 있어 example7이 lift continuation 학습 가능

**대안: example5_2 추가 학습 (조심스럽게)**:
- v46 50M에서 5M씩만 추가 학습 (그 이상은 regression)
- 또는 lr 감소 + 추가 학습
- success rate 0.01% → 1%까지는 가능성 있음

**중요 backup 체크포인트**:
- `example5_2_v46_50M.zip` — **best (현재 example5_2.zip)**, success 0.01%, box 4cm 들기 가능
- `example5_2_v45_40M.zip` — 첫 success 발견 지점
- `example5_2_v41_30M.zip` — best pure grip (no lift)
- `example5_2_v22_15M.zip` — best partial push lift (1.7cm)
- `handoff_states_v46.npz` — 100 samples (best quality, 일부 4cm lift)
- `handoff_states_v41.npz` — 100 samples (pure grip, all at ground)

## handoff 수집 (example7 연결)

example5_2 lift policy가 충분히 잘 되면:
```bash
./isaaclab.sh -p source/example7/scripts/collect_handoff.py \
  --checkpoint checkpoints/example5_2.zip \
  --vecnorm checkpoints/example5_2_vecnorm.pkl \
  --num_samples 100 --upright_threshold 0.95 --headless
```

→ `checkpoints/handoff_states.npz`에 example7용 초기 상태 저장.

## 학습 및 실행 커맨드

```bash
# 신규 학습 (256 envs, 500만 스텝)
python source/example5_2/scripts/train.py --num_envs 256 --timesteps 5000000

# 이어서 학습 (기본 체크포인트: checkpoints/example5_2.zip)
python source/example5_2/scripts/train.py --resume --timesteps 5000000

# 특정 체크포인트에서 이어서 학습
python source/example5_2/scripts/train.py --checkpoint checkpoints/example5_step1000000.zip --timesteps 2000000

# 학습된 정책 재생 (시각화)
python source/example5_2/scripts/play.py --checkpoint checkpoints/example5_2.zip --vecnorm checkpoints/example5_2_vecnorm.pkl

# 저장 없이 학습 (테스트용)
python source/example5_2/scripts/train.py --no_save --timesteps 1000000
```

## 아키텍처

### 파일 구조
```
source/example5/
  scripts/
    train.py          # PPO 학습 루프, TrainCallback, 체크포인트 로드/저장
    play.py           # 학습된 정책 재생 (시각화)
  tasks/lift/
    lift_env.py       # 환경 구현 (DirectRLEnv 상속)
    lift_env_cfg.py   # 설정 dataclass (하이퍼파라미터 포함)

source/omy/
  omy_robot_cfg.py    # OMY 로봇 USD/actuator 설정 (OMY_CFG, OMY_OFF_SELF_COLLISION_CFG)

source/omy_f3m_urdf/
  OMY.usd             # Isaac Lab에서 직접 로드하는 USD 파일
  omy_f3m.urdf        # 참고용 URDF (xacro에서 생성된 것)
```

### 환경 구조 (LiftEnv)

`LiftEnv`는 Isaac Lab의 `DirectRLEnv`를 상속하며, 아래 흐름으로 동작한다:

```
reset → _setup_scene → _reset_idx
  ↓ (매 스텝)
_pre_physics_step   # action → robot_dof_targets 계산
_apply_action       # 물리 엔진에 position target 전달
_compute_intermediate_values  # 공통 중간값 캐싱 (grip center, obj pos 등)
_get_observations   # 34차원 obs 벡터 조립
_get_rewards        # 다단계 reward 계산
_get_dones          # 종료 조건 판정
```

### 로봇 제어 방식

- **Action space**: 7차원 `[-1, 1]` — arm joint 6개 + gripper 1개
- **Position control**: `target += speed_scale × dt × action × action_scale`
- **Gripper**: 4개 관절(rh_r1_joint, rh_r2, rh_l1, rh_l2)을 동일 명령으로 제어 (mimic 구조 모사), action index 6에 ×3 배율 적용
- `actions_to_dof()`: 7차원 action을 전체 관절 수 크기 텐서로 변환 (미제어 관절은 0)

### Observation (34차원)

| 항목 | 차원 | 설명 |
|------|------|------|
| dof_pos_scaled | 10 | arm 6 + gripper 4 관절 위치 (-1~1 정규화) |
| dof_vel_scaled | 10 | arm 6 + gripper 4 관절 속도 |
| obj_pos_rel | 3 | 물체의 환경 원점 기준 상대 위치 |
| obj_to_grip | 3 | grasp_target → grip_center 벡터 |
| left_to_obj_vec | 3 | 왼손가락 끝 → 물체 벡터 |
| right_to_obj_vec | 3 | 오른손가락 끝 → 물체 벡터 |
| gripper_close_state | 1 | gripper 닫힘 정도 (0=열림, 1=닫힘) |
| to_lift_target | 1 | 목표 높이까지 남은 z 거리 |

### Reward 구조

단계적 dense reward 설계:

1. **xy_align_reward** (×15): `exp(-40 * xy_dist²)` — gripper가 물체 XY에 정렬
2. **z_align_reward** (×12): `exp(-60 * z_dist²)`, XY 10cm 이내일 때만 활성화
3. **horizontal_reward** (×0.5): 두 손가락의 z좌표 차이를 최소화
4. **close_reward** (×20): XY/Z 정렬 + aligned 상태에서 gripper 닫기
5. **grasp_close_reward** (×30): 물체 15cm 이내에서 gripper 닫기 보상
6. **lift_reward** (×500): 물체가 초기 높이에서 올라간 만큼 (fingers_near & closed_enough 조건)
7. **premature_close_penalty** (×-5): 물체와 멀 때 gripper 닫으면 패널티
8. **action_penalty** (×-0.001): 큰 action에 L2 패널티

종료 조건: 물체 높이 > `lift_height_threshold`(0.5m) 성공 / 물체 추락(<-0.1m) / 물체 넘어짐(기울기 72도 초과) / 시간 초과(12초).

### 체크포인트 관리

- 저장 위치: `checkpoints/` 디렉터리
- 중간 저장: 500,000 session_step마다 `{name}_step{total_step}.zip` + `_vecnorm.pkl`
- **session_step**(이번 실행 기준)과 **total_step**(SB3 누적)을 분리 관리 — resume 시에도 진행률이 정확하게 표시됨
- `VecNormalize`(obs/reward 정규화) 통계는 반드시 모델과 함께 저장/로드해야 함

### 주요 설정값 (LiftEnvCfg)

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| num_envs | 256 | 병렬 환경 수 |
| decimation | 2 | 물리 120Hz, 제어 60Hz |
| episode_length_s | 12.0초 | 에피소드 최대 길이 |
| object.init_state.pos | (0.45, -0.10, 0.06) | 물체 초기 위치 (로봇 앞 45cm) |
| object_pos_noise | ±0.02m | 에피소드마다 물체 위치 랜덤화 |
| grasp_target_z_offset | 0.04m | 물체 중심보다 4cm 위를 잡도록 유도 |
| lift_height_threshold | 0.5m | 성공 판정 높이 |

### OMY 로봇 설정 (omy_robot_cfg.py)

- USD 경로: `/home/jaewoo/IsaacLab/source/omy_f3m_urdf/OMY.usd` (절대경로 하드코딩)
- Arm actuator: joint1-2 (stiffness 350, damping 25), joint3-6 (stiffness 300, damping 20)
- Gripper actuator: stiffness 500, damping 30 (물체 파지 시 밀림 방지를 위해 높게 설정)
- 자기충돌 비활성화(`enabled_self_collisions=False`)
- `OMY_OFF_SELF_COLLISION_CFG`: `LiftEnvCfg`에서 실제로 사용하는 설정 (명시적으로 self-collision off)

### Gripper Mimic 패치 (example5_2 핵심 차이)

**문제**: `OMY.usd`에서 URDF의 mimic 정보가 손실됨. URDF는 `rh_r2/l1/l2`가 모두 `rh_r1_joint`를 multiplier=1로 mimic하도록 명시되어 있고 limit는 `[0.0, 1.135]`로 동일. 하지만 USD는 4관절이 독립적으로 작동하고 일부 limit가 `1.37`까지 허용되는 등 깨진 상태.

**증상**: 같은 `grip_cmd`를 4관절에 보내도 위치가 제각각 (예: r1=0.37, r2=0.82, l1=0.48, l2=1.31). 결과로 박스가 비대칭으로 잡혀 약 21° 기울어진 상태로 들림.

**패치 (코드 레벨)**:
1. **Limit 강제** ([lift_env.py:67-68](tasks/lift/lift_env.py#L67), [132-138](tasks/lift/lift_env.py#L132)):
   - `soft_joint_pos_limits`를 clone해서 mutable로 만들고
   - 4 grip joint 모두 `[0.0, 1.135]`로 통일
2. **Mimic target 동기화** ([_pre_physics_step](tasks/lift/lift_env.py#L265)):
   - `actions_to_dof + clamp` 후 master target 값을 `mimic_gripper_joint_ids` 에 복사
3. **초기 상태 동기화** ([_reset_idx](tasks/lift/lift_env.py#L728)):
   - `default_joint_pos`를 사용해 초기화한 직후, 4 grip joint 모두 master 값으로 통일

**검증 방법**: 학습 후 박스 들고 정지 상태에서 `joint_pos` 확인 — r1, r2, l1, l2가 모두 동일한 값에 수렴하면 mimic 정상.

### reward_log

`LiftEnv.reward_log` 딕셔너리에 매 스텝 reward 항목별 값이 저장되며, `TrainCallback._on_step()`이 읽어서 100,000 스텝마다 콘솔에 출력한다.
