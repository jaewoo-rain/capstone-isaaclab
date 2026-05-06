# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 개요

OMY 로봇 팔(6DOF + gripper)이 박스를 집어 들어올리는 Lift 태스크를 Isaac Lab + Stable Baselines3 PPO로 학습하는 프로젝트.

## 학습 및 실행 커맨드

```bash
# 신규 학습 (256 envs, 500만 스텝)
python source/example5/scripts/train.py --num_envs 256 --timesteps 5000000

# 이어서 학습 (기본 체크포인트: checkpoints/example5.zip)
python source/example5/scripts/train.py --resume --timesteps 5000000

# 특정 체크포인트에서 이어서 학습
python source/example5/scripts/train.py --checkpoint checkpoints/example5_step1000000.zip --timesteps 2000000

# 학습된 정책 재생 (시각화)
python source/example5/scripts/play.py --checkpoint checkpoints/example5.zip --vecnorm checkpoints/example5_vecnorm.pkl

# 저장 없이 학습 (테스트용)
python source/example5/scripts/train.py --no_save --timesteps 1000000
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

### reward_log

`LiftEnv.reward_log` 딕셔너리에 매 스텝 reward 항목별 값이 저장되며, `TrainCallback._on_step()`이 읽어서 100,000 스텝마다 콘솔에 출력한다.
