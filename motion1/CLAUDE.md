# CLAUDE.md — motion1

> **이 파일은 다른 Claude 세션이 컨텍스트 없이 바로 작업을 이어갈 수 있도록 작성된 프로젝트 레퍼런스입니다.**
> 코드를 먼저 읽기 전에 이 파일을 완독하세요.

---

## 0. 한 줄 요약

OMY-F3M 6-DOF 로봇팔로 박스를 집어서(Grasp) 셀(Cell)에 삽입(Insert)하는 작업을 **Motion Planning + RL Hybrid** 방식으로 구현한다. 큰 이동(reach/transport/retract)은 deterministic 모션 플래닝, 접촉이 필요한 정밀 동작(grasp 정렬 / insert 정렬)만 RL로 학습한다.

**최종 목표**: 카메라(천장 cam + 손목 cam)로 박스·셀 위치를 추정하고, 그 좌표를 RL obs에 넣어 실제 로봇에서 작동시키는 sim2real.

---

## 1. 시스템 정보

| 항목 | 값 |
|---|---|
| 시뮬레이터 | NVIDIA Isaac Lab / Isaac Sim 4.5 |
| RL 라이브러리 | Stable Baselines 3 (PPO) |
| 로봇 | ROBOTIS OMY-F3M (6-DOF arm + 4-joint gripper) |
| 로봇 USD | `/home/jaewoo/IsaacLab/source/omy_f3m_urdf/OMY.usd` (절대 경로 하드코딩) |
| GPU | RTX 4070 12GB |
| Conda 환경 | `env_isaaclab` (`/home/jaewoo/miniconda3/envs/env_isaaclab/`) |
| 실행 명령 형식 | `./isaaclab.sh -p source/motion1/scripts/<파일.py> [args]` |

### 실행 환경 활성화
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd /home/jaewoo/IsaacLab
```

---

## 2. OMY-F3M 로봇 스펙

| 항목 | 값 |
|---|---|
| Arm joints | `joint1` ~ `joint6` (6-DOF) |
| Gripper joints | `rh_r1_joint`, `rh_r2`, `rh_l1`, `rh_l2` (4-joint mimic) |
| EE 정의 | `rh_p12_rn_l2` (left finger) + `rh_p12_rn_r2` (right finger) 월드 좌표 **평균** |
| EE base quat | `(w=0, x=1, y=0, z=0)` wxyz — 수직 아래 + finger Y 방향 고정 |
| EE target quat | `R_z(ee_yaw) ⊗ base_ee_quat` — yaw 회전만 추가 |
| Reach | **580mm** (중요: spawn 반경이 이 이상이면 IK 발산) |
| Forward 방향 | **+X축** (joint1=0 기준, 실제 로봇 코드 `motion2/scripts/jaewoo/` 검증) |
| Gripper tip_ratio | **2.3** — `rh_l2/rh_r2 = rh_l1/rh_r1 × 2.3` (USD mimic 깨짐 보정) |
| Gripper open | 4 joint = 0 |
| Gripper close | cmd 0.8, l2/r2 = 0.8 × 2.3 = 1.84 |
| 안전 Z 범위 | ee_z 0.26~0.30 (LIFT_Z 0.50 이상은 singularity 위험) |
| joint3 실제 한계 | ±150° (URDF에는 ±360° 잘못 기재됨 — sim2real 시 주의) |

### Home 자세 (joint 값)
```python
HOME_JOINT_POS = {
    "joint1": 0.0, "joint2": -1.55, "joint3": 2.66,
    "joint4": -1.1, "joint5": 1.6, "joint6": 0.0,
    "rh_r1_joint": 0.0, "rh_r2": 0.0, "rh_l1": 0.0, "rh_l2": 0.0,
}
```

### Fallback Grasp 자세 (Grasp RL 시작 자세)
```python
fallback_arm_pos = [0.0, 0.06, 1.98, -1.02, 1.26, -0.13]  # joint1~6
# 이 자세에서 ee ≈ (0.46, -0.30, 0.75) world
```

---

## 3. 박스 / 셀 물리 스펙

### 박스
| 항목 | 값 |
|---|---|
| 크기 | 0.139 × 0.044 × 0.118 m (가로 × 세로 × 높이) |
| 잡는 방향 | **short edge Y (4.4cm)** 방향만 가능. long edge X (13.9cm)는 finger sep(11.3cm) 보다 커서 불가 |
| 질량 | 0.3 kg |
| 마찰 | static/dynamic 3.0, combine_mode="max" |
| spawn z | 0.07m (env-rel) |

### 셀 (Cell / 적재함)
| 항목 | 값 |
|---|---|
| 내부 크기 | 0.16 × 0.065 m (x × y) |
| 격벽 두께 | 0.008 m |
| 격벽 높이 | 0.12 m |
| 4개 벽 이름 | `CellWall_VL`, `CellWall_VR`, `CellWall_HF`, `CellWall_HB` |

---

## 4. Motion Planning 공통 상수 (env-relative 좌표)

```python
BOX_SPAWN_Z  = 0.07     # 박스 바닥 z
PRE_GRASP_Z  = 0.17     # Grasp RL 시작 ee z (박스 위 10cm)
GRASP_Z      = 0.115    # 실제 파지 ee z
LIFT_Z       = 0.26     # 들어올린 후 ee z
TRANSPORT_Z  = 0.26     # 운반 중 ee z (= LIFT_Z)
PLACE_Z      = 0.165    # 삽입 시작 ee z
RETRACT_Z    = 0.26     # 빼낼 때 ee z
```

### IK 방식
- Isaac Lab `DifferentialIKController` (DLS, Damped Least Squares)
- EE 위치 = 양 finger 평균 world pos
- EE quat = left finger body quat
- Jacobian = 양 finger jacobian 평균

---

## 5. 전체 파이프라인 (6단계)

```
Stage 1: home → 박스 위 3~5cm offset      [Motion Planning]
Stage 2: Grasp RL 정렬 (xy + yaw)         [RL — GraspEnv]
Stage 3a: ee 하강 → GRASP_Z              [Motion Planning]
Stage 3b: gripper close + hold            [Motion Planning]
Stage 3c: lift → LIFT_Z                   [Motion Planning]
Stage 3d: transport → 셀 위 + ee yaw 정렬 [Motion Planning + slerp]
Stage 4: Insert RL 정렬 (xy + yaw)        [RL — InsertEnv]
Stage 5a: insert descend → PLACE_Z        [Motion Planning]
Stage 5b: gripper open (release)          [Motion Planning]
Stage 6a: retract up → RETRACT_Z         [Motion Planning]
Stage 6b: retract home                    [Motion Planning]
```

---

## 6. Spawn 분포 (2026-05-27 확정)

### 핵심 설계 원칙
- **박스**: 로봇이 바라보는 **전방 180도** (+X ±90°) 안에 위치. 반경 **15~35cm**.
- **셀**: **360도 전체** 어디나 위치 가능. 반경 **30~50cm**.
- **두 물체 모두 yaw ±80°** (1.396 rad) 자유 회전.
- 좌표계: 모두 **robot base(env origin) 기준 env-relative**.

### 박스 spawn (polar)
```python
r     = uniform(0.15, 0.35)         # 반경 15~35cm
theta = uniform(-π/2, π/2)          # 전방 ±90°
box_x = r * cos(theta)              # env-rel x
box_y = r * sin(theta)              # env-rel y
box_z = 0.07                        # 고정
box_yaw = uniform(-1.396, 1.396)    # ±80°
```

### 셀 spawn (polar)
```python
r      = uniform(0.30, 0.50)        # 반경 30~50cm
theta  = uniform(-π, π)             # 360도
cell_x = r * cos(theta)             # env-rel x
cell_y = r * sin(theta)             # env-rel y
cell_yaw = uniform(-1.396, 1.396)   # ±80°
```

> **주의**: OMY reach = 580mm. 셀 반경 50cm + 정렬 여유분 고려 시 로봇 후방에서 IK 발산 가능.
> `collect_insert_handoff.py` 에서 grasp fail 시 자동 skip 처리됨.

---

## 7. RL Task 상세 설계

### 7-1. Grasp RL

**역할**: Stage 2. Fallback 자세(박스 위 대략 위)에서 시작하여 박스 xy + yaw 에 ee 를 정렬하는 RL.

#### State (6차원)
```
obs[0] = obj_rel_x  = box_x_env - ee_x_env         (박스가 ee 기준 얼마나 x 방향)
obs[1] = obj_rel_y  = box_y_env - ee_y_env
obs[2] = obj_yaw_err = wrap(box_yaw - ee_target_yaw) (박스 yaw와 ee target yaw 차이)
obs[3] = ee_vel_x   (실제 finger 평균 world lin vel x)
obs[4] = ee_vel_y
obs[5] = yaw_vel    = (ee_target_yaw - prev_ee_target_yaw) / dt  (target 기반)
```

> 좌표계: **world frame** 유지. `_env` 는 world - env_origin. 상대 벡터라 360도 spawn에서도 동일.

#### Action (3차원, [-1, 1])
```
action[0] = Δx  →  ee_target_x += action[0] * 0.01   (10mm/step)
action[1] = Δy  →  ee_target_y += action[1] * 0.01
action[2] = Δyaw → ee_target_yaw += action[2] * 0.05  (2.86°/step, 누적)
```
- ee_z = 고정 0.17m (PRE_GRASP_Z)
- gripper = open 고정 (4 joint = 0)
- IK target: `(target_xy_w, 0.17) + R_z(ee_target_yaw) ⊗ base_ee_quat`

#### Reward
```python
r_xy_align   = exp(-80.0 * xy_dist²)
r_yaw_align  = exp(-5.0  * yaw_err²)
r_smooth     = -0.01 * (vel_x² + vel_y² + yaw_vel²)
r_success    = 50.0  * aligned  (매 step)
r_lump       = 5000.0 * will_succeed  (hold 도달 시 한 번)
total = r_xy_align + r_yaw_align + r_smooth + r_success + r_lump
```

#### 종료 조건
```python
aligned  = (|rel_x| < 0.005) & (|rel_y| < 0.005) & (|yaw_err| < 0.05)
success  = aligned 30 step(0.5초) 연속 유지
fail     = (|rel_x| > 0.30) | (|rel_y| > 0.30)  # ee 가 박스에서 30cm 이상
truncate = 300 step (5초)
```

#### PPO 하이퍼파라미터
```
n_steps=1024, batch_size=256, lr=3e-4, gamma=0.97, gae=0.95,
clip=0.2, ent=0.005, vf=0.5, n_epochs=5
```

#### 관련 파일
- `tasks/grasp/grasp_env_cfg.py` — cfg (polar spawn 적용됨)
- `tasks/grasp/grasp_env.py` — env 구현 (polar spawn _reset_idx 적용됨)
- `scripts/train_grasp.py` — 학습 스크립트
- `scripts/play_grasp.py` — 시각화

---

### 7-2. Insert RL

**역할**: Stage 4. Transport 끝(박스 잡힌 채 셀 위)에서 시작하여 셀 xy + yaw 에 ee 를 정렬하는 RL.

#### State (7차원)
```
obs[0] = slot_rel_x   = cell_x_env - ee_x_env      (셀 center가 ee 기준 얼마나 x)
obs[1] = slot_rel_y   = cell_y_env - ee_y_env
obs[2] = slot_yaw_err = wrap(cell_yaw - actual_ee_yaw)  (actual ee yaw 기반)
obs[3] = is_grasping  = float(finger↔box dist < 7cm AND box_z > 12cm)
obs[4] = ee_vel_x     (실제 finger 평균 world lin vel x)
obs[5] = ee_vel_y
obs[6] = yaw_vel      (실제 finger 평균 angular vel z — actual 기반)
```

> Grasp와 차이:
> - 타겟이 **박스 → 셀**
> - `is_grasping` 플래그 추가
> - yaw_vel 이 target 기반이 아닌 **actual ee angular vel** 기반
> - yaw action이 **비누적** (매 step `actual_ee_yaw + Δyaw`)

#### Action (3차원, [-1, 1])
```
action[0] = Δx   →  ee_target_x = actual_ee_x + action[0] * 0.005  (5mm/step)
action[1] = Δy   →  ee_target_y = actual_ee_y + action[1] * 0.005
action[2] = Δyaw →  ee_target_yaw = actual_ee_yaw + action[2] * 0.05  (비누적)
```
- ee_z = 고정 0.26m (TRANSPORT_Z)
- gripper = close 고정 (cmd=0.8, l2/r2 = 0.8×2.3=1.84)

#### Reward
```python
r_xy_align       = exp(-80.0  * xy_dist²)   # 멀리서도 exploration 신호
r_xy_align_close = exp(-200.0 * xy_dist²)   # 가까이서 정밀 신호 (dual reward)
r_yaw_align      = exp(-5.0   * yaw_err²)
r_smooth         = -0.01 * (vel_x² + vel_y² + yaw_vel²)
r_success        = 50.0  * aligned  (매 step, is_grasping 포함)
r_lump           = 5000.0 * will_succeed
total = r_xy_align + r_xy_align_close + r_yaw_align + r_smooth + r_success + r_lump
```

#### 종료 조건
```python
aligned  = (|rel_x| < 0.010) & (|rel_y| < 0.010) & (|yaw_err| < 0.087) & is_grasping
success  = aligned 15 step(0.25초) 연속 유지
fail     = (box_z < 0.12)  # 박스 떨어짐
         | (|rel_x| > 0.30) | (|rel_y| > 0.30)
truncate = 300 step (5초, sim_dt=1/60)
```

#### Reset 방식 — Handoff Dataset
Insert RL은 "박스를 잡고 셀 위에 도달한 상태"에서 시작한다.
이 시작 상태를 직접 만드는 것이 어렵기 때문에 **handoff dataset**에서 random sample한다.

dataset 파일: `checkpoints/insert_handoff_states.npz`

```python
data = np.load("checkpoints/insert_handoff_states.npz")
data["joint_pos"]     # (N, 10) — arm 6 + gripper 4
data["box_pos_env"]   # (N, 3)  — env-rel 박스 위치
data["box_quat"]      # (N, 4)  — 박스 quaternion wxyz
data["cell_xy"]       # (N, 2)  — env-rel 셀 center xy
data["cell_yaw"]      # (N, 1)  — 셀 yaw
data["ee_target_yaw"] # (N, 1)  — transport 끝의 ee yaw (≈ cell_yaw)
```

#### PPO 하이퍼파라미터 (Grasp와 동일)
```
n_steps=1024, batch_size=256, lr=3e-4, gamma=0.97, gae=0.95,
clip=0.2, ent=0.005, vf=0.5, n_epochs=5
sim_dt=1/60, decimation=1  → 제어 60Hz
```

#### 관련 파일
- `tasks/insert/insert_env_cfg.py`
- `tasks/insert/insert_env.py`
- `scripts/train_insert.py`
- `scripts/play_insert.py` — cell 4 walls visual marker 포함, keep_alive=True
- `scripts/collect_insert_handoff.py` — handoff dataset 수집 (polar spawn 적용됨)

---

## 8. 현재 진행 상태 (2026-05-27)

### 완료된 것
| 항목 | 상태 | 비고 |
|------|------|------|
| Motion-only chain (6단계) | ✅ | `play_motion_chain.py` 검증 완료 |
| Grasp RL 코드 | ✅ | polar spawn으로 코드 변경 완료 |
| Insert RL 코드 | ✅ | 코드는 그대로 (spawn은 dataset 의존) |
| collect_insert_handoff.py | ✅ | polar spawn으로 코드 변경 완료 |
| Camera chain runner | ✅ | `play_motion_chain_with_grasp_insert_camera.py` 작성 완료 |
| YOLO dataset 수집 스크립트 | ✅ | `collect_yolo_dataset.py` |

### 아직 안 된 것 (해야 할 것)
| 항목 | 상태 | 의존 |
|------|------|------|
| Grasp RL **재학습** | ❌ 안 함 | 코드 변경 완료 → 바로 학습 가능 |
| Insert handoff dataset **재수집** | ❌ 안 함 | Grasp 재학습 완료 후 |
| Insert RL **재학습** | ❌ 안 함 | 재수집 완료 후 |
| Chain runner 검증 | ❌ 안 함 | 두 정책 재학습 후 |
| Sim2Real 검증 | ❌ 안 함 | Chain 검증 후 |

### 기존 checkpoint 상태
| 파일 | 내용 | 사용 여부 |
|------|------|----------|
| `checkpoints/motion1_grasp.zip` | 구버전 (고정 spawn 학습) | ⚠️ 재학습 전까지 임시 사용 |
| `checkpoints/motion1_grasp_vecnorm.pkl` | 위와 쌍 | 위와 동일 |
| `checkpoints/motion1_insert_best.zip` | v14 (고정 spawn 학습, 4M step) | ⚠️ 재학습 전까지 임시 사용 |
| `checkpoints/motion1_insert_best_vecnorm.pkl` | 위와 쌍 | 위와 동일 |
| `checkpoints/insert_handoff_states.npz` | v6_clean (고정 spawn) | ⚠️ 재수집 전까지 임시 사용 |

---

## 9. 다음 세션에서 해야 할 것 (순서대로)

### Step A — 기존 checkpoint 백업
```bash
cd /home/jaewoo/IsaacLab
cp checkpoints/motion1_grasp.zip checkpoints/motion1_grasp_v1_fixed.zip
cp checkpoints/motion1_grasp_vecnorm.pkl checkpoints/motion1_grasp_v1_fixed_vecnorm.pkl
cp checkpoints/motion1_insert_best.zip checkpoints/motion1_insert_v14_fixed.zip
cp checkpoints/motion1_insert_best_vecnorm.pkl checkpoints/motion1_insert_v14_fixed_vecnorm.pkl
cp checkpoints/insert_handoff_states.npz checkpoints/insert_handoff_states_v1_fixed.npz
```

### Step B — Grasp RL 재학습
```bash
./isaaclab.sh -p source/motion1/scripts/train_grasp.py \
    --headless --num_envs 128 --timesteps 2000000
```
- 학습 시간 예상: ~10분 (SPS 6000+, 2M step)
- log: `source/motion1/logs/grasp/`
- checkpoint: `checkpoints/motion1_grasp.zip`
- TensorBoard: `tensorboard --logdir source/motion1/logs/grasp`

**학습 완료 기준**: `success_recent ≥ 0.5` (박스 위치가 넓어졌으므로 수렴이 느릴 수 있음)

학습 후 시각화 검증:
```bash
./isaaclab.sh -p source/motion1/scripts/play_grasp.py --episodes 10
```

### Step C — Insert Handoff Dataset 재수집
```bash
# 소량 GUI 검증 (5개, 30초 hold로 박스 잡히는지 확인)
./isaaclab.sh -p source/motion1/scripts/collect_insert_handoff.py \
    --target 5 --hold_s 30

# 본격 수집 (headless background, 약 30분~1시간)
nohup ./isaaclab.sh -p source/motion1/scripts/collect_insert_handoff.py \
    --headless --target 1000 --hold_s 0 \
    > /tmp/collect_insert_new.log 2>&1 &

tail -f /tmp/collect_insert_new.log  # 진행 확인
```

**진행 확인**:
```bash
ls -la checkpoints/insert_handoff_states.npz  # timestamp 확인
# 50개 단위로 부분 저장되므로 파일이 갱신되면 정상
```

수집 완료 후 데이터 확인:
```python
import numpy as np
d = np.load("checkpoints/insert_handoff_states.npz")
print(d["joint_pos"].shape)    # (N, 10) N≥500 이면 충분
print(d["cell_xy"].min(0), d["cell_xy"].max(0))  # 셀이 다양하게 분포하는지
```

### Step D — Insert RL 재학습
```bash
./isaaclab.sh -p source/motion1/scripts/train_insert.py \
    --headless --num_envs 128 --timesteps 4000000
```
- 학습 시간 예상: ~20분 (SPS 8000+, 4M step)
- checkpoint: `checkpoints/motion1_insert.zip` / `motion1_insert_best.zip`
- TensorBoard: `tensorboard --logdir source/motion1/logs/insert`

**학습 진단 metric** (TensorBoard에서 확인):
```
r_xy_align         → 1에 가까울수록 잘 정렬됨
is_grasping_rate   → 이게 낮으면 dataset 품질 문제 (재수집 필요)
xy_aligned_rate    → xy 정렬 빈도
yaw_aligned_rate   → yaw 정렬 빈도
aligned_rate       → 둘 다 동시에 정렬된 빈도
success_recent     → episode 성공률 (목표 ≥ 0.3)
```

**수렴 안 될 경우 체크리스트**:
1. `is_grasping_rate < 0.2` → handoff dataset 재수집. 박스가 손에서 너무 빨리 떨어짐
2. `xy_aligned_rate ≈ 0` → action_scale_xy 를 0.01로 늘리거나 reward_xy_align_gain 을 40으로 낮추기
3. `yaw_aligned_rate ≈ 0` → reward_yaw_align_gain 을 10으로 올리기

학습 후 시각화:
```bash
./isaaclab.sh -p source/motion1/scripts/play_insert.py
```

### Step E — Chain Runner 통합 검증
```bash
./isaaclab.sh -p source/motion1/scripts/play_motion_chain_with_grasp_insert.py \
    --hold_s 30 --repeat 5
```
- 박스 + 셀 모두 새 polar 분포로 random spawn
- Grasp PPO + Insert PPO inference
- 각 stage 결과 print로 확인

**성공 기준**: insert success rate ≥ 0.3 (repeat 5회 기준)

---

## 10. 파일 구조 전체

```
source/motion1/
├── CLAUDE.md                                    ← 이 파일
├── PLAN.md                                      ← 초기 설계 문서 (히스토리)
├── RL_TASKS_SUMMARY.md                          ← State/Action 요약표
│
├── tasks/
│   ├── __init__.py
│   ├── grasp/
│   │   ├── __init__.py
│   │   ├── grasp_env_cfg.py    ← GraspEnvCfg (polar spawn 적용 완료)
│   │   └── grasp_env.py        ← GraspEnv (polar _reset_idx 적용 완료)
│   ├── insert/
│   │   ├── __init__.py
│   │   ├── insert_env_cfg.py   ← InsertEnvCfg
│   │   └── insert_env.py       ← InsertEnv (handoff dataset reset)
│   └── insert_v2/              ← coded reset 버전 (미사용, 참고용)
│       ├── __init__.py
│       └── insert_env_v2_cfg.py
│
├── scripts/
│   ├── train_grasp.py          ← Grasp PPO 학습 (재학습 예정)
│   ├── train_insert.py         ← Insert PPO 학습 (재학습 예정)
│   ├── play_grasp.py           ← Grasp 시각화
│   ├── play_insert.py          ← Insert 시각화 (cell walls visual + keep_alive)
│   ├── play_motion_chain.py    ← Motion-only 6단계 (검증 완료)
│   ├── play_motion_chain_with_grasp.py              ← chain + Grasp RL
│   ├── play_motion_chain_with_grasp_insert.py       ← chain + Grasp + Insert RL
│   ├── play_motion_chain_with_grasp_insert_camera.py ← 카메라 통합 버전
│   ├── collect_insert_handoff.py   ← Insert dataset 수집 (polar spawn 적용 완료)
│   ├── collect_insert_coded.py     ← coded reset 수집 (미사용)
│   └── collect_yolo_dataset.py     ← YOLO 학습용 이미지 수집
│
├── yolo_dataset/
│   └── data.yaml               ← nc=2, names: [box, cell]
│
└── logs/
    ├── grasp/                  ← TensorBoard grasp 학습 log
    └── insert/                 ← TensorBoard insert 학습 log
```

---

## 11. Camera 통합 설계 (play_motion_chain_with_grasp_insert_camera.py)

현재 파일 작성 완료 상태이며, 아직 실제 검증은 안 됨.

### 카메라 구성
| 카메라 | 위치 | 역할 |
|--------|------|------|
| 천장 cam (`top_cam`) | (0.30, -0.20, 0.80) env-rel | run 시작 시 1회 scan → 박스/셀 xy+yaw 추정 |
| 손목 cam (`wrist_cam`) | link6에 부착 pos=(0,-0.1,0.084) | Grasp RL 매 step → 박스 yaw 추정 |

### 천장 cam 스펙
```python
focal_length=24.0, horizontal_aperture=20.955
height=480, width=640
pos=(0.30, -0.20, 0.80), rot=(0,1,0,0) ROS convention
```

### 손목 cam 스펙
```python
focal_length=11.0  # FOV ≈ 87° (D405 spec 매칭)
height=240, width=320
pos=(0,−0.1,0.084) link6-rel, rot=(0,0,0.7071,−0.7071) ROS
# 매 step set_world_poses()로 link6 world pose 추적
```

### Detection 방식
1. **YOLO** (`--yolo_ckpt` 지정 시): `ultralytics` 패키지, class_id 0=box, 1=cell
2. **Sim instance segmentation** (기본): `instance_id_segmentation_fast` mask 사용

### Sim2Real obs 교체 표
| obs 항목 | Sim (학습) | Real (배포) |
|---------|-----------|------------|
| box_xy | `box.data.root_pos_w` | 천장 cam 추정 |
| box_yaw | `quat_z_yaw(box.data.root_quat_w)` | 손목 cam 추정 |
| cell_xy | wall center 평균 | 천장 cam 추정 |
| cell_yaw | `quat_z_yaw(wall_v_left.root_quat_w)` | 천장 cam 추정 |
| ee_xy | `grip_center_pos()` | TF2 FK |
| ee_yaw | `quat_z_yaw(body_quat_w[:, left_id])` | TF2 link6 |

### 실행 명령
```bash
# 기본 (GT obs, 카메라 켜기만)
./isaaclab.sh -p source/motion1/scripts/play_motion_chain_with_grasp_insert_camera.py \
    --enable_cameras --hold_s 30 --save_camera_debug

# vision obs 사용
./isaaclab.sh -p source/motion1/scripts/play_motion_chain_with_grasp_insert_camera.py \
    --enable_cameras --use_vision --repeat 5 --hold_s 5

# YOLO 사용
./isaaclab.sh -p source/motion1/scripts/play_motion_chain_with_grasp_insert_camera.py \
    --enable_cameras --use_vision \
    --yolo_ckpt runs/segment/source/motion1/yolo_runs/v1_seg/weights/best.pt \
    --repeat 5
```

> **현재 camera 버전 특이사항**: Insert RL stage가 제거되어 있음.
> Transport 끝에서 rule-based descend + release로 처리.
> Insert RL 재학습 완료 후 다시 추가 필요.

---

## 12. 알려진 이슈 및 주의사항

### 미해결 이슈
1. **Insert RL chain runner 미정렬**
   - `play_insert.py` 단독 실행 시 정렬 OK
   - `play_motion_chain_with_grasp_insert.py` chain runner에서 같은 ckpt로 실행 시 안 됨
   - Root cause 불명. 의심 후보:
     - stage 3d 끝 자세가 handoff dataset 분포와 미묘하게 다름
     - chain runner의 obs 측정 타이밍 차이
   - **디버그 방법**: chain runner stage 4 시작 직후 첫 5 step obs/action print + play_insert.py 첫 5 step 비교

### 고정 주의사항
2. **OMY USD 절대 경로**: `omy_robot_cfg.py`에 `/home/jaewoo/IsaacLab/source/omy_f3m_urdf/OMY.usd` 하드코딩
3. **OMY gripper mimic 깨짐**: `tip_ratio=2.3` 보정 필수. `gripper_l2 = gripper_l1 × 2.3`
4. **LIFT_Z 상한**: 0.30 이상에서 singularity 떨림. 현재 0.26 사용
5. **joint3 URDF 오류**: URDF에 ±360°로 기재됐으나 실제 한계는 ±150°. Sim2Real 시 반드시 수정
6. **`os._exit(0)`**: 모든 main 끝에 강제 종료 (PhysX 자원 해제 hang 방지)
7. **Background 프로세스 충돌**: 학습/수집 중 두 번째 instance 실행 시 GPU 충돌. 반드시 확인:
   ```bash
   ps -ef | grep python | grep -v grep
   ```

---

## 13. 사용자 스타일 메모 (향후 대화 시 참고)

- 한국어로 설명 선호
- 코드 변경 후 즉시 시각화로 검증 원함 (`--hold_s` 옵션 활용)
- **페널티(음수 reward) 회피** — all-positive reward 선호. drop은 termination으로 자연 페널티.
- 가중치 1~100 범위 (큰 값 회피, 단 success_lump 5000 OK)
- 학습 200k~4M step 범위
- 상세 계획 먼저 → 확인 후 코드 작업
- 불필요한 추상화/모듈화 최소화

---

## 14. 절대 건드리지 말 것

- `source/example5/` (frozen lift policy)
- `source/omy_f3m_urdf/OMY.usd` (로봇 모델)
- `checkpoints/example5.zip`, `example5_vecnorm.pkl`, `handoff_states.npz` (example5 전용)
- `robotis_lab/` (ROBOTIS 공식 패키지, 참고용)

---

## 15. 참고 — 실제 로봇 코드 (motion2/scripts/jaewoo/)

실제 OMY 로봇 ROS2 제어 코드. Sim2Real 단계에서 참고.

| 파일 | 역할 |
|------|------|
| `run_pick_place.py` | 9단계 pick-and-place 전체 파이프라인 |
| `run_ee_pose_move.py` | 단일 Cartesian 좌표로 EE 이동 |
| `show_ee_pose.py` | 현재 EE 위치 TF2로 읽기 |
| `go_to_start.py` | 홈 자세 복귀 |
| `toggle_gripper.py` | 그리퍼 open/close 토글 |

**실제 로봇 좌표 범위** (link0 기준):
```
X: [-0.10, 0.55] m  (forward = +X)
Y: [-0.45, 0.20] m
Z: [0.10, 0.50] m
```

실제 로봇에서 3단계 안전 guard 있음:
1. Workspace guard (직육면체 범위 체크)
2. MoveIt plan delta guard (최대 관절 변위 ≤ 0.35 rad)
3. Actual delta guard (현재 → 목표 관절 변위 재검증)
