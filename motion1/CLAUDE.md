# CLAUDE.md — motion1

> **다음 Claude 세션이 즉시 컨텍스트 잡고 작업 이어갈 수 있게 작성됨**.
> 설계 전체는 [PLAN.md](PLAN.md). 사용자 디자인 결정과 현재 상태는 이 파일.

---

## ⚡ TL;DR — 지금 즉시 해야 할 일 (2026-05-10)

### 🎯 현재 상태: **v14 정책 학습 완료. chain runner 정렬 안 됨. 다음 = camera (vision) 통합**

**v14 정책** (sim_dt 1/60 + decimation 1, mask 제거 + dual reward + action 5mm + 4M step):
- ckpt: `checkpoints/motion1_insert_best.zip` (num_timesteps=3.94M)
- 학습 metric: `success_recent=1~3%`, r_xy_align=0.93, r_yaw_align=0.97
- play_insert.py 단독 시각화: 정렬 시도 OK
- **chain runner 시각화: 여전히 정렬 안 됨** ← 미해결

### 🔬 chain runner 안 되는 root cause (불명, 후보)

play_insert.py = OK, chain runner = X. 같은 ckpt. obs/action 동일 식. 그런데 행동 다름. 모든 알려진 차이 fix 했지만 안 됨:
- ✅ VecNormalize 직접 호출 (manual normalize 차이 가능 의심)
- ✅ action_scale_xy 분리 (grasp 10mm / insert 5mm)
- ✅ sim_dt 1/120 → 1/60 + decimation 1 (학습 cfg 와 일치)
- ✅ cell wall collision off 시도 (효과 X)
- ✅ insert offset 1cm 줄임 (이전 3~5cm)

남은 의심 — **stage 3d 끝 자세가 학습 분포 (handoff dataset) 와 미묘하게 다름** + chain runner 의 control loop / obs 측정 timing 차이.

### ✅ 다음 step: Camera (vision) 통합

사용자 요청 — `play_motion_chain_with_grasp_insert.py` 의 camera 붙인 버전 새로 만들기.

작업 순서:
1. **카메라 sensor 부착** (IsaacLab `Camera` class 또는 USD camera prim)
2. **detection** (YOLO 또는 단순 segmentation) — 박스 / cell 4 walls 위치 추정
3. **image space → world coord 변환** (camera intrinsic/extrinsic)
4. **그 좌표로 RL 정책 inference** (현재 ground truth obs 자리에 vision 추정 값)

**학습 재학습 필요한가?** — 일단 **불필요** 추정:
- vision 정확도 5mm + 학습 align threshold 10mm = 합산 ~15mm
- cell inner 16×6.5cm vs 박스 13.9×4.4cm → 여유 ±10mm 면 충분
- 만약 vision noise 가 학습 분포 이상이면 → domain randomization 으로 재학습

새 파일 이름 후보: `play_motion_chain_with_grasp_insert_camera.py` 또는 `play_motion_chain_camera.py`

### 📋 Camera 통합 — 시작점

기존 `play_motion_chain_with_grasp_insert.py` 복사 후:
1. `MotionSceneCfg` 에 camera 추가:
   ```python
   from isaaclab.sensors import CameraCfg
   camera = CameraCfg(
       prim_path="{ENV_REGEX_NS}/Camera",
       update_period=1.0/60.0,
       height=480, width=640,
       data_types=["rgb", "depth", "instance_segmentation_fast"],
       offset=CameraCfg.OffsetCfg(pos=(0.5, 0.0, 1.0), rot=(...), convention="opengl"),
   )
   ```
2. detection 모듈 (YOLO 또는 segmentation mask 직접) 추가
3. world coord 변환 — IsaacLab `Camera` 의 intrinsic + extrinsic 사용
4. 변환 결과를 stage_rl_grasp / stage_rl_insert 의 obs (box_xy_env, cell_xy_env) 에 입력

---

## 1. 한 줄 요약

example7/example7_2 의 RL 단독 학습이 박스 적재에서 실패 → 큰 운동 (reach/transport/retract) 은 **motion planning**, contact-rich (grasp / insert) 만 **RL** 로 분해 재설계.

---

## 2. 진행 체크리스트 (현재 시점, 2026-05-10)

- [x] **Step 1**: motion-only 6단계 chain — 검증 완료
- [x] **Step 2**: Grasp RL — 1M step (실제). success_recent 1~3%. play 단독 OK, chain 에서도 stage 2 SUCCESS @ step 50 (잘 작동)
- [x] **chain + grasp 통합** — 작동 OK
- [x] **Step 3 — Insert RL**: 학습 끝 (v14)
  - [x] env / cfg / scripts 작성
  - [x] dataset 재수집 4번 (v1 → v6_clean = 769 sample, cell ±5cm + transport 끝 ee 3~5cm noise + 박스 / cell 위치 0.30, -0.30)
  - [x] 학습 v1~v14 진행 — 자세히 § 12 timeline
  - [x] **v14 best (4M)**: success_recent 1~3%, ep_rew_mean +880, SPS 8925
  - [x] play_insert.py 단독 시각화 — **정렬 OK**
  - [⚠️] chain runner 시각화 — **정렬 안 됨 (root cause 불명)**
- [⚙️] **Step 4 — Chain runner** ([play_motion_chain_with_grasp_insert.py](scripts/play_motion_chain_with_grasp_insert.py))
  - [x] 작성 완료 (cell wall 4개 RigidObjectCfg + kinematic random pose, sim_dt 1/60 + DECIMATION 1)
  - [⚠️] stage 4 RL inference 안 됨. play_insert.py 와 같은 ckpt 인데 chain 에서만 정책이 잘못된 방향
- [⚙️] **Step 7 — Sim2real (Camera vision)**: **다음 작업** ← 사용자 요청
  - [ ] camera 붙인 새 chain runner 파일 생성
  - [ ] detection (YOLO 또는 simple mask)
  - [ ] vision 결과를 RL obs 에 입력
- [ ] **Step 5**: Multi-box / 3×3 grid 확장 (Step 7 이후)
- [ ] **Step 6**: 모듈화

---

## 3. 작성된 파일 — 위치 + 핵심 내용

### Motion-only (Step 1) ✅
| 파일 | 핵심 |
|---|---|
| [scripts/play_motion_chain.py](scripts/play_motion_chain.py) | 6단계 motion chain. RL 미사용. 사용자가 검증 OK |

### Grasp RL (Step 2) ✅
| 파일 | 핵심 |
|---|---|
| [tasks/grasp/grasp_env_cfg.py](tasks/grasp/grasp_env_cfg.py) | DirectRLEnvCfg. action 3 / state 6. cell randomize ±10cm + ±80°. tip_ratio 2.3 |
| [tasks/grasp/grasp_env.py](tasks/grasp/grasp_env.py) | DirectRLEnv. IK + base_ee_quat (0,1,0,0) + R_z(yaw) |
| [scripts/train_grasp.py](scripts/train_grasp.py) | PPO + VecNormalize + early_stop |
| [scripts/play_grasp.py](scripts/play_grasp.py) | 시각화 |
| `checkpoints/motion1_grasp.zip` + `_vecnorm.pkl` | **학습된 정책 — 사용 중. 재학습 전엔 그대로 둘 것** |

### Chain + Grasp 통합 ✅
| 파일 | 핵심 |
|---|---|
| [scripts/play_motion_chain_with_grasp.py](scripts/play_motion_chain_with_grasp.py) | chain 6단계. 단계 2 = 학습된 grasp PPO inference. 단계 3d = transport + ee yaw 점차 0 정렬 (slerp). 단계 4 는 아직 motion placeholder (1cm 정렬) |

### Insert RL (Step 3) — 진행 중
| 파일 | 핵심 |
|---|---|
| [scripts/collect_insert_handoff.py](scripts/collect_insert_handoff.py) | chain 시뮬 + 매 episode cell xy **±5cm** + cell yaw ±80° + **transport 끝 ee xy 3~5cm noise** (dataset 자체에 정렬 학습용 noise 포함). 50개 단위 부분 저장 |
| [tasks/insert/insert_env_cfg.py](tasks/insert/insert_env_cfg.py) | DirectRLEnvCfg. action 3 / state 7. ee_fixed_z 0.26. **drop penalty 제거**. box_drop_z_threshold 0.12 (박스 long edge 누워도 검출) |
| [tasks/insert/insert_env.py](tasks/insert/insert_env.py) | DirectRLEnv. **yaw 비누적**, **drop penalty 코드 제거**, `_extract_ee_yaw` helper, actual ee_ang_vel_z 사용. handoff dataset 그대로 적용 (reset noise 코드 제거) |
| [scripts/train_insert.py](scripts/train_insert.py) | train_grasp.py 패턴 그대로 |
| [scripts/play_insert.py](scripts/play_insert.py) | play_grasp.py 패턴. + **cell 4 walls visual marker** (회색, 정렬 목표 표시), keep_alive (창 자동 안 닫힘), render_interval 4 |
| [scripts/play_motion_chain_with_grasp_insert.py](scripts/play_motion_chain_with_grasp_insert.py) | **Step 4 chain runner — 작성 완료**. cell wall 4개 kinematic RigidObject (random pose 강제 이동). stage 2 = grasp PPO, stage 4 = insert PPO. transport 끝 ee yaw → cell_yaw 회전 |
| `checkpoints/insert_handoff_states.npz` | **dataset v3 재수집 중** (cell ±5cm + transport noise 포함, 1000 row) |
| `checkpoints/insert_handoff_states_v1.npz` | v1 백업 (cell ±2cm, noise 없음 — dead) |

### 향후 (작성 예정)
| 파일 | 언제 |
|---|---|
| `tasks/grid/` | Step 5 — 3×3 grid 확장 |

---

## 4. 사용자가 확정한 디자인 결정

### Insert RL Task (Step 3)
| 항목 | 값 |
|---|---|
| Action (3) | Δx, Δy, Δyaw (cartesian relative, **둘 다 비누적**) |
| **State (7)** | **slot_rel_x, slot_rel_y, slot_yaw_err, is_grasping, ee_vel_x, ee_vel_y, yaw_vel** (모두 actual ee 기준) |
| ee_z 고정 | TRANSPORT_Z = **0.26** (= BOX_SPAWN[2] + 0.19) |
| gripper | 박스 잡힌 상태 유지 (close cmd 0.8 × tip_ratio 2.3) |
| 알고리즘 | PPO. 수렴 안 되면 SAC+HER 검토 |
| episode_length | 5초 (300 step) |
| success_hold_steps | 30 (= 0.5초) |
| align threshold | xy < 5mm, yaw < ~2.86° (0.05 rad) |
| Reward | xy_align(80) + yaw_align(5) + smooth(0.01) + success_step(50) + success_lump(5000) (drop_penalty **제거**) |
| is_grasping 판정 | finger center ↔ box dist < 7cm AND box z > **12cm** (drop threshold 0.12) |

### Insert 시작 분포 (handoff dataset 자체에 noise 포함, 2026-05-09 변경)
| 항목 | 범위 |
|---|---|
| cell base | x=0.21, y=-0.45 |
| cell xy noise (매 episode) | **±5cm** (robot reach 안전 — ±10cm 시 IK 발산) |
| cell yaw range | **±80°** (≈±1.396 rad) |
| **transport 끝 ee xy noise** | **3~5cm random distance + random direction** (collect 시 transport_pos = cell + offset 으로 dataset 자체에 포함) |
| 박스 spawn (collect 시) | xy ±10cm, yaw ±80° (grasp 학습 cfg 와 동일) |

⚠️ 이전엔 `insert_env._reset_idx` 에서 reset noise 추가 시도했으나 **첫 `_pre_physics_step` 에서 덮어써져 무용지물**. 그래서 dataset 자체에 noise 포함 시키는 방식으로 변경 (collect_insert_handoff.py 수정).

### Reward 가중치 (2026-05-09 변경)
| 항목 | 값 |
|---|---|
| `r_xy_align` | exp(-80 × xy_dist²) × is_grasping |
| `r_yaw_align` | exp(-5 × yaw_err²) × is_grasping |
| `r_smooth` | -0.01 × (vel² 합) |
| `r_success` | aligned 매 step bonus 50 (× is_grasping) |
| `r_success_lump` | success terminate 시 5000 |
| ~~`r_drop`~~ | **제거됨** (termination 만으로 자연 페널티. 사용자 스타일 — all-positive reward 선호) |

drop 발생 시 `_get_dones` 의 `dropped → terminated` 으로 episode 즉시 종료 (남은 시간 reward 0 으로 자연 페널티).

### Action scale (grasp / insert 동일)
- Δxy: 10mm/step (= action × 0.01)
- Δyaw: ~2.86°/step (= action × 0.05)

### Action / State 일반 원칙 (사용자 RL 설계)
- Action 은 cartesian Δxyz (+yaw) 만. 관절 각도는 IK 라이브러리로 변환
- State 는 task 별로 직접 판단해서 단순하게
- **Insert state 에 holding_box 플래그 필수**
- **Insert: yaw 도 비누적 (xy 와 동일 패턴)** — `_pre_physics_step` 에서 매 step `cur_ee_yaw + Δyaw` 로 재계산. obs/reward/done 모두 actual ee yaw 기반 (`_extract_ee_yaw()`). 2026-05-09 변경.

---

## 5. 핵심 검증된 디자인 결정 (변경 X)

### Motion-only / chain 공통
- **환경 base**: `SimulationContext` + `InteractiveScene` (DirectRLEnv 안 씀)
- **IK**: Isaac Lab `DifferentialIKController` (DLS). PinkIK / cuRobo 는 sim2real 단계로 미룸
- **EE 정의**: 양 finger body (`rh_p12_rn_l2`, `rh_p12_rn_r2`) **월드 좌표 평균**
- **EE base quat**: `(0, 1, 0, 0)` hardcoded (수직 아래 + finger Y) + R_z(yaw)
- **Reach quat slerp**: home_quat → ee_quat_target (점진적 회전, IK 안정)
- **Gripper**: arm 6 + gripper 4 joint position 직접 명령. tip_ratio = **2.3** (l2/r2 = l1/r1 × 2.3)
- **friction**: combine_mode="max", static/dynamic = 3.0
- **모든 main 끝**: `os._exit(0)` 강제 종료 (PhysX 자원 해제 hang 방지)

### 실행 환경
- Conda env: `env_isaaclab` (`/home/jaewoo/miniconda3/envs/env_isaaclab/`)
- 활성화: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab`
- 그 후 `./isaaclab.sh -p source/motion1/scripts/...`

---

## 6. 알려진 이슈 / 주의사항

1. **OMY USD mimic 깨짐** — 4 gripper joint 같은 명령 시 끝점이 부족하게 굽음. **tip_ratio 2.3** 으로 보정 (l2/r2 = l1/r1 × 2.3)
2. **OMY arm reach 한계** — LIFT_Z 0.50 이상 singularity 떨림. 현재 0.26~0.30 안전
3. **finger sep open = 11.3cm < 박스 long edge 13.9cm** — long edge grasp 물리적 불가능. **short edge Y(4.4cm) 만** 잡기
4. **OMY USD 절대경로** 하드코딩 (`/home/jaewoo/IsaacLab/source/omy_f3m_urdf/OMY.usd`)
5. **joint2 시각적 변동** — home (joint2=-1.55) ↔ grasp (joint2=0.06) 차이 큼. reach 시 큰 회전. **시각적**이지 실제 동작/dataset 수집/학습에는 영향 없음. sim2real 단계에서 PinkIK / 시작 자세 변경 검토
6. **`robotis_lab` 패키지** 가 IsaacLab 안에 이미 설치되어 있음 (`/home/jaewoo/IsaacLab/robotis_lab/`). 공식 OMY USD + cfg 보유. stiffness 만 다름 (우리 350 vs 공식 120)
7. **OMY-F3M spec**: reach 580mm, payload 3kg, joint 1/2/4/5/6 ±360°, **joint 3 ±150°** (URDF 는 ±360° 잘못 — sim2real 시 좁힘)
8. **Cell xy noise reach 한계** (2026-05-09): cell base (0.21, -0.45) 에서 cell xy ±10cm 면 최악 (0.31, -0.55) → 원점 거리 0.63m > reach 0.58m → IK 발산 → collect 무한 루프. **cell xy ±5cm 까지 안전**.
9. **collect 시 print buffer 갇힘** — 1000 sample 도달 후 `np.savez` 는 정상 저장되지만 `saved` print 가 buffer 에 갇혀 log 에 안 보임. 파일 timestamp 로 저장 여부 확인.

### Background process 주의
- 학습 / 수집 background 진행 중일 때 두 번째 instance 띄우면 GPU 충돌
- 이전 conversation 의 background 가 잔존할 수 있음 — `ps -ef | grep python` 으로 확인

---

## 7. 실행 명령 모음

### 1. Motion-only chain (검증 완료)
```bash
./isaaclab.sh -p source/motion1/scripts/play_motion_chain.py --hold_s 30
./isaaclab.sh -p source/motion1/scripts/play_motion_chain.py --headless --repeat 3
```

### 2. Grasp RL (학습 완료, 시각화만)
```bash
./isaaclab.sh -p source/motion1/scripts/play_grasp.py --episodes 10
```

### 3. Chain + Grasp 통합 (검증 완료)
```bash
./isaaclab.sh -p source/motion1/scripts/play_motion_chain_with_grasp.py --hold_s 30
./isaaclab.sh -p source/motion1/scripts/play_motion_chain_with_grasp.py --repeat 5 --hold_s 5
```

### 4. Insert dataset 수집
```bash
# GUI 검증 (작은 수)
./isaaclab.sh -p source/motion1/scripts/collect_insert_handoff.py --target 5 --hold_s 30

# 본격 수집 (background, 약 30분~1시간)
./isaaclab.sh -p source/motion1/scripts/collect_insert_handoff.py --headless --target 1000 --hold_s 0
# → /tmp/collect_insert*.log
# 50개 단위 부분 저장 (stuck 대비)
```

### 5. Insert RL 학습 (수집 끝나면 즉시)
```bash
# 신규 학습 (약 3분 — SPS 6000+)
./isaaclab.sh -p source/motion1/scripts/train_insert.py --headless --num_envs 128 --timesteps 1000000

# resume
./isaaclab.sh -p source/motion1/scripts/train_insert.py --headless --resume --timesteps 500000

# tensorboard (다른 터미널)
tensorboard --logdir source/motion1/logs/insert
```

### 6. Insert 시각화 (cell 4 walls visual + keep_alive)
```bash
./isaaclab.sh -p source/motion1/scripts/play_insert.py
./isaaclab.sh -p source/motion1/scripts/play_insert.py \
    --checkpoint checkpoints/motion1_insert_best.zip \
    --vecnorm checkpoints/motion1_insert_best_vecnorm.pkl
```

### 7. Chain runner — motion + grasp + insert 통합 (Step 4)
```bash
./isaaclab.sh -p source/motion1/scripts/play_motion_chain_with_grasp_insert.py --hold_s 30
./isaaclab.sh -p source/motion1/scripts/play_motion_chain_with_grasp_insert.py --repeat 5 --hold_s 5
# → 박스 + cell 모두 random, grasp PPO + insert PPO inference, cell yaw 정렬 transport
```

---

## 8. 사용자 스타일 메모

- 한국어 설명 선호
- 코드 변경 후 즉시 시각화로 검증 원함 (`--hold_s` 옵션 활용)
- 페널티 (음수 reward) 회피, all-positive reward 선호
- 가중치 1~100 범위 (큰 값 회피, 단 success_lump 5000 OK)
- 학습 200k~2M step 범위
- "마음대로 해" / "필요한 라이브러리 다 가져다 써" 위임 자주
- chain 검증 우선 → sim2real 은 Step 7 으로 미룸 (PinkIK / cuRobo 검토)

---

## 9. 절대 건드리지 말 것

- `source/example5/` (frozen lift policy)
- `source/omy_f3m_urdf/OMY.usd` (로봇 모델 그대로)
- `checkpoints/example5.zip`, `example5_vecnorm.pkl`, `handoff_states.npz` (example5 전용)
- `checkpoints/motion1_grasp.zip` + `_vecnorm.pkl` (학습된 grasp 정책 — 재학습 전엔 그대로)
- `robotis_lab/` 폴더 (ROBOTIS 공식 패키지, 참고용)

---

## 10. 다음 Claude 세션 — 정확한 진행 가이드

### Step A: 진행 상태 확인
```bash
ps -ef | grep -E "collect_insert|train_insert" | grep -v grep   # 프로세스 살아있는지
tail -10 /tmp/collect_insert_v3.log    # 또는 v4 등 최신 log
ls -la checkpoints/insert_handoff_states.npz   # 파일 timestamp
ls -la checkpoints/motion1_insert.zip          # 학습 결과
```

### Step B: dataset 완료 후 학습
```bash
# 기존 ckpt 백업 (env 변경 후 fresh 학습 시)
mv checkpoints/motion1_insert.zip checkpoints/motion1_insert_v2.zip
mv checkpoints/motion1_insert_vecnorm.pkl checkpoints/motion1_insert_v2_vecnorm.pkl
mv checkpoints/motion1_insert_best.zip checkpoints/motion1_insert_v2_best.zip
mv checkpoints/motion1_insert_best_vecnorm.pkl checkpoints/motion1_insert_v2_best_vecnorm.pkl

./isaaclab.sh -p source/motion1/scripts/train_insert.py \
    --headless --num_envs 128 --timesteps 1000000
```
- 학습 시간: 약 3분 (SPS 6000+)
- log: `/tmp/train_insert_v3.log` 같은 곳
- checkpoint: `checkpoints/motion1_insert.zip`

### Step C: 학습 결과 시각화
```bash
./isaaclab.sh -p source/motion1/scripts/play_insert.py
```
- cell 4 walls visual (회색) 보임
- keep_alive=True (창 자동 안 닫힘)
- 정렬 행동 + reset 동작 직접 확인

### Step D: chain runner 검증 (Step 4)
```bash
./isaaclab.sh -p source/motion1/scripts/play_motion_chain_with_grasp_insert.py --hold_s 30 --repeat 5
```
- 박스 + cell 모두 random spawn
- grasp PPO (단계 2) + insert PPO (단계 4) inference
- transport 끝 ee yaw → cell_yaw 정렬
- success rate 보고 다음 결정

### Step E: success rate 따라 결정
- **chain success rate ≥ 0.5**: ✅ Step 5 (multi-box / 3×3 grid) 진행
- **chain success rate < 0.3**: ❌ insert RL 추가 학습 또는 dataset 분포 재검토 (cell 범위, transport noise 등)
- **stuck**: SAC + HER 로 algorithm 전환 (사용자가 이전에 명시한 대안)

### Step F: Step 5 multi-box / 3×3 grid 확장
- `tasks/grid/` 새 폴더
- chain runner 가 박스 9개 / cell 9개 처리
- 박스 마다 grasp + insert inference

---

## 11. 핵심 환경 정보 (cheatsheet)

```
Conda env       : env_isaaclab (/home/jaewoo/miniconda3/envs/env_isaaclab/)
Activate        : source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
Run script      : ./isaaclab.sh -p source/motion1/scripts/<파일.py> [opts]

GPU             : RTX 4070 12GB
RL 학습 사용량  : num_envs 128 → ~3GB (충분)

Box / Cell coordinates (env-rel) — 2026-05-10 변경:
  BOX_SPAWN     = (0.30, -0.10, 0.07)
  CELL_CENTER   = (0.30, -0.30, 0.0)
  CELL_SPAWN_XY_NOISE = 0.05  (±5cm)
  CELL_SPAWN_YAW_MAX  = 1.396 (±80°)
  BOX_SPAWN_XY_NOISE  = 0.10  (±10cm) (chain runner 에서는 0 으로 fixed 시도 가능)
  PRE_GRASP_Z   = 0.17
  GRASP_Z       = 0.115
  TRANSPORT_Z   = 0.26
  PLACE_Z       = 0.165

Insert RL — 2026-05-10 cfg:
  action_scale_xy   = 0.005  (5mm/step) — fine 정렬용
  action_scale_yaw  = 0.05
  ee_fixed_z        = 0.26
  align_xy_threshold = 0.010 (10mm)
  align_yaw_threshold = 0.087 (~5°)
  success_hold_steps = 15 (0.25초)
  reward_xy_align_gain        = 80.0  (exploration)
  reward_xy_align_gain_close  = 200.0 (정밀 정렬, dual reward)
  reward_yaw_align_gain       = 5.0
  reward_success_bonus        = 50.0 (aligned 매 step)
  reward_success_lump         = 5000.0 (success terminate)
  is_grasping mask            = 제거 (grasp env 와 동일)
  drop penalty                = 제거 (termination 만으로 자연 페널티)
  sim_dt = 1/60, decimation = 1, control rate 60Hz
  yaw 비누적 (xy 와 동일 패턴, _extract_ee_yaw 사용)

OMY:
  arm joints    : joint1~6
  gripper joints: rh_r1_joint, rh_r2, rh_l1, rh_l2 (mimic 깨짐, tip_ratio 2.3)
  finger bodies : rh_p12_rn_l2, rh_p12_rn_r2 (월드 좌표 평균 = grip center)

EE base quat (수직 아래) : (0, 1, 0, 0) (w, x, y, z) - hardcoded
EE target quat = R_z(ee_yaw) ⊗ base_ee_quat
```

---

## 12. Insert RL 학습 timeline (2026-05-09 ~ 2026-05-10)

| ver | dataset | cfg 변경 | 결과 (success_recent) |
|-----|---------|---------|---------------------|
| v1  | dirty (cell ±2cm, noise 없음) | yaw 누적 + drop penalty 100 | 0% |
| v2  | dirty (v1 dataset) | **yaw 비누적** + **drop penalty 제거** | 0% (dataset noise 부재가 진짜 원인) |
| v3  | v3_dirty (cell ±5cm + transport noise 3~5cm 추가) | (v2 동일) | 0% (dataset 손상 20.5%) |
| v4  | v3_clean (769 sample, 손상 sample filter) | 동일, **2M step** | 0% |
| v5/6/7 dataset | (재수집 시도들) | — | — |
| v6_clean | dataset 재수집 (박스/cell 0.30,-0.30 가까이, cell ±5cm + transport 3~5cm) | — | — |
| v6  | v6_clean (769 sample) | (v4 동일) | 0% |
| v7  | v6_clean | **align threshold 완화** (5mm→10mm, 2.86°→5°, hold 30→15) | 1% |
| v8  | v6_clean | v7 + 2M resume = **누적 4M** | 2% |
| v9  | v6_clean | v8 + 2M resume = **누적 6M** | 1~2% (정체) |
| v10 | v6_clean | v8 fresh + **action 5mm** + reward gain 200 (단독, 2M) | 0% (gain 200 너무 sharp) |
| v11 | v6_clean | action 5mm + gain 80 (4M) | 0~1% |
| v12 | v6_clean | v11 + **is_grasping mask 제거** (4M) | 0~1%, r_xy_align 0.92 |
| v13 | v6_clean | v12 + **dual reward** (gain 80 + 200, 4M, sim_dt 1/120) | 1%, 진단 metric 추가 |
| v14 | v6_clean | **sim_dt 1/120 → 1/60 + decimation 1** (4M) | **1~3%** |
| v15 | (취소) | EE_OFFSET 0~10cm 첫 시도 | — |
| v16 | (취소) | EE_OFFSET 3~5cm 원복 시도 | — |
| v17 | **v17_clean** (764 sample, drop 23% filter) | v14 + EE_OFFSET 0~10cm 확장 (chain runner 분포 mismatch 해결) | **2~3%, is_grasping 85% (v14 의 9배)** |
| v18 | v17_clean | v17 + reward 정밀화 (xy gain 80→40, yaw 5→15, align 10→5mm, hold 15→30, threshold 5mm) | **14%, xy_dist 5.8cm** |
| v19 | v17_clean | v18 + r_smooth 0.01→0.1, fail_xy 0.30→0.10, r_far_penalty(w=20, 1cm 부터) | **15%, xy_dist 3.5cm** ← 분포 확장 + 페널티 효과 |
| v20 | v17_clean | v19 + action_scale_xy 5→2mm + r_action_penalty 0.5 | **1% 악화** (2mm + penalty 둘 다 너무 강함, cell 도달 능력 잃음) |
| v21 | v17_clean | v19 + r_action_penalty 0.1 (action_scale 5mm 원복) | **17%, yaw_err 2.5°** ← 현재 best |

## 2026-05-12 시도 — 진단 + reward shaping (v17~v21)

### Root cause 진단 (v14 → v17)
- **분포 mismatch 발견**: collect 의 EE_OFFSET 3~5cm vs chain runner 의 offset 0. chain runner 시작 자세가 학습 분포 밖이라 fail
- **해결**: EE_OFFSET 0~10cm 으로 확대 + drop sample 후처리 filter (23% drop)
- **결과 (v17)**: is_grasping_rate **10% → 85%** (9배 ↑). 단 success 그대로 2-3%

### Reward 정밀화 (v17 → v18)
- align threshold 10mm → 5mm, yaw 5° → 3°, hold 15 → 30
- gain 80 (wide) → 40, yaw 5 → 15 (sharper)
- 결과: success 14%, xy_dist_mean 5.8cm

### 발산 방지 (v18 → v19)
- 사용자 본 발산 케이스 (ee 가 cell 에서 15cm 떨어진 채 머무름)
- fail_xy_threshold 0.30 → 0.10 (10cm 밖 즉시 terminate)
- r_far_penalty (w=20, 1cm 초과 거리 비례 페널티)
- 결과: success 15%, xy_dist 3.5cm. 단 fa_rate_5cm 32%, very_far_8cm 11% 남음

### Fine motor 시도 실패 (v19 → v20)
- 진단 결과 `action_norm_mean = 1.33`, **항상 max action**. fine motor X
- action_scale_xy 5mm → 2mm + r_action_penalty 0.5 동시 적용
- 결과: success 1% **악화**. action 작아졌으나 cell 도달 능력 잃음

### 균형점 (v20 → v21)
- action_scale 5mm 원복 + r_action_penalty 0.5 → 0.1 약하게
- 결과: **success 17% (현재 best)**, yaw_err 2.5°, xy_dist 3.4cm, close_rate_2cm 46%

## 미해결 (v21 시점)
- **xy 정렬 정밀도 한계**: xy_aligned_rate 20-23%. 1cm 안 도달은 못 함 (action_scale_xy 5mm 의 정밀도 한계)
- **action_norm 여전히 max** (penalty 0.1 약함): fine motor 학습 X
- 사용자 관찰: 박스가 살짝 잘못 잡혔을 때 정렬 잘 되고, 똑바로 잡힐 때 발산하는 경향 (의문)
- 한 방향 drift 가끔 발생 (box swing torque 가능성)

## ckpt 백업 위치 (절대 건드리지 말 것)
- `motion1_insert_v1.zip` ~ `motion1_insert_v14_best.zip` (이전 시도들)
- `motion1_insert_v17.zip`, `v18.zip`, `v19.zip`, `v20.zip` (v17~v20 시점 백업)
- `motion1_insert_best.zip` (현재) = **v21 best (success_recent 14~17%)**

## Dataset 위치
- `insert_handoff_states_v17_clean.npz` (764 sample, EE_OFFSET 0~10cm, drop filter 후)
- `insert_handoff_states.npz` = default path. 현재 v17_clean 복사본

---

## 13. Camera 통합 (Step 7) — 다음 작업

사용자 요청 — `play_motion_chain_with_grasp_insert.py` 의 camera 붙인 버전 만들기.

### 작업 단계
1. **camera sensor 부착** — IsaacLab `CameraCfg` 또는 USD 카메라
2. **detection** — YOLO (`ultralytics`) 또는 segmentation mask 직접
3. **image → world coord** — camera intrinsic + extrinsic
4. **vision 결과를 RL obs 에** — 현재 ground truth 자리에 vision 추정값

### 학습 영향 분석
- 현재 정책 학습: ground truth obs (sim 의 정확한 cell_xy, box_xy)
- vision noise 추정 (5mm) ≪ align threshold (10mm) → **현재 정책 그대로 사용 가능**
- noise 더 크면 → domain randomization 으로 재학습

### 시작점
- 새 파일: `play_motion_chain_with_grasp_insert_camera.py` (이름 자유)
- 기존 `play_motion_chain_with_grasp_insert.py` 복사 후 camera 추가
- IsaacLab 의 Camera 예제: `source/standalone/...camera...py` 참고

---

## 14. 미해결 의문 — chain runner stage 4 안 되는 이유

play_insert.py (학습 env) = OK, chain runner (SimulationContext + InteractiveScene) = X. 같은 ckpt + 같은 obs 식 + 같은 action 적용 식. 그런데 정책 행동 다름.

알려진 차이 (모두 fix 했음):
- ✅ VecNormalize 직접 호출 (sb3 의 `normalize_obs()` batch (1, 7))
- ✅ action_scale_xy 분리 (grasp 0.01 / insert 0.005)
- ✅ sim_dt 1/120 → 1/60 + decimation 1 (학습 cfg 일치)
- ✅ cell wall collision off 시도 (효과 X)
- ✅ insert offset 0.5~1cm (이전 3~5cm)

**미해결 — 다음 세션에서 분석할 것**:
- chain runner 의 stage 3d 끝 자세가 학습 시작 분포 와 정확히 같은가?
- IK 정확도, settle 시간, ee_vel 잔여 차이?
- obs 의 element 순서, dtype, 정규화 미묘한 차이?

debug print 로 chain runner stage 4 첫 5 step 의 obs/action 출력 + play_insert.py 첫 5 step 비교 권장. 같은 obs 받으면 같은 action 출력 — 다르면 obs 가 진짜 다름.
