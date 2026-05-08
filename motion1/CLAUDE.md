# CLAUDE.md — motion1

다음 Claude 세션이 빠르게 컨텍스트를 잡기 위한 메모.
설계 전체는 [PLAN.md](PLAN.md) 참고. 이 파일은 **사용자가 결정한 사항** 과 **현재 작동 상태** 를 모은다.

---

## 1. 한 줄 요약

example7/example7_2 의 RL 단독 학습이 박스 적재에서 실패 (2~3M step에도 success 0)
→ 큰 운동(reach/transport/retract)은 **motion planning**, contact-rich(grasp / insert)만 **RL** 로 분해 재설계.

**현재 진행**: ✅ Step 1 (motion-only 5단계 pipeline) 완료. 박스 1개 잡고 셀 1개에 적재 OK. 다음은 Step 2 (Grasp RL env) 부터.

---

## 2. 현재 작동하는 코드 / 핵심 파라미터

### 파일
- [scripts/play_motion_chain.py](scripts/play_motion_chain.py) — single-file motion-only pipeline (RL 미사용)

### 실행
```bash
./isaaclab.sh -p source/motion1/scripts/play_motion_chain.py
./isaaclab.sh -p source/motion1/scripts/play_motion_chain.py --gripper_close 0.8 --hold_s 30
./isaaclab.sh -p source/motion1/scripts/play_motion_chain.py --headless --repeat 3
```

### 검증된 파라미터 (현재 작동)

| 변수 | 값 | 의미 |
|---|---|---|
| `BOX_SPAWN` | (0.45, -0.10, 0.07) | 박스 중심 위치. z=0.07 = 박스 바닥 ground 위 1.1cm |
| `BOX_SPAWN_ROT` | (1, 0, 0, 0) | 박스 yaw=0 (회전 없음) |
| `PRE_GRASP_Z` | BOX_SPAWN[2] + 0.06 = 0.13 | reach 끝 — 박스 위 6cm |
| `GRASP_Z` | BOX_SPAWN[2] + 0.06 = 0.13 | 박스 잡는 z (현재는 PRE_GRASP_Z 와 동일 — descend 안 함) |
| `LIFT_Z` | 0.30 | 들어올린 후 z |
| `TRANSPORT_Z` | 0.25 | 셀 위 도달 z (insert 시작점) |
| `PLACE_Z` | BOX_SPAWN[2] + 0.10 = 0.17 | release 시 EE z (insert 최종, 자유낙하 11cm) |
| `RETRACT_Z` | 0.25 | retract 위로 z |
| `--gripper_close` (default) | 0.8 rad | 박스 잡는 grip cmd. tip ratio 2.3 적용 |
| friction | static/dynamic = 3.0, combine = "max" | 박스 슬립 방지 |

### 핵심 디자인 결정 (검증 후 확정)

| 항목 | 결정 | 이유 |
|---|---|---|
| 환경 base | standalone (`SimulationContext` + `InteractiveScene`) | RL 단계와 분리, DirectRLEnv 안 씀 |
| IK 라이브러리 | Isaac Lab `DifferentialIKController` (dls) | jacobian 기반, 양 finger 평균 jacobian 사용 |
| EE / grip center | finger body `rh_p12_rn_l2` + `rh_p12_rn_r2` **월드 좌표 평균** | mimic 깨진 finger 의 안정적인 ee 정의 |
| EE target quat | **`(0, 1, 0, 0)` hardcoded** (180° around world X) | 수직 아래 + finger Y 양옆. URDF 분석 + 시도 결과 확정 |
| Reach quat | **slerp** (home_quat → target_quat) | reach 도중 회전을 점진적으로 풀어 IK 안정 |
| 그리퍼 action | RL action 매핑 무시. **arm 6 + gripper 4 joint position 직접 명령** | mimic 깨짐 → 코드 레벨 강제 |
| Gripper tip ratio | **기저 (l1/r1) : 끝점 (l2/r2) = 1 : 2.3** | example7 fallback 자세 분석. finger 끝점이 더 굽혀져 안쪽 모임 |
| 박스 yaw spawn | **0 (회전 없음)** | RL 단계에서 randomize 예정 |
| Trajectory 보간 | cartesian 직선 + 매 step IK | 자연스럽고 직선 보장 |

---

## 3. 박스 / 셀 / 로봇 사양

### 박스
- 크기 (m): 0.139(x) × 0.044(y) × 0.118(z)
- long edge X(13.9cm) > finger sep open(11.3cm) → **long edge 잡기 불가능**, **short edge Y 잡기만 가능**
- 질량: 0.3 kg
- friction = 3.0 (combine_mode="max")

### 셀 (1×1)
- 셀 내부 (m): 0.16(x) × 0.065(y)
- 격벽: 두께 0.008, 높이 0.12
- 셀 중심 (env-rel): (0.25, -0.45, 0.0)

### OMY 로봇
- USD: `/home/jaewoo/IsaacLab/source/omy_f3m_urdf/OMY.usd`
- arm joints: `joint1` ~ `joint6`
- gripper joints: `rh_r1_joint`, `rh_r2`, `rh_l1`, `rh_l2` (mimic 깨짐 → 코드 레벨 비율 명령)
- finger bodies: `rh_p12_rn_l2`, `rh_p12_rn_r2`
- URDF 분석: `rh_p12_rn_*2` body 의 local frame 은 parent (`rh_p12_rn_*1`) 와 동일 (rpy=0). finger sep = 부모 local Y 축, gripper close 회전축 = 부모 local X.

---

## 4. 5단계 motion-only 파이프라인 (현재 작동)

| 단계 | 동작 | 비고 |
|---|---|---|
| 1. Reach | home → pre_grasp(z=0.13) | quat slerp 로 home_quat → (0,1,0,0) 점진적 회전 |
| 2. Grasp | pre_grasp → grasp(z=0.13) → close(0.8 rad) → hold | tip ratio 2.3 으로 finger 끝점 추가 굽힘 |
| 3. Transport | grasp → lift(z=0.30) → transport(z=0.25) | gripper closed 유지 |
| 4. Place | transport → place(z=0.17) → gripper open | release 후 박스 11cm 자유낙하 → 셀 안착 |
| 5. Retract | place → retract(z=0.25) → home | gripper open 유지 |

각 stage 끝에 `SETTLE_S = 0.8s` hold (IK 수렴). repeat 옵션 + finite hold (`--hold_s`) 로 좀비 프로세스 없음.

---

## 5. 알려진 제약 / 주의사항

1. **OMY USD mimic 깨짐** → 4 gripper joint 같은 값 명령 시 끝점 link 가 부족하게 굽음. **tip_ratio = 2.3** 으로 보정 ([play_motion_chain.py:377-383](scripts/play_motion_chain.py#L377))
2. **OMY arm reach 한계** → LIFT_Z 0.50 이상이면 singularity 근처에서 EE 떨림. 현재 LIFT_Z=0.30 로 안전하게 운영
3. **finger sep open = 11.3cm < 박스 long edge 13.9cm** → 박스 long edge 양옆 grasp 물리적 불가능. short edge Y 만 가능
4. **OMY USD 절대경로 하드코딩** (`/home/jaewoo/IsaacLab/source/omy_f3m_urdf/OMY.usd`)
5. **자동 재시작 주의** — 옛날 Claude session 이 background bash 로 motion1 을 자동 재실행한 사례 있음. session 잘 정리할 것

---

## 6. 다음에 할 일 (Step 2 부터)

### Step 2 — Grasp RL env (예상 1~2시간 + 학습 30분~1시간)

**목표**: 박스 위 5cm 도달한 자세에서 시작 → close 명령 + 미세 위치 조정으로 박스 잡기. motion planner 가 reach 끝낸 자세 randomize 흡수.

**파일**:
- `tasks/grasp/grasp_env.py` (DirectRLEnv 상속, example7 골격 참고)
- `tasks/grasp/grasp_env_cfg.py`
- `scripts/train_grasp.py` (SAC, num_envs=64)
- `scripts/play_grasp.py`

**Action**: cartesian Δxyz 3차원 + gripper 1 = 4. 관절 각도는 IK 로 변환 (motion1 RL 설계 원칙).

**State** (~20 dim):
- arm joint pos/vel (12)
- gripper close state (1)
- ee_pos rel (3)
- box_pos rel (3)
- ee → box 벡터 (3)

**Reward** (단순):
- `r_approach = exp(-50 * dist(ee, box)²)`
- `r_grip_close = (gripper_close > 0.5).float()`
- `r_grasp_success = 박스 z > 시작 z + 1cm AND grip closed AND box near grip`
- 가중치: 5 / 3 / 50

**시작 상태 randomize**:
- 박스 spawn 위치 ±2cm
- ee 시작 자세 = 박스 위 5cm + 약간의 랜덤 (±1~2cm)

**검증**: 박스 1cm 살짝 들기 success rate ≥ 90%.

---

### Step 3 — Insert RL env (예상 1~2시간 + 학습 1시간)

**목표**: 박스 잡힌 채 셀 위 5cm 도달한 자세에서 시작 → 정렬 + 셀 안 삽입 + release.

**파일**:
- `tasks/insert/insert_env.py` (HER GoalEnv 형식)
- `tasks/insert/insert_env_cfg.py`
- `scripts/train_insert.py` (SAC + HER, num_envs=64)
- `scripts/play_insert.py`

**Action**: cartesian Δxyz + gripper = 4 차원.

**State** (~25 dim, GoalEnv):
- core: arm joint pos/vel (12), gripper close (1), ee_pos rel (3), box_pos rel (3), box_quat (4)
- **`holding_box` 플래그 (1)** — 그리퍼-박스 거리 또는 contact 기반 binary. **필수** (motion1 RL 설계 원칙).
- achieved_goal: box xy + endpoint xy (yaw 표현)
- desired_goal: 셀 중심 xy + 고정 endpoint

**Reward**:
- `r_xy_keep = exp(-100 * xy_dist²)`
- `r_yaw = exp(-100 * endpoint_diff²)`
- `r_descent = where(xy_aligned & yaw_aligned, exp(-30*(z-0.06)²), 0)`
- `r_release = (deep_enough & grip_open).float()`
- `r_success = (in_cell & on_floor & yaw_aligned).float()`

**시작 상태**:
- handoff: motion1 chain 의 transport 끝 자세 (셀 위 5cm, 박스 잡힌 상태)
- 박스 yaw randomize ±30°, xy ±2cm

**검증**: 셀 안 정렬 + 삽입 + release success rate ≥ 70%.

---

### Step 4 — Chain Runner (motion + RL 통합)

**파일**: `chain/chain_runner.py`, `scripts/play_chain.py`

**시퀀스**:
1. **Reach** — motion plan (현재 motion1 의 stage 1)
2. **Grasp** — RL policy (Step 2 학습된 정책)
3. **Transport** — motion plan (현재 motion1 의 stage 3)
4. **Place + Insert** — RL policy (Step 3 학습된 정책)
5. **Retract** — motion plan (현재 motion1 의 stage 5)

각 단계 전환 조건: 다음 단계 시작 가능한 상태 충족 시 (예: grasp 성공 신호 + 정해진 ee 위치).

**검증**: 박스 1개 적재 success rate ≥ 70%.

---

### Step 5 — Multi-box / 3×3 grid 확장

- 박스 N 개 + 셀 N 개 (PLAN.md 의 3×3 grid 모듈 추가)
- box → cell 매핑 (row-major 또는 randomize)
- chain runner 에 `run_all()` 추가 (PLAN.md 4.4 참고)

**검증**: 3 박스 적재 success rate ≥ 70%.

---

### Step 6 (옵션) — 모듈화

`scripts/play_motion_chain.py` single-file → PLAN.md 의 모듈 구조 (`motion/`, `chain/`, `tasks/`) 로 분리.

---

### Step 7 (멀리) — Sim2real

- ground truth pose → YOLO 카메라 vision pipeline 으로 교체
- handoff randomization 강화
- example7 의 sim2real 계획 ([memory/project_simreal_plan.md](../../../.claude/projects/-home-jaewoo-IsaacLab/memory/project_simreal_plan.md)) 참고

---

## 7. 환경 정보

- Conda env: `env_isaaclab` (`/home/jaewoo/miniconda3/envs/env_isaaclab/`)
- 실행 활성화: `conda activate env_isaaclab` 후 `./isaaclab.sh -p ...`
- GUI 디버깅: 옵션 안 줌 (default)
- headless 학습: `--headless --num_envs 64`

---

## 8. 절대 건드리지 말 것

- `source/example5/` (frozen lift policy)
- `checkpoints/example5.zip`, `checkpoints/example5_vecnorm.pkl`, `checkpoints/handoff_states.npz`
- `source/omy_f3m_urdf/OMY.usd` (로봇 모델)

---

## 9. 사용자 스타일 메모

- 한국어로 설명 선호
- 코드 변경 후 즉시 시각화로 검증 원함 (`--hold_s` 옵션 활용)
- 페널티 (음수 reward) 회피, all-positive reward 선호
- RL 가중치 1~100 범위
- 학습 200k~2M step 범위
- 코드 직접 수정해서 시도하는 편 — 작은 단위로 변경 → 사진/log 공유

---

## 10. 진행 체크리스트

- [x] PLAN.md 작성 / 사용자 confirm
- [x] CLAUDE.md (이 파일) 작성
- [x] 폴더 구조 생성
- [x] `scripts/play_motion_chain.py` prototype 작성
- [x] GUI 로 5 단계 작동 검증 (gripper close / friction / tip ratio / target quat / lift z 모두 튜닝 완료)
- [ ] **다음**: Step 2 — `tasks/grasp/grasp_env.py` 작성 (Grasp RL env)
- [ ] Step 3 — `tasks/insert/insert_env.py` (Insert RL env, holding_box state 필수)
- [ ] Step 4 — `chain/chain_runner.py` (motion + RL 통합)
- [ ] Step 5 — multi-box / 3×3 grid 확장
- [ ] Step 6 — 모듈화 (single-file → motion/, chain/, tasks/)
- [ ] Step 7 — sim2real (vision, YOLO 통합)
