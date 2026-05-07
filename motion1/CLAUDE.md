# CLAUDE.md — motion1

다음 Claude 세션이 빠르게 컨텍스트를 잡기 위한 메모.
설계 전체는 [PLAN.md](PLAN.md) 참고. 이 파일은 **사용자가 결정한 사항** 과 **합의된 디자인 선택** 만 모음.

---

## 1. 한 줄 요약

example7/example7_2 의 RL 단독 학습이 박스 적재에서 실패 (2~3M step에도 success 0)
→ 큰 운동(reach/transport/retract)은 **motion planning**, contact-rich(grasp / insert)만 **RL** 로 분해 재설계.

**현재 단계**: motion planning **단독** 으로 5단계 파이프라인이 끝까지 작동하는지 시연 (RL 아직 없음).

---

## 2. 사용자가 확정한 디자인 결정 (2026-05-07 대화 기준)

| 항목 | 결정 |
|---|---|
| 박스 / 셀 갯수 | **1박스 / 1셀** 부터 시작 → 작동 시 multi 로 확장 |
| 실행 모드 | **GUI 시각화** 로 디버깅 (num_envs=1) |
| 환경 base | **standalone** (Isaac Lab `SimulationContext` + `InteractiveScene`). DirectRLEnv 안 씀 — RL 단계와 분리 |
| 첫 형태 | **single-file prototype** (`scripts/play_motion_chain.py`) → 작동 시 PLAN.md 모듈 구조로 분리 |
| IK 라이브러리 | Isaac Lab 내장 **`DifferentialIKController`** (jacobian 기반, dls 방식) |
| 박스 yaw spawn | **고정 0** (RL 붙일 때 randomize) |
| 박스 잡는 방향 | 그리퍼가 박스 **y(짧은 변, 4.4cm)** 양옆에서 잡음 |
| 박스 잡는 z 위치 | **박스 중심** |
| Home 자세 | `OMY_CFG.init_state` (joint1=0, joint2=-1.55, joint3=2.66, joint4=-1.1, joint5=1.6, joint6=0) 그대로 |
| EE / grip center | finger body `rh_p12_rn_l2` + `rh_p12_rn_r2` **월드 좌표 평균** |
| 그리퍼 action | RL action 매핑 무시. motion planner 가 **arm 6 + gripper 4 joint position 직접 명령** (4개 동일 값으로 mimic 강제) |
| 그리퍼 닫힘 명령값 | 박스 0.3kg / friction μ=1.0 / stiffness 1500 N/rad 기반 → **초기 0.5 rad** 으로 시작 후 시각화에서 박스 안정성 확인하며 조정 (0.6, 0.7, ...) |
| 그리퍼 열림 명령값 | **0.0 rad** |
| Trajectory 보간 | **cartesian 직선 + 매 step IK** (자연스럽고 직선 보장) |
| Vision | 현재는 ground truth pose. **나중에 YOLO 카메라로 박스/셀 좌표 받아옴** |

---

## 3. 박스 / 셀 / 로봇 사양 (PLAN.md 와 example7 cfg 에서 확정)

### 박스
- 크기 (m): 0.139(x) × 0.044(y) × 0.118(z) — 가늘고 세워진 형태
- 질량: 0.3 kg
- spawn 위치 (env-relative): (0.45, -0.10, 0.06)
- friction: static/dynamic = 1.0

### 셀 (1×1)
- 셀 내부 (m): 0.16(x) × 0.065(y)
- 격벽: 두께 0.008, 높이 0.12
- 셀 중심 (env-relative): (0.25, -0.45, 0.0) — z=0 은 셀 바닥
- spawn 코드: example7 의 `place_env._spawn_grid_walls()` 참고 또는 단순 4 벽 직접 spawn

### 로봇 (OMY)
- USD: `/home/jaewoo/IsaacLab/source/omy_f3m_urdf/OMY.usd`
- arm joints: `joint1` ~ `joint6` (6 DOF)
- gripper joints: `rh_r1_joint`, `rh_r2`, `rh_l1`, `rh_l2` (4-joint mimic — USD 깨져 있어 코드 레벨에서 동일 값 명령)
- finger bodies: `rh_p12_rn_l2`, `rh_p12_rn_r2`
- cfg: `source/omy/omy_robot_cfg.py` → `OMY_OFF_SELF_COLLISION_CFG`

---

## 4. 5단계 파이프라인 (motion-only 시연용)

| 단계 | 동작 | 비고 |
|---|---|---|
| 1. Reach | home → 박스 위 5cm (z ≈ 0.118) | gripper open (0.0) |
| 2. Grasp | ① 박스 위 5cm → 박스 중심 (z ≈ 0.060) 천천히 하강<br>② gripper close (0.5 rad)<br>③ 잠시 hold (그립 안정) | 모두 motion-only |
| 3. Transport | 박스 잡은 채 z=0.30 까지 lift → 셀 위 (0.25, -0.45, z=0.18) 로 이동 | gripper closed 유지, 박스 중력으로 떨어지지 않게 |
| 4. Place + Insert | 셀 위 z=0.18 → 박스 바닥이 셀 바닥 닿도록 천천히 하강 → gripper open | 격벽 0.12, 박스 높이 0.118 → ee z ≈ 0.06 까지 하강 |
| 5. Retract | ee 위로 (z=0.30) → home 자세 | gripper open 유지 |

각 단계 trajectory: cartesian 직선 보간 + 매 step `DifferentialIKController` 로 joint target 계산 + `set_joint_position_target` 적용.

EE orientation: home 측정값을 모든 단계에서 유지 (수직 grasp / 수직 place). 박스 yaw=0 / 셀 yaw=0 이라 회전 불필요.

---

## 5. RL 단계 디자인 (motion-only 검증 후 적용)

사용자 지시 (motion1 RL 설계 원칙):

1. **Action = cartesian Δxyz 정도만** (3 차원, 또는 + gripper 1 = 4 차원). 관절 각도는 IK 라이브러리로 변환.
2. **State 는 task 별로 직접 판단해서 구성** (ee_pos / box_pos / 상대벡터 / gripper_close_state 등).
3. **Insert env state 에 "박스를 잡고 있다(holding)" 플래그 필수** — 그리퍼-박스 거리 또는 contact 기반 binary. 박스 떨어뜨린 후 정렬 보상 받으려는 행동 차단.

→ `tasks/grasp/grasp_env.py`, `tasks/insert/insert_env.py` 에 적용.

---

## 6. 환경 정보

- Conda env: `env_isaaclab` (`/home/jaewoo/miniconda3/envs/env_isaaclab/`)
- 실행: `./isaaclab.sh -p source/motion1/scripts/play_motion_chain.py` (GUI 기본, 디버깅용)
- headless 학습: `./isaaclab.sh -p ... --headless`

---

## 7. 절대 건드리지 말 것

- `source/example5/` (frozen, lift policy 검증됨)
- `checkpoints/example5.zip`, `checkpoints/example5_vecnorm.pkl`, `checkpoints/handoff_states.npz`

---

## 8. 사용자 스타일 메모

- 한국어로 설명 선호
- 코드 변경 후 즉시 시각화로 검증 원함
- 페널티 (음수 reward) 회피, all-positive reward 선호
- RL 가중치 1~100 범위 (큰 값 100~1000 회피)
- 학습 200k~2M step 범위
- "마음대로 해" / "필요한 라이브러리 다 가져다 써" 등 위임 자주 함 → 합리적 default 결정 후 빠르게 시각화로 검증

---

## 9. 진행 체크리스트

- [x] PLAN.md 작성 / 사용자 confirm
- [x] CLAUDE.md (이 파일) 작성
- [ ] 폴더 구조 생성
- [ ] `scripts/play_motion_chain.py` prototype 작성
- [ ] GUI 로 5 단계 작동 검증 (gripper close 값 fix 포함)
- [ ] 작동 시 PLAN.md 모듈 구조 (`motion/`, `chain/`, `tasks/`) 로 분리
- [ ] grasp RL env 작성 (cartesian action + IK)
- [ ] insert RL env 작성 (cartesian action + IK + holding state)
- [ ] chain runner 통합 (motion + RL)
- [ ] multi-box / 3×3 grid 확장
