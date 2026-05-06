# OMY 로봇 박스 적재 RL — 외부 자문 요청

> 6단계 흐름 — **목표 → 접근 → 진행 → 문제 → 자문 → 남은 일**

---

## 1. 무엇을 위해서 (목표)

NVIDIA Isaac Lab 시뮬레이션에서 **OMY 6-DOF 로봇팔 + 병렬 그리퍼** 가 유연 블록을 박스 내부 지정 셀에 적재하는 강화학습 시스템 개발 중.

**최종 시퀀스**: 인식 → 집기 → 들기 → 이동 → 셀 정렬 적재

이후 **sim-to-real** 로 실제 OMY + 카메라 환경 배포 예정.

---

## 2. 어떻게 할 예정인지 (접근)

### 2.1. 시스템 구성

| 항목 | 값 |
| --- | --- |
| 시뮬레이터 | NVIDIA Isaac Lab / Isaac Sim 4.5 (GPU 병렬) |
| 로봇 | ROBOTIS OMY (6-DOF arm + 4-joint mimic gripper) |
| RL 라이브러리 | Stable Baselines 3 (SB3) |
| 알고리즘 | grasp/lift = **PPO**, placement = **SAC + HER** |
| Action | 7-dim continuous (joint position delta + gripper) |
| Observation | 31-dim Dict (HER GoalEnv 형식) |
| 학습 구조 | end-to-end 단일 정책이 아닌 **task 분리 hierarchical 구조** |

### 2.2. 왜 Hierarchical 구조

처음에 reach + grasp + lift + place 를 **단일 정책으로** 학습 시도. 그러나:

- 보상 함수가 단계마다 달라짐
- grasp 만 학습하고 lift 단계로 안 넘어감
- 충분한 reward 얻으면 행동 탐색 멈춤
- sparse reward 문제 심각
- stage transition trigger 설계 어려움

→ 작업을 **3 단계로 분리**, 각각 별도 정책 학습. 각 단계 success state 를 저장 (= **handoff** 데이터) 해서 다음 단계 초기 상태로 사용.

### 2.3. 3-stage 파이프라인

| Stage | 정책 | 입력 | 목표 | 학습 알고리즘 | 상태 |
| --- | --- | --- | --- | --- | --- |
| **1** | example5 | 박스 spawn 위치 | grasp + z=0.20 까지 lift | PPO | ✅ 완료 |
| **2** | example7 | example5 의 handoff (박스 잡고 z≈0.17) | 박스를 셀 위에 xy 정렬 + z=0.30 유지 | SAC + HER | ⏳ **★ 학습 막힘 ★** |
| **3** | example7_2 | example7 의 success state | yaw 정렬 + 셀 안 적재 | SAC + HER | ⏸️ 대기 |

### 2.4. 정책끼리 어떻게 연결되는가

> 핵심: **한 정책의 끝 = 다음 정책의 시작** 으로 만드는 것

#### (1) 학습 시점 연결 — Handoff 데이터

각 stage 의 success state (로봇 관절 위치 + 박스 위치 + 박스 회전) 를 파일로 저장 → 다음 stage 학습 시 reset 마다 그 파일에서 랜덤 샘플링해서 시작 상태로 설정.

흐름:

- Stage 1 학습 완료 → success 순간 100개 캡쳐 → `handoff_states.npz` 저장
- Stage 2 학습 시 reset 마다 그 파일에서 1개 샘플 → "박스 잡고 들고 있는" 상태에서 학습 시작
- Stage 2 학습 완료 → success 순간 캡쳐 → `handoff_xy_aligned.npz` (★ 미수집 ★)
- Stage 3 학습 시 같은 방식, "이미 셀 위 정렬된" 상태에서 학습 시작

> **왜 이렇게**: 다음 stage 가 이전 task 다시 풀 필요 없음. Stage 2 는 grasp 학습 안 하고 "이미 잡은 박스 옮기기" 만, Stage 3 는 lift 안 하고 "이미 셀 위에 있는 박스 정렬" 만 학습. **각 stage task 가 좁아져서 학습 가능해짐**.

**현재 상태**:
- `handoff_states.npz` (Stage 1 → 2): ✅ 존재, 100 samples, z 평균 0.171
- `handoff_xy_aligned.npz` (Stage 2 → 3): ❌ 미수집 (Stage 2 학습이 끝나야 가능)

#### (2) 추론 시점 연결 — State Machine

학습은 단계별 분리, **추론 시에는 단일 환경에서 정책을 순차 호출**해서 end-to-end 작업 수행. 별도 디렉토리 `source/example8_2/` 에서 chain script 운영.

작동 방식:

- Phase 변수가 어느 정책을 호출할지 결정 (`GRASP_LIFT` / `PLACE_HOVER` / `INSERTING`)
- Phase 전환 조건은 **물리량 기반** — 박스 z, xy 거리, 그립 상태. 학습된 success criterion 과 동일하게 맞춤
- 같은 환경 state 라도 정책마다 obs 형식 다름 → chain script 가 매 step 각 정책용 obs 를 변환해서 넘김
   - example5 = 34-dim 벡터 (LiftEnv 형식)
   - example7 / 7_2 = 31-dim Dict (HER GoalEnv 형식)
- example5 는 PPO + VecNormalize → 정규화 통계도 같이 적용
- 박스 3개 동시 환경에서는 active box 만 정책에 노출, 박스 i 완료 시 i+1 박스로 전환

### 2.5. 환경 세부

**박스**:
- 크기 0.139 × 0.044 × 0.118 m (가늘고 세워진 형태)
- 질량 0.3 kg
- example5 의 grasp 가 박스 중심 아닌 **camera-side 한쪽 끝**을 잡음 → 그리퍼에 비대칭 매달림 (Stage 3 yaw 정렬에 중요)

**셀** (3×3 그리드, 현재 1개만 사용):
- 셀 내부 0.17 × 0.06 m, 격벽 0.008 m × 높이 0.12 m
- 셀 중심 (월드): (0.25, -0.45, 0.06)
- 박스 시작 (월드): (0.45, -0.10, 0.06) → **셀까지 xy 거리 ≈ 0.43 m**

---

## 3. 어디까지 했는지 (진행)

### Stage 1 (example5) — ✅ 완료

- PPO 학습
- 박스 잡고 z=0.18~0.20 안정 도달
- frozen 처리, handoff state 100개 npz 저장

### Stage 2 (example7) — ⏳ 학습 진행, 결과 미흡

500k step, 64 envs, SAC + HER, **약 2.5분 소요**.

#### Reward 구조 (현재)

6개 항목 가중합, **모두 양수 (페널티 없음)**:

| 항목 | 가중치 | 역할 |
| --- | --- | --- |
| `r_pull` | 50 | 멀리서도 항상 작동하는 부드러운 끌어당김 (1/(1+5·xy)) |
| `r_xy` | 100 | 셀 가까이 갈수록 강해지는 가우시안 sharpen |
| `r_high` | 80 | 박스 z=0.30 유지 (셀 격벽 위) |
| `r_grip_holding` | 50 | 그립이 박스 가까이 + 닫혀있음 (떨어뜨리지 말기) |
| `r_upright` | 20 | 박스 수직 유지 |
| `r_success` | 1000 | 성공 spike (xy<2.5cm AND z≥0.28 AND 잡고있음) |

#### 종료 조건 (의도적으로 매우 관대)

- terminate = success 유지 또는 박스가 바닥 닿음 + 30°+ 기울어짐
- 셀벽 충돌 OK, 그리퍼 멀어짐 OK, 공중에서 기울어짐 OK
- 시간 초과 (10s) → truncate

#### 학습 결과 — 행동 통계

| 지표 | 값 | 의미 |
| --- | --- | --- |
| 박스 → 셀 xy 거리 | 0.39 ~ 0.55 m | 시작 거리 0.43, 거의 안 줄어듦 |
| xy 정렬 비율 (xy<2.5cm) | **0.02 %** | 정렬 사실상 못함 |
| 충분한 높이 도달 비율 (z≥0.28) | 0.43 ~ 0.53 | 절반은 들어 올림 ✓ |
| 그립 유지 비율 | 0.38 ~ 0.53 | 절반은 잡고 있음 ✓ |
| reset 비율 (tilted on floor) | 0.6 ~ 1 % | 거의 reset 안 됨 |
| 한 step 성공 비율 | **0 %** | 성공 사실상 없음 |
| 누적 성공 (4118 ep 중) | **10번 (0.24%)** | |
| 평균 episode 길이 | 142 ~ 148 step (max 600) | 충분히 유지됨 |

#### 시각적 관찰 (재생 영상)

1. 박스를 들어 올리는 동작은 OK (z=0.30 도달)
2. **들어 올린 후 그 자리에 멈춤**. 셀 (왼쪽 ~45cm) 쪽으로 안 옴
3. 가끔 박스 떨어뜨리거나, 빠른 동작으로 박스 흔들림

### Stage 3 (example7_2) — ⏸️ 대기

Stage 2 안정화되어야 handoff 수집 가능 → 학습 시작 못함. 코드 (reward, done) 는 작성 완료.

### Chain Inference — ✅ Framework 검증 완료

3-stage 분리 정책들을 inference 에서 합치는 시스템:

| 검증 항목 | 결과 |
| --- | --- |
| example5 단독으로 sequential 3박스 처리 | **3 / 3 lift 성공** ✓ |
| example5 + example7 chain (single-box) | framework 작동, lift 성공 |
| 박스 3개 동시 spawn 환경 (MultiBoxEnv) | 정상 spawn / step / reset ✓ |
| Multi-box chain framework | framework 작동 (정책 자체가 약해서 적재까진 미달) |

> **의미**: Pipeline 인프라 (state machine, obs 변환, multi-box env) 는 OK. **단독 정책 (Stage 2) 학습 자체가 막혀** 있어 end-to-end 성공률 미흡. Stage 2 풀리면 chain 즉시 동작 가능.

---

## 4. 무엇이 문제인지 (진단)

> **Local Optimum 가설**: 정책이 "그 자리에서 들고 있기" 라는 쉬운 길로 수렴

step 당 reward 비교:

| 상태 | 기여 합계 |
| --- | --- |
| 그 자리에서 들고만 있기 | **약 165** (높이 + 그립 + 수직 + 약한 pull) |
| 셀 위까지 이동해서 들고 있기 | **약 295** (위 + xy 정렬 보너스) |

이동 시 +130 점 추가지만, 그 보상을 얻으려면 **들고 있는 박스를 흔들지 않으며 0.43m 이동하는 복잡한 동작** 학습 필요. SAC + HER 의 탐험으로는 비용이 큼.

반면 그 자리에서 들고 있기는 example5 의 handoff state 거의 그대로 유지 = **학습 비용 0**.

→ 정책이 쉬운 길로 수렴, **xy 이동 동기 약함**.

---

## 5. 어느 부분을 물어보고 싶은지 (자문 요청)

### A. Reward 설계 검토 (가장 시급)

저희가 제안 중인 **v2 가중치 재배치**:

| 항목 | v1 (현재) | v2 (제안) | 의도 |
| --- | --- | --- | --- |
| `r_pull` | 50 | **200** | 이동 동기 강화 |
| `r_xy` | 100 | 100 | 유지 |
| `r_high` | 80 | **30** | 안주 보상 줄임 |
| `r_grip_holding` | 50 | **30** | 안주 보상 줄임 |
| `r_upright` | 20 | 10 | 안주 보상 줄임 |
| `r_success` | 1000 | 1000 | 유지 |
| `action_penalty` | (없음) | **-0.1** | 빠른 동작 억제 (약한 페널티) |

질문:
- 이 가중치 재배치가 합리적인가?
- potential-based shaping, success bonus 차등화 등 다른 reward 설계 방식이 더 나을지?

### B. 다른 접근 방법 추천

다음 대안 중 추천하실 만한 게 있는지:

1. **Curriculum learning**: 셀을 박스 가까이 (0.10m) 에서 시작 → 점진적으로 0.43m 까지
2. **Behavior Cloning warmstart**: IK 데모 trajectory 로 정책 사전학습 후 SAC fine-tune
3. **Goal randomization**: 셀 위치 고정 (현재) vs 매 episode 랜덤화
4. **Reward 자체 변경**: dense → sparse + HER 강화, 또는 potential-based
5. **HER 전략**: 현재 `future`, `final` 또는 `episode` 가 더 나을지?
6. **Action space 변경**: 현재 joint position delta, end-effector cartesian delta 가 더 학습 쉬울지?

### C. 일반 자문

- 현재 hierarchical 3-stage 구조 자체가 적절한가?
- SAC + HER 조합이 이 task 에 적합한가?
- handoff (grasp → place transition) 구조 개선 방법
- sim-to-real 시 가장 위험한 부분
- dual-camera perception 구조의 현실성
- 산업 환경 수준의 성공률 확보 전략

### D. 개발자 본인 학습 방향

- 강화학습 이론 / AI 이론 자체를 **지금이라도 깊이 공부해야 하는지**, 아니면 **현재처럼 실험 + 협업 도구 (Claude/GPT) + 영상 관찰 반복 방식으로 진행해도 충분한지**
- 깊이 공부한다면 어떤 영역부터 (PPO/SAC 알고리즘? reward shaping 이론? robot manipulation 분야 논문?) 시작하는 게 효율적인지

---

## 6. 뭘 해야 하는지 / 뭐가 남았는지

### Stage 2 (example7) 학습 — 가장 시급
- Reward v2 또는 자문받은 안 적용해서 재학습
- 영상 관찰 → 실패 패턴 분석 → reward 재설계 반복
- Stage 2 success rate 의미 있는 수준 (≥50%) 도달
- example7 success state 100~200개 → `handoff_xy_aligned.npz` 로 저장

### Stage 3 (example7_2) 학습 — Stage 2 끝나면
- 입력: `handoff_xy_aligned.npz` (xy 정렬된 상태에서 시작)
- 학습 목표: yaw 정렬 + 셀 안 적재 (descent + release)
- 코드 자체는 이미 준비됨 (reward / done 작성 완료, 학습 트리거만 남음)

### Chain End-to-End 검증
- 현재 framework 만 검증, 정책이 약해서 end-to-end 성공률 0
- Stage 2, 3 학습 후 chain script 재검증 → 3 박스 적재 목표

### Multi-box 확장
- 박스 1개 → **3개 동시 spawn + 3×3 그리드** (인프라 작동, 적재율 검증만 남음)
- 박스 ↔ 셀 매핑 sequential 처리 시 **robot reset 로직 보완** 필요 (현재 누락)

### Vision 모듈 통합
- 현재: Isaac Sim **ground-truth pose** 그대로 사용
- 목표: 카메라 기반 pose estimation 으로 교체
   - **Eye-in-Hand 카메라**: 근거리 정밀 보정 (grasp alignment, insertion)
   - **천장 고정 카메라**: 박스/grid 전체 인식, 빈 슬롯 좌표 계산
- obs 구조 변경 + camera noise/latency randomization 학습 필요

### Sim-to-Real
- domain randomization (조명, 마찰, 박스 mass, joint friction)
- camera noise / latency randomization (학습 단계에서 미리 적용)
- 실제 OMY + 실제 카메라 + 실제 블록 환경 이전
- 가장 위험한 단계: handoff 분포 mismatch, 카메라 occlusion (그리퍼가 박스 가림)

### 그 외 남은 작업
- `collect_handoff_xy.py` 작성 (example7 success state 수집 스크립트)
- Multi-box chain 의 robot reset 로직 보완
- 박스 3개 동시 처리 시 충돌/간섭 검증
- 산업 환경 적용을 위한 throughput / cycle time 최적화

---

## 부록 — 개발자 배경 (솔직)

저는 강화학습 / AI 이론 전공자가 아닙니다. 현재 개발 방식:

- Isaac Lab 환경 구축 + 학습 실험은 직접 진행
- reward / observation / action 구조 수정은 **Claude / GPT 협업**
- 학습 영상 관찰 + 실패 패턴 분석 + reward 재설계 + 전문가 피드백 반영 → iterative 개선

즉 **"실험 기반 반복 개선"** 방식. 깊은 수학적 해설보다 **실용적 조언** 위주로 부탁드립니다.

---

## 부록 — 추가 정보 (요청 시 공유)

| 항목 | 비고 |
| --- | --- |
| 학습 로그 | 텍스트 파일 |
| Env 코드 | DirectRLEnv 상속, Python |
| Cfg | 모든 파라미터 dataclass |
| Handoff data | npz 파일 (joint_pos / obj_pos / obj_quat) |
| example5 정책 | SB3 PPO + VecNormalize |
| 학습 영상 | GIF / mp4 캡처 |

감사합니다.
