# CLAUDE.md (example8_2)

This file provides guidance to Claude Code when working with example8_2.

---

## 태스크 개요 (example8_2)

**Chain Inference + 3-Box State Machine**

학습된 정책 3개 (example5_2, example7_2, example7_3)를 순차적으로 호출해서
3개 박스를 각각 3개 셀에 배치하는 통합 시스템.

학습 없음 — 추론 전용 코드.

---

## 환경 구성

- **3개 박스** 동시 존재 (사용자 요구사항)
- **9개 셀 grid** (3×3) 중 **3개만 사용** (박스 1개당 1개 셀, 고정 매핑)
- 박스 ↔ 셀 매핑 (고정 순서):
  - 박스 0 → 셀 0 (e.g., (0,0))
  - 박스 1 → 셀 1 (e.g., (0,1))
  - 박스 2 → 셀 2 (e.g., (0,2))

---

## 상태 머신 (State Machine)

각 박스 i (i=0,1,2)에 대해:

```
WAITING
   ↓ env reset
GRASPING (example5_2 정책)
   ↓ grasp_streak ≥ 60 (1초)
PLACE_HOVER (example7_2 정책)
   ↓ 박스 z >= 0.28 + xy align
INSERTING (example7_3 정책)
   ↓ 박스 셀 안 + gripper open + stable
NEXT_BOX (i+1)
   ↓ i < 3
WAITING
   ↓ i == 3
DONE
```

전환 조건:
- GRASPING → PLACE_HOVER: example5_2 streak_success
- PLACE_HOVER → INSERTING: example7_2 success (z + xy align)
- INSERTING → NEXT_BOX: example7_3 success (in cell + released)

---

## 호출 방식

각 정책별 호출:

```python
# Example pseudocode
class StateMachine:
    def __init__(self):
        self.grasp_policy = SB3.load("example5_2.zip")
        self.lift_policy = SAC.load("example7_2.zip")
        self.place_policy = SAC.load("example7_3.zip")
        self.state = "GRASPING"
        self.box_idx = 0
        self.target_cell = self.cell_mapping[0]

    def step(self, obs):
        if self.state == "GRASPING":
            action = self.grasp_policy.predict(self.grasp_obs(obs))
            if self.check_grasp_success():
                self.state = "PLACE_HOVER"
        elif self.state == "PLACE_HOVER":
            action = self.lift_policy.predict(self.lift_obs(obs))
            if self.check_lift_success():
                self.state = "INSERTING"
        elif self.state == "INSERTING":
            action = self.place_policy.predict(self.place_obs(obs))
            if self.check_place_success():
                self.box_idx += 1
                if self.box_idx >= 3:
                    self.state = "DONE"
                else:
                    self.state = "GRASPING"
                    self.target_cell = self.cell_mapping[self.box_idx]
        return action
```

각 정책의 obs format이 다르므로 `*_obs(obs)` 함수가 변환.

---

## 환경 (Multi-box Env)

example8_2 전용 env가 필요:
- robot 1개
- 박스 3개 (다른 초기 위치)
- 9 셀 grid (or 3 셀)
- target box index in obs (현재 처리 중인 박스)
- target cell index in obs

박스 위치 (예시):
- 박스 0: (0.45, -0.10, 0.06)
- 박스 1: (0.45, -0.20, 0.06)
- 박스 2: (0.45, 0.00, 0.06)

(서로 충돌 안 하게 박스 간격 ≥ 박스 길이 + 여유)

---

## TODO 우선순위 (2026-05-06 업데이트)

### ✅ 완료
1. Directory 생성 + framework 코드
2. **chain_mvp.py** — example5 단독 sequential 3-box (3/3 lift 성공)
3. **chain_place_mvp.py** — example7 단독 place test (이슈 발견)
4. **multi_box/multi_box_env_cfg.py** — 3-box env 설정
5. **multi_box/multi_box_env.py** — 3-box env 구현 (skeleton)

### ⏳ 진행 중
6. example7 추가 3M SAC+HER 학습 (handoff 안정성 ↑)

### 📌 남은 작업
7. example7 학습 후 chain_place_mvp 재검증
8. **Unified env**: example5 obs(34) + example7 obs(31) 둘 다 제공
   - LiftEnv 또는 MultiBoxEnv 확장
   - compute_grasp_obs(): 34-dim
   - compute_place_obs(): 31-dim
9. **Real chain script**: state machine
   - Phase 0 (GRASP_LIFT): example5 policy + grasp_obs
   - Phase 1 (PLACE): example7 policy + place_obs
   - 전환 조건: box_z >= lift_threshold (0.15)
   - 박스 i 완료 → reset robot, active_box++
10. Multi-box deployment 검증

**현재 베이스라인**:
- chain_mvp: 3/3 lift 성공 (example5 단독)
- chain_place_mvp: 0/3 place (example7 추가 학습 필요)
- multi_box_env: skeleton 완료, integration 대기

---

## 의존성

- example5_2 정책: `checkpoints/example5_2_v47b_40M.zip` + vecnorm
- example7_2 정책: `checkpoints/example7_2_v14_39M.zip` + replay
- example7_3 정책: 미학습

각 정책의 environment cfg와 같은 robot/object 설정 사용 필요.

---

## 학습/추론 분리

- example5_2/example7_2/example7_3는 **학습** 단계
- example8_2는 **추론** 단계 (학습 안 함, 정책 chain만)

학습 시 각 정책은 독립 task로 학습. example8_2는 학습된 정책 wrapper.
