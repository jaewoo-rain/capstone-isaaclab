# example8_2 설계 문서

## 목표
3개 박스 → 3개 셀 자동 배치 (chain inference, no training)

## 아키텍처

### Multi-box Env 설계

**박스 spawn**:
- 3개 박스 동시 존재
- 박스 위치 (서로 충돌 안 하게):
  - Box 0: (0.45, -0.20, 0.06)
  - Box 1: (0.45, -0.10, 0.06)
  - Box 2: (0.45,  0.00, 0.06)
- y축 10cm 간격 (박스 width 4.4cm + 5.6cm gap)

**박스 ↔ 셀 매핑** (고정):
- Box 0 → Cell (0, 0)
- Box 1 → Cell (0, 1)
- Box 2 → Cell (0, 2)

**환경 상태**:
- `active_box_idx`: 현재 처리 중인 박스 (0/1/2)
- `active_cell_idx`: 목표 셀
- `phase`: GRASP / LIFT / PLACE / DONE

### State Machine

```
[INITIAL]
   ↓ env reset (3개 박스 spawn, active_box_idx=0)
[GRASP_LIFT] (example5 policy)
   ↓ obj_z[active_box] >= 0.17 (handoff threshold)
[PLACE] (example7 policy)
   ↓ success: obj in target cell + on_floor
[NEXT_BOX] (active_box_idx++)
   ↓ if active_box_idx < 3: GRASP_LIFT, else: DONE
[DONE]
```

전환 조건:
- GRASP_LIFT → PLACE: obj_z[active] >= 0.17 AND obj가 그리퍼 근처 (잡혔음 검증)
- PLACE → NEXT_BOX: target cell 안 + on_floor + gripper open
- NEXT_BOX → GRASP_LIFT: 박스 i+1 위치로 reset (other boxes 그대로)

### 정책 호출

각 정책별 obs format:
- example5: 34차원 (joint pos×10 + joint vel×10 + obj_pos_rel×3 + obj_to_grip×3 + l_to_obj×3 + r_to_obj×3 + gripper_close×1 + to_lift_target×1)
- example7: 31차원 (joint pos×6 + joint vel×6 + gripper×1 + obj_pos×3 + obj_vel×3 + ee_pos×3 + obj_to_target×3 + achieved×3 + desired×3)

각 phase에서 active box 정보를 사용해 obs 생성.

### 구현 단계

**Stage 3.1**: PlaceEnvCfg 다중 박스 지원
- `num_boxes: int = 3`
- `box_init_positions: list[tuple]` (3개 위치)
- `cell_mapping: list[int]` (박스 → 셀)

**Stage 3.2**: PlaceEnv 다중 박스 spawn
- _setup_scene에서 3개 RigidObject 생성
- self._objects: list of 3 boxes
- active_box_idx 추가

**Stage 3.3**: chain_inference.py
- example5 + example7 정책 로드
- State machine 구현
- 각 step에서 phase에 따라 정책 선택 + obs 변환

**Stage 3.4**: 박스 i+1 전환 시 active 박스만 reset
- 기존 박스들은 셀에 남음 (placed)
- robot 위치는 default로 복귀 (다음 박스 잡으러)

## 단순화 옵션 (Phase 1 구현)

**MVP**: 3-box를 3번 single-box로 시뮬
- 매번 env reset (1 박스만 spawn)
- example5+example7 한 사이클 완료
- 다음 cell index로 변경하고 다시 reset
- 3번 반복

이게 더 빠르게 구현 가능. true multi-box env는 Phase 2에서.

## Phase 1 (MVP) 구현 우선순위

1. ⏳ chain_inference.py 작성 (single-box, 3-iteration loop)
2. ⏳ example5 + example7 정책 로드 + obs 변환
3. ⏳ State machine 단순 구현
4. ⏳ 3-iteration sequential 실행
5. (Phase 2) Multi-box env 작성

## 주의사항

- example5와 example7 모두 OMY 같은 robot 사용 (동일 USD)
- example7의 init은 handoff에서 가져옴 (z=0.17)
- chain inference에서는 example5가 z=0.06에서 시작 → example5가 lift to z=0.17 → example7 take over
- 같은 env에서 phase 전환만 (env reset 없음)
