# CLAUDE.md (example7_2)

This file provides guidance to Claude Code when working with example7_2.

---

## 태스크 개요 (example7_2)

**Lift + Transport** (insert 제외)

example7_2는 example5_2(grasp) 이후 받은 박스를 **목표 셀 위로 이동**시키는 task.
다음 단계 example7_3(align + insert + release)와 chain되어 example8_2에서 통합 실행.

### 입력 (handoff state)
- 박스가 그리퍼에 잡힌 채로 **바닥 근처(z=0.06)**에서 시작
- `handoff_states_v41.npz` 또는 `handoff_states_v47b.npz` 사용
- handoff 데이터: joint_pos (10), obj_pos_rel (3), obj_quat (4)

### 목표 (Goal)
- 박스를 **목표 셀의 xy 위로 이동**
- 박스 z = 0.30m (격벽 위)
- 박스 정자세 유지 (upright > 0.95)
- 박스 grip 유지 (놓치지 않음)

### Success 판정
- xy_dist(box, target_cell) < 0.025m (cell_tolerance)
- box_z > 0.28m (격벽 위)
- in_grasp > 0.5 (계속 잡혀있음)
- upright > 0.95
- N step 유지 (stable)

---

## 알고리즘: SAC + HER

example7과 동일. example7_2도 goal-conditioned task라 HER 적합.

### Observation (GoalEnv 형식)

| 항목 | 차원 | 설명 |
|---|---|---|
| arm joint pos (정규화) | 6 | joint1~6 위치 |
| arm joint vel | 6 | joint1~6 속도 |
| gripper close state | 1 | 0=열림, 1=닫힘 |
| object pos (env 기준) | 3 | 박스 현재 위치 |
| object vel | 3 | 박스 속도 |
| end_effector pos | 3 | 그리퍼 중심 위치 |
| object to target | 3 | 박스 → 타겟 셀+z 벡터 |

### achieved_goal / desired_goal

- `achieved_goal`: 박스 현재 위치 (x, y, z) — 3D
- `desired_goal`: 타겟 셀 위 위치 (cell_x, cell_y, 0.30) — 3D

---

## Reward 설계 (단순화)

```python
# 핵심 reward 항목
lift_progress = clamp((box_z - 0.06) / 0.24, 0, 1)  # 0 at ground, 1 at z=0.30
xy_to_target = exp(-25 * xy_dist_to_target²)
hold_bonus = in_grasp                                # 박스 잡혀있어야
upright_bonus = clamp(upright_score, 0, 1)
success = (xy_dist < tol) & (z > 0.28) & in_grasp & upright

reward = (
    + 100 * lift_progress         # 들기
    + 100 * xy_to_target          # 셀 xy 정렬
    + 50 * hold_bonus             # 잡기 유지
    + 30 * upright_bonus          # 정자세
    + 1000 * success.float()      # 성공
    - 0.001 * action_penalty
)
```

---

## 그리드 (example7과 동일)

3×3 그리드, 셀 피치 ≈ 17.8 × 6.8 cm.
- 셀 (0, 0) 중심: (0.25 - cell_pitch_x, -0.45 - cell_pitch_y)
- 9 셀 중 3개만 사용 (3개 박스 → 3 셀)

---

## example8_2 (chain inference)

example8_2가 3-box state machine으로:
1. 박스 i (i=0,1,2) 선택
2. example5_2 (grasp) 호출 → grip 성공 시 다음 단계
3. example7_2 (lift+transport) 호출 → 셀 i 위 도달 시 다음 단계
4. example7_3 (align+insert+release) 호출 → 삽입 완료 시 다음 박스
5. i++ 반복

각 정책은 단일 박스 task로 학습. example8_2가 박스 1개씩 처리.

---

## 학습 및 실행 커맨드

```bash
# 신규 학습 (handoff 데이터 필요)
python source/example7_2/scripts/train.py --num_envs 64 --timesteps 3000000 --headless

# 이어서 학습
python source/example7_2/scripts/train.py --resume --timesteps 2000000 --headless

# 재생
python source/example7_2/scripts/play.py --checkpoint checkpoints/example7_2.zip --vecnorm checkpoints/example7_2_vecnorm.pkl
```

---

## 파일 구조 (example7과 거의 동일)

```
source/example7_2/
  CLAUDE.md                  # 이 파일
  scripts/
    train.py                 # SAC+HER 학습
    play.py                  # 정책 재생
    collect_handoff.py       # (예전 example5용)
  tasks/place/
    place_env.py             # PlaceEnv → Lift+Transport용으로 수정 필요
    place_env_cfg.py         # PlaceEnvCfg → 설정 수정 필요
    grid_builder.py          # 3×3 grid 생성
```

---

## TODO (구현 우선순위)

1. ✅ example7 코드 복사 → example7_2
2. ✅ place_env_cfg.py: handoff 경로(v41), lift_target_z=0.30
3. ✅ place_env.py: reward 단순화 (lift + transport 위주)
4. ✅ scripts/train.py, play.py: 경로 example7_2
5. ✅ 학습 시작 — 약 39M SAC+HER step 진행

## 학습 진행 history (v1~v14)

| ver | 변경 | 결과 |
|---|---|---|
| v1 (500k) | 초기 reward | ep_len 19, obj_z 0.062 (안 들림) |
| v2 (1M) | grip 유지 강화 (R8/R9 추가) | ep_len 28, obj_z 0.066 |
| v3 (3M) | lift weight 50→200 + grip gate | ep_len 35, obj_z 0.066 |
| v4 (5M) | 3-gate push 차단 (closed+near+above) | ep_len 48, obj_z 0.062 |
| v5 (7M) | 박스 상승 속도 보상 (r_rising 200) | ep_len 62, obj_z 0.072 (1cm 들림!) |
| v6 (9M) | upright penalty 50→200 | ep_len 61, upright 0.98 회복 |
| v6 cont (12M) | 추가 학습 | ep_len 149, ep_rew 33k (안정 grip) |
| v7 (13M) | LIFT ONLY (xy reward 제거) | regression (ep_len 76) |
| v8 (16M) | v6 reward 복구 + lift_only success | ep_len 126, obj_z 0.063 |
| v9 (19M) | **Action harness — gripper 강제 close** | ep_len 185 (47%↑) |
| v10 (22M) | lift weight 800 + rising 500 | ep_len 154 |
| v11 (25M) | 그리퍼 z velocity 보상 추가 | ep_len 200, ep_rew 48k |
| v12 (28M) | drop penalty 200→30 (lift 시도 격려) | ep_len 220, ep_rew 56k |
| v13 (31M) | mass 0.3→0.1 (physics 검증) | obj_z 그대로 → mass 문제 아님 |
| v14 (34M) | **Physical attach harness** (박스 그리퍼에 강제 부착) | ep_len 191, **cumulative success 6** |
| v14 cont (39M) | 추가 학습 | **cumulative success 13** |

**최종 상태 (v14 39M)**:
- ep_len 206, ep_rew 56k
- success_now 0.001-0.005 (드물게 lift 발생)
- cumulative_success 13/24995 episodes (0.05%)
- on_floor 0.7-0.77 (대부분 박스 바닥)
- abandoned 0 (action harness 작동)

## 결론 — 한계 발견

**문제 본질**: PPO/SAC 정책이 "박스 잡고 바닥에 안정 유지"라는 강력한 local optimum에 갇힘.
- 28M+ step 학습 + action harness + physical attach + 다양한 reward 변형
- 모든 시도에서 success rate 0.05% 이하
- 가능 원인: 정책이 arm을 위로 움직이는 행동 자연 발견 못 함 (6-DOF action space, exploration 부족)

## 다음 단계 (사용자 결정 필요)

**Option A — example7_2 학습 더 시도**:
- Imitation Learning bootstrap (수동 lift trajectory 시범 후 SAC 재학습)
- ent_coef 강제 부스트 (현재 auto)
- 더 긴 학습 (50M+)

**Option B — 시뮬 변경**:
- 박스 attach 더 strict (z도 강제로 raise)
- 또는 별도 IK로 arm 위치 강제

**Option C — example7_3로 진행** (현실적):
- example7_2는 imperfect 상태로 두고
- example7_3 (insert)는 **synthetic init** (박스 z=0.30 above cell, 가정)으로 학습
- example8_2에서 정책 chain 시 example7_2 부분이 약하더라도 예제 동작

**Backup 체크포인트**:
- `example7_2_v14_39M.zip` — best (cumulative 13 successes)
- `example7_2_v14_39M_replay.pkl` — replay buffer (resume 시 필요)
