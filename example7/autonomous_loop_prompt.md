# Autonomous Training Harness — OMY Place+Insert (Curriculum)

You are running autonomously in `/loop` dynamic mode. **DO NOT ask the user anything.** Each iteration: train → parse log → analyze → modify → schedule next.

## 최종 목표 (UPDATED 2026-05-03)

- **누적 성공 episode 수 ≥ 30** (메트릭: `rate_cumulative_success`)
- 정의: 학습 시작 이후 `success_stable=True (xy_tight & yaw_aligned & deep_enough & stable, 50 step 유지)` 도달한 누적 episode 수
- **Stage 무관, 누적 30회 도달 시 즉시 종료 + 사용자 알림 메시지**
- 추적: 매 iter마다 loop_state.json의 `total_cumulative_successes`에 합산 (env 내부 카운터는 iter 시작마다 0으로 초기화되므로 합산 필요)

## Curriculum (3 stages)

학습 어려움 점진적 상승. cfg 값은 `place_env_cfg.py` Edit으로 변경.

| Stage | cell_inner_x | cell_inner_y | wall_height | insertion_target_depth | 진급 기준 |
|-------|--------------|--------------|-------------|------------------------|-----------|
| 1 (easy) | 0.30 | 0.20 | 0.05 | 0.02 | episode_success ≥ 0.50 |
| 2 (medium) | 0.22 | 0.12 | 0.08 | 0.04 | episode_success ≥ 0.70 |
| 3 (target) | 0.17 | 0.06 | 0.12 | 0.06 | episode_success ≥ 0.90 → **DONE** |

진급 시: cfg 4개 값 모두 Edit. 체크포인트는 유지 (resume).

## 매 iteration 프로토콜

### 1. State 읽기
```bash
cat /home/jaewoo/IsaacLab/source/example7/loop_state.json
```

### 2. 종료 조건 확인 (해당 시 즉시 STOP, ScheduleWakeup 호출 안 함)
- **`total_cumulative_successes >= 30`** ← **핵심 종료 조건**
- `iteration >= max_iterations` (30)
- 누적 학습 시간 > 8시간 (벽시계, 안전장치)
- 종료 시 **반드시 사용자 알림** 출력 (마지막 메시지에 "✅ 30회 성공 달성! Iter X에서 도달, total_cumulative_successes=Y" 형태로)

### 3. 학습 실행
```bash
ITER=$(python -c "import json; print(json.load(open('/home/jaewoo/IsaacLab/source/example7/loop_state.json'))['iteration'])")
NEW_ITER=$((ITER + 1))
LOG=/home/jaewoo/IsaacLab/source/example7/logs/iter_$(printf "%03d" $NEW_ITER).log

# 1M step / iter (~6분). resume 사용 (단 stage 진급 후 첫 iter만 새 학습 권장)
RESUME_FLAG="--resume"
if [ -f /tmp/example7_stage_changed ]; then
    rm /tmp/example7_stage_changed
    # stage 진급한 첫 iter는 resume (구조 변경 아니라 cfg값만 바뀌므로 OK)
fi

nohup bash -c "source /home/jaewoo/miniconda3/etc/profile.d/conda.sh && \
  conda activate env_isaaclab && \
  ./isaaclab.sh -p source/example7/scripts/train.py \
    --num_envs 64 --timesteps 1000000 $RESUME_FLAG --headless" > $LOG 2>&1 &

TRAIN_PID=$!
echo "Started: $TRAIN_PID"
```

학습 종료 대기 (Bash run_in_background로 별도 wait):
```bash
tail --pid=$TRAIN_PID -f /dev/null && echo "DONE"
```

### 4. 메트릭 파싱
```bash
python source/example7/scripts/parse_log.py $LOG > /tmp/iter_metrics.json
cat /tmp/iter_metrics.json
```

핵심 메트릭:
- `rate.episode_success` ← **주 평가 지표**
- `rate.success_now` (per-step)
- `rate.deep_enough` (삽입률)
- `rate.xy_tight`, `rate.yaw_aligned` (정렬률)
- `rate.abandoned`, `rate.tilted` (실패 원인)
- `dist.xy`, `dist.endpoint_xy`, `dist.insert_depth`
- `rew.total`, 항목별 rew

### 5. 진단 + 수정 (rule-based + 판단)

**Stage 진급 체크 먼저:**
- stage 1: episode_success ≥ 0.50 → stage 2 (cfg 변경)
- stage 2: episode_success ≥ 0.70 → stage 3 (cfg 변경)
- stage 3: episode_success ≥ 0.90 → DONE

**Stage 진급 시 Edit:**
```python
# stage 1 → 2
cell_inner_x: float = 0.22
cell_inner_y: float = 0.12
wall_height: float = 0.08
insertion_target_depth: float = 0.04

# stage 2 → 3
cell_inner_x: float = 0.17
cell_inner_y: float = 0.06
wall_height: float = 0.12
insertion_target_depth: float = 0.06
```

**진급 안 했을 때 (정체 시) — 증상별 대응:**

| 증상 | 조치 |
|------|------|
| `rate_abandoned > 0.20` | grip_near 가중치 ↑ (10→20) |
| `rate_tilted > 0.15` | upright 가중치 ↑ (30→45) |
| `dist_xy > 0.15` 정체 | center_align_linear ↑ (30→45) |
| `dist_xy < 0.05` 도달 했지만 정렬 안 됨 | endpoint_align_close ↑ (30→50) |
| `rate_xy_tight > 0.3` 인데 `rate_deep_enough = 0` | soft_descent ↑ (60→90), descent_reward ↑ (80→120) |
| `rate_deep_enough > 0.1` 인데 `rate_success_now = 0` | success_hold_steps ↓ (50→25) 임시, release_reward ↑ |
| 5 iter 정체 | reward 구조 큰 변경 (예: cell_tolerance 완화) |

**수정 규모:**
- iter 1~5: 큰 폭 (×1.5~×2)
- iter 6~15: 중간 (×1.2)
- iter 16+ 또는 success>0.5: 미세 (×1.05~×1.1)

### 6. 변경 사항 적용 (Edit tool)
- `source/example7/tasks/place/place_env.py` (reward 가중치)
- `source/example7/tasks/place/place_env_cfg.py` (cfg 값)
- 변경 시 `/tmp/example7_stage_changed` 파일 생성 (단계 진급 표시)

### 7. State 업데이트 (Edit tool)
```json
{
  "iteration": <NEW_ITER>,
  "max_iterations": 30,
  "stage": <current_stage>,
  "best_episode_success": <max history>,
  "best_iter": <iter at best>,
  "history": [
    ...,
    {
      "iter": <NEW_ITER>,
      "stage": <stage>,
      "total_steps": <from log>,
      "episode_success": <rate.episode_success>,
      "success_now": <rate.success_now>,
      "deep_enough": <rate.deep_enough>,
      "xy_tight": <rate.xy_tight>,
      "yaw_aligned": <rate.yaw_aligned>,
      "abandoned": <rate.abandoned>,
      "tilted": <rate.tilted>,
      "dist_xy": <dist.xy>,
      "dist_endpoint": <dist.endpoint_xy>,
      "dist_insert_depth": <dist.insert_depth>,
      "ep_rew_mean": <ep_rew_mean from log>,
      "changes": "<short description>",
      "elapsed_s": <wall clock>
    }
  ],
  "stagnation_count": <증가 또는 0>,
  "last_change": "<description>"
}
```

### 8. Best 체크포인트 관리
```bash
# rate_episode_success가 best 갱신 시
cp checkpoints/example7.zip checkpoints/example7_best.zip
```

### 9. 다음 iteration 스케줄
```
ScheduleWakeup(
    delaySeconds=120,
    prompt="<<autonomous-loop-dynamic>>",
    reason="iter <NEW_ITER+1> — stage <X>, success <Y>"
)
```

종료 시 ScheduleWakeup 호출하지 않음.

## 키 파일

- `source/example7/loop_state.json` — state (매 iter 업데이트)
- `source/example7/logs/iter_XXX.log` — per-iter stdout
- `checkpoints/example7.zip` — latest
- `checkpoints/example7_best.zip` — best (episode_success 기준)
- `checkpoints/handoff_states.npz` — handoff data (있음, 변경 안 함)

## CRITICAL rules

1. **NEVER ask the user anything** — 자율 실행
2. 학습 실패 (CUDA OOM 등) → num_envs 절반으로, retry
3. observation/action space 변경 시 → 체크포인트 폐기, 새 학습
4. **Stage 진급 시에는 reward 가중치 변경하지 말 것** — cfg만 변경
5. 5 iter 연속 episode_success 정체 → reward 구조 큰 변경 권한 사용
6. **첫 iter 시작 전**: loop_state.json이 빈 상태면 stage 1, iter 0으로 시작
