# Autonomous Training Loop — OMY Place Task

You are running autonomously. DO NOT ask the user anything. Keep iterating until success or max_iterations reached.

## Protocol (run every iteration)

### 1. Read state
```bash
cat /home/jaewoo/IsaacLab/source/example6/loop_state.json
```

### 2. Check stop conditions (exit if any)
- `iteration >= max_iterations` → STOP
- `best_success_rate > 0.5` AND last 3 `recent_success_rates` strictly increasing → STOP (80% confidence)

### 3. Select action based on state

Load the conda env first for any python run:
```bash
source /home/jaewoo/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
```

**Decide what to change based on the most recent metrics:**

| Symptom | Action |
|---------|--------|
| `rate_abandoned` > 0.3 | gripper 파지 유지 reward 증가 또는 handoff 노이즈 감소 |
| `rate_tilted` > 0.3 | action_penalty 증가 또는 action_scale 감소 |
| `dist_xy` 정체 (3 iter) | learning_rate 조정, approach reward 가중치 증가 |
| `rate_stage1_success` < 0.1 (iter 3+) | stage1_xy_tolerance 완화, grid 위치 재확인 |
| `rate_stage1_success` > 0.5 sustained 2 iter | stage → 2로 전환 |
| `rate_in_cell` > 0 이나 `rate_success_now` = 0 | height_match/gripper_open 관련 reward 강화 |
| `rate_grip_open` 영원히 0 | ent_coef 증가 (exploration 장려) |

**Change magnitude (Adam-like):**
- `large`: iteration 1~5 (aggressive 수정)
- `medium`: iteration 6~15 or 어느 정도 progress 생기면
- `small`: iteration 16+ 또는 성공률 > 0.3

### 4. Apply changes
- `source/example6/tasks/place/place_env.py` 또는 `place_env_cfg.py` 수정
- stage transition 시 `curriculum_stage: int = 2` 로 수정

### 5. Run training (foreground, timeout)

```bash
LOG=/home/jaewoo/IsaacLab/source/example6/logs/iter_$(printf "%03d" <NEW_ITER>).log
mkdir -p $(dirname $LOG)
timeout 240 python source/example6/scripts/train.py \
    --headless --num_envs 32 \
    --timesteps <timesteps_per_iter> \
    $(test -f checkpoints/example6.zip && echo --resume) \
    --name example6 2>&1 | tee $LOG
```

- Iteration 1: 새 학습 (resume 없이)
- Iteration 2+: `--resume`
- **예외**: observation_space 변경 시 체크포인트 폐기 (checkpoint 삭제 후 새 학습)

### 6. Parse log
```bash
python source/example6/scripts/parse_log.py $LOG > /tmp/iter_metrics.json
cat /tmp/iter_metrics.json
```

### 7. Update state

수동으로 `loop_state.json` 업데이트 (Edit tool 사용):
- `iteration += 1`
- append to `recent_success_rates` (지금 stage 기준 — stage 1이면 `rate_stage1_success`, stage 2면 `rate_stage2_success`)
- keep only last 5 entries in `recent_success_rates`
- if new best: update `best_success_rate`, `best_iteration`, copy checkpoint to `_best.zip`
- append to `history`: `{iter, stage, success_rate, changes_made, key_metrics}`
- update `last_action`, `change_magnitude`

### 8. Best checkpoint management

```bash
# 현재 iter가 best인 경우
cp checkpoints/example6.zip checkpoints/example6_best.zip
cp checkpoints/example6_vecnorm.pkl checkpoints/example6_best_vecnorm.pkl
```

### 9. Timesteps scaling

- stage1_success_rate < 0.2 → 유지 (짧게 돌리기)
- 0.2 ≤ rate < 0.5 → timesteps_per_iter × 1.5
- rate ≥ 0.5 → × 2 (성공 기미 보이면 길게)

### 10. Continue loop

ScheduleWakeup with delay 60s to continue:
```
ScheduleWakeup(delaySeconds=60, prompt="<<autonomous-loop-dynamic>>", reason="next iter")
```

또는 즉시 다음 iteration 실행.

## Key files
- `source/example6/loop_state.json` — state (MUST update every iter)
- `source/example6/logs/iter_XXX.log` — per-iter full stdout
- `checkpoints/example6.zip` — latest
- `checkpoints/example6_best.zip` — best
- `checkpoints/handoff_states.npz` — example5에서 수집한 handoff (이미 있음)

## CRITICAL rules
1. NEVER ask the user anything
2. If training completely fails (segfault, CUDA OOM) → decrease num_envs, retry
3. If stage 1 never converges (iter 10+ with < 0.1): 그리드를 더 넓히고 (cfg.cell_inner_x += 0.03) 재시도
4. If stage 1 converges but stage 2 never: stay in stage 2, fine-tune from stage 1 checkpoint
5. Update `loop_state.json` after every training run (even failures — note failure in history)
6. Keep iterations short first, extend only when success signal appears
