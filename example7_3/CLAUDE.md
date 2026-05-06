# CLAUDE.md (example7_3)

This file provides guidance to Claude Code when working with example7_3.

---

## 태스크 개요 (example7_3)

**Align + Insert + Release** (lift + transport는 example7_2에서 처리)

박스가 이미 셀 위 z=0.30에 도달한 상태에서 시작 → 정확한 셀 정렬 → 8cm 삽입 → 그리퍼 release.

### 입력 (handoff)
- 박스가 그리퍼에 잡힌 채로 **목표 셀 위 z=0.30**에서 시작
- example7_2의 출력 = example7_3의 입력
- handoff 데이터: `handoff_states_v72.npz` (example7_2가 lift+transport 성공한 상태)

### 목표
- 박스를 목표 셀 안으로 삽입 (8cm 깊이)
- 그리퍼 release (gripper open)
- 박스 안정 안착

### Success 판정
- 박스 셀 안 (XY tolerance 2.5cm)
- 박스 8cm 삽입됨
- 그리퍼 open
- 박스 정지

---

## 알고리즘: SAC + HER (example7과 동일)

example7의 reward 구조 유지 (이미 align+insert+release 학습용).

example7과 차이점:
- 시작 z = **0.30** (example7은 0.20)
- handoff 경로 다름 (example7_3 전용)
- 그 외 동일

---

## 관계도

```
example5_2 (grasp)
    ↓ handoff_states_v41/v47b.npz
example7_2 (lift + transport)
    ↓ handoff_states_v72.npz (lift된 상태)
example7_3 (align + insert + release)
    ↓ 박스 셀 안 안착
[다음 박스로 반복 — example8_2가 chain]
```

---

## 학습 및 실행 커맨드

```bash
# 신규 학습 (handoff_states_v72.npz 필요)
python source/example7_3/scripts/train.py --num_envs 64 --timesteps 3000000 --headless

# 이어서 학습
python source/example7_3/scripts/train.py --resume --timesteps 2000000 --headless

# 재생
python source/example7_3/scripts/play.py --checkpoint checkpoints/example7_3.zip
```

---

## 파일 구조

```
source/example7_3/
  CLAUDE.md                  # 이 파일
  scripts/
    train.py                 # SAC+HER 학습 (example7_3 이름으로 저장)
    play.py                  # 정책 재생
  tasks/place/
    place_env.py             # PlaceEnv (example7과 동일, 시작 z만 다름)
    place_env_cfg.py         # PlaceEnvCfg (handoff 경로 변경 필요)
```

---

## TODO

1. ✅ example7 코드 복사 → example7_3
2. ✅ scripts import 경로 example7 → example7_3
3. ⏳ example7_2 학습 완성 → handoff_states_v72.npz 생성
4. ⏳ place_env_cfg.py: handoff_dataset_path = handoff_states_v72.npz
5. ⏳ place_env_cfg.py: 시작 z 조정 (필요 시)
6. ⏳ 학습 시작
