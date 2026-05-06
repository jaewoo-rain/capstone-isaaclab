# example8_2 Status (2026-05-06)

## ✅ 완료된 Framework

### 1. example8_2 디렉토리 구조
```
source/example8_2/
├── CLAUDE.md           # 전체 설명
├── DESIGN.md           # 설계 문서
├── STATUS.md           # 이 파일
├── scripts/
│   ├── chain_mvp.py        # example5 단독 lift demo (3/3 성공)
│   ├── chain_place_mvp.py  # example7 단독 place test
│   ├── chain_full.py       # LiftEnv + chain (single-box, 3 iter)
│   ├── chain_multi.py      # MultiBoxEnv + chain (multi-box)
│   └── inference.py        # skeleton (future state machine)
└── tasks/multi_box/
    ├── __init__.py
    ├── multi_box_env_cfg.py
    └── multi_box_env.py    # 3-box env (skeleton 작동)
```

### 2. 검증 결과

| Script | 상태 | 결과 |
|---|---|---|
| chain_mvp.py | ✅ 작동 | 3/3 lift 성공 (example5 단독) |
| chain_full.py | ✅ framework 작동 | lift 3/3, place 0/3 (example7 약함) |
| chain_multi.py | ✅ framework 작동 | Box 1,2 lift 성공 (Box 0은 학습영역 밖) |
| MultiBoxEnv | ✅ 작동 | 3개 박스 동시 spawn 검증 |

### 3. 발견된 핵심 이슈

**A. VecNormalize 누락** (해결됨)
- example5 obs 정규화 누락 → grasp 실패
- 해결: VecNormalize 통계 로드 + 수동 정규화 적용

**B. example7 place 정책 약함** (학습 진행 중)
- success rate 0.03% (45 successes in 100k+ episodes)
- 현재 10M 추가 학습 중
- 목표: 60% 도달

**C. example5 학습 영역 제한**
- 박스 (0.45, -0.10 ± 2cm)에서만 학습됨
- chain_multi에서 Box 0 (-0.20)은 영역 밖 → 실패
- 해결책: 박스 sequential 처리 (chain_full 방식)

## 📊 현재 정책 상태

### example5 (PPO + LiftEnv)
- 박스 lift to 19-20cm (안정적)
- handoff 데이터 z=0.15-0.20 captured (100 samples)
- chain framework에서 grasp 단계 OK

### example7 (SAC + HER + PlaceEnv)
- cumulative successes: 45 (학습 중)
- xy_aligned rate: 0.03 (3%)
- tilted rate: 0.76 (높음)
- 현재 10M 추가 학습 중

## 🎯 다음 단계

1. **example7 10M 학습 완료** 후 chain_full 재검증
2. **chain_full 3-box demo** (현재 framework 작동, place만 강화하면 됨)
3. 60% 성공률 도달 시 multi_box_env 본격 사용

## 🔄 세션 재개 가이드 (다음 사용자가 이어서 작업할 때)

### 재시작 방법
```bash
# Claude Code 재시작
claude

# 또는 IDE에서 새 세션 시작
```

### 재시작 후 첫 프롬프트 (복사해서 사용)

```
example8_2 chain inference 작업 이어서 진행. 
source/example8_2/CLAUDE.md, STATUS.md, DESIGN.md 읽고 현재 상황 파악해줘.

핵심 정리:
- example5 (grasp+lift): 작동 ✅
- example7 (place): 학습 한계 (success 0.03%, 정체)
- chain_mvp.py: 3/3 lift 성공
- chain_full.py: lift+place chain framework
- chain_multi.py: MultiBoxEnv 3-box (3-phase 구조 추가됨, 검증 미완)
- VecNormalize 누락 발견 + 적용 (핵심)
- Transport trick: example5 + target=cell으로 transport 검증

다음 단계:
1. chain_multi.py 3-phase (GRASP→TRANSPORT→RELEASE) 검증
2. RELEASE 후 박스가 셀에 안착하는지 확인
3. 실패 시 example5 trick 파라미터 튜닝 (target z, xy threshold)

지금 어디까지 됐는지 확인하고 chain_multi 실행해서 결과 분석해줘.
```

### 중요 backup 체크포인트
- `checkpoints/example5.zip` + vecnorm: grasp+lift 정책
- `checkpoints/example7_best.zip`: place 정책 (약함)
- `checkpoints/handoff_states.npz`: 100 samples z=0.17

### 핵심 파일
- `source/example8_2/scripts/chain_multi.py` — 최신 작업 (3-phase chain)
- `source/example8_2/tasks/multi_box/multi_box_env.py` — 3-box env
- `source/example8_2/STATUS.md` — 이 파일 (전체 상태)

---

## 💡 추론 사용법

### 현재 가장 신뢰성 있는 방법: chain_full.py
```bash
python source/example8_2/scripts/chain_full.py \
  --num_envs 1 --num_boxes 3 --headless
```

3박스 sequential 처리 (single-box LiftEnv 3번 reset). example7 학습 완료 후 `Total: N/3 성공` 메트릭 확인.

### Multi-box 동시 시연 (framework 작동, 학습 영역 외): chain_multi.py
```bash
python source/example8_2/scripts/chain_multi.py \
  --num_envs 1 --num_boxes 3 --headless
```

3박스 동시 spawn. Box 1, 2는 lift 성공. Box 0은 학습 영역 밖.
