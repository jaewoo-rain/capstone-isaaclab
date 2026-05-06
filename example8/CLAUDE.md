# CLAUDE.md

example8: example5(Lift PPO) + example6(Place SAC+HER)을 **계층적으로 실행**하는 통합 환경.

---

## 개요

```
에피소드 시작 (물체 바닥, 로봇 기본 자세)
    ↓
LIFT 단계 (example5 PPO)
    ↓
물체 높이 > 0.20m
    ↓
PLACE 단계 (example6 SAC+HER)
    ↓
타겟 셀에 안정 안착 → 종료
```

---

## 파일 구조

```
source/example8/
  CLAUDE.md
  tasks/hierarchy/
    hier_env_cfg.py    # 통합 cfg (lift 시작 + place 그리드)
    hier_env.py        # HierEnv (DirectRLEnv)
  scripts/
    play.py            # 계층 재생 (이번 단계)
    train.py           # (다음 단계)
```

---

## HierEnv 핵심

### 시작 상태 (lift 환경 동일)
- 물체: `(0.45, -0.10, 0.06)` ± 2cm 노이즈
- 로봇: OMY default joint pos
- 그리드: 3×3 셀, 중심 `(0.25, -0.45)` (place 환경 동일)

### Observation
환경은 **두 가지 형식**을 노출:

| 형식 | 차원 | 호출 방법 | 사용처 |
|------|------|-----------|--------|
| Place (flat) | 31 (=25+3+3) | `_get_observations()` (기본 step) | place policy |
| Lift | 34 | `get_lift_observation()` | lift policy |

### 정책 전환
play 스크립트가 `raw_env.get_object_height()[0] > switch_threshold` 조건으로
매 스텝 lift→place 전환.

### Lift VecNormalize 처리
`omy_lift_vecnorm.pkl`에서 `obs_rms.mean / var`만 추출해 lift obs를 수동 정규화.
SB3 wrapper로 wrap하지 않음 (place 형식과 충돌 방지).

---

## 학습 및 실행 커맨드

```bash
# 재생
./isaaclab.sh -p source/example8/scripts/play.py \
    --lift_ckpt checkpoints/omy_lift.zip \
    --lift_vecnorm checkpoints/omy_lift_vecnorm.pkl \
    --place_ckpt checkpoints/example6_best.zip
```

### 옵션
- `--switch_threshold 0.2` : lift→place 전환 높이 (m)
- `--num_envs 1` : 시각화 시 1 권장
- `--place_vecnorm <경로>` : place 모델이 VecNormalize 사용했을 때만

---

## 알려진 제약 / TODO

- `--place_vecnorm`은 place가 학습 시 VecNormalize 미사용이면 비워둠
- 그리드 좌표 `(0.25, -0.45)`는 place env 기본값 그대로 (시뮬레이션에서 검증 필요)
- 종료 조건은 place 기준 (셀 안정 안착) — lift 단계에서 종료 안 됨
- **재학습**: 다음 단계 (이어서 train.py 작성 예정)

---

## 핵심 설계 결정

- **두 obs 형식 동시 지원**: env는 lift/place 양쪽이 필요로 하는 중간값을 모두 계산. `_compute_intermediate_values()`가 grasp_target_pos / cell_centers 둘 다 갱신
- **VecNormalize 수동 적용**: Lift obs는 34차원, Place obs는 31차원 → SB3 wrapper로 동시에 wrap 불가능. Lift는 통계만 추출해 numpy로 정규화
- **시작 상태는 lift 환경 동일**: example6 handoff 데이터셋 사용 안 함. 진짜 grasp부터 시작
- **그리드는 환경에 항상 존재**: lift 단계에서도 그리드가 보이지만 충돌만 안 시키면 무관
