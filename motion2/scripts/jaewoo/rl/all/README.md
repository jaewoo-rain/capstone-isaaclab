# all/ — grasp+insert 전체 적재 파이프라인 (sim2real)

motion3 통합 chain(`play_motion_chain_with_grasp_insert.py`)을 실제 OMY-F3M 로봇에서
**한 번에** 실행한다. 앞쪽 박스를 잡아 → 뒤로 돌려 → 뒤쪽 셀에 적재한다.

> ⚠️⚠️ **실험적 골격(청사진)**. grasp 단독(`../grasp/`)만 검증됐고, 전체 chain은
> 실물 검증 안 됨. 아래 "미검증·실측 필요"를 반드시 읽고 단계별로 끊어 검증하세요.

---

## 전체 시퀀스 (12 stage)

```
앞쪽(책상, +x)                          뒤쪽(셀, -x)
─────────────────────                   ─────────────────────
1. 다가감 (pre_grasp_z)
2. grasp RL  ← 박스 xy/yaw 정렬 (정책)
3a. 하강 (grasp_z)
3b. 잡기 (그리퍼 close)
3c. 들기 (lift_z)
                3c2. turn ── joint1만 -2.90 회전 (뒤로) ──▶
                                        3d1. 셀 위로 (lift_z 유지)
                                        3d2. 호버 하강 (hover_z 0.32)
                                        4. insert RL ← 셀 yaw 정렬 (정책, yaw-only)
                                        5a. place 하강 (place_z) ⚠️ 드리프트
                                        5b. 열기
                                        6a. 올라오기 (retract_z)
                6b. home ── joint 복귀 ──▶
```

- **cartesian 이동**(1,3a,3c,3d,5a,6a) → MoveIt plan + FollowJointTrajectory
- **joint 직접**(3c2 turn, 6b home) → 관절 trajectory 직접 (MoveIt 미사용)
- **RL 정책**(2 grasp, 4 insert) → 내부 롤아웃으로 최종 자세 계산 후 1회 이동
- **그리퍼**(3b, 5b)

---

## ⚠️ 미검증 · 실측 필요 (중요)

grasp 단독과 달리, 전체 chain엔 검증 안 된 부분이 많습니다:

| 항목 | 상태 | 대응 |
|------|------|------|
| **z 높이** (pre_grasp/grasp/lift/hover/place/retract) | config 의 **SIM(ground 기준) 값** | 실제 로봇 link0 기준으로 **실측** 후 `--*-z` 로 지정 |
| **turn/home 관절** (joint1=-2.90, home pose) | **SIM 자세** | 실제 OMY 관절과 일치 확인. base mount 차이 가능 |
| **뒤쪽 셀 좌표계** | 셀이 로봇 **뒤(-x)** | workspace `--x-min -0.55` 기본. reach 한계 확인 |
| **turn 후 자세 매핑** | turn 뒤 그리퍼 yaw 를 TF 로 재측정해 insert 시작 | sim `turned_quat` 과 실물 일치 미검증 |
| **place 하강 드리프트** | sim 도 xy 8~10cm 밀림 (미해결 병목) | 느린 하강(`DUR_PLACE`) + 충분한 검증 |
| **grasp 자세(top-down vs side)** | `../grasp/README` 의 미검증 리스크 동일 | grasp 2단계 검증 먼저 |

→ **이 스크립트는 "어느 단계를 어떤 명령으로 변환하는지"의 청사진**입니다. 바로
end-to-end 실행하지 말고, 아래 단계별 검증으로 하나씩 확인하며 값을 보정하세요.

---

## 실행

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd /home/jaewoo/IsaacLab/source/motion2/scripts/jaewoo/rl/all

# 전체 dry-run (모든 plan 확인, 로봇 안 움직임)
python3 run_pipeline.py \
    --box-x 0.45 --box-y -0.10 --box-yaw 0.0 \
    --cell-x -0.34 --cell-y 0.0 --cell-yaw 0.0
```

### 단계별 검증 순서 (`--stop-after` 로 끊어서)

```bash
# 1) 앞쪽만: 다가감→grasp→잡기→들기 까지 (실제 실행)
python3 run_pipeline.py --box-x .. --box-y .. --cell-x .. --cell-y .. \
    --stop-after grasp_lift --execute --confirm EXECUTE_PIPELINE

# 2) turn 까지 — 뒤로 도는 동작 확인 (큰 회전, 주의)
python3 run_pipeline.py ... --stop-after turn --execute --confirm EXECUTE_PIPELINE

# 3) insert 정렬까지 — 셀 위 호버 + yaw 정렬
python3 run_pipeline.py ... --stop-after insert --execute --confirm EXECUTE_PIPELINE

# 4) 전체 (place 하강 포함 — 드리프트 검증 후)
python3 run_pipeline.py ... --execute --confirm EXECUTE_PIPELINE
```

기본 dry-run. `--execute --confirm EXECUTE_PIPELINE` 둘 다 필요. 단계별 타이핑 확인 있음.

---

## 주요 옵션

| 옵션 | 설명 |
|------|------|
| `--box-x/y/yaw` | 앞쪽 박스 (link0) |
| `--cell-x/y/yaw` | 뒤쪽 셀 (link0) |
| `--pre-grasp-z .. --retract-z` | 각 stage z 높이 (실측값으로 override) |
| `--stop-after grasp_lift\|turn\|insert\|place` | 그 단계까지만 실행 |
| `--no-align-yaw` | grasp yaw 정렬 끄기 (자세 검증) |
| `--turn-max-joint-delta` | turn/home joint 허용 변위 (기본 3.2rad) |
| `--x-min` 등 | workspace (셀 뒤쪽이면 x_min 음수, 기본 -0.55) |
| `--execute --confirm EXECUTE_PIPELINE` | 실제 실행 |

---

## 파일 / 설계

| 파일 | 역할 |
|------|------|
| `config.py` | 전체 파이프라인 상수 (grasp+insert 규약, stage z/관절/duration) |
| `run_pipeline.py` | 6단계 오케스트레이터 (독립 — 정책 로드·롤아웃 자체 포함) |

**왜 독립 모듈인가**: `../grasp/`, `../insert/` 가 각자 `config.py`를 가져서, all/이 그것들을
import 하면 모듈명이 충돌합니다. 그래서 all/은 정책 로드·롤아웃을 자체 포함하고, 두 폴더의
**체크포인트(zip/pkl)만 경로로 참조**합니다 (`config.GRASP_CKPT`, `config.INSERT_CKPT`).

검증(로봇 없이): 두 정책 로드 OK, grasp 롤아웃 수렴(68 step, 3~5mm), insert 롤아웃 수렴(21 step, 0.6°).
