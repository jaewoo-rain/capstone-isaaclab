# 전체 아키텍처 — 카메라 → RL → 실물 파이프라인

카메라에서 **무슨 값**을 받아 **어떤 토픽**으로 흘려 **RL→로봇**까지 잇는지의 설계도.
(개요/상태는 `HANDOFF.md`, 비전 내부는 `camera/PLAN.md`, 학습은 `camera/dataset|train`.)

---

## 1. 카메라가 공급하는 값 = "평면 pose (x, y, yaw)" 뿐

비전은 **link0 기준 평면 위치·각도만** 준다. 6DOF 도, z 높이도 주지 않는다.

| 대상 | 주는 값 | 안 주는 값 | 왜 |
|------|---------|-----------|-----|
| box (grasp 대상) | `x, y` [m], `yaw` [rad] | z, roll/pitch | z 는 YAML/실측 호버·파지 높이. 자세는 수직파지 고정 |
| cell (유압블록 슬롯) | `x, y` [m] | yaw(≈0 고정), z | 셀은 고정구조물 → yaw=0, z 는 실측 |

내부적으로 detector 는 `label, score, center_px, depth_m, mask` 도 갖지만 **토픽엔 (x,y,yaw)만** 인코딩.

> ⚠️ **z 는 vision 에서 받지 않는다.** Pose 의 `position.z` 에 카메라 depth 를 참고용으로 싣지만
> 그건 link0 z 가 아니다(카메라→물체 거리). RL/실행은 z 를 YAML/실측에서 가져온다.

---

## 2. 토픽 계약 (data contract)

### 현재 (PHASE F 적용됨)
| 토픽 | 타입 | 발행 | 내용 |
|------|------|------|------|
| `/vision/box_coarse` | `PoseArray` | ceiling_detector | 천장캠 **모든** 박스 |
| `/vision/box_target` | `PoseStamped` | ceiling_detector | grasp 타깃 1개 = 분포중심 `(0.45,-0.10)` 최근접 |
| `/vision/cell_coarse` | `PoseArray` | ceiling_detector | 천장캠 **모든** 셀(슬롯) |
| `/vision/box_fine` | `PoseStamped` | wrist_detector | 손목캠 박스(근접 정밀) |

### 추가 설계 (미구현 — 아래 §5)
| 토픽 | 타입 | 발행 | 내용 |
|------|------|------|------|
| `/vision/cell_target` | `PoseStamped` | (insert bridge 또는 detector) | 삽입할 슬롯 1개 (규칙/인덱스로 선택) |

### Pose 인코딩 규약 (모든 vision 토픽 공통)
```
position.x, position.y = link0 기준 xy [m]
position.z             = 카메라 depth [m]  ← 참고용. link0 z 아님(소비측은 무시)
orientation            = quat_from_z_yaw(yaw)   # z축 회전만
  → 디코드:  yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
header.frame_id        = "link0"
```

---

## 3. 전체 데이터 흐름 (블록도)

```
┌─ HARDWARE ───────────────────────────────────────────────────────────────────┐
│  천장캠 D435i(top-down)      손목캠 D405(wrist, link6)        OMY-F3M arm       │
│   color+depth                color+depth + TF2               joints / gripper  │
└──────┬────────────────────────────┬─────────────────────────────────▲─────────┘
       │ RealSense                  │ RealSense + TF2(link0←cam)        │
┌──────▼──────────────┐   ┌─────────▼───────────┐                       │
│ ceiling_detector    │   │ wrist_detector      │                       │
│  YOLOv8-seg →NMS    │   │  depth PCA / YOLO   │                       │
│  →긴변 yaw          │   │  →긴변 yaw          │                       │
│  →extrinsic→link0   │   │  →TF2→link0         │                       │
└──────┬──────────────┘   └─────────┬───────────┘                       │
       │ publish                    │ publish                           │
   ╔═══▼═══════════════════════╗╔═══▼════════════╗                      │
   ║ /vision/box_coarse  [Array]║║ /vision/box_   ║   ← 토픽(= 계약)      │
   ║ /vision/box_target  [Pose] ║║   fine  [Pose] ║                      │
   ║ /vision/cell_coarse [Array]║╚════════════════╝                      │
   ║ (/vision/cell_target[Pose])║                                       │
   ╚═══┬════════════════════════╝                                       │
       │ (x, y, yaw)  link0                                              │
┌──────▼──────────────────────────────────────────────────────────┐    │
│  BRIDGE / 오케스트레이션 (글루 — 사람 타이핑을 대체)              │    │
│   run_grasp_vision.py   ← box_target            ✅ 작성됨          │    │
│   run_insert_vision.py  ← cell_target           ⬜ 미작성          │    │
│   run_pipeline_vision.py← box_target+cell_target ⬜ 미작성         │    │
│   (구독 → x,y,yaw 디코드 → 좌표 채워 RL 스크립트 호출)            │    │
└──────┬───────────────────────────────────────────────────────────┘    │
       │ --box-x/--box-y/--box-yaw (또는 --cell-*)                       │
┌──────▼───────────────────────────────────────────────────────────┐    │
│  RL (내부 가상 롤아웃 — 60Hz 폐루프 불가 회피)                     │    │
│   grasp:  obs6  → action[Δx,Δy,Δyaw] → 최종 정렬 EE (xy,yaw)       │    │
│   insert: obs7  → yaw-only           → 최종 정렬 yaw               │    │
└──────┬───────────────────────────────────────────────────────────┘    │
       │ 최종 EE pose "1점"                                              │
┌──────▼───────────────────────────────────────────────────────────┐    │
│  EXECUTION                                                         │    │
│   MoveIt(IK) plan → FollowJointTrajectory  + GripperCommand        │────┘
│   z 높이 = YAML/실측 (vision 아님).  3단계 guard·dry-run 기본       │
└───────────────────────────────────────────────────────────────────┘
```

---

## 4. 제어 흐름 — grasp 구간 (지금 동작하는 경로 ✅)

```
ceiling_detector --ros ──▶ /vision/box_target (x,y,yaw 갱신 발행)
                                   │
run_grasp_vision.py  ── 구독 1개 ──┘  → (x,y,yaw) 디코드
                                   │
                                   ▼  좌표 채워 호출
run_grasp.py --box-x --box-y --box-yaw [--execute --confirm EXECUTE_GRASP]
    ① 현재 EE/joints 읽기   ② 내부 롤아웃(수렴?)   ③ MoveIt 이동
    ④ pre_grasp→grasp→close→lift   (단계 타이핑 확인, guard)
```
> grasp 전 EE 를 박스 근처 수직파지로 pre-position 해 둬야 롤아웃 발산 안 함(수동/별도 단계).

전체 chain(grasp→turn→insert→place)은 `rl/all/run_pipeline.py` 가 좌표 수동입력으로 존재 →
vision 연동판 `run_pipeline_vision.py` 가 그 자리에 box_target+cell_target 을 먹이면 완성(미검증).

---

## 5. 추가로 설계할 것 (미구현)

1. **`/vision/cell_target`** — 삽입 슬롯 1개 선택. 셀이 여러 개(cell_coarse=Array)라 "어느 슬롯에
   넣을지"는 task 결정. **insert bridge 가 cell_coarse 구독→규칙(`--cell-index`/빈슬롯)으로 선택** 권장
   (detector 는 perception 만). 가시화용으로 `/vision/cell_target` 발행도 가능.
2. **`run_insert_vision.py`** — cell_target → insert RL (yaw-only). 셀 뒤쪽(-x) workspace `--x-min` 주의.
3. **`run_pipeline_vision.py`** — box_target + cell_target → grasp→lift→turn→insert→place 자동.
4. **손목캠 fine 보정 stage(선택)** — coarse approach 후 `/vision/box_fine` 로 재정렬 후 파지(정밀↑).
5. **QoS/타이밍** — box_target 등 저주파 → 1개 받으면 됨. 필요 시 `transient_local`(latch)로
   늦게 뜬 구독자도 마지막 값 수신. no-detection 은 bridge 타임아웃으로 처리(이미 run_grasp_vision 구현).

---

## 7. 두 카메라 역할 분담 + 갱신 주기 (의도된 설계)

| 카메라 | 시점 | 역할 | 빈도 | 주는 값 |
|--------|------|------|------|---------|
| 천장 D435i | top-down | **coarse 1회** | 1-shot | box `(x,y,yaw)` + cell `(x,y)` (yaw≈0 고정) |
| 손목 D405 | EE 근접 | **fine 추적** | 연속(10~15Hz) | grasp: box `(x,y,yaw)` / insert: slot yaw |

### ★ 핵심 보정 — "연속 측정 ≠ 연속 제어"
카메라는 연속으로 측정하지만, **팔은 60Hz 폐루프를 못 돈다**(MoveIt plan~5s + exec~6s, blocking).
그래서 손목 값은 approach 중 **몇 개 체크포인트에서만 이산적으로** 소비한다:

```
각 체크포인트: 손목 (x,y,yaw) 1프레임 읽기 → 내부 롤아웃 재계산 → EE 1점 이동
```
즉 "계속 전달"은 **perception 층의 사실**이고, **실행은 이산 재계획(discrete re-plan)** 으로 흡수한다.
(이게 이 프로젝트의 "내부 롤아웃 → 단일 이동" 설계 이유다.)

### 시간 흐름
```
GRASP:
  [천장 1-shot]  box(x,y,yaw) + cell(x,y)
       └▶ coarse approach (박스 위 호버)
            └▶ [손목 연속] box(x,y,yaw) ──(체크포인트마다 재롤아웃)──▶ fine 하강·파지·들기
INSERT:
  turn(뒤로) → 셀 위 호버
       └▶ [손목 연속] slot yaw ──(이산 재정렬)──▶ insert RL yaw 정렬 → place 하강
```

### 현재 구현
- `result/run_grasp_vision.py` ✅ — S1~S4 단일프로세스 오케스트레이터로 **손목 fine 반복수렴 구현됨**(§8).
  천장 coarse 1회 + 손목 box_fine 반복(rollout 재계산→미세이동→수렴/N_max). run_grasp.py 미수정·primitive 재사용.
- insert 때 손목캠이 **잡은 박스에 가려지는지(occlusion)** 는 현장 확인 필요(D405 가 link6 근처라 슬롯이 보여야 함).
- 셀 yaw 는 천장에서 0 고정으로 주지만, 손목으로 **슬롯 yaw 를 다시 재서 정밀 보정**하는 게 네 설계의 insert 부분.

## 6. 파일

| 역할 | 파일 | 상태 |
|------|------|------|
| 천장캠 검출·발행 | `camera/ceiling_detector.py` | ✅ (box_target/coarse, cell_coarse) |
| 손목캠 검출·발행 | `camera/wrist_detector.py` | ✅ (box_fine) |
| 좌표변환·yaw·TF2 | `camera/coord_transform.py` | ✅ |
| **grasp 비전 파이프라인** | `result/grasp/run_grasp_vision.py` | ✅ 신규 (S1~S4 오케스트레이터) |
| insert 비전 파이프라인 | `result/insert/run_insert_vision.py` | ✅ 신규 (I1~I3, self-contained, ⚠️실험적) |
| 전체 chain 비전 파이프라인 | `result/pipeline/` | ⬜ |

> **`result/grasp/`** = **self-contained 배포 폴더.** 필요한 모듈(config·grasp_policy·grasp_rollout·
> run_grasp·run_pick_place·real_moveit_common)과 `checkpoints/`·`teach_pick_place_waypoints.yaml` 을
> 폴더 안에 **복사**해 둠 → 다른 폴더 의존 0, 이 폴더만 통째로 scp/docker cp 하면 로봇에서 바로 실행.
> (원본은 `rl/grasp`·`jaewoo`·`scripts` 에 그대로. 원본 수정 시 이 복사본은 수동 동기화 필요.)
| RL grasp 실행(좌표입력) | `rl/grasp/run_grasp.py` | ✅ |
| RL 전체 chain(좌표입력) | `rl/all/run_pipeline.py` | 🟡 청사진 |

---

## 8. grasp leg 상세 설계 (S1~S4) — 확정 범위

> 결정(2026-06): **범위=grasp leg**, **S3 fine=반복수렴**, **insert=yaw-only(나중)**. 아래는 그 설계.
> 코딩은 나중 — 이 절은 "무엇을 만들지"의 합의 문서.

### 스테이지 정의
| S | 이름 | 입력(토픽) | RL/모션 | 종료조건 | 실패 게이트 |
|---|------|-----------|---------|----------|-------------|
| S1 | SENSE_COARSE | `/vision/box_target` (1회) | — | 1프레임 수신 | timeout→중단 / 분포밖→경고·차단 |
| S2 | APPROACH | (S1 좌표) | MoveIt → (box_x,box_y,**hover_z**) 수직파지 | 이동완료 | plan/workspace guard |
| S3 | FINE_REFINE | `/vision/box_fine` (반복) | 롤아웃 재계산 → 미세 이동 | residual<tol **또는** N_max | 손목 무검출 K회→coarse fallback / 롤아웃 발산→중단 |
| S4 | GRASP | (S3 최종 EE) | 하강 → close → lift | 들기 완료 | 표준 3단계 guard |

### S3 반복수렴 루프 (설계 의사코드 — 실제 코드 아님)
```
box_est = box_coarse                       # S1 천장 값으로 시작
no_det  = 0
for i in range(N_MAX):                      # N_MAX=2~3 (각 iter = MoveIt 1회 ≈10s)
    fine = read("/vision/box_fine", timeout)   # 손목 1프레임 (연속측정 중 1장만 소비)
    if fine is None:
        no_det += 1
        if no_det >= K: break               # 손목 계속 안 보임 → coarse 로 진행(경고)
        continue
    box_est = fine                          # 손목이 더 정확 → 교체 (원하면 EMA 스무딩)
    target_ee = rollout_grasp(box_est, start=current_ee)   # 정렬 EE 재계산
    residual  = | target_ee.xy - current_ee.xy |
    move_to(target_ee, hover_z)             # MoveIt 1점 이동
    current_ee = target_ee
    if residual < REFINE_XY_TOL and dyaw < REFINE_YAW_TOL:
        break                               # 수렴
# → S4 GRASP
```
**수렴 원리**: EE 가 박스 위로 정확해질수록 손목이 박스를 더 중앙에 봄 → 롤아웃 보정량(residual)이
줄어듦 → 보통 2~3회면 mm 단위 수렴.
**파라미터(튜닝)**: `REFINE_XY_TOL≈5mm`, `REFINE_YAW_TOL≈3°`, `N_MAX=2~3`, `K=3`.

### 비용/제약
- 각 iter 가 MoveIt plan+exec(≈10s) → N_MAX=3 이면 fine 단계만 ≈30s 추가. **무한루프 금지, 작은 N.**
- "손목 연속(10~15Hz 측정)"은 각 iter 시작에 **1프레임만** 소비 → 측정은 연속, **제어는 이산**(§7 원칙).

### 코드 구조 (구현됨)
- 반복루프가 [구독 + 롤아웃 + MoveIt 이동]을 **한 프로세스**에서 돌아야 하므로, `result/run_grasp_vision.py`
  는 **단일 프로세스 오케스트레이터**로 구현. `run_grasp.py` 는 **수정하지 않고** 그 primitive(롤아웃·
  run_pick_place 헬퍼·`_compose_yaw_quat`/`_check_box_distribution`)만 import 해 S1~S4 를 얹었다.
- 안전: dry-run 기본·`--confirm`·3단계 guard·단계 타이핑 그대로 상속. 발산 target 으로는 이동 안 함(검토 반영).

### ★ 현장에서 검증할 물리 미지수 (설계 전제 — 코딩 전 확인)
1. **손목 D405 작동거리 vs hover_z**: D405 는 근접센서. 호버(approach_z≈0.37, 박스top≈0.34 → 거리 ~3cm)
   에서 박스가 FOV+depth 범위에 들어오는가? 너무 가까우면 **S2 의 hover_z 를 approach_z 와 별도로**
   "손목이 박스를 잘 보는 높이"로 잡아야 한다(이게 S2 가 grasp_z 가 아닌 hover_z 인 이유).
2. **그리퍼 손가락 occlusion**: 손목 시야를 손가락이 가리는가.
3. → 둘 다 현장 PHASE 5(detection 테스트)에서 손목캠으로 박스 보며 확인 후 파라미터 확정.

