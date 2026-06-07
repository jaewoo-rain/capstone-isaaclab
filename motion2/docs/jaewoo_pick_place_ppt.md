---
marp: true
theme: default
paginate: true
size: 16:9
header: 'Capstone · OMY-F3M Pick-and-Place'
footer: '2026-05-27 · jaewoo'
style: |
  section {
    font-family: 'Pretendard', 'Malgun Gothic', sans-serif;
    font-size: 26px;
  }
  h1 { color: #1f4e79; }
  h2 { color: #2e75b6; border-bottom: 2px solid #2e75b6; padding-bottom: 6px; }
  code { background: #f3f3f3; padding: 2px 6px; border-radius: 4px; }
  pre { background: #1e1e1e; color: #dcdcdc; border-radius: 6px; }
  table { font-size: 22px; }
  .small { font-size: 20px; }
  .tiny { font-size: 16px; }
  .highlight { background: #fff2cc; padding: 2px 6px; border-radius: 4px; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# OMY-F3M Cartesian Pick-and-Place 파이프라인

### 좌표 한 줄로 실제 로봇 제어하기

**실로봇 적용 성공** · 2026-05-27
jaewoo

---

## 한 줄 요약

> **(x, y, z) 좌표만 주면 OMY-F3M이 안전하게 pick-and-place 를 수행하는 스크립트 셋을 구현하고 실로봇 적용에 성공했다.**

핵심 키워드 3가지

1. **MoveIt 은 계획만, 실행은 FollowJointTrajectory** — wraparound 사고 회피
2. **수직 파지 (joint5 = π/2 path constraint)** — OMY-F3M 그리퍼 특성 반영
3. **3단계 안전 가드 + dry-run 기본값 + confirm 문자열** — 실로봇 적용 안전 근거

---

## 배경 — 기존 방식의 한계

**기존**: `run_teach_pick_place.py` — 관절 각도 직접 재생 (teach & replay)

- 물체 위치가 바뀌면 **사람이 매번 다시 자세를 기록**해야 함
- 일반화 불가능, 비전 연동 어려움
- 관절 공간만 다루므로 좌표 단위 사고가 안 됨

**이번 작업의 목표**

좌표만 입력하면 자동으로 관절값을 계산하여 pick-and-place 수행
→ 추후 비전 / Isaac Lab 시뮬 연동을 위한 기반 마련

---

## 대상 시스템 — OMY-F3M

- **6축 매니퓰레이터** + 1축 그리퍼 (`rh_r1_joint`)
- ROS2 Humble · MoveIt2 · FollowJointTrajectory
- 베이스 프레임: `link0` / EE 프레임: `link6`

**작업 좌표계 (top-view)**

```
         +x (앞쪽, 0 ~ 0.50 m)
          ↑
          │
[로봇 베이스] ──────────→ -y (오른쪽, 0 ~ -0.40 m)
 (link0)
```

| 축 | 의미 | 실용 범위 |
|----|------|-----------|
| +x | 로봇 앞쪽 | 0.0 ~ 0.50 m |
| -y | 로봇 오른쪽 | 0.0 ~ -0.40 m |
| +z | 위쪽 | 0.34 ~ 0.75 m |

---

## 핵심 도전 과제

| # | 문제 | 영향 |
|---|------|------|
| 1 | **역기구학 (IK)** — 좌표에서 6개 관절값 어떻게 얻을까? | 수학적으로 다중 해 존재 |
| 2 | **MoveIt 직접 실행의 wraparound 위험** | 관절이 ±2π 회전하는 사고 가능 |
| 3 | **OMY-F3M 은 위에서 잡지 않고 옆에서 수직 파지** | EE orientation 강한 제약 필요 |

→ 이 세 가지를 각각 어떻게 해결했는지가 이번 발표의 본론

---

## 설계 결정 ① — MoveIt 은 "계획만"

```
   목표 (x,y,z)
        ↓
   ┌──────────────┐
   │  MoveIt2     │  ←  IK + path constraint + collision check
   │  (plan only) │
   └──────┬───────┘
          ↓  JointTrajectory (검증된 관절값들)
   ┌──────────────┐
   │  delta guard │  ←  현재 → 목표 변위 검증
   └──────┬───────┘
          ↓
   FollowJointTrajectory  →  실제 로봇
```

- MoveIt 에 `execute` 시키면 관절이 ±2π 돌아가는 trajectory 를 만들 수 있음
- **plan 결과를 받아 직접 검증 후 전송** → wraparound 원천 차단

---

## 설계 결정 ② — 수직 파지 제약

OMY-F3M 은 **위에서 잡지 않고 옆에서 수평으로 접근**해 수직 파지

```
  [그리퍼]
  │  │   ← 손가락이 위아래 방향 (수직)
  └──┘
  → 물체에 앞쪽(+x) 에서 수평으로 접근
```

**두 가지 제약을 항상 적용**

1. EE orientation 고정
   `quat_wxyz ≈ [0.5063, 0.4936, 0.4936, 0.5063]`

2. **joint5 path constraint** ← MoveIt 의 핵심 활용 포인트
   `joint5 = π/2 (≈1.5708 rad)`, tolerance `±0.08 rad`
   계획 **경로 전체**에 걸쳐 이 값 유지 요구

---

## 설계 결정 ③ — 3단계 안전 가드

| 단계 | 검사 내용 | 차단되는 사고 |
|------|-----------|--------------|
| 1. **Workspace guard** | 좌표가 설정된 직육면체 안인지 | 테이블 밖·바닥 충돌 |
| 2. **Plan-level delta guard** | plan 내부 segment 간 변위 ≤ 0.12 rad | 급격한 trajectory |
| 3. **Actual delta guard** | 현재 실제 관절 → 목표 관절 변위 ≤ 0.35 rad | 큰 점프 이동 |

**이중 실행 잠금**

- 기본값은 **dry-run** (계획만, 명령 전송 X)
- 실제 실행: `--execute` **AND** `--confirm <정확한_문자열>` 둘 다 필요
- 단계별로도 step 이름을 직접 타이핑해야 진행 (`--no-step-prompts` 로 끌 수 있음)

---

## 스크립트 6종 — 한눈에 보기

| 스크립트 | 역할 |
|----------|------|
| [`show_ee_pose.py`](../scripts/jaewoo/show_ee_pose.py) | 현재 EE 위치 읽어서 출력 + 다음 명령어 자동 생성 |
| [`run_ee_pose_move.py`](../scripts/jaewoo/run_ee_pose_move.py) | 단일 (x,y,z) 좌표로 EE 이동 |
| [`run_pick_place.py`](../scripts/jaewoo/run_pick_place.py) | **9단계 pick-and-place 전체** |
| [`go_to_start.py`](../scripts/jaewoo/go_to_start.py) | YAML `start` 자세로 홈 복귀 |
| [`go_to_init.py`](../scripts/jaewoo/go_to_init.py) | YAML `init` 자세로 관절 원점 복귀 |
| [`toggle_gripper.py`](../scripts/jaewoo/toggle_gripper.py) | 그리퍼 토글 (열림 ↔ 닫힘) |

공통 유틸: `motion2/scripts/real_moveit_common.py` (MoveIt goal 빌더, delta 검증, TF2 조회)
설정 파일: `motion2/config/teach_pick_place_waypoints.yaml` (Z 좌표, 홈 관절값)

---

## 메인 파이프라인 — 9단계 시퀀스

```
┌─ 1. pre_grasp      pick 위치 위로 호버    (x₁, y₁, approach_z ≈ 0.372)
├─ 2. grasp          pick 위치로 하강       (x₁, y₁, grasp_z    ≈ 0.345)
├─ 3. close_gripper  그리퍼 닫기 → 파지
├─ 4. lift           들어올리기             (x₁, y₁, lift_z = approach_z + 0.06)
├─ 5. transport      place 위로 수평 이동   (x₂, y₂, lift_z)        ← 가장 긴 이동
├─ 6. place_descend  place 위치로 하강      (x₂, y₂, place_z = grasp_z)
├─ 7. open_gripper   그리퍼 열기 → 해제
├─ 8. retract        들어올리기             (x₂, y₂, lift_z)
└─ 9. home           YAML 'start' 관절값으로 직접 재생 (MoveIt 미사용)
```

**Z 좌표는 YAML 에서 자동**: `approach_z`, `grasp_z` 는 사전 기록된 waypoint 값을 그대로 사용
→ 사용자는 (x, y) 두 쌍만 신경 쓰면 됨

---

## 9단계 시퀀스 — 시각화

<div class="small">

```
       pre_grasp        lift           transport         retract
          │              │                  │              │
          ↓              ↑                  →              ↑
        grasp        (close)           place_descend   (open) ────→ home
       (하강)        그리퍼            (하강)          그리퍼      홈 복귀
                     닫기                              열기
```

</div>

각 단계는 **개별 MoveIt plan** 으로 처리
중간에 plan 실패 / delta 초과 발생하면 즉시 멈춤 (이미 들고 있는 물체는 사용자 판단)

---

## 사용법 — 가장 임팩트 있는 한 줄

**선행 조건** (별도 터미널)

```bash
ssh root@omy-SNPR44B1021.local
ros2 launch open_manipulator_bringup omy_f3m.launch.py
```

**Dry-run** — 좌표 미지정 시 YAML 기본값 자동 사용

```bash
python3 motion2/scripts/jaewoo/run_pick_place.py
```

**실제 실행** — 단 한 줄

```bash
python3 motion2/scripts/jaewoo/run_pick_place.py \
    --execute --confirm EXECUTE_PICK_PLACE
```

**좌표 직접 지정**

```bash
python3 motion2/scripts/jaewoo/run_pick_place.py \
    --pick-x 0.496 --pick-y -0.113 \
    --place-x 0.321 --place-y -0.394 \
    --execute --confirm EXECUTE_PICK_PLACE
```

---

## 실로봇 적용 결과

- **실제 OMY-F3M 로봇에서 9단계 시퀀스 전체 성공**
- pick: `(0.496, -0.113)` → place: `(0.321, -0.394)` (YAML 기본 좌표)
- 전체 소요 시간: 약 70 초 (단계당 8 초 × 7 + 그리퍼)
- dry-run → execute 전환 시 사고 0 건

<!-- 여기에 실제 실행 사진/영상 캡처 삽입 -->

**> 영상 또는 사진 4~6 컷 (pre_grasp / grasp / transport / place / home)**

---

## 마주친 문제와 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| `JOINT5 CONSTRAINT NOT MET` | 특정 위치에서 joint5=π/2 IK 해 없음 | `--no-constrain-joint5` 옵션 추가 |
| `plan-level guard violated` | 큰 이동에서 segment delta 초과 | `--max-joint-delta` CLI 오버라이드 추가 |
| `±2π wraparound` 트라젝토리 | MoveIt execute 직접 사용 | plan only → FollowJointTrajectory 분리 |
| confirm 텍스트 오타 시 무조건 거부 | 의도된 안전장치 | 에러 메시지에 예상 텍스트 표시 |

---

## 향후 작업

- **비전 연동** — 카메라로 (x, y, z) 자동 감지 → 이 파이프라인의 입력으로 직접 연결
- **Isaac Lab 시뮬 ↔ 실로봇 sim2real 비교** — 같은 좌표로 시뮬과 실로봇 결과 비교
- **다양한 물체로 일반화** — 그리퍼 폭·파지 깊이 자동 조정
- **궤적 시각화 도구** — dry-run 결과를 RViz 에 미리 표시

---

## 결론

이번 주 작업으로 **좌표 입력 → 실로봇 동작** 의 추상화 계층을 확보했다.

- ✓ Cartesian 좌표 인터페이스 (`run_ee_pose_move.py`, `run_pick_place.py`)
- ✓ 안전 (3단계 가드 · dry-run · confirm)
- ✓ 실로봇 적용 성공
- ✓ 비전·시뮬 연동을 위한 기반

**핵심 학습**

> "MoveIt 의 plan 결과를 신뢰하되, **execute 는 직접** 한다."
> 자동화 도구를 통째로 받아쓰지 않고 **계획과 실행을 분리**한 것이 안전성의 핵심이었다.

---

<!-- _class: lead -->

# Q & A

감사합니다.

`motion2/scripts/jaewoo/`
`motion2/docs/jaewoo_pick_place_ppt.md`
