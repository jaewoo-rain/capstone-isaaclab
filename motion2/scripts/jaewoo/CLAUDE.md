# motion2/scripts/jaewoo — 작업 내용 및 사용법

## 개요

이 폴더는 OMY-F3M 실제 로봇에서 **Cartesian 좌표 기반 pick-and-place**를 수행하기 위해 작성된 스크립트 모음이다.

기존 `run_teach_pick_place.py`(관절 각도 직접 재생 방식)와 달리, 이 폴더의 스크립트들은:
- **x, y, z 좌표만 주면** MoveIt이 역기구학(IK)으로 관절값을 계산한다.
- MoveIt은 **계획(plan)만** 수행하고, 실제 이동은 `FollowJointTrajectory`로 직접 전송한다.
  → MoveIt이 직접 실행하면 발생할 수 있는 **±2π wraparound 위험**을 회피한다.
- **joint5 = π/2 path constraint**를 MoveIt 경로 전체에 걸어 수직 파지 자세를 유지한다.

---

## 파일 목록

| 파일 | 역할 |
|------|------|
| `show_ee_pose.py` | 현재 EE(end-effector) 위치를 읽어 출력 |
| `run_ee_pose_move.py` | 단일 Cartesian 좌표로 EE 이동 |
| `run_pick_place.py` | 두 좌표(pick/place) 받아 9단계 pick-and-place 전체 실행 |

의존하는 공통 유틸리티:

| 파일 | 위치 | 역할 |
|------|------|------|
| `real_moveit_common.py` | `motion2/scripts/` | MoveIt goal 생성, delta 검증, TF2 조회 |
| `teach_pick_place_waypoints.yaml` | `motion2/config/` | Z 좌표 및 홈 관절값 참조 |

---

## 수직 파지(Vertical Grip) 설계

OMY-F3M은 물체를 **위에서 집지 않고 옆에서 수직으로 집는다.**

```
  [그리퍼]
  │  │   ← 손가락이 위아래 방향 (수직)
  └──┘
  → 물체에 앞쪽(+x 방향)에서 수평으로 접근
```

이를 위해 두 가지 제약이 항상 적용된다:

1. **EE orientation 고정**: `quat_wxyz ≈ [0.5063, 0.4936, 0.4936, 0.5063]`  
   (YAML waypoints `'1'`, `'2'` 에서 실측)

2. **joint5 path constraint**: `joint5 = π/2 (≈1.5708 rad)`, 허용 오차 ±0.08 rad  
   MoveIt 계획 경로 전체에 걸쳐 이 관절값 유지를 요구한다.

---

## 안전 설계 (3단계 guard)

모든 암 이동 스크립트에 공통 적용:

| 단계 | 검사 내용 |
|------|-----------|
| 1. Workspace guard | 목표 좌표가 설정된 직육면체 작업 공간 안인지 확인 |
| 2. MoveIt plan delta guard | plan 내 start→goal, segment 간 관절 변위 ≤ 한계 |
| 3. Actual delta guard | 현재 실제 관절값 → 목표 관절값 변위 ≤ 한계 |

기본 delta 한계: `max_joint_delta = 0.35 rad`, `max_segment_delta = 0.12 rad`

실제 실행 시 이중 확인:
- `--execute` 플래그 필수
- `--confirm <TEXT>` 정확한 문자열 필수

---

## 1. show_ee_pose.py — 현재 EE 위치 확인

### 역할
로봇을 원하는 위치로 이동시킨 뒤 실행하면, 현재 EE 위치(link0 기준)를 출력하고  
`run_ee_pose_move.py`에 바로 붙여 쓸 수 있는 명령어를 생성해준다.

### 사용 예시

```bash
# 현재 위치 1회 읽기
python3 motion2/scripts/jaewoo/show_ee_pose.py

# 3회 평균 (로봇이 진동 중일 때)
python3 motion2/scripts/jaewoo/show_ee_pose.py --samples 3

# EE 프레임 변경
python3 motion2/scripts/jaewoo/show_ee_pose.py --ee-frame link6
```

### 출력 형식

```
============================================================
[show-ee-pose] frame: link0 → link6
[show-ee-pose] position  : x=0.496000  y=-0.113000  z=0.345000  [m]
[show-ee-pose] quat_wxyz : w=0.506300  x=0.493600  y=0.493600  z=0.506300
============================================================

# run_ee_pose_move.py 에 사용할 인자 (복사해서 사용):
  --x 0.496000 --y -0.113000 --z 0.345000

# 바로 실행할 명령어 (dry-run):
  python3 motion2/scripts/jaewoo/run_ee_pose_move.py --x 0.496000 --y -0.113000 --z 0.345000

# 실제 실행:
  python3 motion2/scripts/jaewoo/run_ee_pose_move.py --x 0.496000 --y -0.113000 --z 0.345000 --execute --confirm EXECUTE_EE_POSE_MOVE
```

### 전체 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--base-frame` | `link0` | 기준 프레임 |
| `--ee-frame` | `link6` | EE 프레임 |
| `--samples N` | `1` | N회 읽어 평균 |
| `--interval` | `0.3` s | 샘플 간 대기 |
| `--timeout` | `5.0` s | TF2 대기 타임아웃 |

---

## 2. run_ee_pose_move.py — 단일 좌표 이동

### 역할
`--x --y --z`로 목표 Cartesian 위치를 받아 로봇을 이동시킨다.  
기본값으로 수직 파지 orientation + joint5 path constraint가 적용된다.

### 사용 예시

```bash
# dry-run (기본: 수직 파지 orientation + joint5 constraint)
python3 motion2/scripts/jaewoo/run_ee_pose_move.py \
    --x 0.40 --y -0.11 --z 0.37

# 실제 실행
python3 motion2/scripts/jaewoo/run_ee_pose_move.py \
    --x 0.40 --y -0.11 --z 0.37 \
    --execute --confirm EXECUTE_EE_POSE_MOVE

# joint5 constraint 없이 (orientation만 고정)
python3 motion2/scripts/jaewoo/run_ee_pose_move.py \
    --x 0.40 --y -0.11 --z 0.37 \
    --no-constrain-joint5

# 방향 직접 지정 (RPY)
python3 motion2/scripts/jaewoo/run_ee_pose_move.py \
    --x 0.40 --y -0.11 --z 0.37 \
    --rpy 0.0 1.5708 0.0

# 현재 EE 방향 유지
python3 motion2/scripts/jaewoo/run_ee_pose_move.py \
    --x 0.40 --y -0.11 --z 0.37 \
    --use-current-ori
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--x` `--y` `--z` | 필수 | 목표 EE 위치 [m] (link0 기준) |
| `--execute` | False | 실제 실행 (없으면 dry-run) |
| `--confirm TEXT` | `""` | `EXECUTE_EE_POSE_MOVE` 입력 필요 |
| `--duration` | `6.0` s | FollowJointTrajectory 실행 시간 (최소 4.0) |
| `--constrain-joint5` | True | joint5=π/2 path constraint 적용 |
| `--no-constrain-joint5` | — | constraint 비활성화 |
| `--joint5-tolerance` | `0.08` rad | joint5 constraint 허용 오차 |
| `--max-joint-delta` | `0.35` rad | 현재→목표 최대 관절 변위 |
| `--rpy R P Y` | — | 목표 방향 RPY [rad] |
| `--quat W X Y Z` | — | 목표 방향 쿼터니언 |
| `--use-current-ori` | — | 현재 EE 방향 유지 |
| `--planning-time` | `5.0` s | MoveIt 계획 시간 |
| `--attempts` | `5` | MoveIt 계획 재시도 횟수 |

### 작업 공간 기본값

```
x: [-0.10, 0.55]  y: [-0.45, 0.20]  z: [0.10, 0.50]  (단위: m, link0 기준)
```

x_max=0.55는 YAML waypoints '1','2' 위치(x≈0.496)를 커버하도록 여유를 둔 값이다.

---

## 3. run_pick_place.py — 9단계 pick-and-place 파이프라인

### 역할
pick 좌표(x1,y1)와 place 좌표(x2,y2)를 받아 9단계를 순서대로 실행한다.  
좌표를 주지 않으면 YAML `waypoints['2']`(pick), `waypoints['4']`(place) 위치를 자동으로 사용한다.

### 실행 시퀀스

| 단계 | 이름 | 동작 |
|------|------|------|
| 1 | `pre_grasp` | pick 위치 위로 호버 (approach_z) |
| 2 | `grasp` | pick 위치로 하강 (grasp_z) |
| 3 | `close_gripper` | 그리퍼 닫기 (파지) |
| 4 | `lift` | 들어올리기 (lift_z = approach_z + offset) |
| 5 | `transport` | place 위치 위로 수평 이동 (lift_z) |
| 6 | `place_descend` | place 위치로 하강 (place_z = grasp_z) |
| 7 | `open_gripper` | 그리퍼 열기 (해제) |
| 8 | `retract` | 들어올리기 (lift_z) |
| 9 | `home` | 홈 복귀 (YAML 'start' 관절값 직접 재생) |

### Z 좌표 자동 결정

```
approach_z = YAML waypoints['1']['ee_pose']['position']['z']  ≈ 0.372 m
grasp_z    = YAML waypoints['2']['ee_pose']['position']['z']  ≈ 0.345 m
place_z    = grasp_z  (동일)
lift_z     = approach_z + --lift-offset                       기본 +0.06 m
```

### 사용 예시

```bash
# 좌표 미지정 → YAML '2','4' 위치 자동 사용 (dry-run)
python3 motion2/scripts/jaewoo/run_pick_place.py

# 실제 실행 (YAML 기본 좌표)
python3 motion2/scripts/jaewoo/run_pick_place.py \
    --execute --confirm EXECUTE_PICK_PLACE

# 좌표 직접 지정 (dry-run)
python3 motion2/scripts/jaewoo/run_pick_place.py \
    --pick-x 0.496 --pick-y -0.113 \
    --place-x 0.321 --place-y -0.394

# 좌표 직접 지정 (실제 실행)
python3 motion2/scripts/jaewoo/run_pick_place.py \
    --pick-x 0.496 --pick-y -0.113 \
    --place-x 0.321 --place-y -0.394 \
    --execute --confirm EXECUTE_PICK_PLACE

# 홈 복귀 생략
python3 motion2/scripts/jaewoo/run_pick_place.py --no-home

# 이동 시간 늘리기 (기본 6s)
python3 motion2/scripts/jaewoo/run_pick_place.py --duration 8.0
```

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--pick-x`, `--pick-y` | YAML['2'] | 집을 위치 x,y [m] |
| `--place-x`, `--place-y` | YAML['4'] | 놓을 위치 x,y [m] |
| `--execute` | False | 실제 실행 (없으면 dry-run) |
| `--confirm TEXT` | `""` | `EXECUTE_PICK_PLACE` 입력 필요 |
| `--duration` | `6.0` s | 암 이동 시간 (최소 4.0) |
| `--home-duration` | `9.0` s | 홈 복귀 이동 시간 |
| `--lift-offset` | `0.06` m | approach_z 위로 추가 높이 |
| `--approach-z` | YAML['1'].z | 호버 높이 오버라이드 |
| `--grasp-z` | YAML['2'].z | 파지 높이 오버라이드 |
| `--no-home` | — | 9번 홈 복귀 생략 |
| `--no-step-prompts` | — | 단계별 타이핑 확인 생략 |
| `--max-joint-delta` | `0.35` rad | 최대 허용 관절 변위 |
| `--config` | `motion2/config/teach_pick_place_waypoints.yaml` | YAML 파일 경로 |

---

## 일반적인 실행 흐름

### 1단계: 로봇 bringup (별도 터미널)

```bash
ssh root@omy-SNPR44B1021.local
ros2 launch open_manipulator_bringup omy_f3m.launch.py
```

### 2단계: 현재 EE 위치 확인 (선택)

```bash
python3 motion2/scripts/jaewoo/show_ee_pose.py
```

### 3단계: Dry-run으로 계획 검증

```bash
# 기본 좌표로 전체 시퀀스 확인
python3 motion2/scripts/jaewoo/run_pick_place.py
```

출력에서 각 step의 계획 정보와 joint delta를 확인한다.  
모든 step이 `dry-run OK (command_sent=false)`로 나와야 실제 실행 가능.

### 4단계: 실제 실행

```bash
python3 motion2/scripts/jaewoo/run_pick_place.py \
    --execute --confirm EXECUTE_PICK_PLACE
```

각 단계 실행 전 step 이름을 직접 타이핑해야 진행된다 (이중 안전장치).

---

## 자주 발생하는 오류

| 오류 메시지 | 원인 | 해결책 |
|-------------|------|--------|
| `Refusing to execute. --confirm ... 필요` | `--confirm` 문자열 누락 또는 오타 | 정확한 CONFIRM_TEXT 확인 |
| `plan failed or plan-level guard violated` | MoveIt IK 실패 또는 delta 초과 | 목표 좌표 조정, `--max-joint-delta` 증가 |
| `JOINT5 CONSTRAINT NOT MET` | MoveIt이 joint5≈π/2를 만족하는 해를 못 찾음 | 목표 위치 조정 또는 `--no-constrain-joint5` |
| `target ... out of workspace` | 목표 좌표가 작업 공간 밖 | 좌표 수정 또는 `--x-max` 등으로 workspace 확장 |
| `Timed out waiting for /joint_states` | ROS2 연결 없음 | 로봇 bringup 확인 |
| `MoveGroup action server not available` | MoveIt 미실행 | 로봇 bringup 및 MoveIt 실행 확인 |
| `--duration must be >= 4.0 s` | duration이 너무 작음 | `--duration 5.0` 이상 사용 |

---

## 관련 파일 (이 폴더 밖)

| 파일 | 설명 |
|------|------|
| `motion2/scripts/real_moveit_common.py` | MoveIt goal 빌더, delta report, TF2 조회 공통 유틸 |
| `motion2/scripts/run_teach_pick_place.py` | 관절 직접 재생 방식 (MoveIt 불사용, wraparound 위험 없음) |
| `motion2/scripts/record_teach_waypoint.py` | 현재 로봇 자세를 YAML에 저장 |
| `motion2/config/teach_pick_place_waypoints.yaml` | waypoint 저장 파일 (Z 좌표, 홈 관절값 출처) |
| `motion2/docs/teach_pick_place_guide.md` | teach-and-replay 방식 전체 문서 |
