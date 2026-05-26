# OMY-F3M 실제 로봇 제어 스크립트 (jaewoo)

OMY-F3M 실제 로봇에서 Cartesian 좌표 기반 pick-and-place 및 자세 제어를 위한 스크립트 모음.

> 모든 스크립트는 **기본값이 dry-run** 이다. 실제 로봇을 움직이려면 반드시 `--execute --confirm <TEXT>` 를 붙여야 한다.

---

## 사전 조건

```bash
# 별도 터미널에서 로봇 bringup 먼저 실행
ssh root@omy-SNPR44B1021.local
ros2 launch open_manipulator_bringup omy_f3m.launch.py
```

---

## 파일 목록

| 파일 | 한 줄 설명 |
|------|-----------|
| `show_ee_pose.py` | 현재 EE 위치 읽어서 출력 |
| `run_ee_pose_move.py` | 단일 xyz 좌표로 암 이동 |
| `run_pick_place.py` | pick/place 좌표 받아 9단계 전체 실행 |
| `go_to_start.py` | YAML `start` 자세로 홈 복귀 |
| `go_to_init.py` | YAML `init` 자세로 관절 원점 복귀 |
| `toggle_gripper.py` | 그리퍼 토글 (열림↔닫힘) |

---

## 1. show_ee_pose.py

현재 EE(end-effector) 위치를 TF2로 읽어 출력한다.  
`run_ee_pose_move.py` 에 바로 붙여 쓸 수 있는 명령어도 함께 출력한다.

### 사용 예시

```bash
# 현재 위치 1회 읽기
python3 motion2/scripts/jaewoo/show_ee_pose.py

# 3회 평균 (로봇이 미세하게 진동 중일 때)
python3 motion2/scripts/jaewoo/show_ee_pose.py --samples 3
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--samples N` | `1` | N회 읽어 평균값 사용 |
| `--interval` | `0.3` s | 샘플 간 대기 시간 |
| `--base-frame` | `link0` | 기준 프레임 |
| `--ee-frame` | `link6` | EE 프레임 |
| `--timeout` | `5.0` s | TF2 대기 타임아웃 |

### 출력 예시

```
============================================================
[show-ee-pose] frame: link0 → link6
[show-ee-pose] position  : x=0.496000  y=-0.113000  z=0.345000  [m]
[show-ee-pose] quat_wxyz : w=0.506300  x=0.493600  y=0.493600  z=0.506300
============================================================

# run_ee_pose_move.py 에 사용할 인자 (복사해서 사용):
  --x 0.496000 --y -0.113000 --z 0.345000
```

---

## 2. run_ee_pose_move.py

`--x --y --z` 좌표를 주면 MoveIt IK로 계획 후 암을 이동시킨다.  
기본값으로 **수직 파지 orientation + joint5=π/2 constraint** 가 자동 적용된다.  
그리퍼는 움직이지 않는다.

### 사용 예시

```bash
# dry-run (기본: 수직 파지 방향 + joint5 constraint)
python3 motion2/scripts/jaewoo/run_ee_pose_move.py \
    --x 0.40 --y -0.11 --z 0.37

# 실제 실행
python3 motion2/scripts/jaewoo/run_ee_pose_move.py \
    --x 0.40 --y -0.11 --z 0.37 \
    --execute --confirm EXECUTE_EE_POSE_MOVE

# joint5 constraint 없이
python3 motion2/scripts/jaewoo/run_ee_pose_move.py \
    --x 0.40 --y -0.11 --z 0.37 \
    --no-constrain-joint5 \
    --execute --confirm EXECUTE_EE_POSE_MOVE

# 이동 시간 늘리기 (기본 6s)
python3 motion2/scripts/jaewoo/run_ee_pose_move.py \
    --x 0.40 --y -0.11 --z 0.37 \
    --duration 8.0 \
    --execute --confirm EXECUTE_EE_POSE_MOVE

# 현재 EE 방향 유지 (수직 파지 기본값 무시)
python3 motion2/scripts/jaewoo/run_ee_pose_move.py \
    --x 0.40 --y -0.11 --z 0.37 \
    --use-current-ori \
    --execute --confirm EXECUTE_EE_POSE_MOVE
```

### 옵션

**필수**

| 옵션 | 설명 |
|------|------|
| `--x` | 목표 EE x [m] (link0 기준, 앞쪽 방향) |
| `--y` | 목표 EE y [m] (음수 = 오른쪽) |
| `--z` | 목표 EE z [m] (위쪽 방향) |

**실행 제어**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--execute` | False | 실제 실행 (없으면 dry-run) |
| `--confirm` | `""` | `EXECUTE_EE_POSE_MOVE` 입력 필요 |
| `--duration` | `6.0` s | 이동 시간 (최소 4.0) |

**방향 지정 (셋 중 하나, 기본값: 수직 파지)**

| 옵션 | 설명 |
|------|------|
| `--rpy R P Y` | ZYX 오일러각 [rad] |
| `--quat W X Y Z` | 쿼터니언 wxyz |
| `--use-current-ori` | 현재 EE 방향 유지 |

**joint5 constraint**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--no-constrain-joint5` | — | joint5 constraint 비활성화 |
| `--joint5-tolerance` | `0.08` rad | constraint 허용 오차 |
| `--joint5-position` | `1.5708` rad | constraint 목표값 (π/2) |

**안전 guard**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--max-joint-delta` | `0.35` rad | 현재→목표 최대 관절 변위 |
| `--max-segment-delta` | `0.12` rad | plan 내 segment 간 최대 변위 |
| `--position-tolerance` | `0.01` m | IK 위치 허용 오차 |
| `--orientation-tolerance` | `0.15` rad | IK 방향 허용 오차 |

**작업 공간 (기본값)**

| 축 | 범위 |
|----|------|
| x | -0.10 ~ 0.55 m |
| y | -0.45 ~ 0.20 m |
| z | 0.10 ~ 0.50 m |

---

## 3. run_pick_place.py

pick 위치와 place 위치를 받아 **9단계 pick-and-place** 를 순서대로 실행한다.  
좌표를 주지 않으면 YAML `waypoints['2']`(pick), `waypoints['4']`(place) 위치를 자동 사용한다.

### 9단계 시퀀스

```
1. pre_grasp     pick 위 호버       (x1, y1, approach_z ≈ 0.372)
2. grasp         pick 위치 하강     (x1, y1, grasp_z    ≈ 0.345)
3. close_gripper 그리퍼 닫기        (1.05m, 물체에 막히면 stalled=OK)
4. lift          들어올리기         (x1, y1, lift_z = approach_z + 0.06)
5. transport     place 위로 이동    (x2, y2, lift_z)
6. place_descend place 위치 하강    (x2, y2, place_z = grasp_z)
7. open_gripper  그리퍼 열기        (0.0m)
8. retract       들어올리기         (x2, y2, lift_z)
9. home          홈 복귀            (YAML 'start' 관절값 직접 재생)
```

### 단계별 이동 시간 변경

이동 시간은 CLI 옵션이 아니라 **파일 상단 상수**로 고정되어 있다.  
변경하려면 `run_pick_place.py` 상단의 값을 직접 수정한다.

```python
DURATION_PRE_GRASP     = 6.0   # 1단계: 물체 위 호버
DURATION_GRASP         = 5.0   # 2단계: 하강
DURATION_LIFT          = 5.0   # 4단계: 들어올리기
DURATION_TRANSPORT     = 7.0   # 5단계: 수평 이동 (가장 긴 이동)
DURATION_PLACE_DESCEND = 5.0   # 6단계: 하강
DURATION_RETRACT       = 5.0   # 8단계: 들어올리기
DURATION_HOME          = 9.0   # 9단계: 홈 복귀
```

### 사용 예시

```bash
# YAML 기본 좌표로 dry-run
python3 motion2/scripts/jaewoo/run_pick_place.py

# YAML 기본 좌표로 실제 실행
python3 motion2/scripts/jaewoo/run_pick_place.py \
    --execute --confirm EXECUTE_PICK_PLACE

# 좌표 직접 지정 + 실제 실행
python3 motion2/scripts/jaewoo/run_pick_place.py \
    --pick-x 0.496 --pick-y -0.113 \
    --place-x 0.321 --place-y -0.394 \
    --execute --confirm EXECUTE_PICK_PLACE

# 홈 복귀 생략
python3 motion2/scripts/jaewoo/run_pick_place.py \
    --no-home \
    --execute --confirm EXECUTE_PICK_PLACE

# 단계별 타이핑 확인 없이 연속 실행 (주의)
python3 motion2/scripts/jaewoo/run_pick_place.py \
    --no-step-prompts \
    --execute --confirm EXECUTE_PICK_PLACE
```

### 옵션

**좌표 (생략 시 YAML 자동)**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--pick-x` | YAML `['2'].x` | 집을 위치 x [m] |
| `--pick-y` | YAML `['2'].y` | 집을 위치 y [m] |
| `--place-x` | YAML `['4'].x` | 놓을 위치 x [m] |
| `--place-y` | YAML `['4'].y` | 놓을 위치 y [m] |

**Z 좌표 (생략 시 YAML 자동)**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--approach-z` | YAML `['1'].z` ≈ 0.372 | 호버 높이 [m] |
| `--grasp-z` | YAML `['2'].z` ≈ 0.345 | 파지 높이 [m] |
| `--lift-offset` | `0.06` m | approach_z 위로 추가 높이 |

**실행 제어**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--execute` | False | 실제 실행 |
| `--confirm` | `""` | `EXECUTE_PICK_PLACE` 입력 필요 |
| `--no-home` | — | 9번 홈 복귀 생략 |
| `--no-step-prompts` | — | 단계별 타이핑 확인 생략 |

> 이동 시간은 파일 상단 `DURATION_*` 상수로 변경한다.

**안전 guard**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--max-joint-delta` | `0.35` rad | 최대 허용 관절 변위 |
| `--max-segment-delta` | `0.12` rad | plan 내 segment 간 최대 변위 |
| `--no-constrain-joint5` | — | joint5 constraint 비활성화 |
| `--joint5-tolerance` | `0.08` rad | joint5 constraint 허용 오차 |

---

## 4. go_to_start.py

YAML `waypoints['start']` 관절값으로 홈 자세로 복귀한다.  
MoveIt 없이 FollowJointTrajectory 직접 전송. 이상 상황 복귀용.

**start 자세**: 팔이 위쪽으로 접힌 안전 대기 자세, EE x≈-0.05 z≈0.39

### 사용 예시

```bash
# dry-run (계획 확인)
python3 motion2/scripts/jaewoo/go_to_start.py

# 실제 실행
python3 motion2/scripts/jaewoo/go_to_start.py \
    --execute --confirm GO_TO_START

# 느리게 이동 (권장)
python3 motion2/scripts/jaewoo/go_to_start.py \
    --duration 12.0 \
    --execute --confirm GO_TO_START
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--execute` | False | 실제 실행 |
| `--confirm` | `""` | `GO_TO_START` 입력 필요 |
| `--duration` | `9.0` s | 이동 시간 (최소 4.0) |
| `--max-joint-delta` | `6.30` rad | 최대 관절 변위 (≈2π, 어느 자세에서든 가능) |
| `--config` | YAML 경로 | waypoint 파일 경로 |
| `--arm-action` | `/arm_controller/follow_joint_trajectory` | ROS2 action 이름 |

---

## 5. go_to_init.py

YAML `waypoints['init']` 관절값으로 관절 원점(모든 관절=0)으로 복귀한다.  
MoveIt 없이 FollowJointTrajectory 직접 전송.

**init 자세**: 모든 관절 ≈ 0, 팔이 정면 수직으로 완전히 펴진 상태, EE z≈0.75  
> **주의**: 팔이 위로 수직으로 세워지므로 위쪽 공간에 장애물이 없는지 반드시 확인

### 사용 예시

```bash
# dry-run (계획 확인 + 위쪽 공간 확인)
python3 motion2/scripts/jaewoo/go_to_init.py

# 실제 실행
python3 motion2/scripts/jaewoo/go_to_init.py \
    --execute --confirm GO_TO_INIT

# 충분한 시간 확보 (큰 이동이 예상될 때)
python3 motion2/scripts/jaewoo/go_to_init.py \
    --duration 15.0 \
    --execute --confirm GO_TO_INIT
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--execute` | False | 실제 실행 |
| `--confirm` | `""` | `GO_TO_INIT` 입력 필요 |
| `--duration` | `12.0` s | 이동 시간 (최소 4.0, 기본값이 긴 이유: 큰 이동 예상) |
| `--max-joint-delta` | `6.30` rad | 최대 관절 변위 (≈2π, 어느 자세에서든 가능) |
| `--config` | YAML 경로 | waypoint 파일 경로 |
| `--arm-action` | `/arm_controller/follow_joint_trajectory` | ROS2 action 이름 |

---

## 6. toggle_gripper.py

현재 그리퍼 위치를 읽어 열려있으면 닫고, 닫혀있으면 연다.

- 현재 위치 < 0.5 → **열림** 판단 → **닫기** (1.05m)
- 현재 위치 ≥ 0.5 → **닫힘** 판단 → **열기** (0.0m)

### 사용 예시

```bash
# dry-run (현재 상태 및 예정 동작 확인)
python3 motion2/scripts/jaewoo/toggle_gripper.py

# 실제 실행
python3 motion2/scripts/jaewoo/toggle_gripper.py \
    --execute --confirm TOGGLE_GRIPPER

# 닫기 목표값 변경
python3 motion2/scripts/jaewoo/toggle_gripper.py \
    --close-pos 0.8 \
    --execute --confirm TOGGLE_GRIPPER
```

### 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--execute` | False | 실제 실행 |
| `--confirm` | `""` | `TOGGLE_GRIPPER` 입력 필요 |
| `--close-pos` | `1.05` m | 닫을 때 목표 위치 |
| `--open-pos` | `0.0` m | 열 때 목표 위치 |
| `--threshold` | `0.5` m | 열림/닫힘 판단 기준 |
| `--max-effort` | `0.0` | 그리퍼 최대 토크 (0=컨트롤러 기본값) |
| `--gripper-action` | `/gripper_controller/gripper_cmd` | ROS2 action 이름 |

---

## 좌표계 참고

```
(로봇을 위에서 내려다본 top-view)

         +x (앞쪽, 최대 ≈ 0.50m)
          ↑
          │
[로봇 베이스] ──────────────→ -y (오른쪽)
 (link0)
```

| 축 | 의미 | 실용 범위 |
|----|------|-----------|
| +x | 로봇 앞쪽 | 0.0 ~ 0.50 m |
| -y | 로봇 오른쪽 | 0.0 ~ -0.40 m |
| +z | 위쪽 | 0.34 ~ 0.75 m |

**YAML 주요 위치**

| waypoint | x | y | z | 용도 |
|----------|---|---|---|------|
| `start` | -0.05 | -0.11 | 0.39 | 홈 대기 자세 |
| `'1'` | 0.496 | -0.113 | 0.372 | pick 호버 높이 기준 |
| `'2'` | 0.496 | -0.113 | 0.345 | pick 파지 높이 기준 |
| `'4'` | 0.322 | -0.394 | 0.344 | place 위치 기준 |
| `init` | ≈0.0 | -0.113 | 0.753 | 관절 원점 (팔 수직) |

---

## 오류 빠른 참고

| 메시지 | 원인 | 해결 |
|--------|------|------|
| `Refusing. --confirm ... 필요` | confirm 문자열 누락/오타 | 정확한 텍스트 확인 |
| `plan failed or plan-level guard violated` | MoveIt IK 실패 또는 delta 초과 | 좌표 조정 또는 `--max-joint-delta` 증가 |
| `JOINT5 CONSTRAINT NOT MET` | 해당 위치에서 수직 파지 IK 해 없음 | `--no-constrain-joint5` 또는 좌표 조정 |
| `target ... out of workspace` | 좌표가 작업 공간 밖 | 좌표 수정 또는 `--x-max` 등으로 workspace 확장 |
| `Timed out waiting for /joint_states` | ROS2 연결 없음 | 로봇 bringup 확인 |
| `MoveGroup action server not available` | MoveIt 미실행 | 로봇 bringup 확인 |
| `--duration >= 4.0 s 필요` | duration 너무 작음 | `--duration 5.0` 이상 |
