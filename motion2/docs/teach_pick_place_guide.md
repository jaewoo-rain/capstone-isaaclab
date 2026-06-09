# Teach-and-Replay Pick/Place 가이드 (OMY-F3M)

## 개요

이 시스템은 MoveIt pose goal 대신 **사람이 직접 가르친(teach) 관절 자세를 그대로 재생(replay)** 하는 방식으로 pick-and-place를 수행한다.

### MoveIt 대신 teach-and-replay를 쓰는 이유

MoveIt은 end-effector 목표 자세를 받아 역기구학(IK)으로 관절 값을 계산하고 trajectory를 생성한다.
이 과정에서 관절이 ±2π(±360°) 범위를 크게 돌아가는 **wraparound trajectory**가 생성될 수 있어 실제 로봇에서 위험하다.

teach-and-replay는 이 문제를 다음과 같이 우회한다:
1. 사람이 로봇을 안전한 자세로 이동시킨다
2. 그 순간의 관절 값(joint angles)을 YAML에 그대로 저장한다
3. 재생 시에는 저장된 관절 값을 **직접** `FollowJointTrajectory`로 전송한다

---

## 관련 파일 목록

이 두 파일을 이해하려면 다음 파일들을 함께 봐야 한다.

| 파일 | 역할 |
|------|------|
| [scripts/run_teach_pick_place.py](../scripts/run_teach_pick_place.py) | 저장된 waypoint를 순서대로 실행하는 메인 스크립트 |
| [config/teach_pick_place_waypoints.yaml](../config/teach_pick_place_waypoints.yaml) | waypoint 저장 파일 (이 가이드의 주요 대상) |
| [scripts/record_teach_waypoint.py](../scripts/record_teach_waypoint.py) | 현재 로봇 자세를 YAML에 기록하는 스크립트 |
| [scripts/joint_space_smoke_test.py](../scripts/joint_space_smoke_test.py) | 단일 관절 소량 이동 검증용 — teach replay 전 선행 테스트 |
| [scripts/gripper_smoke_test.py](../scripts/gripper_smoke_test.py) | 그리퍼 소량 이동 검증용 — teach replay 전 선행 테스트 |
| [config/joint_space_smoke_waypoints.yaml](../config/joint_space_smoke_waypoints.yaml) | smoke test용 waypoint 설정 |
| `omy_f3m.launch.py` | 실제 로봇 ROS2 bringup launch 파일 (실제 로봇 시스템에 존재) |

### 외부 의존성 (ROS2 패키지)

| 패키지 | 용도 |
|--------|------|
| `sensor_msgs/JointState` | `/joint_states` 토픽으로 현재 관절 값 수신 |
| `control_msgs/action/FollowJointTrajectory` | 암 6축 관절 이동 명령 |
| `control_msgs/action/GripperCommand` | 그리퍼 열기/닫기 명령 |
| `tf2_ros` | TF2 변환으로 end-effector 위치 조회 (`record_teach_waypoint.py`에서 사용) |

---

## 시스템 구조

```
record_teach_waypoint.py          run_teach_pick_place.py
        │                                  │
        │ /joint_states                    │ /joint_states (현재 자세 검증용)
        │ TF2 (ee_pose)                    │
        ▼                                  ▼
teach_pick_place_waypoints.yaml ──────► _build_steps()
                                          │ (safety guard: max_joint_delta)
                                          ▼
                                  _print_steps() ─► dry-run 출력
                                          │
                              (--execute --confirm ...)
                                          │
                               ┌──────────┴──────────┐
                               ▼                     ▼
                    _send_arm_goal()        _send_gripper_goal()
                    FollowJointTrajectory   GripperCommand
                    /arm_controller/...    /gripper_controller/...
```

---

## YAML 파일 구조 설명

### 상단 메타데이터

```yaml
verified: false       # true로 바꾸기 전까지는 실험적 파일
bringup: omy_f3m.launch.py
mode: teach_waypoint_replay
```

### controller 섹션

ROS2 action 서버 이름을 지정한다.
`run_teach_pick_place.py`의 `--arm-action`, `--gripper-action` CLI 인자로 오버라이드 가능.

```yaml
controller:
  arm_action: /arm_controller/follow_joint_trajectory
  gripper_action: /gripper_controller/gripper_cmd
```

### safety 섹션

실행 전 검증에 사용되는 파라미터들. CLI 인자로 오버라이드 가능.

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `max_joint_delta_per_step` | 0.35 rad | 연속 waypoint 간 허용 최대 관절 변위 |
| `min/max_gripper_position` | 0.0 / 1.12 | 그리퍼 위치 범위 가드 (OMY-F3M 물리 한계) |
| `default_arm_duration` | 5.0 s | 각 암 이동에 허용하는 시간 (최소 4.0 강제) |
| `default_gripper_max_effort` | 0.0 | 그리퍼 최대 토크 (0 = 컨트롤러 기본값) |

### sequence 섹션

실행할 waypoint 이름 목록. `close_gripper`와 `open_gripper`는 **특수 키워드**로,
waypoints 딕셔너리에서 찾지 않고 `gripper_targets` 값으로 처리된다.

```yaml
sequence:
- pre_grasp      # waypoints['pre_grasp'] → ArmStep
- grasp          # waypoints['grasp']     → ArmStep
- close_gripper  # 특수 키워드            → GripperStep (gripper_targets.close_gripper)
- lift           # waypoints['lift']      → ArmStep
- place          # waypoints['place']     → ArmStep
- open_gripper   # 특수 키워드            → GripperStep (gripper_targets.open_gripper)
- retract        # waypoints['retract']   → ArmStep
```

### waypoints 섹션

각 waypoint 구조:

```yaml
waypoints:
  pre_grasp:
    arm:
      joint1: 0.000   # 베이스 회전 [rad]
      joint2: 0.709   # 어깨 [rad]
      joint3: 0.813   # 팔꿈치 [rad]
      joint4: 0.023   # 손목 pitch [rad]
      joint5: 1.571   # 손목 roll [rad]
      joint6: 0.000   # 손목 yaw [rad]
    gripper:
      rh_r1_joint: 0.0  # 그리퍼 위치 [m]
    recorded_at_utc: '2026-05-22T11:13:26+00:00'  # 기록 시각 (감사 로그)
    ee_pose:              # 참고용 EE 자세 (실행에 영향 없음)
      frame_id: link0     # 기준 프레임: 로봇 베이스
      link_name: link6    # EE 프레임: 6번 링크(손목)
      position: {x, y, z} # [m]
      quat_wxyz: {w, x, y, z}
```

### 현재 저장된 waypoint 요약

현재 파일에는 시퀀스 이름(`pre_grasp` 등)이 아닌 숫자(`'1'`, `'2'` 등)로 저장되어 있다.
**sequence 섹션의 이름과 waypoints 키가 일치해야 한다** — 불일치 시 실행 전 ValueError 발생.

| 키 | EE 위치(x,y,z) m | 그리퍼 | 의미 |
|----|------------------|--------|------|
| `start` | (-0.05, -0.11, 0.39) | 0.0 열림 | 홈/대기 자세 |
| `'1'` | (0.50, -0.11, 0.37) | 0.0 열림 | 테이블 앞 접근 |
| `'2'` | (0.50, -0.11, 0.34) | 0.0 열림 | 물체 위로 하강 |
| `'3'` | (0.32, -0.39, 0.39) | 1.12 닫힘 | 파지 후 오른쪽 이동 |
| `'4'` | (0.32, -0.39, 0.34) | 1.12 닫힘 | 내려놓기 위치 하강 |
| `init` | (0.0, -0.11, 0.75) | 1.12 닫힘 | 관절 원점 (수직) |

> **주의**: 현재 `sequence`의 이름(pre_grasp, grasp 등)과 waypoints 키(`start`, `'1'` 등)가 일치하지 않는다. 실제 사용 전 sequence를 수정하거나 waypoint를 올바른 이름으로 재기록해야 한다.

---

## 사용 방법

### 1단계: 전제 조건

```bash
# 실제 로봇 ROS2 bringup (별도 터미널)
ssh root@omy-SNPR44B1021.local
ros2 launch open_manipulator_bringup omy_f3m.launch.py
```

### 2단계: Waypoint 기록

로봇을 원하는 자세로 이동시킨 후 각 단계를 기록한다.

```bash
# 파지 준비 자세에서 기록
python3 motion2/scripts/record_teach_waypoint.py --name pre_grasp --overwrite

# 파지 자세에서 기록
python3 motion2/scripts/record_teach_waypoint.py --name grasp --overwrite

# 들어올린 자세에서 기록
python3 motion2/scripts/record_teach_waypoint.py --name lift --overwrite

# 내려놓을 자세에서 기록
python3 motion2/scripts/record_teach_waypoint.py --name place --overwrite

# 복귀 자세에서 기록
python3 motion2/scripts/record_teach_waypoint.py --name retract --overwrite
```

각 명령은 `/joint_states`와 TF2에서 현재 자세를 읽어 YAML에 저장한다. 실제 로봇 명령은 전송하지 않는다.

### 3단계: Dry-run 검증

```bash
python3 motion2/scripts/run_teach_pick_place.py
```

출력 예시:
```
[teach-replay] execute: False
[teach-replay] arm action: /arm_controller/follow_joint_trajectory
[teach-replay] arm duration: 5.00s
[teach-replay] step 01 arm pre_grasp: max_delta=0.234567 joint1=0.000000, ...
[teach-replay] step 02 arm grasp:     max_delta=0.089123 joint1=0.000000, ...
[teach-replay] step 03 gripper close_gripper: position=0.600000
...
[teach-replay] dry-run complete; command_sent=false
```

`max_delta`가 `max_joint_delta_per_step`(0.35 rad)을 넘으면 에러가 발생한다.

### 4단계: 실제 실행

```bash
python3 motion2/scripts/run_teach_pick_place.py \
  --execute \
  --confirm EXECUTE_TEACH_PICK_PLACE
```

각 step 실행 전에 step 이름을 타이핑하도록 요구한다 (이중 안전장치).
`--no-step-prompts`를 추가하면 이 확인을 건너뛸 수 있으나 권장하지 않는다.

### 유용한 옵션

```bash
# 특정 step만 선택 실행
python3 motion2/scripts/run_teach_pick_place.py --sequence pre_grasp,grasp,close_gripper

# 이동 시간 늘리기 (기본 5s, 최소 4s)
python3 motion2/scripts/run_teach_pick_place.py --duration 8.0

# 다른 waypoint 파일 사용
python3 motion2/scripts/run_teach_pick_place.py --config motion2/config/my_waypoints.yaml

# safety limit 조정 (기본 0.35 rad)
python3 motion2/scripts/run_teach_pick_place.py --max-joint-delta 0.5
```

---

## Safety 메커니즘 상세

### 1. 이중 확인 (CONFIRM_TEXT)

실행 시 두 가지를 동시에 제공해야 한다:
- `--execute` 플래그
- `--confirm EXECUTE_TEACH_PICK_PLACE` (정확한 문자열)

### 2. Joint Delta Guard

`_build_steps()`에서 각 연속 waypoint 간 관절 변위를 계산한다:
- 첫 번째 waypoint의 기준: **현재 로봇 자세** (`/joint_states`에서 읽음)
- 이후 waypoint의 기준: 이전 waypoint 목표값
- 어느 관절이든 변위가 `max_joint_delta_per_step`를 초과하면 **실행 전** ValueError

이 검사는 dry-run에서도 수행되므로 실제 실행 전에 반드시 dry-run으로 검증한다.

### 3. 단계별 타이핑 확인

실행 모드에서 각 step마다 step 이름을 직접 타이핑해야 한다.
연속 실행 중 이상이 발견되면 타이핑하지 않고 Ctrl+C로 중단할 수 있다.

### 4. 그리퍼 위치 범위 가드

그리퍼 목표값이 `[min_gripper_position, max_gripper_position]` 범위를 벗어나면 거부.
OMY-F3M 물리적 한계: 0.0 ~ 1.12 m

---

## 코드 흐름 요약 (`run_teach_pick_place.py`)

```
main()
 ├── argparse: CLI 인자 파싱
 ├── CONFIRM_TEXT 검증 (--execute 시 필수)
 ├── _load_yaml(): YAML 파일 로드
 ├── safety 파라미터 결정 (CLI > YAML > 기본값)
 ├── rclpy.init() + Node 생성
 ├── _joint_state_once(): 현재 로봇 관절 값 취득
 ├── _build_steps(): step 리스트 생성 + 모든 safety guard 검증
 ├── _print_steps(): 전체 계획 출력
 │
 ├── [dry-run] → 종료 (command_sent=false)
 │
 └── [--execute]
      ├── action server 대기 (10초 타임아웃)
      └── for step in steps:
           ├── _confirm_step(): 사용자 타이핑 확인
           ├── ArmStep → _send_arm_goal() → FollowJointTrajectory
           └── GripperStep → _send_gripper_goal() → GripperCommand
```

---

## 자주 발생하는 오류

| 오류 메시지 | 원인 | 해결책 |
|-------------|------|--------|
| `Refusing to execute. Re-run with --confirm ...` | `--confirm` 누락 | `--confirm EXECUTE_TEACH_PICK_PLACE` 추가 |
| `missing arm waypoint 'pre_grasp'; available=[...]` | sequence 이름과 waypoint 키 불일치 | waypoint를 올바른 이름으로 재기록 |
| `max joint delta X.XX rad at jointN exceeds limit` | waypoint 간 이동이 너무 큼 | `--max-joint-delta` 증가 또는 중간 waypoint 추가 |
| `Timed out waiting for /joint_states` | ROS2 연결 없음 | 로봇 bringup 확인, `--joint-state-timeout` 증가 |
| `Action server not available: /arm_controller/...` | 컨트롤러 미실행 | 로봇 bringup 확인 |
| `--duration must be >= 4.0 s` | duration이 너무 작음 | `--duration 5.0` 이상 사용 |
