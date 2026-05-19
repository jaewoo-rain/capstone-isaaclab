# OMY-F3M 연구실 실험 가이드: 먼저 성공시키는 teach-and-replay MVP

이 문서는 로봇 sim-to-real 초보자가 연구실에서 `motion2` 프로젝트를
빠르게 진전시키기 위한 실험 가이드입니다.

처음 목표는 완전 자율 sim-to-real이 아닙니다.

처음 목표는 이것입니다.

```text
정해진 위치에 있는 블록을
OMY-F3M이 실제로 잡고,
들어 올리고,
정해진 위치로 옮기는 것
```

이것을 먼저 성공시키면 프로젝트가 살아납니다. 그 다음에 카메라, YOLO,
RL policy, RealAdapter를 하나씩 붙이면 됩니다.

## 지금 방향을 바꾸는 이유

처음에는 IsaacLab에서 성공한 전체 흐름을 실제 로봇에 바로 옮기고 싶을 수
있습니다.

```text
IsaacLab sim 성공
-> RealAdapter 구현
-> MoveIt IK로 실제 로봇 이동
-> D405 카메라
-> YOLO
-> PPO/RL policy
-> 전체 pick-and-place
```

하지만 현재 연구실에서 확인된 가장 큰 문제는 이것입니다.

```text
MoveIt pose goal이 작은 이동에도 관절을 크게 돌리는 trajectory를 만들 수 있다.
```

즉, end-effector를 1cm만 움직이고 싶어도 `joint1`, `joint4`, `joint6` 같은
관절이 크게 돌아가려는 계획이 나올 수 있습니다. 이 상태에서 카메라나 RL까지
붙이면 원인을 분리하기가 너무 어렵습니다.

그래서 지금은 더 쉬운 길로 갑니다.

```text
사람이 안전한 자세를 만든다
-> 그때의 joint 값을 저장한다
-> 저장한 joint 값을 천천히 다시 재생한다
-> 실제 grasp/lift/place를 먼저 성공시킨다
```

이 방식을 `teach-and-replay`라고 부르겠습니다.

## 아주 쉽게 말하면

로봇에게 처음부터 이렇게 시키는 것은 어렵습니다.

```text
카메라로 블록을 찾아서 알아서 잡아.
```

대신 처음에는 이렇게 시킵니다.

```text
내가 안전한 자세들을 하나씩 만들어줄게.
너는 그 자세들을 기억했다가 같은 순서로 다시 따라 해.
```

이게 훨씬 빠르게 실제 결과를 만들 수 있는 방법입니다.

## 절대 처음에 하지 말 것

아래는 아직 하지 않는 것이 좋습니다.

- 실제 로봇에서 `run_chain_once()` 실행
- 실제 로봇에서 full `RealAdapter` 연결
- D405 카메라 결과로 바로 로봇 이동
- YOLO detection 결과로 바로 로봇 이동
- PPO/RL policy action으로 바로 로봇 이동
- MoveIt absolute pose goal로 전체 pick-and-place 실행
- 모터 torque가 켜진 상태에서 손으로 로봇을 억지로 밀기
- `omy_ai.launch.py`로 direct MoveIt/controller 테스트하기

지금 목표는 멋진 자동화가 아니라, 실제 로봇이 안전하게 블록을 잡고 옮기는
최소 성공을 만드는 것입니다.

## 오늘 연구실에서의 전체 흐름

오늘 할 일은 이 순서입니다.

```text
1. 로봇 bringup 확인
2. controller/action/joint_states 확인
3. 아주 작은 smoke test로 로봇이 명령을 받는지 확인
4. pre_grasp, grasp, lift, place, retract 자세 만들기
5. 각 자세의 joint 값을 waypoint로 저장
6. replay dry-run
7. arm-only replay
8. gripper 포함 grasp/lift replay
9. 전체 teach-and-replay MVP 실행
```

중간에 하나라도 이상하면 다음 단계로 넘어가지 않습니다.

## 용어 정리

`waypoint`:

```text
로봇이 기억해야 하는 자세 하나
```

예를 들어 `pre_grasp`는 블록 바로 위에 있는 자세입니다.

`teach`:

```text
로봇을 원하는 자세로 보내고, 그때의 joint 값을 저장하는 것
```

`replay`:

```text
저장한 waypoint들을 순서대로 다시 실행하는 것
```

`dry-run`:

```text
실제로 로봇을 움직이지 않고, 어떤 명령을 보낼지만 출력하는 것
```

`smoke test`:

```text
아주 작은 명령으로 로봇 command path가 살아있는지 확인하는 테스트
```

## 터미널 준비

연구실에서는 터미널을 3개 쓰는 것이 좋습니다.

### Terminal A: 로봇 bringup

```bash
ssh root@omy-SNPR44B1021.local
cd /data/docker/open_manipulator
./docker/container.sh enter
ros2 launch open_manipulator_bringup omy_f3m.launch.py
```

중요:

```text
direct robot test에는 omy_ai.launch.py를 쓰지 않습니다.
```

`omy_ai.launch.py`는 leader/follower teleoperation 쪽이라, 지금 하는 직접
controller/MoveIt 테스트와 헷갈릴 수 있습니다.

### Terminal B: MoveIt/RViz

```bash
ssh root@omy-SNPR44B1021.local
cd /data/docker/open_manipulator
./docker/container.sh enter
ros2 launch open_manipulator_moveit_config omy_f3m_moveit.launch.py start_rviz:=true
```

RViz는 로봇을 원하는 자세 근처로 조심스럽게 보내는 데 사용할 수 있습니다.

### Terminal C: 명령 실행

```bash
ssh root@omy-SNPR44B1021.local
cd /data/docker/open_manipulator
./docker/container.sh enter
cd /root/ros2_ws/src/open_manipulator
```

이 터미널에서 `motion2/scripts/...` 명령을 실행합니다.

## 1단계: 로봇 상태 확인

Terminal C에서 실행합니다.

```bash
ros2 control list_controllers
ros2 action list
ros2 topic echo /joint_states --once
```

성공 기준:

- `joint_state_broadcaster`가 active
- `arm_controller`가 active
- `gripper_controller`가 active
- `/joint_states`에 `joint1`부터 `joint6`, `rh_r1_joint`가 보임
- action list에 `/arm_controller/follow_joint_trajectory`가 보임
- action list에 `/gripper_controller/gripper_cmd`가 보임

실패하면:

- bringup이 제대로 되었는지 확인
- 다른 launch를 켠 것은 아닌지 확인
- Docker container 안에서 실행 중인지 확인
- ROS domain이나 namespace가 다른지 확인

이 단계가 안 되면 로봇을 움직이지 않습니다.

## 2단계: 아주 작은 smoke test

먼저 dry-run입니다.

```bash
python3 motion2/scripts/run_smoke_sequence.py --name full_basic_smoke
```

이 명령은 실제로 움직이지 않고, 어떤 작은 명령을 보낼지 출력합니다.

문제가 없으면 실제 실행합니다.

```bash
python3 motion2/scripts/run_smoke_sequence.py --name full_basic_smoke --execute
```

성공 기준:

- `joint6`가 아주 조금 움직였다가 돌아옴
- `joint5`가 아주 조금 움직였다가 돌아옴
- gripper가 아주 조금 닫혔다가 열림
- `/joint_states` 값이 변함
- 로봇이 예상 밖으로 크게 움직이지 않음

실패하면:

- 여기서 멈춥니다.
- teach waypoint로 넘어가지 않습니다.
- controller, action, motor torque, speed scaling을 확인합니다.

## 3단계: waypoint로 쓸 자세 이해하기

최소 waypoint는 5개입니다.

```text
pre_grasp : 블록 위, 아직 내려가기 전
grasp     : 블록을 잡을 높이
lift      : 블록을 잡고 들어 올린 높이
place     : 블록을 놓을 위치
retract   : 작업 후 빠지는 안전 위치
```

gripper 동작은 waypoint가 아니라 sequence 중간에 들어갑니다.

```text
pre_grasp
-> grasp
-> close_gripper
-> lift
-> place
-> open_gripper
-> retract
```

## 4단계: 로봇을 waypoint 자세로 보내는 방법

여기서 중요한 점:

```text
손으로 억지로 로봇을 미는 것이 아닙니다.
```

로봇이 teach/gravity/freedrive 모드를 안전하게 지원하는 경우가 아니라면,
팔을 잡고 밀지 않습니다.

추천 방법은 이 순서입니다.

### 방법 A: RViz/MoveIt으로 조심스럽게 이동

RViz에서 interactive marker나 planning 기능으로 원하는 자세 근처까지
보냅니다.

주의:

- 한 번에 크게 움직이지 않기
- plan을 보고 이상하면 execute하지 않기
- 로봇 주변을 비우기
- 낮은 z로 바로 내려가지 않기

### 방법 B: 작은 joint 명령으로 조금씩 조정

이미 smoke test가 되는 joint-space 명령을 이용해서 자세를 조금씩 맞춥니다.

예:

```bash
python3 motion2/scripts/joint_space_smoke_test.py \
  --joint joint6 \
  --delta 0.01 \
  --execute \
  --confirm EXECUTE_JOINT_SPACE_SMOKE_TEST
```

이 방법은 느리지만, pose goal보다 예측하기 쉽습니다.

### 방법 C: teach/gravity mode

OMY-F3M 설정에서 안전한 teach/gravity/freedrive 모드가 확인된 경우에만
사용합니다.

확실하지 않으면 쓰지 않습니다.

## 4.5단계: teleoperation으로 움직여서 저장해도 되는가?

가능합니다. 오히려 지금 상황에서는 좋은 방법입니다.

다만 지금 바로 목표로 삼을 것은 엄밀한 의미의 “모방학습 policy 학습”이
아닙니다. 먼저 목표로 삼을 것은 이것입니다.

```text
teleoperation으로 사람이 로봇을 움직인다
-> 중요한 순간의 joint 값을 저장한다
-> 저장한 waypoint를 다시 replay한다
```

즉, 지금은 “학습”보다 “시연 기록 후 재생”에 가깝습니다.

### 왜 teleoperation이 좋은가?

MoveIt pose goal은 작은 EE 이동에도 관절이 크게 돌아가는 계획을 만들 수
있었습니다. 반면 teleoperation으로 사람이 직접 안전한 자세를 만들어두고
그때의 joint 값을 저장하면, 우선 원하는 실제 동작을 더 빠르게 만들 수
있습니다.

쉽게 말하면:

```text
MoveIt에게 "이 좌표로 알아서 가"라고 시키는 대신,
사람이 "이 자세가 안전하고 좋다"를 직접 보여준다.
```

### teleoperation을 쓸 때 중요한 주의점

teleoperation launch와 direct replay launch를 섞지 않는 것이 좋습니다.

추천 흐름:

```text
1. teleoperation 모드에서 로봇을 움직여 자세를 만든다
2. 각 자세에서 /joint_states를 저장한다
3. teleoperation을 종료한다
4. regular bringup인 omy_f3m.launch.py로 다시 실행한다
5. run_teach_pick_place.py로 dry-run/replay한다
```

즉, teleoperation 중에 바로 `run_teach_pick_place.py --execute`를 섞어서
실행하지 않습니다. controller remap이나 leader/follower setup 때문에
명령 경로가 헷갈릴 수 있습니다.

### teleoperation으로 저장할 자세

teleoperation으로 아래 자세를 하나씩 만들어 저장합니다.

```text
pre_grasp : 블록 위, 아직 내려가기 전
grasp     : 블록을 잡을 높이
lift      : 블록을 잡고 들어 올린 높이
place     : 블록을 놓을 위치
retract   : 작업 후 빠지는 안전 위치
```

각 자세에서 저장 명령은 같습니다.

```bash
python3 motion2/scripts/record_teach_waypoint.py --name pre_grasp --overwrite
```

이후 `grasp`, `lift`, `place`, `retract`도 같은 방식으로 저장합니다.

### trajectory 전체를 저장할 수도 있나?

가능합니다. teleoperation 중 모든 `/joint_states`를 시간 순서대로 저장하면
더 자연스러운 궤적 replay가 가능합니다.

하지만 지금 프로젝트를 빠르게 진전시키려면 먼저 waypoint 방식이 좋습니다.

우선순위:

```text
1순위: 중요한 자세 몇 개를 waypoint로 저장하고 replay
2순위: 필요하면 중간 waypoint 추가
3순위: 시간이 남으면 teleop 전체 trajectory 기록/replay
4순위: 나중에 진짜 imitation learning policy 학습
```

발표에서는 이렇게 말하면 됩니다.

```text
실제 로봇에서는 MoveIt pose goal의 joint wraparound 문제가 있어, 우선
teleoperation 기반 demonstration으로 주요 자세를 만들고 joint-space
teach-and-replay 방식으로 pick-and-place MVP를 검증하는 방향으로 전환했다.
```

## 5단계: waypoint 저장

로봇이 원하는 자세에 도착하면 그때의 joint 값을 저장합니다.

```bash
python3 motion2/scripts/record_teach_waypoint.py --name pre_grasp --overwrite
```

그 다음 같은 방식으로 저장합니다.

```bash
python3 motion2/scripts/record_teach_waypoint.py --name grasp --overwrite
python3 motion2/scripts/record_teach_waypoint.py --name lift --overwrite
python3 motion2/scripts/record_teach_waypoint.py --name place --overwrite
python3 motion2/scripts/record_teach_waypoint.py --name retract --overwrite
```

저장되는 파일:

```text
motion2/config/teach_pick_place_waypoints.yaml
```

성공 기준:

- 명령이 `command_sent=false`로 끝남
- YAML 파일에 waypoint가 추가됨
- 각 waypoint에 `joint1`부터 `joint6` 값이 들어 있음
- gripper joint `rh_r1_joint` 값도 기록됨

## 6단계: replay dry-run

이제 저장된 waypoint를 실제로 움직이기 전에 dry-run합니다.

```bash
python3 motion2/scripts/run_teach_pick_place.py
```

성공 기준:

- 각 단계가 출력됨
- `pre_grasp`, `grasp`, `lift`, `place`, `retract`가 모두 있음
- 각 단계의 max joint delta가 너무 크지 않음
- 마지막에 `command_sent=false`가 출력됨

실패하면:

- waypoint 이름이 빠졌는지 확인
- 현재 로봇 자세가 첫 waypoint와 너무 먼지 확인
- waypoint 사이 joint 변화가 너무 큰지 확인
- 필요하면 중간 waypoint를 추가

## 7단계: arm-only 부분 replay

처음부터 gripper까지 포함하지 말고, arm만 움직여봅니다.

Dry-run:

```bash
python3 motion2/scripts/run_teach_pick_place.py \
  --sequence pre_grasp,grasp,lift,retract \
  --duration 8.0
```

실행:

```bash
python3 motion2/scripts/run_teach_pick_place.py \
  --sequence pre_grasp,grasp,lift,retract \
  --duration 8.0 \
  --execute \
  --confirm EXECUTE_TEACH_PICK_PLACE
```

성공 기준:

- 로봇이 천천히 waypoint들을 따라감
- 관절이 예상 밖으로 크게 돌지 않음
- 블록이나 테이블과 부딪히지 않음
- gripper 명령은 아직 보내지 않음

## 8단계: gripper 포함 grasp/lift 테스트

arm-only가 안전하면 gripper를 포함합니다.

```bash
python3 motion2/scripts/run_teach_pick_place.py \
  --sequence pre_grasp,grasp,close_gripper,lift,open_gripper,retract \
  --duration 8.0 \
  --close-gripper 0.60 \
  --open-gripper 0.00 \
  --execute \
  --confirm EXECUTE_TEACH_PICK_PLACE
```

성공 기준:

- grasp 위치에서 gripper가 닫힘
- 블록을 잡음
- lift에서 블록이 따라 올라옴
- open_gripper에서 블록을 놓음

실패하면:

- `grasp` waypoint가 블록 중심에 맞는지 확인
- grasp 높이가 너무 높거나 낮은지 확인
- `--close-gripper` 값을 조금 조정
- 블록 위치가 recording 때와 replay 때 같은지 확인

## 9단계: 전체 MVP replay

위 단계들이 성공하면 전체 sequence를 실행합니다.

먼저 dry-run:

```bash
python3 motion2/scripts/run_teach_pick_place.py --duration 8.0
```

그 다음 실행:

```bash
python3 motion2/scripts/run_teach_pick_place.py \
  --duration 8.0 \
  --execute \
  --confirm EXECUTE_TEACH_PICK_PLACE
```

성공 기준:

```text
pre_grasp
-> grasp
-> close_gripper
-> lift
-> place
-> open_gripper
-> retract
```

이 흐름으로 실제 블록을 잡고 옮기면 1차 MVP 성공입니다.

## 문제가 생겼을 때 바로 보는 표

### 로봇이 아예 안 움직임

확인할 것:

- `omy_f3m.launch.py`를 썼는가?
- `omy_ai.launch.py`를 쓰고 있지 않은가?
- controller들이 active인가?
- action 이름이 맞는가?
- `/joint_states`가 들어오는가?
- motor torque가 켜져 있는가?
- controller state의 `speed_scaling_factor`가 0은 아닌가?

### 로봇이 너무 크게 움직이려 함

확인할 것:

- MoveIt pose goal을 쓰고 있는가?
- teach replay인데 waypoint 사이가 너무 먼가?
- 첫 waypoint가 현재 로봇 자세와 너무 먼가?
- `joint1`, `joint4`, `joint6`가 `+/-2*pi` 근처로 돌아가려 하는가?
- 중간 waypoint가 필요한가?

### gripper가 블록을 못 잡음

확인할 것:

- 블록 위치가 고정되어 있는가?
- `grasp` waypoint가 블록 중심인가?
- grasp 높이가 맞는가?
- gripper close 값이 충분한가?
- 블록이 미끄럽거나 너무 작은가?

### dry-run에서 waypoint가 없다고 나옴

확인할 것:

- `record_teach_waypoint.py`를 실행했는가?
- 이름을 정확히 썼는가?
- YAML에 `pre_grasp`, `grasp`, `lift`, `place`, `retract`가 있는가?

## 오늘 성공의 기준

오늘의 성공은 full autonomous sim-to-real이 아닙니다.

오늘의 성공은 이것입니다.

```text
실제 OMY-F3M이 저장된 joint waypoint를 따라 움직이고,
정해진 위치의 블록을 잡고,
들어 올리고,
놓는 것
```

이걸 성공하면 다음 단계로 갈 수 있습니다.

## 다음 단계

teach-and-replay MVP가 성공한 뒤에만 다음을 진행합니다.

1. D405 RGB/depth 이미지 저장
2. D405에서 YOLO detection 확인
3. D405로 block yaw 추정
4. manual/fixed target 기반 `RealAdapter` 일부 구현
5. 작은 guarded motion에만 camera result 연결
6. PPO/RL alignment는 마지막에 연결
7. D435 top camera 또는 marker 기반 localization 추가

## 보고서나 발표에서 이렇게 말할 수 있음

```text
IsaacLab에서는 전체 pick-and-place pipeline을 검증했지만, 실제 OMY-F3M에서는
MoveIt pose goal이 큰 joint wraparound trajectory를 만들 수 있음을 확인했다.
따라서 실제 로봇 MVP는 먼저 joint-space teach-and-replay 방식으로 구성하여
안전하게 command path와 grasp/lift/place 동작을 검증하고, 이후 camera/YOLO/RL
기반의 자율 sim-to-real로 확장하는 단계적 전략을 사용한다.
```
