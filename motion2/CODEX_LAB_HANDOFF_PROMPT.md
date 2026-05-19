# 연구실에서 Ubuntu Codex CLI에 처음 넣을 프롬프트

아래 프롬프트를 연구실에서 Codex CLI를 켠 뒤 그대로 붙여넣으세요.

이 프롬프트의 목적은 Codex가 지금 상황을 바로 이해하게 만드는 것입니다.
Codex Desktop의 이전 대화는 Ubuntu Codex CLI로 자동 연동되지 않는다고
생각하는 것이 안전합니다. 그래서 이 프롬프트가 필요합니다.

## 첫 프롬프트

```text
나는 로봇 sim-to-real 초보자이고, 지금 연구실에서 OMY-F3M 실제 로봇으로
capstone-isaaclab 저장소의 motion2 프로젝트를 진행하려고 한다.

중요한 목표:

나는 지금 완전 자율 sim-to-real을 바로 하려는 것이 아니다.
먼저 실제 로봇이 정해진 위치의 블록을 잡고, 들어 올리고, 놓는
teach-and-replay MVP를 빠르게 성공시키고 싶다.

먼저 아래 파일들을 읽고 현재 프로젝트 상태를 파악해줘.

- motion2/SIM_TO_REAL_CHECKLIST.md
- motion2/MVP_DEMO.md
- motion2/TEACH_REPLAY_LAB_GUIDE.md
- motion2/config/teach_pick_place_waypoints.yaml
- motion2/scripts/record_teach_waypoint.py
- motion2/scripts/run_teach_pick_place.py
- motion2/scripts/run_smoke_sequence.py
- motion2/scripts/joint_space_smoke_test.py
- motion2/scripts/gripper_smoke_test.py

현재 상황:

- 친구가 IsaacLab에서 OMY-F3M pick-and-place 시뮬레이션을 성공했다.
- 원래 목표는 IsaacLab sim 코드를 실제 OMY-F3M으로 sim-to-real 하는 것이다.
- 하지만 나는 초보자라서 전체 RealAdapter, camera, YOLO, RL policy를 한 번에
  붙이기 어렵다.
- 연구실에서 MoveIt pose goal을 테스트했을 때 작은 EE 이동에도 관절이 크게
  돌아가는 wraparound 문제가 있었다.
- 그래서 지금은 MoveIt absolute pose goal이나 PPO/RL policy로 로봇을 바로
  움직이면 안 된다.
- D435 top camera는 아직 천장 카메라로 mount/calibration이 안 되어 있다.
- D405 wrist camera, YOLO, PPO/RL policy는 아직 실제 로봇에서 검증되지 않았다.
- 현재 가장 현실적인 목표는 joint-space teach-and-replay 방식으로 MVP를
  먼저 성공시키는 것이다.
- teleoperation으로 로봇을 안전하게 움직여서 주요 자세를 만든 뒤,
  그 순간의 joint 값을 waypoint로 저장하는 것도 허용되는 방향이다.
- 다만 지금 당장 목표는 진짜 imitation learning policy 학습이 아니라,
  teleoperation demonstration을 waypoint로 기록하고 replay하는 MVP이다.
- teleoperation launch와 direct replay launch를 섞으면 controller/remap이
  헷갈릴 수 있으니, teleop으로 저장한 뒤 regular bringup에서 replay하는
  흐름을 우선한다.

내가 오늘 연구실에서 하고 싶은 순서:

1. regular bringup 확인
   ros2 launch open_manipulator_bringup omy_f3m.launch.py

2. MoveIt/RViz는 필요하면 조심스럽게 자세를 만드는 용도로만 사용
   ros2 launch open_manipulator_moveit_config omy_f3m_moveit.launch.py start_rviz:=true

3. controller/action/joint_states 확인
   ros2 control list_controllers
   ros2 action list
   ros2 topic echo /joint_states --once

4. tiny smoke test dry-run
   python3 motion2/scripts/run_smoke_sequence.py --name full_basic_smoke

5. tiny smoke test execute
   python3 motion2/scripts/run_smoke_sequence.py --name full_basic_smoke --execute

6. 로봇을 안전한 방법으로 pre_grasp, grasp, lift, place, retract 자세로 보낸 뒤
   각 자세의 joint 값을 저장
   - 가능하면 RViz/MoveIt 또는 teleoperation으로 자세를 만든다.
   - 손으로 억지로 밀지 않는다.
   - teleoperation을 썼다면 저장 후 regular bringup으로 돌아와 replay한다.

   python3 motion2/scripts/record_teach_waypoint.py --name pre_grasp --overwrite
   python3 motion2/scripts/record_teach_waypoint.py --name grasp --overwrite
   python3 motion2/scripts/record_teach_waypoint.py --name lift --overwrite
   python3 motion2/scripts/record_teach_waypoint.py --name place --overwrite
   python3 motion2/scripts/record_teach_waypoint.py --name retract --overwrite

7. replay dry-run
   python3 motion2/scripts/run_teach_pick_place.py

8. arm-only partial replay dry-run
   python3 motion2/scripts/run_teach_pick_place.py --sequence pre_grasp,grasp,lift,retract --duration 8.0

9. arm-only partial replay execute
   python3 motion2/scripts/run_teach_pick_place.py --sequence pre_grasp,grasp,lift,retract --duration 8.0 --execute --confirm EXECUTE_TEACH_PICK_PLACE

10. gripper 포함 grasp/lift 테스트
   python3 motion2/scripts/run_teach_pick_place.py --sequence pre_grasp,grasp,close_gripper,lift,open_gripper,retract --duration 8.0 --close-gripper 0.60 --open-gripper 0.00 --execute --confirm EXECUTE_TEACH_PICK_PLACE

네가 나를 도와줄 방식:

- 나는 초보자니까 한 번에 여러 단계를 말하지 말고, 다음 한 단계만 명확하게 알려줘.
- 내가 명령 결과를 붙여넣으면, 성공인지 실패인지 쉽게 설명해줘.
- 실패하면 원인을 추측만 하지 말고, 다음에 확인할 명령을 알려줘.
- 위험한 명령이면 실행하지 말라고 먼저 말해줘.
- full autonomous pick-and-place, RealAdapter 전체 구현, YOLO/RL 실제 구동은 아직 제안하지 마.
- 지금은 joint-space teach-and-replay MVP를 우선한다.
- MoveIt pose goal이 필요하면 반드시 plan-only 또는 guard를 먼저 확인하게 해줘.
- 로봇이 실제로 움직이는 명령은 항상 dry-run 또는 상태 확인 후에만 제안해줘.
- 내가 손으로 로봇을 억지로 밀려고 하면 torque/teach mode가 확인되지 않는 한 하지 말라고 말해줘.
- 내가 teleoperation을 쓰겠다고 하면, 그것은 가능한 방향이라고 안내하되
  "모방학습 policy 학습"이 아니라 "teleop demonstration 기록 후 waypoint replay"로
  먼저 진행하게 도와줘.
- teleoperation 중에는 direct replay execute를 섞지 말고, teleop 종료 후 regular
  `omy_f3m.launch.py` bringup에서 replay하게 안내해줘.

지금부터 나와 같이 연구실 실험을 단계별로 진행해줘.
먼저 네가 이해한 현재 전략을 5줄 이내로 요약하고,
그 다음 내가 지금 실행해야 할 첫 번째 명령만 알려줘.
```

## Codex에게 결과를 붙여넣을 때 쓰는 문장

명령 결과를 붙여넣을 때는 이렇게 말하면 됩니다.

```text
방금 이 명령을 실행했어:

<명령어>

출력은 아래와 같아:

<터미널 출력 붙여넣기>

이게 성공인지 실패인지 설명해주고, 다음에 뭘 해야 하는지 한 단계만 알려줘.
```

## 로봇이 움직이는 명령 전 확인 프롬프트

실제로 로봇을 움직이기 직전에 Codex에게 이렇게 물어보세요.

```text
이 명령은 실제 로봇을 움직이는 명령이야.
내가 실행하기 전에 위험한 점이 있는지 확인해줘.
지금 단계에서 이 명령을 실행해도 되는지, 실행 전 확인할 것을 체크리스트로 알려줘.

명령:
<실행하려는 명령어>
```

## smoke test 이후 상황 설명 프롬프트

```text
tiny smoke test는 성공했어.
이제 teach-and-replay waypoint를 만들려고 해.
pre_grasp, grasp, lift, place, retract가 각각 어떤 자세인지 초보자 기준으로 설명해주고,
어떤 순서로 저장하면 되는지 알려줘.
```

## teleoperation으로 waypoint를 만들고 싶을 때 쓰는 프롬프트

```text
teleoperation으로 로봇을 움직여서 waypoint를 저장하고 싶어.
지금 목표는 진짜 모방학습 policy 학습이 아니라, teleop으로 주요 자세를 만든 뒤
그 joint 값을 저장해서 replay하는 MVP야.

teleoperation으로 pre_grasp, grasp, lift, place, retract 자세를 만들 때
주의할 점과 저장 순서를 알려줘.
그리고 teleop launch와 replay launch를 섞지 않도록 어떤 순서로 종료/재시작해야 하는지도 알려줘.
```

## waypoint 저장 후 확인 프롬프트

```text
waypoint를 저장했어.
motion2/config/teach_pick_place_waypoints.yaml 내용을 확인해서
pre_grasp, grasp, lift, place, retract가 모두 있는지,
값이 이상해 보이는지,
replay 전에 조심해야 할 점이 있는지 봐줘.
```

## replay dry-run 후 확인 프롬프트

```text
run_teach_pick_place.py dry-run 결과를 붙여넣을게.
각 단계의 max joint delta가 안전해 보이는지 봐줘.
너무 큰 이동이 있으면 어떤 waypoint 사이가 문제인지 알려줘.
아직 실제 실행하지 말고, 먼저 판단만 해줘.
```

## 프로젝트 방향이 흔들릴 때 다시 붙일 문장

```text
지금 나는 full sim-to-real보다 실제 MVP 성공이 우선이야.
카메라/YOLO/RL은 나중이고, 지금은 joint-space teach-and-replay로
정해진 위치의 블록을 잡고 옮기는 것부터 성공시키고 싶어.
이 기준으로 다음 단계를 추천해줘.
```

## 자주 쓰는 명령 모음

상태 확인:

```bash
ros2 control list_controllers
ros2 action list
ros2 topic echo /joint_states --once
```

smoke test dry-run:

```bash
python3 motion2/scripts/run_smoke_sequence.py --name full_basic_smoke
```

smoke test execute:

```bash
python3 motion2/scripts/run_smoke_sequence.py --name full_basic_smoke --execute
```

waypoint 저장:

```bash
python3 motion2/scripts/record_teach_waypoint.py --name pre_grasp --overwrite
python3 motion2/scripts/record_teach_waypoint.py --name grasp --overwrite
python3 motion2/scripts/record_teach_waypoint.py --name lift --overwrite
python3 motion2/scripts/record_teach_waypoint.py --name place --overwrite
python3 motion2/scripts/record_teach_waypoint.py --name retract --overwrite
```

replay dry-run:

```bash
python3 motion2/scripts/run_teach_pick_place.py
```

arm-only partial replay dry-run:

```bash
python3 motion2/scripts/run_teach_pick_place.py \
  --sequence pre_grasp,grasp,lift,retract \
  --duration 8.0
```

arm-only partial replay execute:

```bash
python3 motion2/scripts/run_teach_pick_place.py \
  --sequence pre_grasp,grasp,lift,retract \
  --duration 8.0 \
  --execute \
  --confirm EXECUTE_TEACH_PICK_PLACE
```

grasp/lift replay execute:

```bash
python3 motion2/scripts/run_teach_pick_place.py \
  --sequence pre_grasp,grasp,close_gripper,lift,open_gripper,retract \
  --duration 8.0 \
  --close-gripper 0.60 \
  --open-gripper 0.00 \
  --execute \
  --confirm EXECUTE_TEACH_PICK_PLACE
```
