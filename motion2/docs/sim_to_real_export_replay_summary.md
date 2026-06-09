# IsaacLab-to-Real OMY-F3M Trajectory Export/Replay 요약

## 1. 이번 작업의 목적

기존 프로젝트의 최종 목표는 IsaacLab 시뮬레이션에서 성공한 OMY-F3M pick-and-place 동작을 실제 로봇으로 옮기는 것이다.

이전 단계에서는 실제 로봇에서 다음을 이미 확인했다.

- RViz/MoveIt으로 사람이 만든 waypoint를 저장하고 replay하는 방식
- `link0` 기준, `link6` end-effector 기준의 real waypoint pick-and-place
- 좌표 기반 Cartesian pick-and-place (`motion2/scripts/jaewoo/run_pick_place.py`)
- 실제 그리퍼 action `/gripper_controller/gripper_cmd`를 통한 open/close
- joint delta guard와 confirm 문자열을 통한 안전 실행

이번 작업의 목표는 그 다음 단계로, IsaacLab에서 성공한 시뮬레이션 동작을 실제 로봇이 사용할 수 있는 데이터로 추출하는 것이다.

즉 이번 작업은 아래 흐름을 만들기 위한 것이다.

```text
IsaacLab 성공 동작
-> trajectory export
-> NPZ 파일 저장
-> real robot replay용 검사
-> real OMY-F3M에서 dry-run / execute 후보 생성
```

이 과정은 카메라가 아직 설치/보정되지 않은 상황에서도 진행 가능하다.

## 2. 왜 바로 RL policy를 실로봇에 넣지 않았는가

현재 실제 로봇 환경에서는 다음이 아직 준비되지 않았다.

- D435 top camera mount/calibration
- D405 wrist camera real image pipeline
- real camera 기반 YOLO detection 검증
- real robot에서 PPO/RL policy action 검증
- RealAdapter의 full autonomous execution

또한 이전 real robot 실험에서 MoveIt pose goal이 작은 EE 이동에도 큰 joint wraparound trajectory를 만들 수 있음을 확인했다.

따라서 지금 바로 아래 구조로 가는 것은 위험하다.

```text
camera detection
-> RL policy action
-> real robot motion
```

대신 이번 작업에서는 안전하게 다음 구조를 택했다.

```text
sim에서 성공한 동작을 먼저 숫자 파일로 기록
-> 기록된 joint trajectory를 분석
-> 실제 로봇에 천천히 replay 가능한 형태로 변환
```

## 3. Exporter란 무엇인가

Exporter는 시뮬레이션 동작 녹화기이다.

기존 `motion2/scripts/play_chain_sim.py`는 IsaacLab 안에서 로봇이 움직이지만, 그 움직임을 파일로 저장하지 않는다.

이번에 추가한 exporter는 기존 sim chain을 실행하면서 매 step마다 다음 정보를 저장한다.

- target end-effector position
- target end-effector quaternion
- gripper command
- 현재 simulated arm joint position
- 현재 simulated gripper joint position
- 현재 EE pose
- box 위치/yaw
- cell 위치/yaw
- run index
- timestamp

추가 파일:

```text
motion2/scripts/export_sim_pick_place_trajectory.py
```

기본 실행 예시:

```bash
./isaaclab.sh -p source/motion2/scripts/export_sim_pick_place_trajectory.py \
  --enable_cameras \
  --repeat 1 \
  --hold_s 0 \
  --out motion2/config/sim_exported_pick_place_trajectory.npz
```

## 4. NPZ 파일이란 무엇인가

`.npz`는 Numpy 배열 저장 파일이다.

영상 파일이 사람이 보는 녹화 파일이라면, `.npz`는 로봇 프로그램이 읽는 숫자 녹화 파일이다.

이번 exporter가 만드는 NPZ에는 다음 배열이 들어간다.

```text
time_s
target_pos
target_quat_wxyz
gripper_command
ee_pos
ee_quat_wxyz
ee_lin_vel
arm_joint_pos
gripper_joint_pos
all_joint_pos
box_xy
box_yaw
cell_xy
cell_yaw
run_index
```

이 중 실제 robot replay에 가장 중요한 값은 다음이다.

```text
arm_joint_pos
gripper_command
time_s
run_index
```

## 5. Scripted Export 모드

처음 exporter는 기존 chain 전체를 그대로 실행했다.
그 경우 RL grasp 단계가 들어가므로, policy/checkpoint/vision 조건이 맞지 않으면 export 결과가 불안정할 수 있었다.

그래서 exporter에 `--scripted` 옵션을 추가했다.

```bash
./isaaclab.sh -p source/motion2/scripts/export_sim_pick_place_trajectory.py \
  --enable_cameras \
  --scripted \
  --repeat 3 \
  --hold_s 0 \
  --out motion2/config/sim_exported_pick_place_trajectory.npz
```

`--scripted` 모드는 다음을 우회한다.

- camera
- YOLO
- RL policy

대신 IsaacLab의 ground-truth box/cell pose를 이용해서 deterministic pick-and-place trajectory를 생성한다.

이 모드의 목적은 RL 성능을 증명하는 것이 아니라, sim-to-real replay로 옮길 수 있는 안정적인 trajectory 데이터를 만드는 것이다.

## 6. 친구가 생성한 성공 Export 파일

친구가 `--scripted --repeat 3`로 export를 실행했고, 3개 run 중 첫 번째 run이 적재 성공한 것으로 확인했다.

Git에 추가된 파일:

```text
motion2/config/sim_exported_pick_place_trajectory copy.npz
motion2/config/sim_exported_pick_place_trajectory.npz.summary copy.json
```

summary 기준:

```text
runs: 3
samples: 4014
mode: scripted
```

각 run 결과:

```text
run_index 0:
  grasp_success: true
  insert_success: true
  cell_xy_dist_m: 0.0076

run_index 1:
  grasp_success: true
  insert_success: false
  cell_xy_dist_m: 0.1415

run_index 2:
  grasp_success: true
  insert_success: false
  cell_xy_dist_m: 0.2157
```

따라서 real replay 후보로는 `run_index 0`만 사용한다.

## 7. 왜 run_index 필터가 필요한가

`--repeat 3`로 export하면 하나의 NPZ 파일 안에 3개 run이 이어붙어서 저장된다.

이 상태에서 전체 trajectory를 그대로 보면 다음 문제가 생긴다.

```text
run0 마지막 자세
-> run1 시작 자세
```

이 구간은 실제 작업 동작이 아니라, 시뮬레이션 reset으로 인해 생기는 큰 점프이다.

그래서 inspect/replay 스크립트에 `--run-index` 옵션을 추가했다.

```bash
python3 motion2/scripts/inspect_sim_exported_trajectory.py \
  --npz "motion2/config/sim_exported_pick_place_trajectory copy.npz" \
  --run-index 0
```

## 8. Inspect 스크립트

추가 파일:

```text
motion2/scripts/inspect_sim_exported_trajectory.py
```

역할:

- NPZ 파일 구조 확인
- 특정 run만 선택
- sample 수 확인
- gripper transition 확인
- raw joint delta 확인
- downsample 후 segment delta 확인
- guard 통과 여부 출력

성공 run 검사 결과:

```text
run_index: 0
samples: 1338
duration: 22.283333 s
gripper transitions: 2
raw max per-step delta: 0.027203 rad
downsample stride: 5
downsampled points: 271
downsampled max segment delta: 0.122475 rad
guard_ok=True
```

이 결과는 real replay 후보로 사용하기에 이전 전체 3-run 검사보다 훨씬 안전하다.

## 9. Real Replay 스크립트

추가 파일:

```text
motion2/scripts/run_sim_exported_trajectory_real.py
```

역할:

- NPZ에서 `run_index` 하나를 선택
- `arm_joint_pos`를 downsample
- gripper command 변화 지점에서 arm trajectory segment 분리
- arm segment는 `/arm_controller/follow_joint_trajectory`로 전송
- gripper event는 `/gripper_controller/gripper_cmd`로 전송
- 기본값은 dry-run
- 실제 실행은 confirm 문자열이 있어야만 가능

실제 실행 안전장치:

```text
--execute
--confirm EXECUTE_SIM_EXPORTED_TRAJECTORY
```

dry-run 예시:

```bash
python3 motion2/scripts/run_sim_exported_trajectory_real.py \
  --npz "motion2/config/sim_exported_pick_place_trajectory copy.npz" \
  --run-index 0
```

실제 실행 예시:

```bash
python3 motion2/scripts/run_sim_exported_trajectory_real.py \
  --npz "motion2/config/sim_exported_pick_place_trajectory copy.npz" \
  --run-index 0 \
  --duration-scale 2.0 \
  --execute \
  --confirm EXECUTE_SIM_EXPORTED_TRAJECTORY
```

## 10. Gripper 매핑

시뮬레이션 gripper command는 다음 범위를 사용한다.

```text
open: 0.0
close: 0.8
```

실제 OMY-F3M gripper에서는 기존 실험 기준으로 다음 값을 사용한다.

```text
open: 0.0
close: 1.05
```

따라서 replay 스크립트는 기본적으로 다음 매핑을 사용한다.

```text
sim 0.0 -> real 0.0
sim 0.8 -> real 1.05
```

CLI 옵션으로 조정 가능하다.

```bash
--sim-close 0.8
--real-close 1.05
--real-open 0.0
```

## 11. Docker 복사 방식

로봇 Docker 내부 Git remote가 실험 브랜치를 바로 pull하지 못하는 문제가 있었다.

따라서 앞으로 로봇 Docker에는 로컬에서 필요한 파일만 직접 복사하는 방식을 사용한다.

예시:

```bash
tar -cf - \
  motion2/scripts/inspect_sim_exported_trajectory.py \
  motion2/scripts/run_sim_exported_trajectory_real.py \
  "motion2/config/sim_exported_pick_place_trajectory copy.npz" \
  "motion2/config/sim_exported_pick_place_trajectory.npz.summary copy.json" \
| ssh root@omy-SNPR44B1021.local \
  "docker exec -i open_manipulator tar -xf - -C /root/ros2_ws/src/open_manipulator"
```

## 12. 발표에서 설명할 핵심 문장

이번 작업은 다음 한 문장으로 요약할 수 있다.

```text
IsaacLab에서 성공한 OMY-F3M pick-and-place trajectory를 NPZ로 export하고,
실제 OMY-F3M에서 replay 가능한 joint trajectory로 변환하는 sim-to-real bridge를 구현했다.
```

조금 더 쉽게 말하면:

```text
시뮬레이션에서 성공한 로봇 동작을 숫자 파일로 녹화하고,
그 파일을 실제 로봇이 따라 할 수 있는 형태로 바꾸는 파이프라인을 만들었다.
```

## 13. 현재 한계

아직 완전한 autonomous sim-to-real은 아니다.

남아 있는 한계:

- real camera 기반 object detection은 아직 사용하지 않음
- RL policy를 실제 로봇에 직접 연결하지 않음
- sim joint trajectory가 real robot에 완전히 같은 동작을 만든다는 보장은 없음
- 실제 execute 전에는 반드시 dry-run과 안전 확인이 필요함
- 물체 위치는 sim export 기준과 비슷하게 맞춰야 함
- table/collision object는 아직 real replay에 완전히 반영되지 않음

## 14. 다음 단계

추천 다음 단계:

1. Docker 안에서 success run inspect 실행
2. real replay dry-run 실행
3. 로봇을 start/home 자세로 맞춤
4. arm-only replay로 먼저 확인
5. gripper 포함 replay execute
6. 실패 지점이 있으면 해당 구간만 잘라서 partial replay

Docker 안에서 먼저 실행할 명령:

```bash
cd /root/ros2_ws/src/open_manipulator

python3 motion2/scripts/inspect_sim_exported_trajectory.py \
  --npz "motion2/config/sim_exported_pick_place_trajectory copy.npz" \
  --run-index 0
```

그 다음 dry-run:

```bash
python3 motion2/scripts/run_sim_exported_trajectory_real.py \
  --npz "motion2/config/sim_exported_pick_place_trajectory copy.npz" \
  --run-index 0
```

실제 실행은 dry-run 결과와 로봇 주변 안전을 확인한 뒤에만 진행한다.
