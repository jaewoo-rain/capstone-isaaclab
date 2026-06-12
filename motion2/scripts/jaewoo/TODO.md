# 현장 실행 런북 (Field Runbook)

로봇 앞에 가서 **처음부터 순서대로** 실행하는 가이드.
최종 목표: 천장캠 좌표 → 모션플래닝 → 손목캠 정밀 보정 → RL grasp → lift → insert 자동화.

> 한 PHASE 끝나면 다음 PHASE. 막히면 그 PHASE 안에서 해결하고 넘어간다.

---

## 머신 / 터미널 범례

| 표기 | 의미 |
|------|------|
| **[PC]** | 개발 PC(윈도우). YOLO 학습·ONNX export·파일 송신 |
| **[HOST]** | 로봇 호스트(Pi5). `ssh root@omy-SNPR44B1021.local` 로 접속 |
| **[DOCKER]** | 컨테이너 안. `docker exec -it open_manipulator bash` → `source /root/ros2_ws/install/setup.bash` |

**[DOCKER] 안에서는 항상 먼저:**
```bash
cd /root/ros2_ws/src/open_manipulator        # 이하 모든 python3 motion2/... 명령의 기준 경로
source /root/ros2_ws/install/setup.bash
```

bringup·카메라노드·RL 은 **동시에 여러 [DOCKER] 터미널**이 필요하다(T1 bringup, T2 천장캠, T3 손목캠, T4 명령).
각 터미널마다 `docker exec -it open_manipulator bash` 로 새로 들어가 `source` 한다.

---

## PHASE 0 — 최신 코드 컨테이너에 복사 ⚠️ 이번에 camera 코드 바뀜, 필수

> 이번 수정: 천장캠 yaw 좌표변환 버그, minAreaRect 90° 모호성, TF2 buffer 재생성 +
> 새 캘리브 스크립트(`dump_intrinsics.py`, `calibrate_ceiling_extrinsic.py`).
> **이거 안 옮기면 옛 버그 그대로 돈다.**

**[PC] 최신 jaewoo 폴더 → [HOST]**
```cmd
scp -r "C:\Users\jaewoo\Desktop\캡스톤\capstone-isaaclab\motion2\scripts\jaewoo\camera" ^
    root@omy-SNPR44B1021.local:/root/open_manipulator/motion2/scripts/jaewoo/
scp -r "C:\Users\jaewoo\Desktop\캡스톤\capstone-isaaclab\motion2\scripts\jaewoo\rl" ^
    root@omy-SNPR44B1021.local:/root/open_manipulator/motion2/scripts/jaewoo/
```
> ⚠️ 최신 코드가 이 리눅스 데스크(`/home/jaewoo/IsaacLab`)에만 있으면, 먼저 그쪽에서
> PC 로 동기화(git pull 등) 후 위 scp 를 하거나, 데스크에서 직접 scp 한다.

**[HOST] → [DOCKER]**
```bash
docker cp /root/open_manipulator/motion2/scripts/jaewoo/camera \
    open_manipulator:/root/ros2_ws/src/open_manipulator/motion2/scripts/jaewoo/
docker cp /root/open_manipulator/motion2/scripts/jaewoo/rl \
    open_manipulator:/root/ros2_ws/src/open_manipulator/motion2/scripts/jaewoo/
```
- [ ] camera/ 복사 완료 (`calibrate_ceiling_extrinsic.py`, `dump_intrinsics.py` 보이는지 확인)
- [ ] rl/ 복사 완료

---

## PHASE 1 — 컨테이너 진입 & 의존성 확인

**[DOCKER]**
```bash
docker exec -it open_manipulator bash
cd /root/ros2_ws/src/open_manipulator && source /root/ros2_ws/install/setup.bash
```

**카메라/비전 의존성:**
```bash
pip install onnxruntime opencv-python-headless pyrealsense2 numpy pyyaml
```
- [ ] 설치 OK

**RL 의존성(롤아웃용 — 컨테이너에 없을 수 있음):**
```bash
python3 -c "import stable_baselines3, torch, gymnasium; print('rl deps OK')"
# 실패하면:  pip install 'stable-baselines3==2.7.1' torch gymnasium
```
- [ ] `rl deps OK` 출력 확인 (torch 가 aarch64 라 느릴 수 있음 — 한 번만 설치)

---

## PHASE 2 — bringup 정상화 (T1 터미널, 계속 켜둠)

**[DOCKER · T1]**
```bash
ros2 launch open_manipulator_bringup omy_f3m.launch.py
```
- [ ] `stack smashing detected` 없이 부팅
- [ ] 다른 터미널에서 `/joint_states` 수신 확인:
  ```bash
  ros2 topic echo /joint_states --once
  ```
- [ ] MoveIt action 서버 확인:
  ```bash
  ros2 action list | grep move_action
  ```

> bringup 안 켜지면 이후 전부 막힘. 여기서 해결하고 진행.

---

## PHASE 3 — 카메라 캘리브 (1회성, bringup 불필요한 것도 있음)

### 3-1. intrinsics 2개 (카메라 USB만, bringup 불필요)
**[DOCKER]** — detector 가 쓰는 해상도(640x480)와 같은 해상도로 측정됨(스크립트 기본값)
```bash
python3 motion2/scripts/jaewoo/camera/dump_intrinsics.py --name d435i   # 천장캠
python3 motion2/scripts/jaewoo/camera/dump_intrinsics.py --name d405    # 손목캠
```
- [ ] `camera/calib/d435i_intrinsics.yaml` 생성
- [ ] `camera/calib/d405_intrinsics.yaml` 생성

### 3-2. 천장캠 extrinsic (체커보드 필요, bringup 불필요)
> ★★ 이 행렬의 **회전**이 박스 yaw 에 직접 들어간다(이번 수정 `yaw_cam_to_base`).
> 체커보드를 link0 **축까지** 맞춰 놓아야 yaw 오차가 안 생긴다:
> - 체커보드 (0,0)코너 = link0 원점(로봇 발 밑)
> - 긴 줄(X축) = 로봇 앞쪽(+x), Y축 = +y, 수평 유지

**[DOCKER]**
```bash
python3 motion2/scripts/jaewoo/camera/calibrate_ceiling_extrinsic.py --show
# 체커보드가 기본 7x5 내부코너·30mm 아니면: --cols N --rows M --square 0.025
```
- [ ] 재투영 오차 RMS **< ~1px** (3px 넘으면 조명/위치 바꿔 재측정)
- [ ] 출력된 "카메라 위치(link0 기준)" 가 실제 천장캠 대략 위치와 맞는지 눈으로 확인
- [ ] `camera/calib/ceiling_extrinsic.yaml` 생성
- [ ] 검증: `python3 motion2/scripts/jaewoo/camera/coord_transform.py --extrinsic motion2/scripts/jaewoo/camera/calib/ceiling_extrinsic.yaml`

### 3-3. 손목캠 TF2 프레임 확인 (캘리브 불필요 — URDF 자동, bringup 필요)
**[DOCKER]** (bringup 켜진 상태)
```bash
python3 motion2/scripts/jaewoo/camera/coord_transform.py --check-tf
# timeout 나면 프레임명이 다른 것 — 어떤 게 있는지 확인:
ros2 run tf2_tools view_frames    # frames.pdf 에서 camera_* 프레임 찾기
# 다른 이름이면:  --check-tf --cam-frame camera_link
```
- [ ] `link0 ← camera_depth_optical_frame` 4×4 행렬 출력됨 (실패 시 쓸 프레임명 기록)

---

## PHASE 4 — YOLO 모델 (box/cell) — ⚠️ 제작 중

기본 `yolov8n-seg.pt` 는 COCO(80클래스)라 **box/cell 을 못 잡는다.** 커스텀 모델 필요.

### 4-A. 모델이 완성되면
**[PC]**
```bash
pip install ultralytics
yolo export model=best.pt format=onnx imgsz=640     # 학습한 box/cell 모델
scp best.onnx root@omy-SNPR44B1021.local:~/
```
**[HOST]**  `docker cp ~/best.onnx open_manipulator:/root/yolov8n-seg.onnx`
- [ ] 컨테이너 `/root/yolov8n-seg.onnx` 배치

### 4-B. 모델 아직 없을 때 — 대안 경로 (모델 완성 전에도 진행 가능)
- **손목캠**: `--pca-only` 로 **모델 없이** depth PCA 로 박스 yaw/위치 추정 (그대로 사용 가능)
- **천장캠**: 모델 없으면 자동검출 불가 → **박스를 알려진 위치에 놓고 좌표를 수동 측정**해서
  PHASE 6 의 `--box-x/--box-y` 로 직접 입력 (아래 6-0 참고). 천장캠 자동연동은 모델 완성 후.

---

## PHASE 5 — 카메라 detection 단독 테스트 (RL 전, 눈으로 검증)

### 5-1. 천장캠 (모델 있을 때만)
**[DOCKER · T2]**
```bash
python3 motion2/scripts/jaewoo/camera/ceiling_detector.py --ros \
    --model /root/yolov8n-seg.onnx \
    --extrinsic  motion2/scripts/jaewoo/camera/calib/ceiling_extrinsic.yaml \
    --intrinsics motion2/scripts/jaewoo/camera/calib/d435i_intrinsics.yaml \
    --classes box,cell
```
다른 터미널에서:
```bash
ros2 topic echo /vision/box_coarse --once     # PoseArray — 검출된 모든 박스
ros2 topic echo /vision/box_target --once     # grasp 타깃 1개(분포중심 최근접). yaw=2*atan2(qz,qw)
ros2 topic echo /vision/cell_coarse --once    # PoseArray — 모든 셀
```
- [ ] box xy 가 실제 박스 위치와 대략 맞음
- [ ] 박스를 20° 돌렸을 때 yaw 도 그만큼 변하는지 (이번 yaw 수정 확인)
- [ ] 박스 여러 개 놓았을 때 `/vision/box_target` 이 분포중심(0.45,-0.10) 최근접을 가리킴

### 5-2. 손목캠 (모델 없이 PCA)
**[DOCKER · T3]**
```bash
python3 motion2/scripts/jaewoo/camera/wrist_detector.py --ros \
    --intrinsics motion2/scripts/jaewoo/camera/calib/d405_intrinsics.yaml \
    --pca-only
```
```bash
ros2 topic hz /vision/box_fine      # 발행 주파수 (TF2 buffer 수정으로 안정적이어야)
ros2 topic echo /vision/box_fine --once
```
- [ ] `/vision/box_fine` 가 끊김 없이 publish (TF2 timeout 경고 안 뜸)
- [ ] 박스 가까이서 yaw 안정적

---

## PHASE 6 — RL grasp (★ 안전 단계별로, 비상정지 손 닿는 곳에)

> RL README 의 sim2real 검증 순서 그대로. **자세 매핑이 미검증**이라 작게 시작한다.

### 6-0. 박스 좌표 준비
- 모델 있으면: 5-1 의 `/vision/box_coarse` 에서 x,y 읽기 (yaw 디코드: `yaw = 2*atan2(qz, qw)`)
- 모델 없으면: 박스를 작업공간(`x≈0.45±0.10, y≈-0.10±0.10`)에 놓고, EE 를 박스 중심 바로 위로
  이동 후 좌표 읽기:
  ```bash
  python3 motion2/scripts/jaewoo/show_ee_pose.py
  ```
- **시작 EE 를 박스 근처(30cm 이내)·수직파지 자세로** 먼저 보내둔다(롤아웃 발산 방지):
  ```bash
  python3 motion2/scripts/jaewoo/run_ee_pose_move.py --x 0.45 --y -0.10 --z 0.3724 \
      --execute --confirm EXECUTE_EE_POSE_MOVE --no-constrain-joint5 --max-joint-delta 3.0
  ```

### 6-1. dry-run (로봇 안 움직임)
```bash
python3 motion2/scripts/jaewoo/rl/grasp/run_grasp.py \
    --box-x <값> --box-y <값> --box-yaw 0.0
```
- [ ] `rollout status=converged`
- [ ] `align err: xy=[..] mm` 5mm 이내
- [ ] 각 step `plan ok=True`, `actual max_delta < 0.35`

### 6-2. yaw 끄고 실제 실행 — 자세 검증 (가장 중요)
```bash
python3 motion2/scripts/jaewoo/rl/grasp/run_grasp.py \
    --box-x <값> --box-y <값> --box-yaw 0.0 \
    --no-align-yaw --execute --confirm EXECUTE_GRASP \
    --no-constrain-joint5 --max-joint-delta 3.0
```
눈으로 확인:
- [ ] pre_grasp: EE 가 박스 **중심 바로 위** 정지
- [ ] grasp 하강: 그리퍼가 박스를 **양옆에서** 감싸며 내려옴 (위에서 찍으면 자세 매핑 문제 → 멈춤)
- [ ] close: 짧은 변(4.4cm) 물고 정지
- [ ] lift: 박스 딸려 올라옴
> 자세 명백히 틀리면 **여기서 멈추고** `rl/grasp/config.py` 의 `GRIP_CENTER_OFFSET_XY` /
> orientation 매핑 조정. 3단계로 무리해서 넘어가지 말 것.

### 6-3. yaw 정렬 켜고 실행
```bash
# 박스를 20°(0.35rad) 비스듬히 놓고
python3 motion2/scripts/jaewoo/rl/grasp/run_grasp.py \
    --box-x <값> --box-y <값> --box-yaw 0.35 \
    --execute --confirm EXECUTE_GRASP \
    --no-constrain-joint5 --max-joint-delta 3.0
```
- [ ] 그리퍼가 박스 회전각만큼 손목 돌려 짧은 변에 정렬
- [ ] 반대로 돌거나 90° 어긋나면 → yaw 부호/축, 또는 extrinsic(3-2) 재확인

---

## PHASE 7 — (다음 단계, grasp 검증 후)

- [ ] 손목캠 정밀보정 연동: approach 후 `/vision/box_fine` 좌표로 run_grasp 재실행
- [ ] insert: `rl/insert/run_insert.py` (셀이 뒤쪽 -x → `--x-min -0.55`, README 한계 3가지 숙지)
- [ ] 전체 파이프라인: `rl/all/run_pipeline.py --stop-after grasp_lift|turn|insert|place` 로 끊어서
- [ ] 비전 자동연동 스크립트 `run_grasp_vision.py`(미작성) — 토픽 자동구독, 데스크에서 작성 예정

---

## 빠른 복귀 / 안전

```bash
# 홈 자세 복귀
python3 motion2/scripts/jaewoo/go_to_start.py --execute --confirm GO_TO_START
# 그리퍼 토글
python3 motion2/scripts/jaewoo/toggle_gripper.py --execute --confirm TOGGLE_GRIPPER
```
- 모든 스크립트 **기본 dry-run**. `--execute --confirm <TEXT>` 둘 다 있어야 움직인다.
- 단계별 타이핑 확인이 기본(생략은 `--no-step-prompts`, 권장 안 함).

---

## 캘리브/모델 산출물 체크

| 파일 | 생성 위치 | 상태 |
|------|-----------|------|
| `camera/calib/d435i_intrinsics.yaml` | PHASE 3-1 | ⬜ |
| `camera/calib/d405_intrinsics.yaml` | PHASE 3-1 | ⬜ |
| `camera/calib/ceiling_extrinsic.yaml` | PHASE 3-2 | ⬜ |
| 손목캠 hand-eye | 불필요(URDF/TF2) | ✅ |
| `/root/yolov8n-seg.onnx` (box/cell 커스텀) | PHASE 4 | ⬜ 제작 중 |
