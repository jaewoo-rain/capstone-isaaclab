# camera — 비전 파이프라인 사용 가이드

## 개요

천장 카메라(D435i)와 손목 카메라(D405)로 박스·셀 위치/yaw 를 추정하고,
RL grasp/insert 스크립트에 좌표를 자동으로 공급하는 파이프라인.

```
천장캠 ─→ ceiling_detector ─→ /vision/box_coarse, /vision/cell_coarse
                                          │
                                  모션플래닝 (rough approach)
                                          │
손목캠 ─→ wrist_detector   ─→ /vision/box_fine
                                          │
                                  RL rollout → pre_grasp → grasp → lift
```

---

## 파일 구조

```
camera/
├── README.md              ← 이 파일
├── PLAN.md                ← 설계 문서 (아키텍처, 구현 순서)
├── __init__.py
├── coord_transform.py     ← 카메라 좌표 → link0 변환 유틸
├── ceiling_detector.py    ← 천장캠 detection + ROS2 노드
├── wrist_detector.py      ← 손목캠 detection + ROS2 노드
└── calib/
    ├── ceiling_extrinsic.yaml   ← 천장캠 외부 파라미터 (캘리브 후 생성)
    ├── wrist_handeye.yaml       ← 손목캠 hand-eye 결과 (캘리브 후 생성)
    ├── d435i_intrinsics.yaml    ← 천장캠 내부 파라미터
    └── d405_intrinsics.yaml     ← 손목캠 내부 파라미터
```

---

## 설치 (Pi 5 기준, Docker 내부)

```bash
pip install onnxruntime opencv-python-headless pyrealsense2 numpy pyyaml

# YOLO 모델은 PC에서 ONNX export 후 scp 로 복사
# PC에서:
#   pip install ultralytics
#   yolo export model=yolov8n-seg.pt format=onnx imgsz=640
#   scp yolov8n-seg.onnx root@omy-SNPR44B1021.local:~/
```

---

## 사전 필수: 캘리브레이션

비전 파이프라인을 쓰기 전에 반드시 두 개의 캘리브레이션 yaml 이 필요하다.

### 1. 카메라 내부 파라미터 (intrinsics)

렌즈의 초점거리(fx, fy)와 주점(cx, cy). RealSense 에서 직접 읽는다.

```python
# 터미널에서 한 번만 실행 — intrinsics 확인
import pyrealsense2 as rs
pipeline = rs.pipeline()
profile = pipeline.start()
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print(f"fx={intr.fx} fy={intr.fy} cx={intr.ppx} cy={intr.ppy}")
pipeline.stop()
```

결과를 yaml 로 저장:
```yaml
# calib/d435i_intrinsics.yaml  (천장캠)
fx: 615.3
fy: 615.2
cx: 319.8
cy: 241.1

# calib/d405_intrinsics.yaml  (손목캠)
fx: 428.5
fy: 428.5
cx: 212.3
cy: 117.8
```

---

### 2. 천장캠 외부 파라미터 (extrinsic) — ceiling_extrinsic.yaml

**무엇인가?**
천장 카메라가 로봇 기준(link0)에서 어느 위치/각도에 있는지 나타내는 4×4 변환 행렬.
이걸 알아야 "카메라에서 픽셀 좌표" → "로봇이 움직일 link0 좌표" 로 바꿀 수 있다.

**측정 방법 (체커보드 PnP):**

```bash
# 1. 체커보드 (7×5 내부 코너, 정사각형 30mm) 를 link0 원점 위 테이블에 수평 고정
# 2. 아래 스크립트 실행
python3 - <<'EOF'
import cv2, numpy as np, pyrealsense2 as rs, yaml, pathlib

CHECKERBOARD = (7, 5)   # 내부 코너 개수 (가로, 세로)
SQUARE_SIZE  = 0.030    # 정사각형 한 변 [m]

# 체커보드 3D 좌표 (link0 에 놓였다고 가정 → z=0, 원점에서 시작)
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# RealSense 에서 한 프레임
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile = pipeline.start(cfg)
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
K  = np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]])
dist = np.array(intr.coeffs)

import time; time.sleep(2)
for _ in range(30): frames = pipeline.wait_for_frames()
img = np.asanyarray(frames.get_color_frame().get_data())
pipeline.stop()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ok, corners = cv2.findChessboardCorners(gray, CHECKERBOARD)
if not ok:
    raise RuntimeError("체커보드를 찾지 못했습니다. 조명/위치 확인")

corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
    (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

_, rvec, tvec = cv2.solvePnP(objp, corners2, K, dist)
R, _ = cv2.Rodrigues(rvec)
T = np.eye(4)
T[:3,:3] = R
T[:3, 3] = tvec.flatten()

# T 는 "link0 → cam" 이므로 역변환
T_cam2base = np.linalg.inv(T)

out = {"T_cam_to_base": T_cam2base.tolist()}
pathlib.Path("calib").mkdir(exist_ok=True)
with open("calib/ceiling_extrinsic.yaml", "w") as f:
    yaml.dump(out, f, default_flow_style=None)
print("저장 완료: calib/ceiling_extrinsic.yaml")
print(T_cam2base)
EOF
```

생성 결과 예시:
```yaml
# calib/ceiling_extrinsic.yaml
T_cam_to_base:
- [ 0.998, -0.012,  0.054, -0.023]
- [ 0.013,  0.999, -0.032,  0.317]
- [-0.053,  0.033,  0.998,  0.842]
- [ 0.000,  0.000,  0.000,  1.000]
```

---

### 3. 손목캠 — hand-eye 캘리브레이션 불필요

**이유:**
`omy_f3m.urdf` 238번째 줄에 `camera_joint` 가 이미 정의돼 있다.

```
link6 (fixed) → camera_bottom_screw_frame
                  xyz="-0.021 -0.07385 0.0803"
                  rpy="-π/2 0 -π/2"
              → camera_link  (D405 본체)
```

bringup 을 켜면 TF2 가 이 체인을 자동 브로드캐스트한다.
따라서 `lookup_T_frame(node, rclpy, "link0", "camera_depth_optical_frame")` 하나로
카메라 → link0 변환을 실시간으로 얻을 수 있다.

**TF2 프레임 확인 (bringup 실행 중일 때):**
```bash
# 어떤 프레임이 있는지 확인
ros2 run tf2_tools view_frames
# → frames.pdf 생성. camera_link, camera_depth_optical_frame 등이 link6 아래 있어야 정상

# 또는 직접 조회
ros2 run tf2_ros tf2_echo link0 camera_depth_optical_frame
```

**만약 camera_depth_optical_frame 이 없으면:**
```bash
# camera_link 로 대체
python3 motion2/scripts/jaewoo/camera/wrist_detector.py --ros \
    --intrinsics calib/d405_intrinsics.yaml \
    --cam-frame camera_link
```

---

## 사용법

### Step 1. 모듈 독립 테스트 (ROS2 없음)

#### 좌표 변환 유틸 확인
```bash
cd /root/ros2_ws/src/open_manipulator

python3 motion2/scripts/jaewoo/camera/coord_transform.py \
    --extrinsic motion2/scripts/jaewoo/camera/calib/ceiling_extrinsic.yaml
# → 이미지 중심 픽셀 (320, 240), depth=0.8m 가 link0 어디인지 출력
```

#### 천장캠 — 이미지 파일 테스트 (카메라 없어도 됨)
```bash
# 테스트 이미지로 detection 확인 (depth 없으므로 xy 변환은 안 됨, 시각화만)
python3 motion2/scripts/jaewoo/camera/ceiling_detector.py \
    --model /root/yolov8n-seg.onnx \
    --extrinsic motion2/scripts/jaewoo/camera/calib/ceiling_extrinsic.yaml \
    --intrinsics motion2/scripts/jaewoo/camera/calib/d435i_intrinsics.yaml \
    --source /path/to/test_image.jpg
```

#### 천장캠 — RealSense D435i 실시간
```bash
python3 motion2/scripts/jaewoo/camera/ceiling_detector.py \
    --model /root/yolov8n-seg.onnx \
    --extrinsic motion2/scripts/jaewoo/camera/calib/ceiling_extrinsic.yaml \
    --intrinsics motion2/scripts/jaewoo/camera/calib/d435i_intrinsics.yaml \
    --source 0
# → OpenCV 창 팝업. q 로 종료
# → 터미널에 box xy (link0), yaw 출력
```

#### 손목캠 — RealSense D405 (PCA only, ROS2 없음)
```bash
# hand-eye yaml 불필요 — intrinsics 만 있으면 됨
python3 motion2/scripts/jaewoo/camera/wrist_detector.py \
    --intrinsics motion2/scripts/jaewoo/camera/calib/d405_intrinsics.yaml \
    --pca-only
# → OpenCV 창에 depth PCA + yaw 화살표 시각화
# (단독 실행: TF2 없으므로 xy 변환 불가. link0 좌표는 --ros 모드만 가능)
```

---

### Step 2. ROS2 노드로 실행 (Docker 내부)

**터미널 1 — bringup**
```bash
docker exec -it open_manipulator bash
cd /root/ros2_ws && source install/setup.bash
ros2 launch open_manipulator_bringup omy_f3m.launch.py
```

**터미널 2 — 천장캠 노드**
```bash
docker exec -it open_manipulator bash
cd /root/ros2_ws/src/open_manipulator
python3 motion2/scripts/jaewoo/camera/ceiling_detector.py --ros \
    --model /root/yolov8n-seg.onnx \
    --extrinsic  motion2/scripts/jaewoo/camera/calib/ceiling_extrinsic.yaml \
    --intrinsics motion2/scripts/jaewoo/camera/calib/d435i_intrinsics.yaml \
    --classes box,cell
# → /vision/box_coarse, /vision/cell_coarse 토픽 publish 시작
```

**터미널 3 — 손목캠 노드**
```bash
docker exec -it open_manipulator bash
cd /root/ros2_ws/src/open_manipulator
# hand-eye yaml 불필요 — URDF camera_joint → TF2 자동 처리
python3 motion2/scripts/jaewoo/camera/wrist_detector.py --ros \
    --intrinsics motion2/scripts/jaewoo/camera/calib/d405_intrinsics.yaml \
    --pca-only
# → /vision/box_fine 토픽 publish 시작
# (YOLO seg 도 쓰려면: --model /root/yolov8n-seg.onnx  --pca-only 제거)
```

**토픽 확인 (별도 터미널)**
```bash
# 천장캠 박스 좌표 확인
ros2 topic echo /vision/box_coarse

# 손목캠 박스 좌표 확인
ros2 topic echo /vision/box_fine

# 발행 주파수 확인
ros2 topic hz /vision/box_fine
```

출력 예시:
```
header:
  frame_id: link0
pose:
  position:
    x: 0.4512       ← link0 기준 박스 x [m]
    y: -0.1034      ← link0 기준 박스 y [m]
    z: 0.023        ← 박스 깊이 (참고용)
  orientation:
    w: 0.9998       ← z축 yaw 인코딩 (quat_from_z_yaw)
    x: 0.0
    y: 0.0
    z: 0.0174       ← sin(yaw/2) ≈ yaw ≈ 0.035rad ≈ 2°
```

---

### Step 3. RL grasp 와 연동

현재는 토픽을 받아서 수동으로 좌표를 넘기는 방식. (run_grasp_vision.py 완성 전)

```bash
# 1. 천장캠 토픽에서 좌표 읽기 (box_coarse 는 PoseArray=전체 박스, grasp 는 box_target=타깃 1개)
ros2 topic echo /vision/box_target --once
# → x=0.451, y=-0.103, z=0.35 (orientation 에서 yaw 계산)

# 2. 읽은 좌표로 grasp 실행 (dry-run 먼저)
python3 motion2/scripts/jaewoo/rl/grasp/run_grasp_full.py \
    --box-x 0.451 --box-y -0.103 --box-yaw 0.0

# 3. 확인 후 실행
python3 motion2/scripts/jaewoo/rl/grasp/run_grasp_full.py \
    --box-x 0.451 --box-y -0.103 --box-yaw 0.0 \
    --execute --confirm EXECUTE_GRASP \
    --max-joint-delta 3.0
```

orientation → yaw 변환 (필요 시):
```python
import math
# quat_wxyz = [w, x, y, z]
w, z = 0.9998, 0.0174
yaw = 2 * math.atan2(z, w)   # ≈ 0.035 rad
```

---

## 트러블슈팅

### "모델 파일 없음" 오류
```
FileNotFoundError: yolov8n-seg.onnx
```
→ PC에서 ONNX export 후 scp:
```bash
# PC에서
pip install ultralytics
yolo export model=yolov8n-seg.pt format=onnx imgsz=640
scp yolov8n-seg.onnx root@omy-SNPR44B1021.local:~/

# 로봇 호스트에서 Docker로 복사
docker cp ~/yolov8n-seg.onnx open_manipulator:/root/
```

### "체커보드를 찾지 못했습니다"
- 조명이 너무 어두거나 반사됨 → 간접 조명 사용
- 체커보드가 너무 작음 → 천장 높이 대비 A3 이상 크기 필요
- 카메라 초점 거리가 너무 멀면 패턴이 흐릿 → 1.5m 이내 권장

### "TF2 lookup timeout"
- bringup 이 실행 중인지 확인
- Docker 내부에서 실행하는지 확인 (호스트에서 실행 시 TF2 없음)

### depth PCA 결과가 불안정 (손목캠)
- `--max-depth` 값을 줄여 배경 포함 방지 (기본 0.6m)
- 박스가 충분히 가깝게 approach 된 상태에서만 호출
- 조명이 너무 밝으면 depth 노이즈 증가 → 간접광 사용

### minAreaRect yaw 가 90° 씩 튐
- 직사각형 박스의 180° 대칭 문제 (정상 동작)
- grasp 는 어느 방향이든 집을 수 있으므로 영향 없음
- insert 는 cell_yaw ≈ 0 이라 문제없음

---

## 파일 Windows → 로봇 복사

```bash
# 1. Windows → 로봇 호스트
scp -r "C:\Users\jaewoo\Desktop\캡스톤\capstone-isaaclab\motion2\scripts\jaewoo\camera" ^
    root@omy-SNPR44B1021.local:/root/open_manipulator/motion2/scripts/jaewoo/

# 2. 로봇 호스트 → Docker
docker cp /root/open_manipulator/motion2/scripts/jaewoo/camera \
    open_manipulator:/root/ros2_ws/src/open_manipulator/motion2/scripts/jaewoo/
```

---

## 구현 현황

| 모듈 | 상태 | 비고 |
|------|------|------|
| `coord_transform.py` | ✅ 완성 | 캘리브 yaml 필요 |
| `ceiling_detector.py` | ✅ 완성 | ONNX 모델 + 캘리브 yaml 필요 |
| `wrist_detector.py` | ✅ 완성 | hand-eye yaml 필요 |
| `calib/ceiling_extrinsic.yaml` | ⬜ 미완성 | 현장 체커보드 캘리브 필요 |
| `calib/wrist_handeye.yaml` | ✅ 불필요 | URDF camera_joint → TF2 자동 처리 |
| `calib/d435i_intrinsics.yaml` | ⬜ 미완성 | RealSense 에서 읽어서 저장 |
| `calib/d405_intrinsics.yaml` | ⬜ 미완성 | RealSense 에서 읽어서 저장 |
| `rl/grasp/run_grasp_vision.py` | ⬜ 미완성 | 비전 자동연동 스크립트 |
