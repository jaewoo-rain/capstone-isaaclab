# Camera Vision Pipeline — 설계 문서

## 목적

천장 카메라 + 손목 카메라 두 개를 연동해 박스 위치·yaw 를 추정하고,
기존 RL grasp/insert 파이프라인에 자동으로 좌표를 공급한다.

---

## 시스템 구성

```
천장캠 (D435i)                         손목캠 (D405)
    │ 1~5 Hz                               │ 10~15 Hz
    │ inst.seg → minAreaRect               │ depth ROI → PCA / minAreaRect
    ▼                                      ▼
/vision/box_coarse  (geometry_msgs/Pose)  /vision/box_fine  (geometry_msgs/Pose)
/vision/cell_coarse (geometry_msgs/Pose)
    │                                      │
    ▼                                      ▼
 모션 플래닝 (rough approach)          RL rollout (정밀 정렬)
 run_grasp_full.py ← box_xy, box_yaw ──────┘
```

---

## 카메라 사양

| 항목 | 천장캠 | 손목캠 |
|------|--------|--------|
| 모델 | Intel RealSense D435i | Intel RealSense D405 |
| 역할 | 박스·셀 초기 위치 (coarse) | 박스 정밀 위치·yaw (fine) |
| 주파수 | 1~5 Hz (한 번만 or 저주파) | 10~15 Hz (approach 후 활성화) |
| 좌표계 | static extrinsic → link0 | hand-eye → TF2 → link0 |
| 탑재 위치 | 천장 고정 | 로봇 손목 (link6 근처) |

---

## 파일 구조

```
camera/
├── PLAN.md                  ← 이 파일
├── __init__.py
├── coord_transform.py       ← 카메라 좌표 → link0 변환 공통 유틸
├── ceiling_detector.py      ← 천장캠: inst.seg + minAreaRect → Pose
├── wrist_detector.py        ← 손목캠: depth ROI + PCA → Pose
└── calib/
    ├── ceiling_extrinsic.yaml   ← 천장캠 외부 파라미터 (캘리브레이션 후 저장)
    └── wrist_handeye.yaml       ← 손목캠 hand-eye 결과 (easy_handeye2 출력)
```

---

## 좌표계 정의

- **link0** : 로봇 베이스 프레임 (모든 좌표의 기준)
- **link6** : EE (손목) 프레임
- **camera_color_optical_frame** : 카메라 픽셀 좌표 기준

변환 체인:
- 천장캠: `pixel + depth → cam3d → T_ceil2base → link0`
- 손목캠: `pixel + depth → cam3d → T_wrist2ee(hand-eye) → T_ee2base(TF2) → link0`

---

## 모듈별 역할

### `coord_transform.py`
- `pixel_to_cam3d(u, v, depth, K)` : 픽셀 + 깊이 → 카메라 3D 좌표
- `CeilingTransform` : static extrinsic yaml 로드, `cam3d_to_base(pt)`
- `WristTransform` : hand-eye yaml 로드 + TF2 조회, `cam3d_to_base(pt, node, rclpy)`

### `ceiling_detector.py`
- `CeilingDetector(model_path, extrinsic_yaml)` : YOLOv8n-seg (ONNX) 로드
- `detect(color_img, depth_img, K)` → `list[DetectionResult]`
  - `DetectionResult` : `xy_base [m]`, `yaw_base [rad]`, `label str`, `score float`
- ROS2 노드로 래핑 가능: `/vision/box_coarse`, `/vision/cell_coarse` publish

### `wrist_detector.py`
- `WristDetector(model_path, handeye_yaml)` : ONNX 로드
- `detect(color_img, depth_img, K, node, rclpy)` → `DetectionResult | None`
  - approach 후 박스가 뷰 중앙에 있을 때 호출
- approach 중 주기적 호출 → 갱신된 box_xy, box_yaw 로 rollout 재실행

---

## Detection 파이프라인 (공통)

```
color frame
    │
    ▼
YOLOv8n-seg (ONNX Runtime)          ← ONNX: ~2~6 fps on Pi5
    │  instance mask
    ▼
mask → findContours → minAreaRect   ← OpenCV, 즉시
    │  center_px, angle_deg
    ▼
center_px + depth[center_px]        ← depth frame lookup
    │  (u, v, d)
    ▼
pixel_to_cam3d                      ← K(intrinsics) 사용
    │  (X, Y, Z) in cam frame
    ▼
coord_transform (extrinsic/TF2)
    │
    ▼
(x, y) in link0 frame  +  yaw
```

---

## yaw 180° 대칭 처리

minAreaRect 반환 각도: `[-90°, 0°)` (OpenCV 규약)

| 용도 | 처리 |
|------|------|
| grasp box_yaw | `wrap_to_pi` 후 그대로 사용. grasp 정책은 ±80° 학습, 180° 대칭으로 어느 방향이든 집을 수 있음 |
| insert cell_yaw | 셀은 고정 구조물 → yaw ≈ 0 (±10°), 대칭 문제 없음 |
| 손목캠 시계열 | 직전 프레임 yaw 와 비교 → `abs(diff) > 90°` 이면 ±180° flip 후 사용 |

---

## 캘리브레이션 절차 (사전 필수)

### 천장캠 extrinsic (1회)
1. 체커보드(7×5, 30mm) 를 link0 원점 위에 고정
2. `rs-enumerate-devices` 로 intrinsics 확인
3. PnP 풀어서 `T_cam_to_base` 4×4 행렬 저장 → `calib/ceiling_extrinsic.yaml`

```yaml
# ceiling_extrinsic.yaml 예시
T_cam_to_base:
  - [r00, r01, r02, tx]
  - [r10, r11, r12, ty]
  - [r20, r21, r22, tz]
  - [0,   0,   0,   1 ]
```

### 손목캠 hand-eye (1회)
1. ROS2 `easy_handeye2` 패키지 사용
2. `ros2 launch easy_handeye2 calibrate.launch.py ...`
3. 결과 → `calib/wrist_handeye.yaml` (easy_handeye2 포맷 그대로)

---

## ROS2 토픽 규약

| 토픽 | 메시지 타입 | 발행자 | 내용 |
|------|------------|--------|------|
| `/vision/box_coarse` | `geometry_msgs/PoseArray` | ceiling_detector | 천장캠 **모든** 박스 pose (link0) |
| `/vision/box_target` | `geometry_msgs/PoseStamped` | ceiling_detector | grasp 타깃 1개(분포중심 최근접) |
| `/vision/cell_coarse` | `geometry_msgs/PoseArray` | ceiling_detector | 천장캠 **모든** 셀 pose (link0) |
| `/vision/box_fine` | `geometry_msgs/PoseStamped` | wrist_detector | 손목캠 박스 pose (link0) |

> 다중 박스 대응: coarse 는 PoseArray(전체 보존), grasp 는 `box_target`(분포중심 `(0.45,-0.10)` 최근접) 구독.

Pose 규약:
- `position.x/y` : link0 기준 xy [m]
- `position.z` : 검출된 물체 z (참고용)
- `orientation` : `quat_from_z_yaw(yaw)` — z축 yaw 만 인코딩

---

## 통합 실행 순서 (최종 목표)

```bash
# 1. ROS2 bringup (Docker 내부)
ros2 launch open_manipulator_bringup omy_f3m.launch.py

# 2. 천장캠 detector 노드 (Docker 내부)
python3 motion2/scripts/jaewoo/camera/ceiling_detector.py \
    --model yolov8n-seg.onnx \
    --extrinsic camera/calib/ceiling_extrinsic.yaml

# 3. 손목캠 detector 노드 (Docker 내부)
python3 motion2/scripts/jaewoo/camera/wrist_detector.py \
    --model yolov8n-seg.onnx \
    --handeye camera/calib/wrist_handeye.yaml

# 4. RL grasp 실행 (비전 연동 버전 — 추후 구현)
python3 motion2/scripts/jaewoo/rl/grasp/run_grasp_vision.py \
    --execute --confirm EXECUTE_GRASP
```

---

## 구현 순서 (단계별)

| Phase | 파일 | 내용 | 선행 조건 |
|-------|------|------|----------|
| P1 | `coord_transform.py` | 좌표 변환 유틸 (캘리브 없이 테스트 가능) | 없음 |
| P2 | `ceiling_detector.py` | 천장캠 detection + link0 변환 | P1, 캘리브 yaml |
| P3 | (검증) | 천장캠 좌표 → `run_grasp_full.py --dry-run` 으로 확인 | P2, ONNX 모델 |
| P4 | `wrist_detector.py` | 손목캠 detection (hand-eye 필요) | P1, P4 캘리브 |
| P5 | `rl/grasp/run_grasp_vision.py` | 비전 연동 grasp 통합 스크립트 | P2, P4 |
| P6 | insert 연동 | run_insert_vision.py | P5 완료 후 |

---

## 의존성 (Pi 5 설치 목록)

```bash
# ONNX Runtime (aarch64)
pip install onnxruntime

# OpenCV
pip install opencv-python-headless

# RealSense SDK Python binding
pip install pyrealsense2

# NumPy (이미 설치돼 있을 가능성 높음)
pip install numpy

# YOLOv8 — 추론만 필요하면 onnxruntime 로 충분
# (ultralytics 패키지는 Pi5에서 무거움 → ONNX export 후 사용 권장)
# PC에서: yolo export model=yolov8n-seg.pt format=onnx
```

---

## 현재 상태 (TODO)

- [x] PLAN.md 작성
- [ ] `coord_transform.py` 구현
- [ ] `ceiling_detector.py` 구현
- [ ] `wrist_detector.py` 구현
- [ ] 천장캠 extrinsic 캘리브레이션 (로봇 현장)
- [ ] 손목캠 hand-eye 캘리브레이션 (로봇 현장)
- [ ] ONNX 모델 준비 (PC에서 export → scp)
- [ ] `run_grasp_vision.py` 구현
