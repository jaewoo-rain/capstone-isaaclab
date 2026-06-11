# TODO — 다음에 할 것들

최종 목표: 천장캠 좌표 → 모션플래닝 → 손목캠 정밀 보정 → RL grasp → lift → insert 자동화

---

## 🔴 즉시 필요 (로봇 실행 전 필수)

### 1. 로봇 bringup 정상화
- [ ] Docker 내부에서 bringup 이 `stack smashing detected` 없이 켜지는지 확인
  ```bash
  docker exec -it open_manipulator bash
  cd /root/ros2_ws && source install/setup.bash
  ros2 launch open_manipulator_bringup omy_f3m.launch.py
  ```
- [ ] 켜지면 `/joint_states` 토픽 수신 확인
  ```bash
  ros2 topic echo /joint_states
  ```

### 2. 파일 복사 (Windows → 로봇)
```bash
# camera 폴더 전체 복사
scp -r "C:\Users\jaewoo\Desktop\캡스톤\capstone-isaaclab\motion2\scripts\jaewoo\camera" ^
    root@omy-SNPR44B1021.local:/root/open_manipulator/motion2/scripts/jaewoo/

docker cp /root/open_manipulator/motion2/scripts/jaewoo/camera \
    open_manipulator:/root/ros2_ws/src/open_manipulator/motion2/scripts/jaewoo/

# rl 폴더도 같이 (run_grasp_full.py 최신 버전 포함)
scp -r "C:\Users\jaewoo\Desktop\캡스톤\capstone-isaaclab\motion2\scripts\jaewoo\rl" ^
    root@omy-SNPR44B1021.local:/root/open_manipulator/motion2/scripts/jaewoo/

docker cp /root/open_manipulator/motion2/scripts/jaewoo/rl \
    open_manipulator:/root/ros2_ws/src/open_manipulator/motion2/scripts/jaewoo/
```

---

## 🟡 카메라 캘리브레이션 (1회성 작업)

### 3. 카메라 내부 파라미터 저장 (intrinsics)
bringup 없이도 가능. 카메라 USB 꽂힌 상태에서:
```bash
# Docker 내부
python3 - <<'EOF'
import pyrealsense2 as rs, yaml, pathlib

for name, w, h, fps in [("d435i", 1280, 720, 30), ("d405", 640, 480, 90)]:
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    profile = pipeline.start(cfg)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pipeline.stop()
    data = {"fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy}
    path = f"motion2/scripts/jaewoo/camera/calib/{name}_intrinsics.yaml"
    pathlib.Path(path).parent.mkdir(exist_ok=True)
    yaml.dump(data, open(path, "w"))
    print(f"{name}: {data}")
EOF
```
- [ ] `calib/d435i_intrinsics.yaml` 생성 확인
- [ ] `calib/d405_intrinsics.yaml` 생성 확인

### 4. 천장캠 외부 파라미터 측정 (extrinsic)
bringup 불필요. 체커보드(7×5, 30mm 격자) 준비 필요.
→ `camera/README.md` "천장캠 외부 파라미터" 섹션 스크립트 실행
- [ ] 체커보드를 link0 원점(로봇 발 위) 에 수평으로 고정
- [ ] 스크립트 실행 → `calib/ceiling_extrinsic.yaml` 생성
- [ ] 검증: `python3 coord_transform.py --extrinsic calib/ceiling_extrinsic.yaml`

### 5. TF2 손목캠 프레임 확인
bringup 켜진 상태에서:
```bash
ros2 run tf2_tools view_frames
# frames.pdf 에서 camera_depth_optical_frame 이 link6 아래 있는지 확인

# 없으면 어떤 프레임이 있는지 확인
ros2 topic echo /tf --once
```
- [ ] `camera_depth_optical_frame` 또는 `camera_link` 중 사용할 프레임명 확인
- [ ] 필요 시 `wrist_detector.py --ros --cam-frame <프레임명>` 으로 변경

---

## 🟡 YOLO 모델 준비

### 6. PC에서 ONNX export
```bash
# PC (Windows) — ultralytics 설치된 환경
pip install ultralytics
yolo export model=yolov8n-seg.pt format=onnx imgsz=640
```
- [ ] `yolov8n-seg.onnx` 생성
- [ ] scp 로 로봇 전송:
  ```bash
  scp yolov8n-seg.onnx root@omy-SNPR44B1021.local:~/
  docker cp ~/yolov8n-seg.onnx open_manipulator:/root/
  ```

### 7. Pi 5 에 의존성 설치
```bash
# Docker 내부
pip install onnxruntime opencv-python-headless pyrealsense2 pyyaml
```
- [ ] 설치 완료 확인

---

## 🟢 단계별 테스트

### 8. 카메라 단독 테스트 (detection 확인)
```bash
# 천장캠 — 박스/셀 검출되는지 확인
python3 motion2/scripts/jaewoo/camera/ceiling_detector.py \
    --model /root/yolov8n-seg.onnx \
    --extrinsic motion2/scripts/jaewoo/camera/calib/ceiling_extrinsic.yaml \
    --intrinsics motion2/scripts/jaewoo/camera/calib/d435i_intrinsics.yaml \
    --source 0

# 손목캠 — PCA yaw 확인 (TF2 없어도 됨)
python3 motion2/scripts/jaewoo/camera/wrist_detector.py \
    --intrinsics motion2/scripts/jaewoo/camera/calib/d405_intrinsics.yaml \
    --pca-only
```
- [ ] 천장캠: 박스 bbox + 마스크 + yaw 화살표 표시 확인
- [ ] 손목캠: 박스 근처에서 depth PCA yaw 안정적으로 나오는지 확인

### 9. RL grasp dry-run (카메라 좌표로)
천장캠 토픽에서 좌표 읽어서 수동으로 입력:
```bash
# bringup + 천장캠 노드 켠 상태에서
ros2 topic echo /vision/box_coarse --once
# x, y, yaw 읽기

python3 motion2/scripts/jaewoo/rl/grasp/run_grasp_full.py \
    --box-x <x> --box-y <y> --box-yaw <yaw>
# dry-run 으로 rollout 수렴 확인
```
- [ ] rollout 수렴 (status=converged) 확인
- [ ] `--execute --confirm EXECUTE_GRASP` 붙여서 실제 실행

### 10. 손목캠 → RL 정밀 보정 테스트
approach 완료 후 손목캠 좌표로 rollout 재실행:
```bash
ros2 topic echo /vision/box_fine --once
python3 motion2/scripts/jaewoo/rl/grasp/run_grasp_full.py \
    --box-x <fine_x> --box-y <fine_y> --box-yaw <fine_yaw> \
    --approach-noise 0.0   # 이미 가까이 있으므로 노이즈 0
```
- [ ] 손목캠 좌표가 천장캠보다 정확한지 비교

---

## 🔵 추후 구현 (코드 작업 필요)

### 11. `run_grasp_vision.py` 작성
비전 토픽을 자동으로 구독해서 좌표를 받아 RL 실행하는 통합 스크립트.
현재 없음 — `rl/grasp/run_grasp_vision.py` 로 작성 예정.

### 12. Insert 비전 연동
`rl/insert/run_insert_vision.py` — 셀 좌표 자동 감지 + insert RL.

### 13. 전체 파이프라인 통합
`run_full_pipeline.py` — grasp → lift → (turn) → insert 한 번에.

---

## 📋 현재 파일 상태

| 파일 | 상태 |
|------|------|
| `camera/PLAN.md` | ✅ 완성 |
| `camera/coord_transform.py` | ✅ 완성 |
| `camera/ceiling_detector.py` | ✅ 완성 |
| `camera/wrist_detector.py` | ✅ 완성 (hand-eye 불필요) |
| `camera/README.md` | ✅ 완성 |
| `rl/grasp/run_grasp_full.py` | ✅ 완성 |
| `rl/grasp/run_grasp.py` | ✅ 완성 |
| `rl/insert/run_insert.py` | ✅ 완성 |
| `calib/d435i_intrinsics.yaml` | ⬜ 현장 측정 필요 |
| `calib/d405_intrinsics.yaml` | ⬜ 현장 측정 필요 |
| `calib/ceiling_extrinsic.yaml` | ⬜ 현장 측정 필요 |
| `calib/wrist_handeye.yaml` | ✅ 불필요 (URDF 활용) |
| `rl/grasp/run_grasp_vision.py` | ⬜ 미작성 |
| `rl/insert/run_insert_vision.py` | ⬜ 미작성 |
