# HANDOFF — motion2/scripts/jaewoo (다음 Claude가 먼저 읽는 문서)

> 이 폴더는 **OMY-F3M 실제 로봇**에서 sim2real 로 pick→place→insert 를 수행하는 코드다.
> 최종 목표: **천장캠 좌표 → 모션플래닝 → 손목캠 정밀보정 → RL grasp → lift → insert 자동화.**
> 이 문서 하나로 "무엇이 되어 있고 / 다음에 무엇을 / 어떤 명령으로" 가 다 파악되게 썼다.
> 더 깊은 내용은 각 폴더 README 로 링크해 둠.

---

## 0. 한 장 요약 (현재 상태)

| 영역 | 상태 |
|------|------|
| Cartesian 제어(show_ee_pose/run_ee_pose_move/run_pick_place/go_to_*/toggle_gripper) | ✅ 실물 검증됨 |
| RL grasp (`rl/grasp/`) | ✅ 롤아웃 검증, 자세 매핑은 실물 미검증(아래 §6) |
| RL insert / 전체 chain (`rl/insert/`, `rl/all/`) | 🟡 청사진·롤아웃만, 실물 미검증 |
| 카메라 코드 (`camera/`) | ✅ 코드 완성+버그수정. 캘리브 yaml·커스텀 모델은 현장 필요 |
| YOLO box/cell 학습 파이프라인 (`camera/dataset/`, `camera/train/`) | ✅ 도구 완성·검증. 실측 데이터만 있으면 학습 가능 |
| 비전→RL 자동연동 | ✅ `result/grasp/`(self-contained, S1~S4 fine 반복수렴). insert/chain은 ⬜ |
| 전체 아키텍처 도식 | ✅ `ARCHITECTURE.md` (카메라 값·토픽계약·흐름) |

**모든 ROS2/RL/카메라는 Pi5 Docker 컨테이너(`open_manipulator`) 안에서 실행한다.**
학습(YOLO)만 이 리눅스 데스크 GPU(`env_isaaclab`)에서 한다.

---

## 1. 이번 세션에 한 일 (2026-06-11~12)

1. **카메라 코드 점검 → 위험 버그 3개 수정** (로봇 없이 단위검증, `camera/`):
   - 천장캠 yaw 가 link0 가 아니라 이미지프레임 값이던 버그 → `coord_transform.yaw_cam_to_base()` 신설
     (optical 방향벡터를 extrinsic 회전으로 돌려 base 투영, **Y-down 부호**까지 처리). ceiling·wrist 통일.
   - minAreaRect 90° 모호성(긴 변에 정렬해 파지 실패) → `ceiling_detector.rect_long_axis_yaw()` (긴 변 결정론적).
   - TF2 buffer 를 매 프레임 새로 만들던 것(손목캠 노드 timeout) → `coord_transform._ensure_tf_buffer()` 1회 캐시.
2. **캘리브 스크립트 작성** (`camera/dump_intrinsics.py`, `camera/calibrate_ceiling_extrinsic.py`).
3. **YOLO box/cell 학습 파이프라인 설계·구축** (`camera/dataset/`, `camera/train/`):
   - 실측 + **depth 자동라벨**(박스) + **고정 ROI 템플릿**(셀). 박스가 광택금속이라 **depth-seed→FastSAM**(`--method sam`) 추가.
   - 데스크 GPU 학습 + ONNX export. **ONNX 출력이 현 detector 와 호환됨을 실측 검증**(nc=2 → `(1,38,8400)`/`(1,32,160,160)`).
4. **detector 다중박스/셀 대응(PHASE F)** — NMS 추가, 토픽을 PoseArray+target 으로, 셀 yaw=0 분기, 손목캠 box-only.

**확정된 사실**(다음 Claude가 다시 묻지 말 것):
- 실행 위치: 전부 Pi Docker. 학습만 데스크 GPU.
- **박스·셀 둘 다 광택 금속**(셀 = 유압블록, 구멍 있는 직육면체). → 반사 대응 필수.
- 다중 박스 중 grasp 타깃 = **분포중심 (0.45,-0.10) 최근접**.
- 모델 = YOLOv8n-seg, 데이터 = 실측+depth/FastSAM 자동라벨.

---

## 2. ⚠️ 다음 Claude가 반드시 아는 함정

1. **천장캠 extrinsic 회전 정확도 = 박스 yaw 정확도.** `yaw_cam_to_base` 가 extrinsic 회전으로 yaw 를
   계산하므로, 체커보드를 link0 **축까지** 맞춰 캘리브해야 한다(안 그러면 상수 yaw 오차).
2. **광택 금속.** RealSense depth 가 정반사로 깨진다. 박스 라벨은 `--method sam`. 💡 박스 파지 안 하는 면에
   **무광 테이프/스프레이** 한 겹이면 depth·비전 난이도 급감 — 가장 효과 큰 물리 조치.
3. **토픽 계약**: grasp 는 `/vision/box_target`(단일 PoseStamped) 구독. `/vision/box_coarse`·`/vision/cell_coarse`
   는 PoseArray(전체). 미작성 `run_grasp_vision.py` 도 `box_target` 전제.
4. **start_yaw=0.0 가정** (`rl/grasp/run_grasp.py:220`): 롤아웃이 "실물은 수직파지 자세에서 시작"을 가정만 함.
   grasp 전에 반드시 수직파지로 pre-position 할 것.
5. **커스텀 모델 필수**: 기본 `yolov8n-seg.pt`(COCO)는 box/cell 못 잡음. §5 로 학습해야 함.
6. **Docker 에 RL deps**(sb3/torch) 가 있어야 롤아웃이 돈다(없으면 설치, aarch64라 김).
7. (미수정, 무해) 발행 Pose 의 `z` 는 카메라 depth(거리)지 link0 z 가 아님. RL 은 z 무시라 현재 무해.

---

## 3. 다음에 할 일 — 데스크에서 (로봇 없이 가능)

- [x] **`result/grasp/run_grasp_vision.py` ✅** (S1~S4 단일프로세스, **self-contained 폴더**): S1 box_target coarse
      1회 → S2 hover_z 이동 → S3 box_fine **fine 반복수렴** → S4 하강·close·lift. 의존 모듈·checkpoints·YAML 을
      `result/grasp/` 안에 복사(폴더 통째로 배포). `--dry-coords X Y YAW` 로 비전 없이 배선테스트(검증됨).
      검토 완료(안전+로직): 발산 target 이동 금지 수정. **현장 전 hover_z 실측 필요**(§8 미지수).
- [x] **`result/insert/run_insert_vision.py` ✅** (I1~I3, self-contained, ⚠️**실험적**): I1 cell_coarse(PoseArray)→
      `--cell-index` 슬롯선택 → I2 insert RL yaw-only 정렬(align) → I3 (옵션 `--place-z`) 하강·release·복귀.
      앞단 grasp→turn 은 전제(이 스크립트 밖). 안전검토 반영: z-min 0.10 복원 + place-z 하한 + **하강 직전
      EE-셀 드리프트 가드**(place 드리프트 충돌 방어). **실물 첫 실행은 align-only(`--place-z` 미지정)** 권장.
- [ ] **`result/pipeline/`** (전체 chain grasp→turn→insert→place 자동). grasp·insert 실물 검증 후. `ARCHITECTURE.md` §5.
- [ ] (선택) YOLO 모델 학습 데이터가 생기면 §5 학습 — 데스크 GPU.

## 4. 다음에 할 일 — 현장 런북 (로봇 앞에서, 순서대로)

**머신**: [PC]=개발PC(윈도우) / [HOST]=`ssh root@omy-SNPR44B1021.local` / [DOCKER]=`docker exec -it open_manipulator bash`
[DOCKER] 안에서 항상 먼저: `cd /root/ros2_ws/src/open_manipulator && source /root/ros2_ws/install/setup.bash`
동시 터미널: T1 bringup / T2 천장캠 / T3 손목캠 / T4 명령.

### PHASE 0 — 최신 코드 컨테이너로 (이번에 camera/rl 바뀜, 필수)
```bash
# [PC] → [HOST]   (최신본이 데스크에만 있으면 데스크에서 scp)
scp -r "<jaewoo>\camera" "<jaewoo>\rl"  root@omy-SNPR44B1021.local:/root/open_manipulator/motion2/scripts/jaewoo/
# [HOST] → [DOCKER]
docker cp /root/open_manipulator/motion2/scripts/jaewoo/camera open_manipulator:/root/ros2_ws/src/open_manipulator/motion2/scripts/jaewoo/
docker cp /root/open_manipulator/motion2/scripts/jaewoo/rl     open_manipulator:/root/ros2_ws/src/open_manipulator/motion2/scripts/jaewoo/
```

### PHASE 1 — 의존성
```bash
pip install onnxruntime opencv-python-headless pyrealsense2 numpy pyyaml
python3 -c "import stable_baselines3, torch, gymnasium; print('rl deps OK')"  # 실패시 pip install 'stable-baselines3==2.7.1' torch gymnasium
```

### PHASE 2 — bringup (T1, 켜둠)
```bash
ros2 launch open_manipulator_bringup omy_f3m.launch.py
ros2 topic echo /joint_states --once      # 수신 확인
```

### PHASE 3 — 카메라 캘리브 (1회)
```bash
python3 motion2/scripts/jaewoo/camera/dump_intrinsics.py --name d435i
python3 motion2/scripts/jaewoo/camera/dump_intrinsics.py --name d405
# 체커보드를 link0 원점·축정렬·수평 고정 후 (★ §2-1)
python3 motion2/scripts/jaewoo/camera/calibrate_ceiling_extrinsic.py --show   # 재투영오차<~1px
python3 motion2/scripts/jaewoo/camera/coord_transform.py --check-tf           # 손목캠 TF2 확인(bringup 필요)
```

### PHASE 4 — YOLO 모델 배치 (학습은 §5)
```bash
# [PC] export 후:  scp best.onnx root@...:~/  →  [HOST] docker cp ~/best.onnx open_manipulator:/root/yolov8n-seg.onnx
```
모델 없으면: 손목캠은 `--pca-only`(모델 불필요), 천장캠은 박스를 알려진 위치에 놓고 좌표 수동입력(PHASE 6-0).

### PHASE 5 — detection 단독 테스트
```bash
# T2 천장캠
python3 motion2/scripts/jaewoo/camera/ceiling_detector.py --ros --model /root/yolov8n-seg.onnx \
    --extrinsic  motion2/scripts/jaewoo/camera/calib/ceiling_extrinsic.yaml \
    --intrinsics motion2/scripts/jaewoo/camera/calib/d435i_intrinsics.yaml --classes box,cell
ros2 topic echo /vision/box_target --once     # grasp 타깃 1개. yaw=2*atan2(qz,qw)
# T3 손목캠 (모델 없이 PCA)
python3 motion2/scripts/jaewoo/camera/wrist_detector.py --ros \
    --intrinsics motion2/scripts/jaewoo/camera/calib/d405_intrinsics.yaml --pca-only
ros2 topic hz /vision/box_fine                # 끊김 없어야(TF2 수정 효과)
```

### PHASE 6 — RL grasp (★ 단계별, 비상정지 손 닿는 곳에)
```bash
# 6-0 박스좌표: box_target echo 또는 (모델없을때) show_ee_pose 로 측정. grasp 전 박스 근처 수직파지로 이동:
python3 motion2/scripts/jaewoo/run_ee_pose_move.py --x 0.45 --y -0.10 --z 0.3724 \
    --execute --confirm EXECUTE_EE_POSE_MOVE --no-constrain-joint5 --max-joint-delta 3.0
# 6-1 dry-run
python3 motion2/scripts/jaewoo/rl/grasp/run_grasp.py --box-x <x> --box-y <y> --box-yaw 0.0
# 6-2 yaw 끄고 실제 — 자세검증(가장 중요): 그리퍼가 박스를 양옆에서 감싸는지
python3 motion2/scripts/jaewoo/rl/grasp/run_grasp.py --box-x <x> --box-y <y> --box-yaw 0.0 \
    --no-align-yaw --execute --confirm EXECUTE_GRASP --no-constrain-joint5 --max-joint-delta 3.0
# 6-3 yaw 정렬 켜고 (박스 20° 비스듬히)
python3 motion2/scripts/jaewoo/rl/grasp/run_grasp.py --box-x <x> --box-y <y> --box-yaw 0.35 \
    --execute --confirm EXECUTE_GRASP --no-constrain-joint5 --max-joint-delta 3.0
```
자세 틀리면 6-2 에서 멈추고 `rl/grasp/config.py` 의 `GRIP_CENTER_OFFSET_XY`/orientation 매핑 조정.

### PHASE 7 — 그 다음
손목캠 정밀보정 연동 → `rl/insert/run_insert.py`(셀 뒤쪽 `--x-min -0.55`, README 한계 3가지) →
`rl/all/run_pipeline.py --stop-after grasp_lift|turn|insert|place`.

### 빠른 복귀/안전
```bash
python3 motion2/scripts/jaewoo/go_to_start.py --execute --confirm GO_TO_START
python3 motion2/scripts/jaewoo/toggle_gripper.py --execute --confirm TOGGLE_GRIPPER
```
모든 스크립트 **기본 dry-run**. `--execute --confirm <TEXT>` 둘 다 있어야 움직인다.

---

## 5. YOLO box/cell 모델 학습 (이 데스크 GPU)

도구는 다 만들어져 검증됨. 실측 데이터만 있으면 됨. (`camera/dataset/README.md`, `camera/train/*.md` 에 상세)

```bash
# A. 캡처 (카메라 연결된 머신. 박스 광택면 무광처리+확산광 권장)
python3 camera/dataset/capture_dataset.py --cam d435i --n 80 --trigger key   # 천장캠 60~100장
python3 camera/dataset/capture_dataset.py --cam d405  --n 40 --trigger key   # 손목캠 30~50장
# B. 셀 템플릿 1회 (천장캠 대표 프레임)
python3 camera/dataset/annotate_cells.py --frame dataset/raw/d435i/0000_color.png
# C. 자동 라벨 → 데이터셋 (박스=광택금속이므로 --method sam)
python3 camera/dataset/auto_label_depth.py --raw dataset/raw/d435i \
    --cell-template dataset/cell_template.yaml --method sam --show   # --show 로 검수
python3 camera/dataset/auto_label_depth.py --raw dataset/raw/d405 --method sam
# D. 학습 (데스크 GPU, env_isaaclab)
conda activate env_isaaclab && bash camera/train/train.sh
# E. export + Pi 전송
bash camera/train/export.sh
python3 camera/train/verify_onnx.py <onnx>     # _YoloSegONNX 로드 스모크
```
첫 실측 후 `--height-thresh`(depth), `--sam-conf/--sam-iou`(FastSAM) 1회 미세조정 필요.

---

## 6. 컴포넌트별 상세 위치

| 무엇 | 어디 | 상세문서 |
|------|------|----------|
| Cartesian 제어 스크립트 | `*.py` (루트) | `README.md` |
| RL grasp sim2real | `rl/grasp/` | `rl/grasp/README.md` |
| RL insert (yaw-only) | `rl/insert/` | `rl/insert/README.md` (단독실행 3대 한계 필독) |
| RL 전체 chain | `rl/all/` | `rl/all/README.md` (실물 미검증 청사진) |
| 카메라 비전 코드 | `camera/*.py` | `camera/README.md`, `camera/PLAN.md` |
| 캘리브 스크립트 | `camera/dump_intrinsics.py`, `camera/calibrate_ceiling_extrinsic.py` | (스크립트 docstring) |
| YOLO 데이터셋 도구 | `camera/dataset/` | `camera/dataset/README.md` |
| YOLO 학습/export | `camera/train/` | `camera/train/{setup,train,export}.md` |
| **배포 실행 + 모델 핸드오프** | `result/README.md` | 팀원(모델)·현장(SSH 실행) 한 문서 |
| 프로젝트 설계 컨텍스트 | `CLAUDE.md` | — |

**핵심 미검증 리스크 (실물 확인 필요)**: grasp 자세 매핑(sim `(0,1,0,0)` vs 실물 수직파지 quat — top-down vs
side 가능성, `rl/grasp/README.md` §2), insert place 하강 xy 드리프트(sim 도 미해결), 뒤쪽 셀 좌표계.
