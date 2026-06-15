# result/ — 배포 실행 가이드 (모델 핸드오프 + 현장 실행)

카메라→YOLO→RL→실물 파이프라인을 **실제로 돌리는** 폴더. 이 문서 하나로
**① 팀원이 모델을 어떻게 뽑아 무엇을 전달하는지**, **② 모델이 무엇을 감지해 어떤 값을
어디로 보내는지**, **③ 내가 현장에서 무엇을 어떤 명령으로 실행하는지** 가 다 파악되게 썼다.

> 더 깊은 내용: 설계 `../ARCHITECTURE.md`, 학습 도구 `../camera/dataset/`·`../camera/train/`,
> 전체 현황 `../HANDOFF.md`.

---

## 0. 큰 그림 (한 장)

```
[팀원이 만듦]                         [내가 현장에서 실행]
 YOLO 모델(.onnx) ──전달──▶  ceiling_detector / wrist_detector ──▶ /vision/* 토픽 ──▶ result/ 파이프라인 ──▶ 로봇
                            (카메라 USB→Pi, YOLO 추론)            (좌표 x,y,yaw)      (RL→MoveIt→FollowJointTrajectory)
```
- **전부 로봇 컴퓨터(Pi)의 Docker 컨테이너 `open_manipulator` 안에서** 실행.
- **노트북 = SSH 터미널 역할만** (계산·카메라·ROS2 전부 Pi). 카메라는 **Pi USB**에 꽂는다.

---

# PART ① — 팀원에게 (모델 만드는 사람)

## 1-1. 무엇을 감지하는 모델인가
**2-클래스 instance segmentation (YOLOv8-seg):**
| 클래스 id | 이름 | 정체 | 비고 |
|---|---|---|---|
| 0 | `box` | 로봇이 **잡을** 직육면체 박스(약 4.4×8cm) | 광택 금속 — 반사 주의 |
| 1 | `cell` | 적재할 **유압블록 슬롯** | 광택 금속, 고정 구조물 |

> ⚠️ **기본 `yolov8n-seg.pt`(COCO)는 box/cell 을 못 잡는다.** 반드시 box/cell 로 **커스텀 학습**한 모델이어야 함.

## 1-2. 데이터·학습 (요약)
실측 RealSense 이미지 + **depth/FastSAM 자동 라벨**(박스 광택 금속이라 `--method sam`) → 이 데스크 GPU 학습.
상세 절차·명령은 `../camera/dataset/README.md`, `../camera/train/{train,export}.md` 참고.

## 1-3. ★ ONNX export — 반드시 이대로 뽑아주세요
학습 끝난 `best.pt` 를 **ONNX 로 변환**해서 전달:
```bash
yolo export model=<runs/.../weights/best.pt> format=onnx imgsz=640 opset=12 nms=False dynamic=False half=False
```
**고정 제약 (어기면 추론 코드가 못 읽거나 라벨이 뒤바뀜):**
- `imgsz=640`, `nms=False` — detector 가 letterbox 640 전제 + conf필터/NMS 를 직접 함.
- **클래스 순서 `names: [box, cell]` (box=0, cell=1) 절대 고정** — `data.yaml` 의 names 순서가 곧 onnx cls_id.
- 출력 형상이 이래야 함(2클래스 기준): `output0 (1, 38, 8400)` = 4+2(클래스)+32(mask),
  `output1 (1, 32, 160, 160)`, 입력 `images (1,3,640,640)`.

## 1-4. 전달물 (이 3개만 주면 됨)
1. **`best.onnx`** 파일.
2. **클래스 순서 확인 문구**: "names = [box, cell] (box=0, cell=1)".
3. (선택) 검증 통과 여부 — 전달 전 한 번 돌려보면 좋음:
   ```bash
   python3 ../camera/train/verify_onnx.py best.onnx   # _YoloSegONNX 로 로드·추론 스모크
   ```

---

# PART ② — 모델이 내는 값 / 어디로 가는가 (데이터 계약)

## 2-1. 감지 → 어떤 값으로
detector 가 마스크 → minAreaRect(긴 변) → extrinsic/TF2 변환을 거쳐 **link0 기준 평면 pose** 로 만든다.
| 대상 | 내는 값 | 안 내는 값 |
|---|---|---|
| box | `x, y` [m], `yaw` [rad] | z(=YAML/실측), roll/pitch(=수직파지 고정) |
| cell | `x, y` [m] | yaw(고정구조물 → 0), z |

> z 는 vision 이 **안 준다.** 호버·파지·하강 높이는 YAML/실측에서. (Pose.z 엔 카메라 depth 가 참고로 실리지만 무시)

## 2-2. 어디로 보내나 (ROS2 토픽)
| 토픽 | 타입 | 발행자 | 내용 | 구독자 |
|---|---|---|---|---|
| `/vision/box_coarse` | PoseArray | ceiling_detector | 천장캠 **모든** 박스 | (참고/확장용) |
| `/vision/box_target` | PoseStamped | ceiling_detector | **grasp 타깃 1개** = 분포중심(0.45,-0.10) 최근접 | `result/grasp/run_grasp_vision.py` |
| `/vision/cell_coarse` | PoseArray | ceiling_detector | 천장캠 **모든** 셀 | `result/insert/run_insert_vision.py` (--cell-index 로 선택) |
| `/vision/box_fine` | PoseStamped | wrist_detector | 손목캠 박스(근접 정밀) | grasp S3 fine 반복수렴 |

## 2-3. Pose 인코딩 규약 (모든 vision 토픽 공통)
```
position.x, position.y = link0 기준 xy [m]
position.z             = 카메라 depth (참고, 무시)
orientation            = quat_from_z_yaw(yaw)   # z축 회전만
  → yaw 디코드: yaw = atan2( 2*(w*z + x*y), 1 - 2*(y*y + z*z) )
header.frame_id        = "link0"
```

## 2-4. 두 카메라 역할
- **천장캠 D435i (coarse, 1회)**: box+cell 의 대략 위치/yaw. extrinsic(체커보드 캘리브)로 link0 변환.
- **손목캠 D405 (fine, 연속)**: approach 후 box 정밀 위치/yaw. TF2 로 link0 변환(캘리브 불필요). 모델 없이 `--pca-only` 도 가능.

---

# PART ③ — 내가 현장에서 (실행하는 사람)

## 3-0. 토폴로지
카메라→**Pi USB**. 모든 노드(bringup·detector·파이프라인)는 **Pi Docker 안**. 노트북은 **SSH 접속만.**

각 터미널 공통 진입:
```bash
ssh root@omy-SNPR44B1021.local
docker exec -it open_manipulator bash
cd /root/ros2_ws/src/open_manipulator && source /root/ros2_ws/install/setup.bash
```

## 3-1. 1회성 준비 (이게 빠지면 아래 명령이 에러남)
1. **파일 복사** — `camera/`·`result/` 폴더를 Pi 컨테이너로 (scp → docker cp). `../HANDOFF.md` PHASE 0.
2. **모델 올리기** — 팀원 `best.onnx` → 컨테이너 `/root/yolov8n-seg.onnx`
   ```bash
   # 노트북: scp best.onnx root@omy-SNPR44B1021.local:~/
   # Pi:     docker cp ~/best.onnx open_manipulator:/root/yolov8n-seg.onnx
   ```
3. **캘리브 yaml 3개** (`../camera/calib/` 에 생성):
   ```bash
   python3 motion2/scripts/jaewoo/camera/dump_intrinsics.py --name d435i
   python3 motion2/scripts/jaewoo/camera/dump_intrinsics.py --name d405
   python3 motion2/scripts/jaewoo/camera/calibrate_ceiling_extrinsic.py --show   # 체커보드 link0 축정렬
   ```
4. **RL 의존성 확인** — `python3 -c "import stable_baselines3, torch, gymnasium; print('OK')"` (없으면 설치).
5. **hover_z 실측** — 손목캠이 박스를 잘 보는 높이(아래 grasp `--hover-z`). 한 번 측정해 둠.

## 3-2. 매번 실행 — SSH 터미널 4개
```bash
# T1  bringup (로봇+MoveIt+TF2, 계속 켜둠)
ros2 launch open_manipulator_bringup omy_f3m.launch.py

# T2  천장캠 detector → /vision/box_target, /vision/cell_coarse
python3 motion2/scripts/jaewoo/camera/ceiling_detector.py --ros \
    --model /root/yolov8n-seg.onnx \
    --extrinsic  motion2/scripts/jaewoo/camera/calib/ceiling_extrinsic.yaml \
    --intrinsics motion2/scripts/jaewoo/camera/calib/d435i_intrinsics.yaml --classes box,cell

# T3  손목캠 detector → /vision/box_fine  (모델 없이 PCA)
python3 motion2/scripts/jaewoo/camera/wrist_detector.py --ros \
    --intrinsics motion2/scripts/jaewoo/camera/calib/d405_intrinsics.yaml --pca-only

# T4  파이프라인 (토픽 자동 구독 → RL → 로봇)
python3 motion2/scripts/jaewoo/result/grasp/run_grasp_vision.py    # 먼저 dry-run
```

## 3-3. 연결 확인 (별도 터미널)
```bash
ros2 topic echo /vision/box_target --once    # 좌표 나오나 (yaw=2*atan2(qz,qw))
ros2 topic hz   /vision/box_fine             # 손목캠 끊김 없나
ros2 topic list | grep vision
```

## 3-4. grasp 실행 — 안전 단계별 (T4)
```bash
# (1) dry-run — 로봇 안 움직임, 토픽→롤아웃→plan 확인
python3 .../result/grasp/run_grasp_vision.py

# (2) 자세검증: yaw·fine 끄고 실제 (그리퍼가 박스를 양옆에서 감싸는지)
python3 .../result/grasp/run_grasp_vision.py --no-fine --no-align-yaw \
    --hover-z <실측값> --execute --confirm EXECUTE_GRASP --no-constrain-joint5 --max-joint-delta 3.0

# (3) 전체: 천장 coarse + 손목 fine 반복수렴 + yaw 정렬
python3 .../result/grasp/run_grasp_vision.py \
    --hover-z <실측값> --execute --confirm EXECUTE_GRASP --no-constrain-joint5 --max-joint-delta 3.0
```
> 비전 없이 배선만: `--dry-coords 0.45 -0.10 0.0` (토픽/카메라 없이 x,y,yaw 직접 주입)

## 3-5. insert 실행 — ⚠️ 실험적, align-only 먼저 (T4)
```bash
# (1) yaw 회전만 (place 하강 X) — 먼저 이걸로 검증
python3 .../result/insert/run_insert_vision.py --cell-index 0 \
    --execute --confirm EXECUTE_INSERT --no-constrain-joint5 --max-joint-delta 3.0
# (2) place 하강까지 (드리프트 검증 후): --place-z <실측> 추가
```
> insert 는 박스를 **이미 잡고 셀 위 호버**한 상태 전제(앞단 grasp→turn 은 이 스크립트 밖). place 하강 드리프트 가드 내장.

---

# PART ④ — 트러블슈팅 / 함정

| 증상 | 원인 | 해결 |
|---|---|---|
| 천장캠 토픽이 안 나옴 / 검출 0 | COCO 기본모델 / 모델 누락 | box·cell **커스텀 onnx** 확인, `--model` 경로 |
| cell 이 box 로 발행됨 | 클래스 순서 뒤바뀜 | 모델 `names=[box,cell]` (box=0) 확인 |
| 손목캠 TF2 timeout | bringup 안 켜짐 | T1 먼저, Docker 안에서 실행 |
| 손목캠 fine 무검출 | hover_z 너무 가까워 박스 FOV 밖 | `--hover-z` 실측값으로 |
| RealSense 2대 충돌 | detector 가 serial 미지정 | 한 Pi에 2대면 detector 에 serial 바인딩 필요(현재 옵션 없음 → 요청 시 추가) |
| SSH 인데 창 안 뜸 | `--show`/`--standalone` 은 디스플레이 필요 | `--ros` 모드는 토픽만 발행(SSH OK), 확인은 `ros2 topic echo` |

---

# 요약
- **팀원**: box·cell 커스텀 YOLOv8-seg 학습 → `yolo export ... format=onnx imgsz=640 nms=False`, **names=[box,cell]** → `best.onnx` 전달.
- **모델 출력**: box `(x,y,yaw)`·cell `(x,y)` (link0) → `/vision/box_target`·`/vision/cell_coarse`·`/vision/box_fine` 토픽.
- **나**: 준비(파일·모델·캘리브) 1회 → 이후 **SSH 접속 → 터미널 4개 명령**(bringup, 천장캠, 손목캠, 파이프라인).
