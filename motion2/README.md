# motion2 — Pick-and-place chain (sim + real 분리 구조)

OMY 로봇으로 박스를 잡고 셀(상자)에 삽입하는 pick-and-place 작업. 시뮬레이션에서 학습/검증 후 real RealSense D435/D405 + OMY ROS 로 deploy 하기 위한 모듈화 구조.

## Current MVP demo status

현재 실제 OMY-F3M 데모는 전체 pick-and-place execute가 아니라,
`dry-run / MoveIt plan-only / guarded smoke execution`으로 분리해서 진행한다.

빠른 시연 절차와 금지 항목은 아래 문서를 기준으로 한다.

```text
motion2/MVP_DEMO.md
motion2/SIM_TO_REAL_CHECKLIST.md
```

현재 승인된 real-robot 범위:

- tiny joint/gripper smoke sequence
- `run_capstone_mvp_pipeline.py`의 전체 캡스톤 파이프라인 요약
- `run_pick_place_mvp.py`의 dry-run chain 출력
- `run_pick_place_mvp.py --demo-safe` 발표용 안전 흐름 출력
- MoveIt plan-only waypoint 검증
- trajectory guard를 통과한 단일 relative pose smoke stage

아직 금지:

- full pick-and-place real execution
- `grasp_z=0.115` execution
- absolute `transport`/`insert` pose execution
- camera/RL output을 실제 로봇 명령으로 직접 사용

---

## 폴더 구조

```
motion2/
├── inference/                  ← sim/real 무관 (수정 X)
│   ├── unproject.py            # pixel → world coord
│   ├── yolo_box_detector.py    # YOLO seg → mask → world pose
│   ├── grasp_policy.py         # SB3 PPO inference
│   └── chain_state_machine.py  # stage 흐름 (motion + RL)
├── adapters/
│   ├── base_adapter.py         # 추상 interface (수정 X)
│   ├── sim_adapter.py          # IsaacLab 구현 (시뮬레이션 검증용)
│   └── real_adapter.py         # ← 친구가 여기 채워 넣음
├── scripts/
│   └── play_chain_sim.py       # sim entry
├── models/                     # 모델 ckpt 보관 위치 (별도 다운로드)
└── README.md
```

---

## 흐름 개요

1. **천장 cam scan** (D435, 위 80cm) — 1 frame 캡처 → YOLO seg → 박스/셀 world xy + yaw 추정
2. **Stage 1 (motion)**: home → 박스 위 17cm 으로 이동 (천장 cam 의 박스 xy)
3. **Stage 2 (Grasp RL)**: 손목 cam (D405, link6 부착) 매 step → YOLO 박스 yaw 추정 → SB3 PPO inference → ee 정렬 (5mm/2.86° 안에 30 step)
4. **Stage 3a-d (motion)**: descend 1cm → close gripper → lift → transport (천장 cam 의 셀 xy/yaw 정확히)
5. **Stage 5a-b (motion)**: insert descend → release
6. **Stage 6a-b (motion)**: retract up → home

---

## 빠른 시작 (sim 검증)

```bash
# Isaac Sim + IsaacLab 환경 (사용자 쪽)
./isaaclab.sh -p source/motion2/scripts/play_chain_sim.py \
    --enable_cameras --repeat 3 --hold_s 5
```

---

## real 배포 — 친구 작성 가이드

### Step 1. 환경 셋업

```bash
# Python 3.10+, ROS (1 또는 2)
pip install ultralytics opencv-python pyrealsense2 stable-baselines3 numpy
# OMY SDK 설치 (ROBOTIS 공식)
```

### Step 2. `adapters/real_adapter.py` 채워 넣기

필수 구현 메서드 7개 (자세한 docstring 은 파일 안에):

| 메서드 | 역할 |
|---|---|
| `get_top_cam()` | RealSense D435 RGB + depth + intrinsic + extrinsic |
| `get_wrist_cam()` | RealSense D405 (link6 부착) 동일 |
| `get_ee_pose()` | OMY FK → ee world pose + velocity |
| `set_ee_target(pos, quat, gripper)` | OMY IK + joint publish |
| `step(n)` | control loop n 회 (60Hz, sleep) |
| `reset_to_home()` | home joint pose 이동 (blocking) |
| `get_home_ee_pose()` | home 자세의 ee pose hardcode |

### Step 3. 모델 파일 위치

```
motion2/models/
├── yolo11n_seg_v2_box_cell.pt        # YOLO segmentation (class 0=box, 1=cell)
├── motion1_grasp.zip                  # SB3 PPO grasp policy
└── motion1_grasp_vecnorm.pkl          # VecNormalize 통계
```

(별도 다운로드 또는 사용자가 전달)

### Step 4. 실행

```python
from source.motion2.adapters.real_adapter import RealAdapter
from source.motion2.inference.yolo_box_detector import YoloBoxDetector
from source.motion2.inference.grasp_policy import GraspPolicy
from source.motion2.inference.chain_state_machine import ChainConfig, run_chain_once

adapter = RealAdapter()  # 친구가 채운 클래스
yolo = YoloBoxDetector("motion2/models/yolo11n_seg_v2_box_cell.pt")
policy = GraspPolicy("motion2/models/motion1_grasp.zip",
                     "motion2/models/motion1_grasp_vecnorm.pkl")
cfg = ChainConfig()

result = run_chain_once(adapter, yolo, policy, cfg)
print(result)
```

`run_chain_once()` 가 stage 1~6 다 처리.

---

## 좌표 / 단위 / Convention

- 모든 world 좌표 = robot base frame 기준 (sim env_origin 적용된 world 와 동일)
- Length: meters
- Angle: radians (yaw 는 world Z-axis 회전)
- Quaternion: `(w, x, y, z)`
- Camera frame:
  - sensor 의 `quat_w_world` = "world convention" (forward=+X, up=+Z)
  - 내부 unproject 는 ROS convention (forward=+Z, right=+X, down=+Y) 도 자동 처리
- EE = gripper 양 finger 의 world 좌표 평균 (OMY OpenManipulator 기준)

---

## RealSense 카메라 mount spec

### 천장 cam (D435)
- Robot base 위 약 80cm
- Forward 방향 = world -Z (정 아래)
- FOV: D435 의 native 87° (87.5°x58° depth)
- Pre-calibrated extrinsic (eye-to-base) 필요

### 손목 cam (D405)
- OMY `link6` 끝에 부착
- link6 frame 기준 offset:
  - position: `(0.0, -0.1, 0.084)` m
  - rotation (ROS convention): `(0, 0, 0.7071068, -0.7071068)` (wxyz)
- 즉 카메라 forward = link6 `-Y` 방향 (gripper 향)
- FOV: D405 의 native 87° (≈87°x58°)
- Real 에선 매 frame `get_link6_pose()` 읽어서 offset 적용 → world pose 계산

---

## 학습 분포 (sim domain randomization 적용된 분포)

| 항목 | 분포 |
|---|---|
| 박스 spawn yaw | ±π/2 (sim 학습 분포) |
| 셀 spawn yaw | ±π/2 |
| 조명 intensity | 800~4000 (dome light) |
| 박스 material | metallic 0.85, roughness 0.12 |
| 카메라 view | 박스 위 ±5cm, z 0.15~0.35m / 셀 위 ±5cm (50/50) |

Real 시 박스/셀 material, 조명, 카메라 view 가 위 분포 안에 들면 zero-shot 가능. 안 되면 real 이미지로 YOLO fine-tune (50~200장).

---

## 주의 사항

- **Control rate**: 60Hz. sim 학습 정책의 dt = 1/60 와 일치해야 정책 분포 안정.
- **Gripper**: sim 의 `gripper_close` = 0.8 + tip_ratio 2.3 (4 finger joint 명령). Real 에선 OMY gripper API 의 0~1 normalized 와 맞게 변환.
- **Joint 3 limits**: URDF 는 ±360° 잘못 — 실제 OMY 는 ±150°. real IK 솔버에서 좁힌 limit 사용.
- **Hand-eye calibration**: 천장 cam extrinsic 은 robot base 기준 정확히 calibration 필요 (mm 단위 정확도 영향).

---

## 디버깅 흐름

1. sim 으로 `play_chain_sim.py` 돌려서 grasp/insert 성공 확인
2. real 에서 천장 cam scan 만 호출 → YOLO 추정 vs 실제 박스 위치 비교
3. 손목 cam 추정 yaw vs 실제 박스 yaw 비교 (수동 측정)
4. ee pose 정확도 확인 (IK + FK round-trip)
5. RL 정책 inference 만 dummy obs 로 호출 → 출력 action 범위 확인
6. 단계별 결합 → 전체 chain

---

## 문의

질문 있으면 [원본 sim 코드 motion1](../motion1/scripts/play_motion_chain_with_grasp_insert_camera.py) 참고.
