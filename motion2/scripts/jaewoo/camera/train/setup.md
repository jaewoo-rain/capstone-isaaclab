# setup.md — YOLOv8-seg 학습 환경 (OMY-F3M box/cell)

학습 머신: 이 리눅스 데스크 (`/home/jaewoo/IsaacLab`), GPU = RTX 4070 12GB.
추론 머신: Pi5 Docker (onnxruntime CPU). 여기서는 **학습 + ONNX export** 만 한다.

---

## 결론: `env_isaaclab` 환경을 그대로 쓴다 (별도 venv 불필요)

이미 검증된 사실 (2026-06-12, 실측):

| 패키지 | 버전 | 상태 |
|--------|------|------|
| python | 3.10.20 | OK |
| torch | 2.7.0+cu128 | `cuda.is_available()=True`, cuda 12.8 |
| ultralytics | 8.4.36 | 설치됨 |
| onnx | 1.16.1 | 설치됨 |
| onnxruntime | 1.23.2 | **이번에 설치함** (없었음) |
| opencv (cv2) | 4.9.0 | 설치됨 |
| numpy | 1.26.0 | 설치됨 |

근거:
- ultralytics 8.4.36 이 이미 `env_isaaclab` 에 있고, IsaacLab 의 torch(2.7.0+cu128)와 충돌 없이 동작.
  실제로 이 환경에서 yolov8n-seg.yaml → 1-epoch train → ONNX export 를 끝까지 돌려 검증함.
- 별도 venv 를 만들면 CUDA torch 를 다시 받아야 하고(수 GB), 버전 핀이 IsaacLab 과 갈려 유지보수만 늘어난다.
- ultralytics 가 IsaacLab/SB3 의존성을 깨는 정황은 관측되지 않음. 학습은 RL 학습과 **동시 실행하지 말 것**(GPU 12GB 공유, OOM 위험) — 시간만 분리하면 됨.

> 주의: `conda run -n env_isaaclab ...` 는 이 머신에서 `pydantic_core` 경고를 뱉지만 무해.
> 안정적으로는 인터프리터 절대경로를 직접 호출하라:
> `/home/jaewoo/miniconda3/envs/env_isaaclab/bin/python`
> 또는 `source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab`.

---

## 0. 환경 활성화

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd /home/jaewoo/IsaacLab
PY=/home/jaewoo/miniconda3/envs/env_isaaclab/bin/python   # 직접 호출용
```

## 1. CUDA / GPU 확인

```bash
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
$PY -c "import torch; print('torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'cuda', torch.version.cuda)"
```

기대: `cuda_avail True`. False 면 학습이 CPU 로 떨어져 매우 느림 → 드라이버/torch 재확인.

## 2. 패키지 확인 (재설치 필요 시)

```bash
$PY -c "import ultralytics, onnx, onnxruntime, cv2, numpy; \
print('ultralytics', ultralytics.__version__); \
print('onnx', onnx.__version__); \
print('onnxruntime', onnxruntime.__version__)"
```

onnxruntime 가 없다고 나오면 (export 후 verify_onnx.py 스모크에 필요):
```bash
$PY -m pip install onnxruntime
```

ultralytics 가 없거나 너무 구버전이면:
```bash
$PY -m pip install -U ultralytics
```

## 3. pretrained 가중치 캐시 (오프라인 대비)

`yolov8n-seg.pt` (COCO, ~6.5MB) 를 base 로 fine-tune 한다. 첫 train 시 자동 다운로드되지만,
인터넷 없는 환경이면 미리 받아 두라:

```bash
cd /home/jaewoo/IsaacLab
$PY -c "from ultralytics import YOLO; YOLO('yolov8n-seg.pt')"   # 캐시에 다운로드
```

## 다음 단계
- 데이터셋: 다른 에이전트가 `camera/dataset/` 에 ultralytics seg 형식 + `data.yaml` 로 생성.
  (참고: `source/motion1/yolo_dataset/data.yaml` 가 이미 `nc=2, names=[box, cell]` 로 동일 규약)
- 학습: `train.sh` 참조
- export: `export.sh` 참조
- 검증: `verify_onnx.py` 참조
