#!/usr/bin/env bash
# train.sh — YOLOv8n-seg fine-tune (box=0, cell=1) on env_isaaclab GPU.
#
# 사용법:
#   bash train.sh                       # 기본값으로 학습
#   DATA=/path/data.yaml EPOCHS=150 bash train.sh
#
# 데이터셋은 다른 에이전트가 camera/dataset/ 에 ultralytics seg 형식으로 생성.
# data.yaml 은 nc=2, names=[box, cell] (순서 고정: box=0, cell=1) 이어야 함.
set -euo pipefail

PY="${PY:-/home/jaewoo/miniconda3/envs/env_isaaclab/bin/python}"

# ── 경로 (placeholder — 데이터 준비되면 그대로 동작) ─────────────────────────
DATA="${DATA:-/home/jaewoo/IsaacLab/source/motion2/scripts/jaewoo/camera/dataset/data.yaml}"
MODEL="${MODEL:-yolov8n-seg.pt}"          # COCO pretrained nano-seg 에서 fine-tune
PROJECT="${PROJECT:-/home/jaewoo/IsaacLab/source/motion2/scripts/jaewoo/camera/train/runs}"
NAME="${NAME:-box_cell_seg}"

# ── 하이퍼파라미터 (소규모 실측 데이터셋 기준 — 과적합 억제) ────────────────
IMGSZ="${IMGSZ:-640}"                     # 추론 래퍼 INPUT_SIZE=640 와 반드시 일치
EPOCHS="${EPOCHS:-120}"
BATCH="${BATCH:-16}"                      # RTX 4070 12GB 에서 nano-seg 640 여유. OOM 시 8
PATIENCE="${PATIENCE:-30}"               # early stop: val 30 epoch 정체 시 중단
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"

echo "[train] DATA=$DATA"
echo "[train] MODEL=$MODEL  IMGSZ=$IMGSZ  EPOCHS=$EPOCHS  BATCH=$BATCH"
echo "[train] out -> $PROJECT/$NAME"

if [ ! -f "$DATA" ]; then
  echo "[train][경고] data.yaml 없음: $DATA"
  echo "[train]        다른 에이전트가 camera/dataset/ 생성 후 다시 실행하세요."
  exit 1
fi

# ── 학습 ────────────────────────────────────────────────────────────────────
# augment 설계 근거 (소규모 + 천장 top-down vs 손목 closeup 시점 차이 커버):
#  - degrees=180, flipud=0.5  : 박스/셀은 평면상 임의 회전 → 전방향 회전 불변 학습
#                               (rect long-axis yaw 는 하류 minAreaRect 가 다시 재므로 안전)
#  - scale=0.5, translate=0.1 : 천장(멀다) vs 손목(가깝다) 스케일/위치 차이 흡수
#  - hsv_v=0.5, hsv_s=0.5     : 조명/노출 변화(실측 환경 조명 편차) 흡수
#  - mosaic=1.0, close_mosaic=15 : 초반 mosaic 로 데이터 다양성 ↑, 마지막 15ep 끄고 실분포 정렬
#  - mixup=0.0, copy_paste=0.1 : 2-클래스 단순 장면 → mixup 불필요, copy_paste 소량만
#  - perspective=0.0          : 천장/손목 모두 거의 정사영 → 원근 왜곡 비활성
"$PY" -m ultralytics 2>/dev/null || true   # ultralytics CLI 사용 가능 확인용 (무해)

"$PY" - "$DATA" "$MODEL" "$IMGSZ" "$EPOCHS" "$BATCH" "$PATIENCE" "$DEVICE" "$WORKERS" "$PROJECT" "$NAME" <<'PYEOF'
import sys
from ultralytics import YOLO
data, model, imgsz, epochs, batch, patience, device, workers, project, name = sys.argv[1:]
m = YOLO(model)
m.train(
    data=data,
    task="segment",
    imgsz=int(imgsz),
    epochs=int(epochs),
    batch=int(batch),
    patience=int(patience),
    device=int(device),
    workers=int(workers),
    project=project,
    name=name,
    exist_ok=True,
    # ── 정규화/안정화 ──
    optimizer="auto",
    cos_lr=True,
    # ── augment ──
    degrees=180.0,
    flipud=0.5,
    fliplr=0.5,
    scale=0.5,
    translate=0.1,
    shear=0.0,
    perspective=0.0,
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.5,
    mosaic=1.0,
    close_mosaic=15,
    mixup=0.0,
    copy_paste=0.1,
    plots=True,
)
print("[train] done. best ->", f"{project}/{name}/weights/best.pt")
PYEOF

echo "[train] 완료. 다음: bash export.sh  (best.pt -> ONNX)"
