#!/usr/bin/env bash
# export.sh — 학습된 best.pt -> ONNX (추론 래퍼 _YoloSegONNX 호환) + Pi5 전송.
#
# 사용법:
#   bash export.sh
#   WEIGHTS=/path/best.pt bash export.sh
#   PI_HOST=root@omy-SNPR44B1021.local bash export.sh   # 전송까지
set -euo pipefail

PY="${PY:-/home/jaewoo/miniconda3/envs/env_isaaclab/bin/python}"

WEIGHTS="${WEIGHTS:-/home/jaewoo/IsaacLab/source/motion2/scripts/jaewoo/camera/train/runs/box_cell_seg/weights/best.pt}"
IMGSZ="${IMGSZ:-640}"     # 추론 래퍼 INPUT_SIZE=640 와 반드시 일치
OPSET="${OPSET:-12}"      # 확정값 (아래 근거 참조)
OUT_NAME="${OUT_NAME:-yolov8n-seg.onnx}"  # 런북/README 와 일치하는 파일명

if [ ! -f "$WEIGHTS" ]; then
  echo "[export][에러] weights 없음: $WEIGHTS"
  echo "[export]        먼저 train.sh 로 학습하세요."
  exit 1
fi

echo "[export] WEIGHTS=$WEIGHTS  IMGSZ=$IMGSZ  OPSET=$OPSET"

# ── ONNX export ──────────────────────────────────────────────────────────────
# 호환 조건 (검증됨):
#   - format=onnx, imgsz=640, opset=12
#   - nms=False (래퍼가 conf 필터 + argmax 직접 수행. NMS 내장 모델은 출력 형상이 달라 X)
#   - dynamic=False, half=False (Pi5 onnxruntime CPU 는 fp32, 고정 640x640)
#   - simplify 여부 무관 (래퍼는 output0/output1 두 텐서만 읽음). 기본 끔.
# 산출: best.onnx (WEIGHTS 와 같은 폴더). nc=2 -> output0 (1,38,8400), output1 (1,32,160,160).
"$PY" - "$WEIGHTS" "$IMGSZ" "$OPSET" <<'PYEOF'
import sys
from ultralytics import YOLO
weights, imgsz, opset = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m = YOLO(weights)
path = m.export(format="onnx", imgsz=imgsz, opset=opset, nms=False,
                dynamic=False, half=False, simplify=False)
print("[export] ONNX ->", path)
PYEOF

ONNX_SRC="${WEIGHTS%.pt}.onnx"   # ultralytics 가 best.pt -> best.onnx 로 저장
ONNX_DST="$(dirname "$WEIGHTS")/$OUT_NAME"
if [ "$ONNX_SRC" != "$ONNX_DST" ]; then
  cp -f "$ONNX_SRC" "$ONNX_DST"
fi
echo "[export] 산출 ONNX: $ONNX_DST"

# ── 출력 형상 자가검증 (onnx 로 그래프만 확인, onnxruntime 불필요) ───────────
"$PY" - "$ONNX_DST" <<'PYEOF'
import sys, onnx
mo = onnx.load(sys.argv[1])
def shp(v): return [d.dim_value for d in v.type.tensor_type.shape.dim]
outs = {o.name: shp(o) for o in mo.graph.output}
ins  = {i.name: shp(i) for i in mo.graph.input}
print("[export] inputs :", ins)
print("[export] outputs:", outs)
o0 = outs.get("output0"); o1 = outs.get("output1")
ok = (o0 == [1,38,8400]) and (o1 == [1,32,160,160])
print("[export] opset  :", mo.opset_import[0].version)
md = {p.key:p.value for p in mo.metadata_props}
print("[export] names  :", md.get("names"))
print("[export] COMPAT  :", "OK (1,38,8400 / 1,32,160,160)" if ok else "MISMATCH -> 래퍼 호환 안됨!")
sys.exit(0 if ok else 2)
PYEOF

echo "[export] 스모크 테스트는: $PY verify_onnx.py --model $ONNX_DST"

# ── Pi5 전송 (PI_HOST 지정 시) ───────────────────────────────────────────────
# 런북 PHASE 4 / camera/README.md 와 일관: 컨테이너 안 경로 /root/yolov8n-seg.onnx
if [ -n "${PI_HOST:-}" ]; then
  CONTAINER="${CONTAINER:-open_manipulator}"
  echo "[export] scp -> $PI_HOST:~/$OUT_NAME"
  scp "$ONNX_DST" "$PI_HOST:~/$OUT_NAME"
  echo "[export] Pi 호스트에서 Docker 로 복사하려면 (Pi 에서 실행):"
  echo "         docker cp ~/$OUT_NAME $CONTAINER:/root/$OUT_NAME"
  echo "[export] 또는 한 줄로 (로컬에서 ssh 경유):"
  echo "         ssh $PI_HOST \"docker cp ~/$OUT_NAME $CONTAINER:/root/$OUT_NAME\""
else
  echo "[export] Pi 전송 생략 (PI_HOST 미지정). 전송 예시:"
  echo "         scp $ONNX_DST root@omy-SNPR44B1021.local:~/$OUT_NAME"
  echo "         ssh  root@omy-SNPR44B1021.local 'docker cp ~/$OUT_NAME open_manipulator:/root/$OUT_NAME'"
fi
