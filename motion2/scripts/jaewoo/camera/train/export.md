# export.md — best.pt -> ONNX (Pi5 추론 호환)

실제 export 는 `export.sh` 가 수행. 이 문서는 호환 조건 확정값과 전송 런북.

---

## 확정 export 명령 (CLI 등가)

```bash
PY=/home/jaewoo/miniconda3/envs/env_isaaclab/bin/python
$PY -c "from ultralytics import YOLO; \
YOLO('<runs>/box_cell_seg/weights/best.pt').export(\
format='onnx', imgsz=640, opset=12, nms=False, dynamic=False, half=False, simplify=False)"
```

또는 ultralytics CLI:
```bash
yolo export model=<runs>/box_cell_seg/weights/best.pt \
  format=onnx imgsz=640 opset=12 nms=False dynamic=False half=False simplify=False
```

## 호환 조건 (추론 래퍼 `_YoloSegONNX` 기준 — 모두 검증됨)

| 항목 | 값 | 이유 |
|------|----|------|
| `format` | `onnx` | onnxruntime CPU on Pi5 |
| `imgsz` | **640** | 래퍼 `INPUT_SIZE=640`, letterbox 114-pad 중앙정렬이 640 전제 |
| `opset` | **12** | 검증 완료. onnxruntime 1.x 모두 지원. 더 높여도 되지만 12 가 가장 호환 폭 넓음 |
| `nms` | **False** | 래퍼가 `out0.T` 후 conf 필터 + `argmax` 직접 수행. NMS 내장 시 출력 형상 달라져 파싱 깨짐 |
| `dynamic` | False | Pi5 고정 640x640 입력. 동적축 불필요 + 일부 ort 빌드에서 느려짐 |
| `half` | False | onnxruntime CPU 는 fp32. fp16 export 시 CPU 추론 비호환/저하 |
| `simplify` | False (무관) | 래퍼는 output0/output1 두 텐서명만 읽음. simplify 해도 텐서명/형상 유지되면 OK |

## 출력 형상 (실측 검증, 2026-06-12)

nc=2 (box,cell) export 결과:
```
inputs : images   [1, 3, 640, 640]
outputs: output0  [1, 38, 8400]      # 38 = 4(box) + 2(cls) + 32(mask coef)
         output1  [1, 32, 160, 160]  # prototype masks (nc 와 무관, 항상 32)
opset  : 12
names  : {0: 'box', 1: 'cell'}
```

래퍼 `_postprocess` 의 `num_cls = out0.shape[0] - 4 - 32` → `38 - 36 = 2`. **정확히 일치.**

> 검증 방법: yolov8n-seg.yaml 로 nc=2 head 를 만들어(dummy 1-epoch train) 실제 ONNX export 후
> onnx 그래프 출력 형상 확인 + `_YoloSegONNX` 로 로드해 추론까지 돌려 `(1,38,8400)/(1,32,160,160)`,
> `num_cls=2`, input name=`images` 를 확인함.

## 클래스 순서 — 중요

- 래퍼는 실행 시 `--classes box,cell` 로 이름을 **주입**하므로 ONNX 메타데이터 이름과 무관.
- 단, **인덱스 순서(box=0, cell=1)** 는 `data.yaml` 의 `names` 순서가 그대로 ONNX `cls_id` 가 된다.
- 따라서 `data.yaml` 의 `names: [box, cell]` 순서를 **절대 바꾸지 말 것**.
  (`source/motion1/yolo_dataset/data.yaml` 도 동일 순서 — 일관됨.)

## Pi5 전송 런북 (camera/README.md PHASE 4 와 일관)

```bash
OUT=yolov8n-seg.onnx
# 1) 로컬 -> Pi 호스트
scp <runs>/box_cell_seg/weights/$OUT root@omy-SNPR44B1021.local:~/$OUT
# 2) Pi 호스트 -> Docker 컨테이너 (컨테이너 안 경로 = /root/yolov8n-seg.onnx)
ssh root@omy-SNPR44B1021.local "docker cp ~/$OUT open_manipulator:/root/$OUT"
```

컨테이너 안 추론 호출(예):
```bash
python3 motion2/scripts/jaewoo/camera/ceiling_detector.py \
  --model /root/yolov8n-seg.onnx \
  --extrinsic  motion2/scripts/jaewoo/camera/calib/ceiling_extrinsic.yaml \
  --intrinsics motion2/scripts/jaewoo/camera/calib/d435i_intrinsics.yaml \
  --classes box,cell
```

`export.sh PI_HOST=root@omy-SNPR44B1021.local bash export.sh` 로 전송까지 한 번에 가능.
