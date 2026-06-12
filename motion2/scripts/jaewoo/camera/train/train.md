# train.md — YOLOv8n-seg fine-tune 전략 (box/cell)

실제 학습은 `train.sh` 가 수행한다. 이 문서는 **왜 이 설정인지** 와
순수 ultralytics CLI 등가 명령을 정리한다.

---

## 모델 선택: `yolov8n-seg.pt` (COCO) fine-tune

- 추론 머신 Pi5 (onnxruntime CPU) → **nano(n)** 고정. s/m 은 Pi5 에서 너무 느림(목표 2~6fps).
- scratch 학습 X. COCO pretrained backbone fine-tune → 수십~수백 장 소규모 데이터셋에서도
  수렴 빠르고 일반화 좋음.

## 핵심 등가 CLI (placeholder 경로)

```bash
PY=/home/jaewoo/miniconda3/envs/env_isaaclab/bin/python
$PY -m ultralytics  # (선택) CLI 확인

yolo segment train \
  model=yolov8n-seg.pt \
  data=/home/jaewoo/IsaacLab/source/motion2/scripts/jaewoo/camera/dataset/data.yaml \
  imgsz=640 epochs=120 batch=16 patience=30 device=0 \
  project=/home/jaewoo/IsaacLab/source/motion2/scripts/jaewoo/camera/train/runs \
  name=box_cell_seg exist_ok=True cos_lr=True close_mosaic=15 \
  degrees=180 flipud=0.5 fliplr=0.5 scale=0.5 translate=0.1 \
  hsv_s=0.5 hsv_v=0.5 mosaic=1.0 copy_paste=0.1
```

> `imgsz=640` 은 **반드시** 고정. 추론 래퍼 `_YoloSegONNX.INPUT_SIZE=640` 과 letterbox
> (114 패딩, 중앙 정렬) 로직이 640 전제로 짜여 있다. 다른 값으로 export 하면 좌표/마스크 어긋남.

## 소규모 데이터셋 과적합 억제 전략

| 수단 | 설정 | 이유 |
|------|------|------|
| pretrained fine-tune | `model=yolov8n-seg.pt` | scratch 대비 적은 데이터로 일반화 |
| early stopping | `patience=30` | val 정체 시 자동 중단 (과적합 직전) |
| cosine LR | `cos_lr=True` | 후반 LR 감쇠로 안정 수렴 |
| close_mosaic | `close_mosaic=15` | 마지막 15ep mosaic 끄고 실분포에 정렬 |
| 강한 회전/플립 | `degrees=180, flipud/fliplr=0.5` | box/cell 평면 임의회전 → 회전불변, 데이터 증강 |
| 스케일/조명 | `scale=0.5, hsv_s/v=0.5` | 시점·조명 편차 흡수 |

### freeze 는 기본 안 함 (조건부 권장)
- 데이터 **< 50장** 으로 심하게 적으면 backbone 동결로 과적합 추가 억제:
  ```bash
  ... freeze=10   # 처음 10개 레이어(backbone) 동결, head 만 학습
  ```
- 50~수백 장이면 freeze 없이 전체 fine-tune 가 보통 더 정확. 먼저 freeze 없이 학습해 보고
  val seg mAP 가 train 대비 크게 낮으면 (과적합) `freeze=10` 추가.

## 시점 차이(천장 top-down vs 손목 closeup) 일반화 — 어디까지 augment 로 커버되나

- **스케일 차이**: `scale=0.5` + `translate=0.1` 로 상당 부분 커버. 천장(작게 보임)·손목(크게 보임)
  양쪽 데이터가 데이터셋에 **둘 다 포함**되어 있으면 augment 가 그 사이를 메운다.
- **회전 차이**: `degrees=180, flipud` 로 완전 커버 (평면 물체).
- **한계 — augment 로 못 메우는 것**: 원근/렌즈 왜곡, 천장 vs 손목의 조명 색온도 극단 차이,
  손목캠에서만 보이는 그리퍼 가림(occlusion). → 이건 **두 시점 실측 이미지를 데이터셋에 모두
  넣어야** 한다. augment 만으로는 한 시점 데이터로 다른 시점을 일반화하지 못한다.
- 권장: 천장캠 이미지와 손목캠 이미지를 **같은 data.yaml 한 데이터셋**에 섞어 단일 모델 학습.
  클래스(box/cell)는 동일하므로 분리할 이유 없음.

## 학습 모니터링

```bash
# 진행 로그
ls -la /home/jaewoo/IsaacLab/source/motion2/scripts/jaewoo/camera/train/runs/box_cell_seg/
# results.csv, weights/best.pt, weights/last.pt, 곡선 png 생성됨
```

확인 지표 (results.csv / 콘솔):
- `metrics/mAP50(M)` (mask mAP@0.5) → box/cell seg 품질. 0.9+ 목표(단순 2-클래스).
- `metrics/mAP50-95(M)` → 마스크 경계 정밀도. minAreaRect yaw 정확도에 직결.
- train loss ↘ 인데 val loss ↗ → 과적합 → `patience` 가 잡아줌, 또는 `freeze=10`.

## 산출물
- best: `.../runs/box_cell_seg/weights/best.pt`  ← export 대상
- last: `.../runs/box_cell_seg/weights/last.pt`

다음: `export.md` / `export.sh` 로 ONNX 변환.
