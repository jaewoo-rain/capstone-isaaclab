# YOLO box/cell 학습 플랜 (카메라 연결 → 바로 학습)

천장캠/손목캠으로 `box`·`cell` 을 검출하는 **YOLOv8-seg** 모델을, 실측 이미지 +
**depth 자동 라벨링**으로 만들어 이 데스크 GPU에서 학습하고, 기존 detector가 그대로
읽는 ONNX로 내보내 Pi5에 올린다.

## 확정 결정 (잠금)
| 항목 | 결정 | 근거 |
|------|------|------|
| 모델 | YOLOv8n-seg (instance seg) | 현 `_YoloSegONNX` 파싱 그대로, Pi5 nano ~2-6fps |
| 데이터 | 실측 RealSense + **depth 자동 라벨** | 박스는 테이블 위로 솟은 강체 → 평면 제거로 마스크 자동. 수동라벨 최소 |
| 셀 라벨 | **고정 ROI 템플릿 1회** 후 전 프레임 재사용 | 셀=유압블록(광택 금속) → depth 신뢰불가. 정적 천장캠이라 템플릿이 정답 |
| 학습 머신 | 이 데스크 GPU (`env_isaaclab`, RTX 4070) | ultralytics 8.4.36 + torch cu128 이미 있음(검증). onnxruntime 1.23.2 설치 완료 |
| 대상 | 박스 여러 개 + 셀(유압블록) | 다중 박스 → NMS·타깃선택(PHASE F). 타깃=**분포중심(0.45,-0.10) 최근접** 잠금 |

## ⚠️ 셀=유압블록(광택 금속) — 반사 대응
셀은 광 나는 금속 직육면체(작은 구멍 다수). 반사가 두 곳에 영향:
- **depth**: 정반사로 셀 위 depth 가 구멍/노이즈 → 셀은 depth 라벨 안 함(템플릿 전략으로 회피 ✓).
- **color/YOLO**: 하이라이트가 조명·각도로 변동 → 검출 불안정. 대응:
  - 촬영 시 **확산광**(직사광/스팟 금지), 셀 각도·조명을 **여러 조건**으로 다양화.
  - 학습 augment 의 `hsv_v`(밝기) 높게(train.sh 기본 0.5) 유지. 데이터에 하이라이트 케이스 충분히 포함.
- 셀은 **속 찬 직육면체** → `min_mask_ratio`(0.25) 면제 불필요. 단 yaw 는 고정구조물이라 `cell_yaw=0`.
- ⚠️ **박스도 광택 금속(확정)**: 박스 윗면 depth 도 정반사로 구멍/노이즈 → **depth 단독 자동라벨 부적합**.
  → **depth-seed + FastSAM** 하이브리드 채택: depth 유효 픽셀로 박스 대략 위치(seed) → FastSAM(ultralytics
  내장, 새 무거운 의존성 X)이 깨끗한 인스턴스 마스크 완성. `auto_label_depth.py --method sam` 로 추가(작업 중).
  - **촬영 확산광 필수**, 하이라이트 다양 조건 포함. 학습 augment hsv_v↑.
  - 💡 **강력 추천(물리)**: 박스 파지면 아닌 면에 **무광 테이프/스프레이** 한 겹 → depth·비전 난이도 급감.
    파지는 짧은 변(4.4cm)을 무니 윗면 무광처리는 grasp 에 무관.

## 산출물 (이미 작성·컴파일 검증됨)
```
camera/dataset/
  capture_dataset.py    RealSense color+depth 캡처 (--from-folder 로 카메라 없이 라벨링도)
  annotate_cells.py     셀 고정 ROI 1회 클릭 라벨 → cell_template.yaml
  auto_label_depth.py   depth 평면제거 → 박스 인스턴스 마스크 → ultralytics seg 라벨 + data.yaml
  README.md             데이터 흐름·촬영 가이드
camera/train/
  setup.md              env_isaaclab 에 ultralytics/onnxruntime (별도 venv 불필요 결론)
  train.sh / train.md   yolo segment train 명령 + 과적합 억제 전략
  export.sh / export.md  best.pt → ONNX(opset12, nms=False) + Pi5 전송
  verify_onnx.py        ONNX 를 _YoloSegONNX 로 로드·추론 스모크 테스트
```
**ONNX 호환성 실측 검증 완료**: nc=2 → `output0 (1,38,8400)`(=4+2+32), `output1 (1,32,160,160)`,
input `images (1,3,640,640)` — 현 detector 파싱과 정확히 일치.

---

## 실행 순서

### PHASE A — 캡처 (카메라 연결된 머신에서)
```bash
# 천장캠 60~100장 (박스 위치/개수1~4/각도0~180°/조명 다양화)
python3 camera/dataset/capture_dataset.py --cam d435i --n 80 --trigger key
# 손목캠 30~50장 (박스 근접 다양화)
python3 camera/dataset/capture_dataset.py --cam d405 --n 40 --trigger key
```
→ `dataset/raw/<cam>/NNNN_color.png + NNNN_depth.png + K.yaml`

### PHASE B — 셀 템플릿 1회 (천장캠 대표 프레임에서)
```bash
python3 camera/dataset/annotate_cells.py --frame dataset/raw/d435i/0000_color.png
```
→ `dataset/cell_template.yaml` (천장캠 위치 바뀌면 재생성)

### PHASE C — 자동 라벨 → 데이터셋 빌드
```bash
python3 camera/dataset/auto_label_depth.py \
    --raw dataset/raw/d435i --cell-template dataset/cell_template.yaml --show
# --show 로 박스 자동 마스크 검수. 박스 안 잡히면 --height-thresh 튜닝(기본 12mm)
python3 camera/dataset/auto_label_depth.py --raw dataset/raw/d405   # 손목캠(셀 라벨 X)
```
→ `dataset/dataset/yolo_seg/{images,labels}/{train,val}/` + `data.yaml` (`names: {0:box,1:cell}`)

### PHASE D — 학습 (이 데스크 GPU)
```bash
conda activate env_isaaclab
bash camera/train/train.sh    # data.yaml 없으면 안전 종료. 데이터 50장↓면 freeze=10
```

### PHASE E — ONNX export + Pi5 전송
```bash
bash camera/train/export.sh                 # best.pt → onnx + 형상 자가검증
python3 camera/train/verify_onnx.py <onnx>  # _YoloSegONNX 로드 스모크
# → scp/docker cp 로 컨테이너 /root/yolov8n-seg.onnx (export.sh 자동)
```
이후 현장 런북(TODO.md) PHASE 5(detection 테스트) → PHASE 6(RL grasp)로.

### PHASE F — detector 코드 보정 (커스텀 다중 박스/셀 대응) ✅ 적용 완료
현 `ceiling_detector.py` 가 단일 객체/단일 토픽 가정이라 다중 박스+셀에서 깨지던 것 수정:
- ✅ **per-class IoU NMS** — `_postprocess` 에 추가(중복앵커 제거, 생존분만 마스크 합성→성능↑). 단위검증.
- ✅ **다중 박스 토픽** — `/vision/box_coarse`=**PoseArray**(전체 박스) + `/vision/box_target`=**PoseStamped**(분포중심 `(0.45,-0.10)` 최근접 1개) + `/vision/cell_coarse`=PoseArray. `--target-center` override.
- ✅ **셀 분기** — cell 은 `yaw=0` 고정(정사각 long-axis 토글 회피), 경계필터 box 전용(외곽 셀 유지). min_mask_ratio 는 셀=속찬블록이라 유지.
- ✅ **wrist** `_detect_seg` box-only 필터(2-클래스 모델에서 cell 섞임 방지).

> ⚠️ **토픽 계약 변경**: grasp 연동은 이제 `/vision/box_target`(단일 PoseStamped)를 구독/echo.
> `box_coarse`/`cell_coarse` 는 PoseArray. 미작성 `run_grasp_vision.py` 는 `box_target` 구독 전제.

---

## 확인 상태
1. ✅ 셀 외형: 유압블록(광택 금속, 속찬 직육면체) → 반사대응 위 섹션. mask_ratio 유지, cell yaw=0.
2. ✅ 다중 박스 타깃: 분포중심(0.45,-0.10) 최근접.
3. 🔶 캡처 머신: 미정(capture_dataset.py 가 Pi/데스크 둘 다 지원 — 나중 결정 가능).
4. ✅ 박스=광택 금속 → depth-seed + FastSAM 라벨러(`--method sam`) 추가. 확산광·무광처리 권장.
