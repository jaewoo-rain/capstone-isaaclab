# camera/dataset — RealSense → YOLOv8-seg 학습 데이터셋

"RealSense 카메라 연결 → 바로 YOLOv8-seg 학습 데이터셋"을 만드는 도구 모음.
박스는 **depth 자동 라벨**(테이블 평면 위로 솟은 강체 분할), 셀은 **고정 ROI 템플릿**으로 처리한다.

클래스: `box`(id 0), `cell`(id 1). 라벨 포맷: ultralytics seg(`labels/*.txt`, 각 줄 `cls x1 y1 x2 y2 ...` 정규화 폴리곤).

---

## 데이터 흐름

```
RealSense (color+depth aligned)
        │  capture_dataset.py
        ▼
dataset/raw/<cam>/
  NNNN_color.png   (BGR uint8)
  NNNN_depth.png   (uint16, mm)
  K.yaml           (fx,fy,cx,cy,depth_scale_mm)
  cell_template.yaml  ← annotate_cells.py 로 1회 생성(셀 고정 ROI)
        │  auto_label_depth.py  (박스=depth 자동, 셀=템플릿 주입)
        ▼
dataset/yolo_seg/
  images/{train,val}/<cam>_NNNN.png
  labels/{train,val}/<cam>_NNNN.txt
  data.yaml        ← 학습 담당에게 넘기는 진입점
```

---

## 실행 순서

### 1. 프레임 캡처 (`capture_dataset.py`)
```bash
# 천장캠 2초마다 자동 저장, 80장
python3 capture_dataset.py --cam d435i --interval 2.0 --n 80
# 손목캠 수동(스페이스=저장)
python3 capture_dataset.py --cam d405 --trigger key --n 60
# 카메라 없이 기존 프레임 점검(머신 위치 독립)
python3 capture_dataset.py --from-folder dataset/raw/d435i --list
```
color↔depth 는 `rs.align(rs.stream.color)` 로 정렬되어 같은 픽셀이 같은 물체를 가리킨다.
`K.yaml` 에 color intrinsics 와 `depth_scale_mm`(보통 1mm/LSB)이 함께 저장된다.

### 2. 셀 고정 ROI 1회 라벨 (`annotate_cells.py`) — 셀 쓸 때만
```bash
python3 annotate_cells.py --image dataset/raw/d435i/0000_color.png
# 좌클릭=점추가  n=셀확정  u=점취소  s=저장(cell_template.yaml)  q=종료
```
정적 천장캠 기준 1회만 그리면 모든 프레임에 재사용된다.

### 3. 박스 자동 라벨 + 데이터셋 빌드 (`auto_label_depth.py`)
```bash
# 검수(시각화, 저장 안 함)
python3 auto_label_depth.py --from-folder dataset/raw/d435i --show --no-write
# 빌드(셀 템플릿 있으면 자동 주입)
python3 auto_label_depth.py --from-folder dataset/raw/d435i
# 테이블 평면 깊이 고정(RANSAC 대신) + 셀 템플릿 지정
python3 auto_label_depth.py --from-folder dataset/raw/d435i \
    --table-z 0.80 --cell-template dataset/raw/d435i/cell_template.yaml
```
박스 분할: 테이블 평면 추정(기본 RANSAC, `--table-z` 로 고정 가능) → 평면보다
`--height-thresh`(기본 12mm) 이상 카메라에 가까운 픽셀 = 박스 윗면 → connected
components 인스턴스 분리 → approxPolyDP 폴리곤. 필터: 최소 면적, 화면 경계 닿은 것 제외.

#### 광택 금속 박스 → `--method sam` (depth-seed + FastSAM 하이브리드)
박스가 광택 금속이면 정반사로 박스 윗면 depth 에 구멍/노이즈가 생겨 depth 단독
자동라벨이 깨진다. `--method sam` 은 depth 로 박스 **대략 위치(seed)** 만 잡고,
color 이미지에 **FastSAM**(ultralytics 내장, 새 의존성 없음)을 point-prompt 로 돌려
깨끗한 인스턴스 마스크를 얻는다. 출력 포맷/디렉터리/필터는 depth 경로와 동일.
```bash
# 검수(seed 점=빨강 십자 + FastSAM 마스크 시각 확인, 저장 안 함)
python3 auto_label_depth.py --from-folder dataset/raw/d435i --method sam --show --no-write
# 빌드(데스크 GPU 라벨링 단계, 가중치 FastSAM-s.pt 최초 1회 자동 다운로드)
python3 auto_label_depth.py --from-folder dataset/raw/d435i --method sam --device cuda:0
```
seed→마스크: depth 전경 컴포넌트의 distance-transform peak 를 seed 점으로 뽑아
FastSAM point-prompt → seed 를 포함하는 마스크를 선택(없으면 seed 포함 마스크 중
최소면적). depth seed 가 0개(박스 depth 완전 소실)면 FastSAM **everything 모드** 후
크기/위치 휴리스틱으로 박스 후보를 고르는 fallback 으로 자동 전환한다. 옵션:
`--device`(cuda:0/cpu), `--sam-weights`(FastSAM-s.pt/FastSAM-x.pt), `--sam-imgsz`,
`--sam-conf`, `--sam-iou`. **FastSAM 은 이 데스크 GPU 에서만 도는 라벨링 단계**이며
Pi5 추론(ONNX YOLOv8-seg)과 무관하다.
> 촬영 권장: 광택 금속은 **확산광**(직사광/스팟 금지)으로 찍고, 가능하면 박스
> 비파지면에 **무광 테이프/스프레이** 한 겹 → depth·비전 난이도가 급감한다.

---

## 셀 라벨링 전략: (a) 고정 ROI 템플릿 재사용 — 채택 근거

| 후보 | 평가 |
|------|------|
| (a) 1회 수동/템플릿 라벨 후 전 프레임 재사용 | **채택** |
| (b) depth 로 셀 구조 검출 | 기각 — 셀 슬롯은 오목/평면이라 '솟은 강체' depth 분할이 안 통함 |
| (c) 셀 별도 고정 ROI 처리 | (a)와 사실상 동일, (a)로 통합 |

근거:
- 천장캠은 **정적**, 셀 그리드는 **고정 구조물** → 모든 프레임에서 셀 픽셀 위치가 동일.
  매 프레임 검출할 이유가 없다.
- 셀은 박스와 depth 특성이 정반대(솟지 않음/오목) → 박스 자동 분할 로직 재사용 불가.
- 1회 클릭 라벨이 가장 안정적이고 노이즈가 없다. 정적 카메라 가정과 정확히 일치.

구현: `annotate_cells.py` 로 `cell_template.yaml` 생성 → `auto_label_depth.py` 가
전 프레임에 `cls=1` 로 주입. 템플릿이 없으면 박스만 라벨하고 경고를 출력한다.
(주의: 카메라를 옮기면 템플릿을 다시 그려야 한다.)

---

## 천장캠 vs 손목캠 시점 — 데이터 혼합 권장안

모델은 두 시점 모두 추론에 쓰이므로 **둘을 한 데이터셋에 섞어 학습**한다(시점 불변성 확보).
`auto_label_depth.py` 를 cam 별로 두 번 돌리면 `<cam>_NNNN` 접두사로 같은
`dataset/yolo_seg/` 에 누적된다.

- 천장캠(D435i, top-down): 주 시점. 박스가 평면 위로 또렷이 솟아 자동 라벨 품질이 가장 좋음.
  전체의 약 **70%** 권장.
- 손목캠(D405, 근접/사선): 시점이 기울어 평면 RANSAC 이 더 중요(`--table-z` 고정보다 RANSAC).
  약 **30%** 권장. 셀 템플릿은 손목캠이 움직이면 무효이므로, 손목캠 프레임은 **박스 위주**로
  쓰고 셀 라벨은 천장캠에서만 확보하는 것을 권장.

---

## 권장 촬영 가이드

- **장수**: 천장캠 60~100장 + 손목캠 30~50장(합 ≈ 100~150장)이면 seg 미세조정에 충분.
- **다양화**: 박스 위치(테이블 전 영역), 개수(1~4개), 각도(yaw 0~180° 골고루),
  박스 간 간격(붙은 경우/떨어진 경우), 조명(밝기·그림자), 약간의 가림.
- **분할**: train/val = **80/20**(`--val-frac 0.2`, seed 고정 셔플).
- **검수**: 빌드 전 `--show --no-write` 로 자동 라벨이 박스를 정확히 감싸는지 눈으로 확인.
  `ceiling_detector` 가 마스크→minAreaRect→yaw 를 뽑으므로, 폴리곤이 박스 긴 변/짧은 변을
  잘 따라가야 yaw 추정이 정확하다.

---

## 학습 담당에게 넘길 것

- `data.yaml` 경로: `camera/dataset/dataset/yolo_seg/data.yaml`
- 학습 예: `yolo segment train model=yolov8n-seg.pt data=.../data.yaml imgsz=640`
- 추론 시 ONNX 로 export 하면 `ceiling_detector._YoloSegONNX`(output0 (1,116,8400),
  output1 (1,32,160,160)) 가 그대로 파싱한다.
