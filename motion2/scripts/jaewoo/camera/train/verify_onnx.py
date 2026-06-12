#!/usr/bin/env python3
"""verify_onnx.py — 학습/export 된 YOLOv8n-seg ONNX 스모크 테스트 (카메라 불필요).

검증 항목:
  1) onnx 그래프 출력 형상이 추론 래퍼 _YoloSegONNX 기대와 일치하는가
       output0 == (1, 4+nc+32, 8400),  output1 == (1, 32, 160, 160)
       nc(클래스 수) 는 output0 채널에서 역산 (num_cls = ch - 4 - 32)
  2) 실제 추론 래퍼 _YoloSegONNX 로 로드해 더미/샘플 이미지를 통과시켜
       num_cls == 2 (box, cell) 이고 추론이 예외 없이 도는가

사용:
  PY=/home/jaewoo/miniconda3/envs/env_isaaclab/bin/python
  $PY verify_onnx.py --model <path/to/yolov8n-seg.onnx>
  $PY verify_onnx.py --model <onnx> --image <sample.jpg>   # 실제 검출 개수까지 출력
  $PY verify_onnx.py --model <onnx> --conf 0.25

종료 코드: 0 = 모든 검사 통과, 2 = 형상/호환 불일치, 1 = 로드/추론 오류.

문법 확인만:  python3 -m py_compile verify_onnx.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

# 추론 래퍼가 있는 jaewoo/ 를 import path 에 추가 (camera.ceiling_detector 로 import)
_THIS = pathlib.Path(__file__).resolve()
_JAEWOO_DIR = _THIS.parents[2]           # .../scripts/jaewoo
if str(_JAEWOO_DIR) not in sys.path:
    sys.path.insert(0, str(_JAEWOO_DIR))

EXPECTED_NC = 2                          # box(0), cell(1)
EXPECTED_OUT1 = [1, 32, 160, 160]
INPUT_SIZE = 640


def check_graph_shapes(model_path: str) -> int:
    """onnx 그래프만으로 출력 형상 검증 (onnxruntime 불필요)."""
    import onnx
    mo = onnx.load(model_path)

    def shp(v):
        return [d.dim_value for d in v.type.tensor_type.shape.dim]

    ins = {i.name: shp(i) for i in mo.graph.input}
    outs = {o.name: shp(o) for o in mo.graph.output}
    print(f"[verify] inputs : {ins}")
    print(f"[verify] outputs: {outs}")
    print(f"[verify] opset  : {mo.opset_import[0].version}")
    md = {p.key: p.value for p in mo.metadata_props}
    print(f"[verify] names  : {md.get('names')}")

    ok = True

    # 입력
    img_in = ins.get("images")
    if img_in != [1, 3, INPUT_SIZE, INPUT_SIZE]:
        print(f"[verify][FAIL] input 'images' != [1,3,640,640]: {img_in}")
        ok = False

    # output0: (1, 4+nc+32, 8400)
    o0 = outs.get("output0")
    if not o0 or len(o0) != 3 or o0[0] != 1 or o0[2] != 8400:
        print(f"[verify][FAIL] output0 형상 비정상 (기대 (1, 4+nc+32, 8400)): {o0}")
        ok = False
    else:
        num_cls = o0[1] - 4 - 32
        print(f"[verify] output0 채널 {o0[1]} -> num_cls = {o0[1]}-4-32 = {num_cls}")
        if num_cls != EXPECTED_NC:
            print(f"[verify][FAIL] num_cls={num_cls} != {EXPECTED_NC} (box,cell). "
                  f"data.yaml nc 확인 필요.")
            ok = False

    # output1: (1, 32, 160, 160)
    o1 = outs.get("output1")
    if o1 != EXPECTED_OUT1:
        print(f"[verify][FAIL] output1 != {EXPECTED_OUT1}: {o1}")
        ok = False

    print("[verify] 그래프 형상 검사:", "OK" if ok else "FAIL")
    return 0 if ok else 2


def check_runtime(model_path: str, image: str | None, conf: float) -> int:
    """실제 추론 래퍼 _YoloSegONNX 로 로드 + 추론 (onnxruntime 필요)."""
    try:
        from camera.ceiling_detector import _YoloSegONNX
    except Exception as e:  # noqa: BLE001
        print(f"[verify][ERROR] _YoloSegONNX import 실패: {e}")
        return 1

    try:
        ys = _YoloSegONNX(model_path, conf_thresh=conf)
        ys.set_class_names(["box", "cell"])
    except Exception as e:  # noqa: BLE001
        print(f"[verify][ERROR] ONNX 로드 실패 (onnxruntime 설치 확인): {e}")
        return 1

    print(f"[verify] 래퍼 로드 OK. input name={ys._in_name}, INPUT_SIZE={ys.INPUT_SIZE}")

    # 이미지 준비: 지정 없으면 더미 노이즈 (검출 0 이어도 추론이 도는지만 확인)
    if image:
        import cv2
        img = cv2.imread(image)
        if img is None:
            print(f"[verify][ERROR] 이미지 못 읽음: {image}")
            return 1
        print(f"[verify] 샘플 이미지 {image} ({img.shape[1]}x{img.shape[0]})")
    else:
        img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        print("[verify] 더미 480x640 노이즈 이미지 사용 (검출 0 정상)")

    # 원시 출력 형상 (래퍼가 실제로 소비하는 형태)
    blob, ratio, (dw, dh) = ys._preprocess(img)
    o0, o1 = ys._sess.run(None, {ys._in_name: blob})
    print(f"[verify] 런타임 out0={o0.shape} out1={o1.shape}")
    num_cls = o0.shape[1] - 4 - 32
    if num_cls != EXPECTED_NC:
        print(f"[verify][FAIL] 런타임 num_cls={num_cls} != {EXPECTED_NC}")
        return 2

    # 전체 infer 파이프라인
    try:
        dets = ys.infer(img)
    except Exception as e:  # noqa: BLE001
        print(f"[verify][ERROR] infer() 예외: {e}")
        return 1

    print(f"[verify] infer() OK — 검출 {len(dets)}개 (conf>={conf})")
    for label, score, box_xyxy, mask in dets[:10]:
        print(f"         {label:5s} score={score:.3f} box={box_xyxy.tolist()} "
              f"mask_px={int(mask.sum())}")
    if image and len(dets) == 0:
        print("[verify][경고] 샘플 이미지인데 검출 0 — 학습 품질/conf 임계 확인 필요.")
    print("[verify] 런타임 검사: OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="YOLOv8n-seg ONNX 호환 스모크 테스트")
    ap.add_argument("--model", required=True, help="export 된 ONNX 경로")
    ap.add_argument("--image", default=None, help="(선택) 샘플 이미지로 실제 검출 확인")
    ap.add_argument("--conf", type=float, default=0.25, help="검출 conf 임계값")
    ap.add_argument("--skip-runtime", action="store_true",
                    help="onnxruntime 없이 그래프 형상만 검사")
    args = ap.parse_args()

    if not pathlib.Path(args.model).is_file():
        print(f"[verify][ERROR] 모델 파일 없음: {args.model}")
        return 1

    print("=" * 70)
    print("[verify] 1) onnx 그래프 출력 형상 검사")
    rc_graph = check_graph_shapes(args.model)

    rc_rt = 0
    if not args.skip_runtime:
        print("-" * 70)
        print("[verify] 2) _YoloSegONNX 런타임 추론 검사")
        rc_rt = check_runtime(args.model, args.image, args.conf)
    else:
        print("[verify] (런타임 검사 건너뜀: --skip-runtime)")

    print("=" * 70)
    rc = rc_graph or rc_rt
    print("[verify] 최종:", "ALL OK" if rc == 0 else f"FAIL (rc={rc})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
