"""camera/dataset/annotate_cells.py — 셀 그리드 고정 ROI 1회 클릭 라벨링.

셀 라벨링 전략 (a): 천장캠은 정적, 셀 그리드는 고정 구조물 → 셀 폴리곤을 1회만
정의해 cell_template.yaml 로 저장하고 auto_label_depth.py 가 전 프레임에 재사용한다.
(셀 슬롯은 오목/평면이라 박스용 'depth 솟음' 분할이 안 통하므로 별도 처리.)

사용:
  # 한 장의 대표 color 프레임 위에 셀 폴리곤을 클릭으로 그린다
  python3 annotate_cells.py --image dataset/raw/d435i/0000_color.png

조작:
  좌클릭        = 현재 폴리곤에 점 추가
  n            = 현재 폴리곤 확정하고 다음 셀 시작
  u            = 마지막 점 취소
  s            = cell_template.yaml 저장(이미지와 같은 폴더)
  q            = 저장 없이 종료

출력: <image 와 같은 폴더>/cell_template.yaml
  cells:
    - [[u0,v0],[u1,v1],...]   # 셀 1
    - [[...]]                  # 셀 2
"""
from __future__ import annotations

import argparse
import pathlib
from typing import List, Tuple

import cv2
import numpy as np
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(
        description="셀 그리드 고정 ROI 클릭 라벨 → cell_template.yaml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="대표 color 프레임 경로")
    parser.add_argument("--out", default=None,
                        help="출력 yaml(기본: 이미지 폴더/cell_template.yaml)")
    args = parser.parse_args()

    img_path = pathlib.Path(args.image)
    base = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if base is None:
        raise FileNotFoundError(f"이미지 없음: {img_path}")
    out_path = pathlib.Path(args.out) if args.out else img_path.parent / "cell_template.yaml"

    cells: List[List[Tuple[int, int]]] = []
    current: List[Tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append((x, y))

    win = "annotate cells (n=next s=save u=undo q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("[annotate] 좌클릭=점추가  n=셀확정  u=점취소  s=저장  q=종료")

    while True:
        vis = base.copy()
        for poly in cells:
            cv2.polylines(vis, [np.array(poly, np.int32)], True, (0, 165, 255), 2)
        if current:
            cv2.polylines(vis, [np.array(current, np.int32)], False, (0, 255, 0), 1)
            for p in current:
                cv2.circle(vis, p, 3, (0, 0, 255), -1)
        cv2.putText(vis, f"cells={len(cells)} cur_pts={len(current)}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("q"):
            print("[annotate] 저장 없이 종료")
            break
        if k == ord("u") and current:
            current.pop()
        if k == ord("n") and len(current) >= 3:
            cells.append(current[:])
            current.clear()
            print(f"  셀 {len(cells)} 확정")
        if k == ord("s"):
            if current and len(current) >= 3:
                cells.append(current[:])
                current.clear()
            with open(out_path, "w") as f:
                yaml.safe_dump(
                    {"cells": [[[int(u), int(v)] for u, v in poly] for poly in cells]},
                    f, default_flow_style=None, sort_keys=False)
            print(f"[annotate] 저장: {out_path} (셀 {len(cells)}개)")
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
