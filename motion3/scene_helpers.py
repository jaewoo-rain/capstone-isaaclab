"""motion3 — chain runner / collect 가 공유하는 scene 기하 헬퍼.

3×3 grid 적재 셀의 격벽(wall) 기하와 셀 타깃 좌표를 layout.py 단일 출처에서 생성한다.
순수 기하(파이썬 float/list) — isaaclab 무의존, 수치 검증 가능. chain·collect 양쪽이 import.

격벽 규약 (example7 _spawn_grid_walls 패턴):
  - grid: GRID_NUM_X(깊이,X) × GRID_NUM_Y(좌우,Y) 셀.
  - X(깊이) 칸 경계 벽: nx+1 개. X에 얇고 Y로 길다.
  - Y(좌우) 칸 경계 벽: ny+1 개. Y에 얇고 X로 길다.
  - 모든 좌표는 grid 중심 기준 local. 실제 배치 시 grid_center + (선택)yaw 회전.
"""
from __future__ import annotations

import math

from source.motion3 import layout


def grid_extent() -> tuple[float, float]:
    """grid 전체 footprint (X 깊이, Y 좌우) — 벽 두께 포함."""
    ex = layout.GRID_NUM_X * layout.CELL_INNER_X + (layout.GRID_NUM_X + 1) * layout.WALL_THICKNESS
    ey = layout.GRID_NUM_Y * layout.CELL_INNER_Y + (layout.GRID_NUM_Y + 1) * layout.WALL_THICKNESS
    return ex, ey


def wall_specs() -> list[tuple[str, tuple[float, float, float], tuple[float, float]]]:
    """8개 격벽: (이름, size(x,y,z), grid중심 기준 local_xy(x,y)).

    X 경계벽(깊이 칸막이): nx+1개, size=(t, Ey, h).
    Y 경계벽(좌우 칸막이): ny+1개, size=(Ex, t, h).
    """
    nx, ny = layout.GRID_NUM_X, layout.GRID_NUM_Y
    sx, sy = layout.CELL_INNER_X, layout.CELL_INNER_Y   # 칸 내부 깊이(X)/좌우(Y)
    t, h = layout.WALL_THICKNESS, layout.WALL_HEIGHT
    ex, ey = grid_extent()
    specs: list[tuple[str, tuple[float, float, float], tuple[float, float]]] = []
    # X(깊이) 경계벽: i=0..nx, x = -ex/2 + t/2 + i*(sx+t), 전체 Y길이
    for i in range(nx + 1):
        x = -ex / 2.0 + t / 2.0 + i * (sx + t)
        specs.append((f"gw_x{i}", (t, ey, h), (x, 0.0)))
    # Y(좌우) 경계벽: j=0..ny, y = -ey/2 + t/2 + j*(sy+t), 전체 X길이
    for j in range(ny + 1):
        y = -ey / 2.0 + t / 2.0 + j * (sy + t)
        specs.append((f"gw_y{j}", (ex, t, h), (0.0, y)))
    return specs


def cell_centers_local() -> list[tuple[float, float]]:
    """9개 셀 중심의 grid중심 기준 local xy (row-major: j=좌우행, i=깊이열)."""
    nx, ny = layout.GRID_NUM_X, layout.GRID_NUM_Y
    px, py = layout.CELL_PITCH_X, layout.CELL_PITCH_Y   # 깊이 0.073, 좌우 0.168
    cells: list[tuple[float, float]] = []
    for j in range(ny):
        for i in range(nx):
            x = -(nx - 1) * px / 2.0 + i * px
            y = -(ny - 1) * py / 2.0 + j * py
            cells.append((x, y))
    return cells


def rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    return c * x - s * y, s * x + c * y


def grid_world_poses(grid_cx: float, grid_cy: float, grid_yaw: float):
    """주어진 grid pose에서 (벽 world pose 목록, 셀 중심 world xy 목록) 반환.

    벽: [(name, (wx, wy, wz=WALL_HEIGHT/2), yaw)], 셀: [(cx, cy)].
    grid 전체에 grid_yaw 회전 + (grid_cx, grid_cy) 평행이동 적용. (env-rel, ground=0 기준)
    """
    wz = layout.WALL_HEIGHT / 2.0
    walls = []
    for name, size, (lx, ly) in wall_specs():
        rx, ry = rotate_xy(lx, ly, grid_yaw)
        walls.append((name, size, (grid_cx + rx, grid_cy + ry, wz), grid_yaw))
    cells = []
    for (lx, ly) in cell_centers_local():
        rx, ry = rotate_xy(lx, ly, grid_yaw)
        cells.append((grid_cx + rx, grid_cy + ry))
    return walls, cells
