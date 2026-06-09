"""motion3 — 책상 위 로봇(앞쪽 블록) + 책상 아래 뒤쪽 3×3 셀 레이아웃의 단일 출처.

S0 측정으로 확정한 좌표 규약 (robotis_lab 규약 응용):
  ground plane z = 0  = 뒤쪽 적재 셀이 놓이는 바닥 (cell floor)
  로봇 base / 책상 상판 = z = TABLE_HEIGHT  (cell floor보다 12cm 위로 elevated)

  → 앞쪽(블록 pick, grasp, lift)은 전부 "책상 위" 평면(motion1 값 + TABLE_HEIGHT).
    앞쪽은 robot base와 함께 통째로 +TABLE_HEIGHT 올라가므로 블록과의 상대 기하가
    motion1과 동일 → grasp 정책(motion1_grasp.zip) 그대로 재사용 + reach 변화 없음.
  → 뒤쪽(insert 정렬, place)은 ground 프레임(z=0 바닥). 셀이 base보다 12cm 아래라
    reach 여유 있음(외곽 셀까지 ~0.48m < OMY 0.58m).

⚠️ 이 파일은 isaaclab을 import하지 않는다(순수 float/tuple). app 런칭 전/후 아무 때나
   import 가능 — cfg / chain / collect 가 전부 여기서 z 상수를 가져다 써서 "3곳 손동기화"
   미스매치(v1~v14 실패 원인)를 구조적으로 차단.
"""
from __future__ import annotations

# =========================================================================
# 1. 책상 높이 (= robot base z, cell floor(ground=0) 기준).
#    셀을 로봇 가까이 두면 reach 구(√(수평²+수직²)≤0.58)에서 수직 여유가 커져 더 깊이 가능.
#    앞쪽(블록/grasp/lift)은 base와 통째로 +H 올라가 상대기하 motion1과 동일 → H 무관하게 도달.
# =========================================================================
TABLE_HEIGHT: float = 0.30   # 책상 상판(=robot base) z. 셀 바닥보다 ~24cm 위. S1: 0.35는 far 열 reach 초과 → 0.30.

# =========================================================================
# 2. 박스(pick) — 책상 위 앞쪽(+x). motion1 분포 유지 + 책상 높이만큼 z shift.
# =========================================================================
# 박스: 서있는 형태(높이 z=0.139 최장, 벽 0.12 위로 빠져나오는 게 정상).
# 앞뒤(x)=0.118(깊은 셀로 들어감), 좌우(y)=0.044(잡는 변), 높이(z)=0.139.
BOX_SIZE: tuple[float, float, float] = (0.118, 0.044, 0.139)
_BOX_HALF_Z = BOX_SIZE[2] / 2.0   # 0.0695
BOX_SPAWN: tuple[float, float, float] = (0.35 , 0.0, _BOX_HALF_Z + 0.005 + TABLE_HEIGHT)  # 책상 위 안착
BOX_SPAWN_XY_NOISE: float = 0.10   # ±10cm
BOX_SPAWN_YAW_MAX: float = 1.396   # ±80°

# =========================================================================
# 3. z 상수
#    앞쪽(책상 평면) = motion1 값 + TABLE_HEIGHT. 뒤쪽(insert/place) = ground 프레임.
# =========================================================================
# --- 앞쪽(책상 위): 박스 서있음(높이 0.139, 꼭대기=책상+0.139). 손가락이 아래로 뻗어
#     접근 시 박스 꼭대기에 닿지 않게 호버를 충분히 높임. ---
PRE_GRASP_Z: float = 0.25 + TABLE_HEIGHT    # 0.57 — 박스 꼭대기(0.439) 위 13cm 호버 (더 높이)
GRASP_Z: float = 0.155 + TABLE_HEIGHT        # 0.45 — 파지 깊이 (박스 위쪽을 잡음)
LIFT_Z: float = 0.26 + TABLE_HEIGHT         # 0.56 — 책상 위로 들어올림 → 뒤로 운반 높이

# --- 뒤쪽(ground 프레임): insert 정렬은 저고도 hover, place는 셀 바닥 안착 ---
INSERT_HOVER_Z: float = 0.20   # insert RL ee_fixed_z. 셀 wall top(0.12) 위 → 박스 밑면 충돌 회피.
PLACE_Z: float = 0.070         # 셀 바닥(ground) 안착 시 박스 중심(=_BOX_HALF_Z 0.0695). motion 전담 하강.

# 운반/하강: LIFT_Z 0.56(앞,책상) → INSERT_HOVER_Z 0.20(뒤,ground) 운반 + descend 0.20→0.065.
# grasp→place 순 낙차 0.415→0.065 = 0.35m. 셀은 책상 상판(0.30)보다 아래.

# =========================================================================
# 4. 3×3 grid 셀 — 로봇 뒤쪽(-x) ground(z=0 바닥), 로봇 가까이 붙여 배치.
#    "일자 wide" 배치: 좌우(Y)로 길게 펼치고 뒤쪽(X) 깊이는 얕게 → 전부 base 가까이.
#    각 셀: 긴 변 0.16(박스 long edge 0.139 수용)을 Y(좌우)로, 짧은 변 0.065를 X(깊이)로.
# =========================================================================
CELL_GRID_CENTER: tuple[float, float] = (-0.38, 0.0)   # 뒤쪽(-x). 2깊이가 reach band[0.27~0.50] 안에 들도록.
GRID_NUM_X: int = 2   # X(깊이, 뒤쪽) 방향 셀 수 — 3깊이는 reach 초과 → 2깊이
GRID_NUM_Y: int = 5   # Y(좌우) 방향 셀 수 → 5좌우 × 2깊이 = 10칸
# 앞뒤로 깊은 셀: 박스 긴변(0.139)을 X(앞뒤/depth)에, 짧은변(0.044)을 Y(좌우)에.
CELL_INNER_X: float = 0.16     # 셀 앞뒤(깊이) — 박스 long edge 0.139 + 여유
CELL_INNER_Y: float = 0.065    # 셀 좌우 — 박스 short edge 0.044 + 여유
WALL_THICKNESS: float = 0.008
WALL_HEIGHT: float = 0.12
CELL_PITCH_X: float = CELL_INNER_X + WALL_THICKNESS   # 0.168 (앞뒤, 깊게)
CELL_PITCH_Y: float = CELL_INNER_Y + WALL_THICKNESS   # 0.073 (좌우, 좁게)
# 그리드 footprint: X(깊이) 3*0.168=0.504, Y(좌우) 3*0.073=0.219  ← 앞뒤로 긴 그리드

# 셀 randomization (타깃 셀 기준). 적재 장소는 "거의 항상 일자" → yaw ±10°만.
CELL_SPAWN_XY_NOISE: float = 0.03    # ±3cm (그리드 전체가 거의 고정 → 작게)
CELL_SPAWN_YAW_MAX: float = 0.175    # ±10° (≈0.175 rad). 일자 적재.

# transport 끝(=insert RL 시작) ee xy noise — dataset 자체에 주입(정렬 학습용)
EE_OFFSET_MIN_M: float = 0.03
EE_OFFSET_MAX_M: float = 0.05

# grasp(앞쪽) 단계 joint1 클램프 — 베이스가 크게 휘청이지 않게 제한.
# 박스 정면(~-0.3 rad) 향하는 정도는 허용, 큰 swing 차단. 박스 yaw는 손목(joint6)이 처리.
GRASP_JOINT1_CLAMP: tuple[float, float] = (-1.0, 0.9)

# =========================================================================
# 5. 책상 플랫폼(앞쪽 블록이 놓이는 콜리전 면). chain/collect에서 사용.
#    상판 top = TABLE_HEIGHT. 뒤쪽(-x)은 비워 셀이 ground에 놓이게 한다.
# =========================================================================
DESK_TOP_Z: float = TABLE_HEIGHT          # 0.12
DESK_CENTER: tuple[float, float] = (0.30, -0.10)   # 앞쪽 작업영역 중심
DESK_SIZE_XY: tuple[float, float] = (0.70, 0.70)   # 앞쪽 working area 덮기 (x: -0.05~0.65)


def build_cell_centers() -> list[tuple[float, float, float]]:
    """3×3 셀 중심 좌표 (env-rel, z=PLACE_Z 안착 평면). example7 _build_cell_centers 패턴.

    row-major (j=행, i=열).
    """
    cx, cy = CELL_GRID_CENTER
    nx, ny = GRID_NUM_X, GRID_NUM_Y
    px, py = CELL_PITCH_X, CELL_PITCH_Y
    cells: list[tuple[float, float, float]] = []
    for j in range(ny):
        for i in range(nx):
            x = cx - (nx - 1) * px / 2 + i * px
            y = cy - (ny - 1) * py / 2 + j * py
            cells.append((x, y, PLACE_Z))
    return cells
