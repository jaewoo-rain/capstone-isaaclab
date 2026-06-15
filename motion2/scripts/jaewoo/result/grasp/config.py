"""rl/config.py — grasp 정책 sim2real 실행 상수 (단일 출처).

motion1 grasp 정책(checkpoints/motion1_grasp.zip)을 실제 OMY-F3M 로봇에서
"내부 롤아웃 → 단일 이동" 방식으로 실행하기 위한 모든 상수를 한 곳에 모은다.

⚠️ 여기 값들은 sim 학습 env(source/motion1/tasks/grasp/grasp_env_cfg.py)에서
   그대로 가져온 것. sim 정책을 1:1 재현하려면 절대 임의로 바꾸지 말 것.
   (값을 바꾸면 정책이 학습한 dynamics 와 어긋나 정렬이 깨진다.)
"""
from __future__ import annotations

import math
import pathlib


# ─────────────────────────────────────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────────────────────────────────────
RL_DIR = pathlib.Path(__file__).resolve().parent
CKPT_PATH = RL_DIR / "checkpoints" / "motion1_grasp.zip"
VECNORM_PATH = RL_DIR / "checkpoints" / "motion1_grasp_vecnorm.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# 정책 규약 — sim grasp_env_cfg.py 에서 그대로 (변경 금지)
#   obs(6): [obj_rel_x, obj_rel_y, obj_yaw_err, ee_vel_x, ee_vel_y, yaw_vel]
#   action(3): [Δx, Δy, Δyaw] ∈ [-1, 1], 누적 적용
# ─────────────────────────────────────────────────────────────────────────────
ACTION_SCALE_XY = 0.01          # grasp_env_cfg.action_scale_xy — 10 mm / step
ACTION_SCALE_YAW = 0.05         # grasp_env_cfg.action_scale_yaw — ~2.86° / step
EE_YAW_MIN = -math.pi / 2       # grasp_env_cfg.ee_yaw_min
EE_YAW_MAX = math.pi / 2        # grasp_env_cfg.ee_yaw_max

# 제어 주파수: sim dt(1/120) * decimation(2) = 1/60 (60 Hz)
CONTROL_DT = 1.0 / 60.0

# 정렬 성공 판정 (grasp_env_cfg.py) — 롤아웃 수렴 종료 조건
ALIGN_XY_THRESHOLD = 0.005      # 5 mm (각 축)
ALIGN_YAW_THRESHOLD = 0.05      # ~2.86°
SUCCESS_HOLD_STEPS = 30         # 0.5초 연속 정렬 유지 → 수렴
FAIL_XY_THRESHOLD = 0.30        # EE가 박스에서 30cm 이상 벌어지면 발산으로 간주

# 롤아웃 안전 상한 (sim episode_length_s=5초 → 300 step). 넉넉히 2배.
ROLLOUT_MAX_STEPS = 600


# ─────────────────────────────────────────────────────────────────────────────
# 정책 학습 분포 (grasp_env_cfg.py) — 입력 박스 좌표 검증용 경고 범위
#   sim env-rel = OMY base(link0) 기준. 실제 로봇과 같은 로봇이므로 동일 좌표계로 본다.
#   박스가 이 범위 밖이면 정책이 학습하지 않은 영역 → 경고.
# ─────────────────────────────────────────────────────────────────────────────
BOX_DIST_CENTER_XY = (0.45, -0.10)   # box_spawn_xy
BOX_DIST_XY_NOISE = 0.10             # box_spawn_xy_noise (±10cm)
BOX_DIST_YAW_MAX = 1.396             # box_spawn_yaw_max (±80°)

# EE 초기 위치 (sim reset 기준). 시작 EE가 박스에서 너무 멀면 학습 분포 밖.
EE_INIT_XY = (0.46, -0.30)


# ─────────────────────────────────────────────────────────────────────────────
# grip center ↔ link6 offset 보정
#   sim 정책은 "양 finger 평균(grip center)" 을 박스에 정렬한다.
#   실제 로봇 TF는 link6 를 읽는다. link6 와 grip center 사이에 offset 이 있으면
#   그만큼 정렬이 어긋난다. base(link0) 기준 xy offset [m].
#   ⚠️ 1차값 0 — 실물 검증(show_ee_pose 로 link6 vs 실제 그리퍼 중심 비교) 후 채울 것.
# ─────────────────────────────────────────────────────────────────────────────
GRIP_CENTER_OFFSET_XY = (0.0, 0.0)   # (dx, dy): grip_center = link6_xy + offset


# ─────────────────────────────────────────────────────────────────────────────
# Z 높이 / orientation — 정책과 무관. 기존 motion2 값 그대로.
#   xy/yaw 만 정책이 결정하고, z 는 아래 값(또는 YAML)을 쓴다.
# ─────────────────────────────────────────────────────────────────────────────
# YAML waypoints 에서 z 를 읽는다. 못 읽을 때의 fallback (run_pick_place 기본과 동일).
APPROACH_Z_FALLBACK = 0.372     # 호버 높이 (박스 위)
GRASP_Z_FALLBACK = 0.345        # 파지 하강 높이
LIFT_OFFSET = 0.06              # approach_z 위로 추가로 들어올리는 높이

# 실행 확인 문자열
CONFIRM_TEXT = "EXECUTE_GRASP"
