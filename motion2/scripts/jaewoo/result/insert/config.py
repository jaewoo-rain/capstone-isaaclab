"""rl/insert/config.py — insert 정책 sim2real 실행 상수 (단일 출처).

motion3 insert 정책(checkpoints/motion3_insert_v25.zip)을 실제 OMY-F3M 로봇에서
실행하기 위한 상수. motion3/tasks/insert/insert_env_cfg.py 에서 그대로 가져옴.

⚠️ insert 는 grasp 와 다르다:
   - obs 7 차원 (is_grasping 추가)
   - yaw_only 모드: 정책의 xy action 무시, xy 는 IK 가 셀에 고정. RL 은 yaw 만.
   - yaw 는 reference-anchored unwrap: 시작 그리퍼 yaw(_yaw_ref) ± yaw_margin 로만 제한.
   - 박스 180° 대칭 → yaw 오차 ±90° fold (fold_yaw_sym).
   값 변경 금지 (sim 학습 dynamics 와 어긋나면 정렬 깨짐).
"""
from __future__ import annotations

import math
import pathlib


# ─────────────────────────────────────────────────────────────────────────────
# 경로
# ─────────────────────────────────────────────────────────────────────────────
RL_DIR = pathlib.Path(__file__).resolve().parent
CKPT_PATH = RL_DIR / "checkpoints" / "motion3_insert_v25.zip"
VECNORM_PATH = RL_DIR / "checkpoints" / "motion3_insert_v25_vecnorm.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# 정책 규약 — insert_env_cfg.py 에서 그대로 (변경 금지)
#   obs(7): [slot_rel_x, slot_rel_y, slot_yaw_err, is_grasping, ee_vel_x, ee_vel_y, yaw_vel]
#   action(3): [Δx, Δy, Δyaw] — ★ yaw_only 라서 Δx,Δy 는 무시, Δyaw 만 사용
# ─────────────────────────────────────────────────────────────────────────────
ACTION_SCALE_XY = 0.005         # insert_env_cfg.action_scale_xy (yaw_only 라 미사용 — 참고)
ACTION_SCALE_YAW = 0.05         # insert_env_cfg.action_scale_yaw — ~2.86° / step

# ★ yaw-only: 정책은 obs 7개를 다 받지만(xy 포함), 출력 action 의 yaw 만 쓴다.
#   xy 는 IK 가 셀에 holding (정책 제어 X).
YAW_ONLY = True

# ★ reference-anchored unwrap (v25): 누적 yaw setpoint 를 _yaw_ref ± yaw_margin 로만 제한.
#   _yaw_ref = 시작(=turn 후) 그리퍼 yaw. 절대 ±π clamp 폐기 (wrap 경계 함정 회피).
YAW_MARGIN = 1.75               # insert_env_cfg.yaw_margin — ~100° (fold 한계 90° + 여유)

# 제어 주파수: sim dt(1/60) * decimation(1) = 1/60 (60 Hz)
CONTROL_DT = 1.0 / 60.0

# 정렬 성공 판정 (insert_env_cfg.py) — yaw-only 라 yaw + is_grasping 만 본다
ALIGN_XY_THRESHOLD = 0.010      # 10 mm (yaw_only 에선 success 판정에 미사용)
ALIGN_YAW_THRESHOLD = 0.087     # ~5°
SUCCESS_HOLD_STEPS = 15         # 0.25초 연속 정렬 유지 → 수렴
FAIL_XY_THRESHOLD = 0.30        # (yaw_only 에선 xy 고정이라 미발동)

# 롤아웃 안전 상한 (episode_length_s=15, 60Hz → 900 step). 넉넉히.
ROLLOUT_MAX_STEPS = 1200


# ─────────────────────────────────────────────────────────────────────────────
# 박스 대칭 / 셀 분포 (참고·경고용)
# ─────────────────────────────────────────────────────────────────────────────
BOX_SYMMETRIC = True            # 박스 180° 대칭 → yaw 오차 ±90° fold (fold_yaw_sym)
CELL_YAW_MAX = 0.175            # 셀 yaw 분포 ±10° (거의 일자). 실제 셀 각도가 이 안이어야.


# ─────────────────────────────────────────────────────────────────────────────
# Z 높이 / gripper — 정책과 무관. 기존 motion3 layout 값.
#   xy/yaw 만 정책/IK 가 결정. z 는 호버→place 하강(motion planning 전담).
# ─────────────────────────────────────────────────────────────────────────────
EE_FIXED_Z = 0.20               # INSERT_HOVER_Z — 셀 위 호버 (정렬 시 고정 높이)
# 박스 잡은 상태 유지용 그리퍼 명령 (insert 내내 close)
GRIPPER_CLOSE_CMD = 0.8
GRIPPER_TIP_RATIO = 2.3

# 실행 확인 문자열
CONFIRM_TEXT = "EXECUTE_INSERT"
