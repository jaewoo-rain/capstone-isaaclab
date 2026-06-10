"""rl/all/config.py — 전체 파이프라인(grasp→insert 적재) 상수 (단일 출처).

motion3 통합 chain(play_motion_chain_with_grasp_insert.py)을 실제 OMY-F3M 로봇에서
한 번에 실행하기 위한 상수. grasp + insert 규약을 모두 포함한다.

⚠️ all/ 은 독립 모듈이다 — grasp/insert 폴더의 config.py 와 이름이 충돌하므로
   import 하지 않고, 두 폴더의 체크포인트(zip/pkl)만 경로로 참조한다.

⚠️⚠️ 좌표(z 높이)·관절(turn/home)은 sim(ground 기준) 값이다. 실제 로봇 link0 기준으로
     반드시 **실측·보정**해야 한다. 아래 SIM 표시 값은 출발점일 뿐 검증 안 됨.
"""
from __future__ import annotations

import math
import pathlib


# ─────────────────────────────────────────────────────────────────────────────
# 정책 체크포인트 — grasp/insert 폴더 것을 재사용 (복사 안 함)
# ─────────────────────────────────────────────────────────────────────────────
_RL_ROOT = pathlib.Path(__file__).resolve().parents[1]   # .../rl
GRASP_CKPT = _RL_ROOT / "grasp" / "checkpoints" / "motion1_grasp.zip"
GRASP_VECNORM = _RL_ROOT / "grasp" / "checkpoints" / "motion1_grasp_vecnorm.pkl"
INSERT_CKPT = _RL_ROOT / "insert" / "checkpoints" / "motion3_insert_v25.zip"
INSERT_VECNORM = _RL_ROOT / "insert" / "checkpoints" / "motion3_insert_v25_vecnorm.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# 제어 공통
# ─────────────────────────────────────────────────────────────────────────────
CONTROL_DT = 1.0 / 60.0
ROLLOUT_MAX_STEPS = 1200


# ─────────────────────────────────────────────────────────────────────────────
# GRASP 규약 (grasp_env_cfg.py) — obs6, xy+yaw
# ─────────────────────────────────────────────────────────────────────────────
GRASP_ACTION_SCALE_XY = 0.01
GRASP_ACTION_SCALE_YAW = 0.05
GRASP_EE_YAW_MIN = -math.pi / 2
GRASP_EE_YAW_MAX = math.pi / 2
GRASP_ALIGN_XY = 0.005
GRASP_ALIGN_YAW = 0.05
GRASP_HOLD_STEPS = 30
GRASP_FAIL_XY = 0.30
# 학습 분포 (앞쪽 박스, 경고용)
GRASP_BOX_CENTER_XY = (0.45, -0.10)
GRASP_BOX_XY_NOISE = 0.10
GRASP_BOX_YAW_MAX = 1.396


# ─────────────────────────────────────────────────────────────────────────────
# INSERT 규약 (insert_env_cfg.py) — obs7, yaw-only
# ─────────────────────────────────────────────────────────────────────────────
INSERT_ACTION_SCALE_YAW = 0.05
INSERT_YAW_MARGIN = 1.75
INSERT_ALIGN_YAW = 0.087
INSERT_HOLD_STEPS = 15
INSERT_CELL_YAW_MAX = 0.175      # ±10°


# ─────────────────────────────────────────────────────────────────────────────
# EE 기준자세 (양쪽 공통, 실제 로봇 수직파지)
#   sim base_ee_quat=(0,1,0,0). 실제 로봇은 run_ee_pose_move 의 VERTICAL_GRIP_QUAT.
#   run_pipeline 에서 R_z(yaw) ⊗ VERTICAL_GRIP_QUAT 로 합성.
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# ★ Stage z 높이 — ⚠️ SIM(ground 기준) 값. 실제 로봇 link0 기준으로 실측·보정 필요.
#   grasp sim2real 과 동일 원칙: 정책은 z 무관(xy/yaw만), z 는 실측값을 쓴다.
# ─────────────────────────────────────────────────────────────────────────────
PRE_GRASP_Z = 0.55     # SIM layout.PRE_GRASP_Z — 앞쪽 박스 위 호버
GRASP_Z = 0.455        # SIM GRASP_Z — 파지 깊이
LIFT_Z = 0.56          # SIM LIFT_Z — 들어올림
HOVER_Z = 0.32         # SIM RL_HOVER_Z — 뒤쪽 셀 위 insert 정렬 높이
PLACE_Z = 0.07         # SIM PLACE_Z — 셀 바닥 안착  ⚠️ 하강 드리프트 병목
RETRACT_Z = 0.56       # SIM RETRACT_Z — 복귀 높이


# ─────────────────────────────────────────────────────────────────────────────
# ★ Turn-around / Home 관절 — ⚠️ SIM 자세. 실제 로봇 관절과 일치 확인 필요.
#   turn = joint1 만 회전(joint2~6 은 lift 자세 유지). 실제론 lift 후 현재 관절 읽고 joint1 만 변경.
# ─────────────────────────────────────────────────────────────────────────────
TURN_JOINT1 = -2.90              # SIM BACK_HOME_JOINTS["joint1"] — 뒤로 ~180° 회전
HOME_JOINTS = {                  # SIM HOME_JOINT_POS (arm6)
    "joint1": 0.0, "joint2": -1.55, "joint3": 2.66,
    "joint4": -1.1, "joint5": 1.6, "joint6": 0.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# 단계별 이동 시간 [s] (최소 4.0)
# ─────────────────────────────────────────────────────────────────────────────
DUR_APPROACH = 8.0
DUR_DESCEND = 6.0
DUR_LIFT = 6.0
DUR_TURN = 8.0          # joint-space 큰 회전 — 여유
DUR_TRANSPORT = 8.0
DUR_HOVER_DESCEND = 6.0
DUR_PLACE = 8.0         # 느린 하강 (드리프트 감소)
DUR_RETRACT = 6.0
DUR_HOME = 9.0


# ─────────────────────────────────────────────────────────────────────────────
# Gripper
# ─────────────────────────────────────────────────────────────────────────────
GRIPPER_CLOSE_CMD = 0.8
GRIPPER_OPEN_CMD = 0.0
GRIPPER_TIP_RATIO = 2.3

# 실행 확인 문자열
CONFIRM_TEXT = "EXECUTE_PIPELINE"
