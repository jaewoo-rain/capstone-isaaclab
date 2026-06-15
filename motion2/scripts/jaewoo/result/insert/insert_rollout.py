"""rl/insert/insert_rollout.py — insert 정책 내부 롤아웃 (yaw-only).

insert 는 yaw_only RL: xy 는 IK 가 셀에 고정하고, 정책은 손목 yaw 만 정렬한다.
그래서 롤아웃도 yaw 1차원만 적분한다. xy 는 셀에 고정(slot_rel ≈ 0)으로 둔다.

sim 재현 (source/motion3/tasks/insert/insert_env.py):
  obs(7) = [cell_x-ee_x, cell_y-ee_y, fold_yaw_sym(cell_yaw-ee_yaw), is_grasping,
            ee_vel_x, ee_vel_y, yaw_vel]
  action(3) → ee_yaw = clamp(ee_yaw + a[2]*scale_yaw, yaw_ref±margin)   # a[:2] 무시
  yaw_err 은 ±90° fold (박스 180° 대칭). is_grasping 은 잡은 상태라 1.0.

수렴: |fold(cell_yaw - ee_yaw)| < 5° 를 15 step 연속 유지 (is_grasping=1 가정).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import config as C


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def fold_yaw_sym(a: float) -> float:
    """박스 180° 대칭 → yaw 오차를 ±90° 로 fold (insert_env.fold_yaw_sym 과 동일)."""
    return (a + math.pi / 2.0) % math.pi - math.pi / 2.0


@dataclass
class InsertRolloutResult:
    ee_yaw: float                # 최종 정렬 EE yaw [rad]
    converged: bool
    status: str                  # "converged" | "max_steps"
    n_steps: int
    start_yaw: float
    cell_yaw: float
    final_yaw_err: float         # |fold(cell_yaw - ee_yaw)| [rad]


def rollout_insert(
    policy,
    cell_yaw: float,
    start_ee_yaw: float,
    *,
    cell_xy=(0.0, 0.0),
    start_ee_xy=None,
) -> InsertRolloutResult:
    """insert 정책(yaw-only)을 내부에서 굴려 최종 정렬 EE yaw 를 계산한다.

    Args:
        policy:        InsertPolicy 인스턴스
        cell_yaw:      목표 셀 yaw [rad]
        start_ee_yaw:  롤아웃 시작 EE yaw [rad] (= turn 후 그리퍼 yaw, ~±180°)
        cell_xy:       셀 xy (slot_rel 계산용. IK 고정이라 보통 ee_xy 와 같다고 봄)
        start_ee_xy:   시작 EE xy (None 이면 cell_xy 와 동일 — IK holding 가정 → slot_rel=0)

    Returns:
        InsertRolloutResult (핵심 출력 = ee_yaw)
    """
    cell_xy = np.asarray(cell_xy, dtype=np.float64)
    ee_xy = cell_xy.copy() if start_ee_xy is None else np.asarray(start_ee_xy, dtype=np.float64)

    ee_yaw = float(start_ee_yaw)
    yaw_ref = float(start_ee_yaw)         # ★ reference anchor (v25)
    prev_ee_yaw = ee_yaw

    aligned_count = 0
    status = "max_steps"

    for step in range(C.ROLLOUT_MAX_STEPS):
        # ── obs 구성 (sim _get_observations 재현) ──
        slot_rel = cell_xy - ee_xy        # IK 고정 → ≈ 0
        yaw_err = fold_yaw_sym(cell_yaw - ee_yaw)
        yaw_vel = (ee_yaw - prev_ee_yaw) / C.CONTROL_DT
        is_grasping = 1.0                 # 박스 잡은 상태 (insert 내내 close)
        obs = np.array(
            [slot_rel[0], slot_rel[1], yaw_err, is_grasping, 0.0, 0.0, yaw_vel],
            dtype=np.float64,
        )

        # ── 정렬 판정 (yaw_only: yaw + is_grasping) ──
        if abs(yaw_err) < C.ALIGN_YAW_THRESHOLD:
            aligned_count += 1
        else:
            aligned_count = 0
        if aligned_count >= C.SUCCESS_HOLD_STEPS:
            status = "converged"
            break

        # ── 정책 추론 + yaw 적용 (xy action 무시 — yaw_only) ──
        action = policy.predict(obs)
        prev_ee_yaw = ee_yaw
        ee_yaw = float(np.clip(
            ee_yaw + action[2] * C.ACTION_SCALE_YAW,
            yaw_ref - C.YAW_MARGIN, yaw_ref + C.YAW_MARGIN))

    final_yaw_err = abs(fold_yaw_sym(cell_yaw - ee_yaw))
    return InsertRolloutResult(
        ee_yaw=ee_yaw,
        converged=(status == "converged"),
        status=status,
        n_steps=step + 1,
        start_yaw=float(start_ee_yaw),
        cell_yaw=float(cell_yaw),
        final_yaw_err=final_yaw_err,
    )


if __name__ == "__main__":
    from insert_policy import InsertPolicy

    pol = InsertPolicy()
    # turn 후 그리퍼 yaw ~ π(180°). 셀 yaw 가 그와 0.3rad 어긋난 상황 (fold 로 봄).
    start_ee_yaw = math.pi - 0.3
    cell_yaw = math.pi
    res = rollout_insert(pol, cell_yaw, start_ee_yaw)
    print(f"[insert rollout] status={res.status} converged={res.converged} steps={res.n_steps}")
    print(f"  start ee yaw : {res.start_yaw:.4f} rad ({math.degrees(res.start_yaw):.1f}°)")
    print(f"  final ee yaw : {res.ee_yaw:.4f} rad ({math.degrees(res.ee_yaw):.1f}°)")
    print(f"  cell yaw     : {res.cell_yaw:.4f} rad ({math.degrees(res.cell_yaw):.1f}°)")
    print(f"  final yaw err: {math.degrees(res.final_yaw_err):.2f}°  (목표 < 5°)")
