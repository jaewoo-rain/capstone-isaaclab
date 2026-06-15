"""rl/grasp_rollout.py — grasp 정책 내부 롤아웃 (박스 고정 + EE 가상 적분).

실제 로봇 인프라(MoveIt)는 60Hz 폐루프를 못 돌리므로, 정책을 "파이썬 안에서"
가상으로 굴려 최종 정렬 EE pose(xy, yaw)를 미리 계산한다. 박스는 고정이고
EE 만 sim 의 action 적용식과 똑같이 적분하므로, sim 의 폐루프 정렬과 거의 동일하다.

sim 재현 (source/motion1/tasks/grasp/grasp_env.py):
  obs(6) = [box_x-ee_x, box_y-ee_y, wrap(box_yaw-ee_yaw), ee_vel_x, ee_vel_y, yaw_vel]
  action(3) → ee_xy += a[:2]*scale_xy ; ee_yaw = clip(ee_yaw + a[2]*scale_yaw, ±π/2)
  ee_vel/yaw_vel = 명령 미분 (즉시 도달 가정 — action_scale 이 작아 IK 1-step 수렴 OK)

수렴: |Δx|,|Δy| < 5mm AND |yaw_err| < 2.86° 를 30 step 연속 유지.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import config as C


def wrap_to_pi(angle: float) -> float:
    """각도를 [-π, π] 로 wrap (sim grasp_env.wrap_to_pi 와 동일)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class RolloutResult:
    ee_xy: np.ndarray            # 최종 정렬 EE xy (base 기준) [m]
    ee_yaw: float                # 최종 정렬 EE yaw [rad]
    converged: bool              # 정렬 수렴 여부
    status: str                  # "converged" | "diverged" | "max_steps"
    n_steps: int                 # 소요 step 수
    start_xy: np.ndarray         # 시작 EE xy
    start_yaw: float             # 시작 EE yaw
    final_xy_err: np.ndarray     # 최종 |box_xy - ee_xy|
    final_yaw_err: float         # 최종 |wrap(box_yaw - ee_yaw)|
    trajectory: list = field(default_factory=list)  # [(ee_x, ee_y, ee_yaw), ...]


def rollout_grasp(
    policy,
    box_xy,
    box_yaw: float,
    start_ee_xy,
    start_ee_yaw: float = 0.0,
    *,
    align_yaw: bool = True,
    record_traj: bool = False,
) -> RolloutResult:
    """grasp 정책을 내부에서 굴려 최종 정렬 EE pose 를 계산한다.

    Args:
        policy:        GraspPolicy 인스턴스 (predict(obs6)->action3)
        box_xy:        박스 중심 (base x, y) [m]
        box_yaw:       박스 yaw [rad]
        start_ee_xy:   롤아웃 시작 EE xy (보통 현재 실제 grip center) [m]
        start_ee_yaw:  롤아웃 시작 EE yaw [rad] (sim reset 기본 0.0)
        align_yaw:     False 면 yaw 를 0 으로 고정(검증 1단계용 — xy 만 정렬)
        record_traj:   True 면 step 별 (x,y,yaw) 궤적 기록

    Returns:
        RolloutResult
    """
    box_xy = np.asarray(box_xy, dtype=np.float64)
    ee_xy = np.asarray(start_ee_xy, dtype=np.float64).copy()
    ee_yaw = float(start_ee_yaw)

    prev_ee_xy = ee_xy.copy()
    prev_ee_yaw = ee_yaw

    aligned_count = 0
    traj: list = []
    status = "max_steps"

    for step in range(C.ROLLOUT_MAX_STEPS):
        # ── obs 구성 (sim _get_observations 재현) ──
        obj_rel = box_xy - ee_xy
        yaw_err = wrap_to_pi(box_yaw - ee_yaw)
        ee_vel = (ee_xy - prev_ee_xy) / C.CONTROL_DT
        yaw_vel = (ee_yaw - prev_ee_yaw) / C.CONTROL_DT
        obs = np.array(
            [obj_rel[0], obj_rel[1], yaw_err, ee_vel[0], ee_vel[1], yaw_vel],
            dtype=np.float64,
        )

        # ── 정렬/발산 판정 (action 적용 전 상태 기준) ──
        is_aligned = (
            abs(obj_rel[0]) < C.ALIGN_XY_THRESHOLD
            and abs(obj_rel[1]) < C.ALIGN_XY_THRESHOLD
            and abs(yaw_err) < C.ALIGN_YAW_THRESHOLD
        )
        aligned_count = aligned_count + 1 if is_aligned else 0
        if aligned_count >= C.SUCCESS_HOLD_STEPS:
            status = "converged"
            break
        if abs(obj_rel[0]) > C.FAIL_XY_THRESHOLD or abs(obj_rel[1]) > C.FAIL_XY_THRESHOLD:
            status = "diverged"
            break

        # ── 정책 추론 + action 적용 (sim _pre_physics_step 재현) ──
        action = policy.predict(obs)
        prev_ee_xy = ee_xy.copy()
        prev_ee_yaw = ee_yaw
        ee_xy = ee_xy + action[:2] * C.ACTION_SCALE_XY
        if align_yaw:
            ee_yaw = float(np.clip(
                ee_yaw + action[2] * C.ACTION_SCALE_YAW, C.EE_YAW_MIN, C.EE_YAW_MAX))
        # align_yaw=False 면 ee_yaw 그대로 유지 (0)

        if record_traj:
            traj.append((float(ee_xy[0]), float(ee_xy[1]), float(ee_yaw)))

    final_obj_rel = np.abs(box_xy - ee_xy)
    final_yaw_err = abs(wrap_to_pi(box_yaw - ee_yaw))

    return RolloutResult(
        ee_xy=ee_xy,
        ee_yaw=ee_yaw,
        converged=(status == "converged"),
        status=status,
        n_steps=step + 1,
        start_xy=np.asarray(start_ee_xy, dtype=np.float64),
        start_yaw=float(start_ee_yaw),
        final_xy_err=final_obj_rel,
        final_yaw_err=final_yaw_err,
        trajectory=traj,
    )


if __name__ == "__main__":
    # 단독 검증: sim 학습 분포 안의 박스로 롤아웃 → ee 가 box xy/yaw 로 수렴하는지 확인
    from grasp_policy import GraspPolicy

    pol = GraspPolicy()
    # 박스: sim 분포 중심 근처. 시작 EE: sim reset 기본 (0.46, -0.30)
    box_xy = np.array([0.45, -0.10], dtype=np.float64)
    box_yaw = 0.30
    res = rollout_grasp(pol, box_xy, box_yaw, C.EE_INIT_XY, 0.0, align_yaw=True)
    print(f"[rollout] status={res.status} converged={res.converged} steps={res.n_steps}")
    print(f"  start ee   : xy={np.round(res.start_xy,4).tolist()} yaw={res.start_yaw:.4f}")
    print(f"  final ee   : xy={np.round(res.ee_xy,4).tolist()} yaw={res.ee_yaw:.4f}")
    print(f"  box        : xy={box_xy.tolist()} yaw={box_yaw:.4f}")
    print(f"  final err  : xy={np.round(res.final_xy_err*1000,2).tolist()} mm "
          f"yaw={math.degrees(res.final_yaw_err):.2f}°")
