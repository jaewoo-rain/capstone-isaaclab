"""motion2 — Pick-and-place chain state machine.

sim/real 무관. adapter 만 받으면 같은 흐름:
 1. 천장 cam scan → 박스/셀 pose vision 추정
 2. Stage 1 (motion): home → 박스 위
 3. Stage 2 (Grasp RL): 손목 cam 으로 박스 yaw 추정 → PPO inference → ee 정렬
 4. Stage 3a-d (motion): descend → close → lift → transport → 셀 위
 5. Stage 5a-b (motion): insert descend → release
 6. Stage 6a-b (motion): retract → home

Adapter 인터페이스 (sim/real 동일):
    adapter.get_top_cam() / get_wrist_cam() / get_ee_pose()
    adapter.set_ee_target(pos, quat, gripper) / step()
    adapter.spawn_random_box() / spawn_random_cell()
    adapter.reset_to_home()
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np

from ..adapters.base_adapter import BaseAdapter, CamData, EePose
from .unproject import wrap_to_pi, quat_z_yaw
from .yolo_box_detector import YoloBoxDetector, CLASS_BOX, CLASS_CELL
from .grasp_policy import GraspPolicy


# ===== Geometry constants (sim/real 동일) =====
@dataclass
class ChainConfig:
    box_top_z: float = 0.129       # 박스 윗면 z (world)
    cell_wall_top_z: float = 0.12  # 셀 wall 윗면 z

    pre_grasp_z: float = 0.17      # stage 1 끝 z
    grasp_z: float = 0.115         # stage 3a 끝 z
    lift_z: float = 0.26
    transport_z: float = 0.26
    place_z: float = 0.165         # stage 5a 끝 z
    retract_z: float = 0.26

    # stage durations (sec). sim 60Hz 면 *60 step.
    move_above_box_s: float = 2.5
    descend_s: float = 1.0
    close_s: float = 1.5
    lift_s: float = 2.0
    transport_s: float = 3.0
    insert_s: float = 1.5
    release_s: float = 0.7
    retract_up_s: float = 1.5
    retract_home_s: float = 3.0
    settle_s: float = 0.8

    # RL grasp
    rl_max_steps: int = 300
    rl_action_scale_xy: float = 0.01   # m / unit
    rl_action_scale_yaw: float = 0.05  # rad / unit
    rl_ee_yaw_min: float = -math.pi / 2
    rl_ee_yaw_max: float = math.pi / 2
    rl_align_xy: float = 0.005
    rl_align_yaw: float = 0.05
    rl_hold_steps: int = 30

    gripper_open: float = 0.0
    gripper_close: float = 0.8


# ===== util =====
def _slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """quat (wxyz) slerp."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1 = -q1; dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return (out / np.linalg.norm(out)).astype(np.float32)
    theta_0 = math.acos(min(1.0, max(-1.0, dot)))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return ((s0 * q0 + s1 * q1) / np.linalg.norm(s0 * q0 + s1 * q1)).astype(np.float32)


def _quat_from_z_yaw(yaw: float) -> np.ndarray:
    half = yaw / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """wxyz."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)


# ===== Vision scan (천장 cam) =====
def top_cam_scan(adapter: BaseAdapter, yolo: YoloBoxDetector, cfg: ChainConfig) -> dict:
    """천장 cam 1 frame 캡처 → 박스/셀 world xy + yaw 추정."""
    cam: CamData = adapter.get_top_cam()
    rgb = cam.rgb

    box_xy = None; box_yaw = None; cell_xy = None; cell_yaw = None

    # 박스
    mask = yolo.predict_mask(rgb, CLASS_BOX)
    if mask is not None and mask.any():
        est = yolo.estimate_pose_from_topdown(
            mask, cam.depth, cam.K, cam.pos_w, cam.quat_w_world,
            z_known=cfg.box_top_z)
        if est is not None:
            box_xy = (est[0], est[1]); box_yaw = est[2]

    # 셀
    mask = yolo.predict_mask(rgb, CLASS_CELL)
    if mask is not None and mask.any():
        est = yolo.estimate_pose_from_topdown(
            mask, cam.depth, cam.K, cam.pos_w, cam.quat_w_world,
            z_known=cfg.cell_wall_top_z)
        if est is not None:
            cell_xy = (est[0], est[1]); cell_yaw = est[2]

    return {"box_xy": box_xy, "box_yaw": box_yaw,
            "cell_xy": cell_xy, "cell_yaw": cell_yaw}


# ===== Grasp RL (손목 cam obs) =====
def wrist_cam_estimate_box(adapter: BaseAdapter, yolo: YoloBoxDetector,
                           cfg: ChainConfig) -> tuple[float, float, float] | None:
    """손목 cam → 박스 world xy + yaw (yaw 가 핵심)."""
    cam: CamData = adapter.get_wrist_cam()
    mask = yolo.predict_mask(cam.rgb, CLASS_BOX)
    if mask is None or not mask.any():
        return None
    return yolo.estimate_pose_general(
        mask, cam.depth, cam.K, cam.pos_w, cam.quat_w_world,
        z_known=cfg.box_top_z)


# ===== Motion helpers =====
def _stage_move(adapter: BaseAdapter, cfg: ChainConfig,
                start_pos: np.ndarray, end_pos: np.ndarray,
                duration_s: float, gripper_val: float,
                start_quat: np.ndarray, end_quat: np.ndarray):
    """linear interp + slerp + 매 step set_ee_target + adapter.step(1)."""
    n = max(1, int(duration_s / adapter.control_dt))
    s = max(1, int(cfg.settle_s / adapter.control_dt))
    for i in range(n):
        tau = (i + 1) / n
        target_pos = start_pos + (end_pos - start_pos) * tau
        q_i = _slerp_quat(start_quat, end_quat, tau)
        adapter.set_ee_target(target_pos, q_i, gripper_val)
        adapter.step(1)
    for _ in range(s):
        adapter.set_ee_target(end_pos, end_quat, gripper_val)
        adapter.step(1)


def _stage_hold(adapter: BaseAdapter, cfg: ChainConfig,
                pos: np.ndarray, duration_s: float, gripper_val: float,
                quat: np.ndarray):
    n = max(1, int(duration_s / adapter.control_dt))
    for _ in range(n):
        adapter.set_ee_target(pos, quat, gripper_val)
        adapter.step(1)


# ===== Stage 2: Grasp RL =====
def stage_rl_grasp(adapter: BaseAdapter, yolo: YoloBoxDetector,
                   policy: GraspPolicy, cfg: ChainConfig,
                   fallback_box_xy: tuple[float, float],
                   fallback_box_yaw: float) -> tuple[float, bool]:
    """PPO inference 으로 ee 를 박스 위로 정렬.

    obs (6) = [box_x - ee_x, box_y - ee_y, box_yaw - ee_target_yaw, ee_vx, ee_vy, yaw_rate]
    xy 는 fallback (천장 cam scan) 사용. yaw 만 손목 cam est (wrist_cam_estimate_box).

    Returns: (final_ee_target_yaw, success)
    """
    ee_target_yaw = 0.0
    prev_yaw = 0.0
    aligned = 0
    success = False
    base_ee_quat = adapter.get_base_ee_quat()

    for step_i in range(cfg.rl_max_steps):
        ee: EePose = adapter.get_ee_pose()

        # box xy = 천장 cam scan (정확). yaw = 손목 cam (실시간).
        box_x, box_y = fallback_box_xy
        est = wrist_cam_estimate_box(adapter, yolo, cfg)
        if est is not None:
            _, _, box_yaw_v = est
        else:
            box_yaw_v = fallback_box_yaw

        ee_x, ee_y = float(ee.pos_w[0]), float(ee.pos_w[1])
        obj_rel_x = box_x - ee_x
        obj_rel_y = box_y - ee_y
        obj_yaw_err = wrap_to_pi(box_yaw_v - ee_target_yaw)
        ee_vx, ee_vy = float(ee.lin_vel[0]), float(ee.lin_vel[1])
        yaw_rate = (ee_target_yaw - prev_yaw) / max(adapter.control_dt, 1e-6)

        obs = np.array([obj_rel_x, obj_rel_y, obj_yaw_err,
                        ee_vx, ee_vy, yaw_rate], dtype=np.float32)
        action = policy.predict(obs)

        dxy = action[:2] * cfg.rl_action_scale_xy
        new_target_xy = (ee_x + dxy[0], ee_y + dxy[1])
        prev_yaw = ee_target_yaw
        ee_target_yaw = max(cfg.rl_ee_yaw_min,
                            min(cfg.rl_ee_yaw_max,
                                ee_target_yaw + float(action[2]) * cfg.rl_action_scale_yaw))

        target_pos = np.array(
            [new_target_xy[0], new_target_xy[1], cfg.pre_grasp_z], dtype=np.float32)
        target_quat = _quat_mul(_quat_from_z_yaw(ee_target_yaw), base_ee_quat)
        adapter.set_ee_target(target_pos, target_quat, cfg.gripper_open)
        adapter.step(1)

        if (abs(obj_rel_x) < cfg.rl_align_xy and
                abs(obj_rel_y) < cfg.rl_align_xy and
                abs(obj_yaw_err) < cfg.rl_align_yaw):
            aligned += 1
            if aligned >= cfg.rl_hold_steps:
                success = True
                break
        else:
            aligned = 0

    return ee_target_yaw, success


# ===== Full chain =====
def run_chain_once(adapter: BaseAdapter, yolo: YoloBoxDetector,
                   policy: GraspPolicy, cfg: ChainConfig) -> dict:
    """1 run: home → 박스 위 → grasp RL → close → lift → transport → insert → retract.

    Returns: dict with 'grasp_success', 'insert_success', 'box_final_pos', 'cell_xy_dist'
    """
    adapter.reset_to_home()
    home_pos, home_quat = adapter.get_home_ee_pose()
    base_ee_quat = adapter.get_base_ee_quat()

    bx, by, byaw = adapter.spawn_random_box()
    cx, cy, cyaw = adapter.spawn_random_cell()
    adapter.step(30)

    # ---- 천장 cam scan ----
    top_est = top_cam_scan(adapter, yolo, cfg)
    box_xy = top_est["box_xy"] or (bx, by)
    box_yaw = top_est["box_yaw"] if top_est["box_yaw"] is not None else byaw
    cell_xy = top_est["cell_xy"] or (cx, cy)
    cell_yaw = top_est["cell_yaw"] if top_est["cell_yaw"] is not None else cyaw

    print(f"[scan] box est={box_xy}, yaw={math.degrees(box_yaw):+.1f}°  "
          f"cell est={cell_xy}, yaw={math.degrees(cell_yaw):+.1f}°")

    cell_quat = _quat_mul(_quat_from_z_yaw(cell_yaw), base_ee_quat)
    bx_wp, by_wp = box_xy
    cx_wp, cy_wp = cell_xy

    # ---- 1. Move above box ----
    pre_grasp = np.array([bx_wp, by_wp, cfg.pre_grasp_z], dtype=np.float32)
    _stage_move(adapter, cfg, home_pos, pre_grasp,
                cfg.move_above_box_s, cfg.gripper_open,
                home_quat, base_ee_quat)

    # ---- 2. Grasp RL ----
    final_yaw, grasp_success = stage_rl_grasp(
        adapter, yolo, policy, cfg, box_xy, box_yaw)
    print(f"[stage2] grasp RL: final_yaw={math.degrees(final_yaw):+.1f}°, success={grasp_success}")

    yaw_q_final = _quat_from_z_yaw(final_yaw)
    ee_after_align_quat = _quat_mul(yaw_q_final, base_ee_quat)

    # ---- 3a. Descend ----
    cur_ee = adapter.get_ee_pose()
    descend_start = np.array([cur_ee.pos_w[0], cur_ee.pos_w[1], cfg.pre_grasp_z], dtype=np.float32)
    grasp_pos = np.array([bx_wp, by_wp, cfg.grasp_z], dtype=np.float32)
    _stage_move(adapter, cfg, descend_start, grasp_pos,
                cfg.descend_s, cfg.gripper_open,
                ee_after_align_quat, ee_after_align_quat)

    # ---- 3b. Close ----
    _stage_hold(adapter, cfg, grasp_pos, cfg.close_s, cfg.gripper_close,
                ee_after_align_quat)

    # ---- 3c. Lift ----
    lift_pos = np.array([bx_wp, by_wp, cfg.lift_z], dtype=np.float32)
    _stage_move(adapter, cfg, grasp_pos, lift_pos,
                cfg.lift_s, cfg.gripper_close,
                ee_after_align_quat, ee_after_align_quat)

    # ---- 3d. Transport → cell + cell_yaw ----
    transport_target = np.array([cx_wp, cy_wp, cfg.transport_z], dtype=np.float32)
    _stage_move(adapter, cfg, lift_pos, transport_target,
                cfg.transport_s, cfg.gripper_close,
                ee_after_align_quat, cell_quat)

    # ---- 5a. Insert descend ----
    cur_ee = adapter.get_ee_pose()
    descend_end = np.array([cur_ee.pos_w[0], cur_ee.pos_w[1], cfg.place_z], dtype=np.float32)
    descend_start_2 = np.array([cur_ee.pos_w[0], cur_ee.pos_w[1], cfg.transport_z], dtype=np.float32)
    _stage_move(adapter, cfg, descend_start_2, descend_end,
                cfg.insert_s, cfg.gripper_close, cell_quat, cell_quat)

    # ---- 5b. Release ----
    _stage_hold(adapter, cfg, descend_end, cfg.release_s, cfg.gripper_open, cell_quat)

    # ---- 6a/6b. Retract ----
    retract_pos = np.array([cur_ee.pos_w[0], cur_ee.pos_w[1], cfg.retract_z], dtype=np.float32)
    _stage_move(adapter, cfg, descend_end, retract_pos,
                cfg.retract_up_s, cfg.gripper_open, cell_quat, base_ee_quat)
    _stage_move(adapter, cfg, retract_pos, home_pos,
                cfg.retract_home_s, cfg.gripper_open, base_ee_quat, home_quat)

    # ---- Eval ----
    box_gt = adapter.get_box_gt()
    if box_gt is not None:
        bx_final, by_final = box_gt.xy
        dist = math.hypot(bx_final - cx_wp, by_final - cy_wp)
        insert_ok = dist < 0.05  # 5cm 안
    else:
        dist = float("nan")
        insert_ok = None

    return {
        "grasp_success": grasp_success,
        "insert_success": insert_ok,
        "cell_xy_dist_m": dist,
        "box_xy_est": box_xy, "box_yaw_est": box_yaw,
        "cell_xy_est": cell_xy, "cell_yaw_est": cell_yaw,
    }
