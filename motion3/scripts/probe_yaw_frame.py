"""motion3 — yaw 프레임 진단 probe.

가설: insert_env._extract_ee_yaw (gripper 좌-finger body yaw) 는 box/cell yaw 와
~90° 상수 offset 이 있어, yaw_err 측정이 틀린 프레임이다. 박스는 물리적으로 정렬됨.

확인:
  1) reset(handoff) 직후 cell_yaw / box_yaw / gripper_yaw / offset(gripper-box) 분포
  2) yaw action(+max)을 줬을 때 box_yaw 와 gripper_yaw 가 함께 회전하는지 (제어 가능성)
실행:
  ./isaaclab.sh -p source/motion3/scripts/probe_yaw_frame.py --headless
"""
from __future__ import annotations
import argparse, math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys, functools
print = functools.partial(print, flush=True)  # os._exit 가 buffer flush 안 하므로

import torch
from source.motion3.tasks.insert.insert_env import InsertEnv, fold_yaw_sym
from source.motion3.tasks.insert.insert_env_cfg import InsertEnvCfg


def quat_z_yaw(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def stats(name, deg):
    print(f"  {name:28s} mean={deg.mean():7.2f}  std={deg.std():6.2f}  "
          f"min={deg.min():7.2f}  max={deg.max():7.2f}")


def main():
    cfg = InsertEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = InsertEnv(cfg=cfg)
    env.reset()
    # 한 번 더 0-action step 으로 안정화
    zero = torch.zeros(env.num_envs, 3, device=env.device)
    for _ in range(3):
        env.step(zero)

    cell_yaw = env._cell_yaw
    box_yaw = quat_z_yaw(env._object.data.root_quat_w)
    grip_yaw = env._extract_ee_yaw()

    print("\n================ RESET(handoff) yaw 프레임 ================")
    stats("cell_yaw (deg)", torch.rad2deg(cell_yaw))
    stats("box_yaw (deg)", torch.rad2deg(box_yaw))
    stats("gripper_yaw _extract (deg)", torch.rad2deg(grip_yaw))
    print("  ---- 오차 (fold ±90°) ----")
    stats("|fold(cell - box)|  =물리정렬", torch.rad2deg(fold_yaw_sym(cell_yaw - box_yaw).abs()))
    stats("|fold(cell - gripper)| =현재env", torch.rad2deg(fold_yaw_sym(cell_yaw - grip_yaw).abs()))
    print("  ---- gripper↔box offset (이게 상수면 = mounting angle) ----")
    stats("wrap(gripper - box) (deg)", torch.rad2deg(wrap(grip_yaw - box_yaw)))
    stats("fold(gripper - box) (deg)", torch.rad2deg(fold_yaw_sym(grip_yaw - box_yaw)))

    # ---- 제어 가능성: yaw action +max 20 step ----
    box_yaw0 = box_yaw.clone()
    grip_yaw0 = grip_yaw.clone()
    act = torch.zeros(env.num_envs, 3, device=env.device)
    act[:, 2] = 1.0  # +max yaw
    for _ in range(20):
        env.step(act)
    box_yaw1 = quat_z_yaw(env._object.data.root_quat_w)
    grip_yaw1 = env._extract_ee_yaw()
    print("\n================ yaw action +max ×20 step (제어 가능성) ================")
    stats("Δbox_yaw (deg)", torch.rad2deg(wrap(box_yaw1 - box_yaw0)))
    stats("Δgripper_yaw (deg)", torch.rad2deg(wrap(grip_yaw1 - grip_yaw0)))
    print("  → 둘 다 같은 방향/크기로 돌면 box yaw 도 action 으로 제어 가능 (rigid grasp).")

    sys.stdout.flush()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    import os
    os._exit(0)
