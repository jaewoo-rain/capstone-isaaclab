"""motion3 — yaw 제어 검증 (env.step 경로 = 박스 정상 grasp 유지).

누적 setpoint 복구 후:
  Phase A) zero-action 60 step → yaw_err 드리프트 사라졌는지 (setpoint 유지/복원력).
  Phase B) 상수 +yaw action 40 step → box_yaw 가 명령(_ee_target_yaw)에 추종하는지 + 박스 유지.

실행: ./isaaclab.sh -p source/motion3/scripts/probe_yaw_verify.py --headless
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
print = functools.partial(print, flush=True)
import torch
from source.motion3.tasks.insert.insert_env import InsertEnv, fold_yaw_sym
from source.motion3.tasks.insert.insert_env_cfg import InsertEnvCfg


def quat_z_yaw(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def snap(env):
    cell = env._cell_yaw
    box = quat_z_yaw(env._object.data.root_quat_w)
    yaw_err = torch.rad2deg(fold_yaw_sym(cell - box).abs())
    tgt = torch.rad2deg(env._ee_target_yaw)
    box_d = torch.rad2deg(box)
    held = env._is_grasping().float().mean() * 100
    return yaw_err.mean().item(), tgt.mean().item(), box_d.mean().item(), held.item()


def main():
    cfg = InsertEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = InsertEnv(cfg=cfg)

    # ---------- Phase A: zero-action hold ----------
    env.reset()
    zero = torch.zeros(env.num_envs, 3, device=env.device)
    print("\n===== Phase A: zero-action 60 step (드리프트/유지) =====")
    print(f"{'step':>4} {'yaw_err°':>9} {'tgt°':>8} {'box°':>8} {'held%':>6}")
    for i in range(61):
        if i % 10 == 0:
            ye, tg, bx, hd = snap(env)
            print(f"{i:>4} {ye:>9.2f} {tg:>8.2f} {bx:>8.2f} {hd:>6.0f}")
        env.step(zero)

    # ---------- Phase B: 상수 +yaw 추종 ----------
    env.reset()
    act = torch.zeros(env.num_envs, 3, device=env.device)
    act[:, 2] = 0.6  # +0.6 * scale(0.05) = +1.72°/step 누적
    print("\n===== Phase B: 상수 +yaw action(0.6) 40 step (추종/유지) =====")
    print(f"{'step':>4} {'yaw_err°':>9} {'tgt°':>8} {'box°':>8} {'held%':>6}  (tgt↑ 따라 box↑ 면 제어 OK)")
    for i in range(41):
        if i % 5 == 0:
            ye, tg, bx, hd = snap(env)
            print(f"{i:>4} {ye:>9.2f} {tg:>8.2f} {bx:>8.2f} {hd:>6.0f}")
        env.step(act)

    sys.stdout.flush()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    import os
    os._exit(0)
