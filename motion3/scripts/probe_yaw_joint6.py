"""motion3 — 손목 관절 직접 yaw 제어 가능성 probe (grip 단단히 hold 버전).

가설: insert hover 자세에서 full-pose IK 는 yaw authority 가 없다.
손목 관절(joint6 등)을 직접 돌리면 box yaw 가 깨끗이(monotonic, 일관) 따라오고
박스를 떨어뜨리지(z drop) 않으면 → 그 관절 직접제어가 yaw 해법.

핵심: gripper 를 reset 직후의 '실제 handoff 관절값'으로 hold (느슨한 0.8 cmd 금지).
천천히(0.5°/step) 돌려 박스 안 날아가게. is_grasping/Δz/Δxy 도 같이 봐서 신뢰성 판단.

실행: ./isaaclab.sh -p source/motion3/scripts/probe_yaw_joint6.py --headless
"""
from __future__ import annotations
import argparse, math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys, functools
print = functools.partial(print, flush=True)
import torch
from source.motion3.tasks.insert.insert_env import InsertEnv
from source.motion3.tasks.insert.insert_env_cfg import InsertEnvCfg


def quat_z_yaw(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


def main():
    cfg = InsertEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    env = InsertEnv(cfg=cfg)
    robot = env._robot
    arm_ids = env._arm_joint_ids
    grip_ids = env._gripper_joint_ids
    all_ids = env._all_joint_ids

    STEP = 24
    DELTA = math.radians(0.6)   # 0.6°/step → 24 step = 14.4° 명령
    CMD = math.degrees(DELTA) * STEP

    print("\n각 arm joint 를 +%.1f° 천천히 돌렸을 때 박스 반응 "
          "(grip=handoff hold). box yaw 깨끗+z유지+held → 그 joint=수직yaw\n" % CMD)
    print(f"{'joint':7s} {'Δbox_yaw mean/std':22s} {'Δbox_z(cm)':12s} "
          f"{'Δbox_xy(cm)':12s} {'held%':6s}")

    for jslot in range(6):
        env.reset()
        # reset 직후 실제 관절값 (handoff 자세 + grip) 으로 hold target 고정
        hold = robot.data.joint_pos[:, arm_ids].clone()
        grip_hold = robot.data.joint_pos[:, grip_ids].clone()

        def apply(arm_t):
            full = torch.cat([arm_t, grip_hold], dim=-1)
            robot.set_joint_position_target(full, joint_ids=all_ids)
            env.sim.step(); env.scene.update(env.sim.get_physics_dt())

        for _ in range(6):       # 현재자세 안정화
            apply(hold)

        byaw0 = quat_z_yaw(env._object.data.root_quat_w).clone()
        bpos0 = env._object.data.root_pos_w.clone()
        envz = env.scene.env_origins[:, 2]

        arm_t = hold.clone()
        for _ in range(STEP):
            arm_t[:, jslot] += DELTA
            apply(arm_t)

        byaw1 = quat_z_yaw(env._object.data.root_quat_w)
        bpos1 = env._object.data.root_pos_w
        dyaw = torch.rad2deg(wrap(byaw1 - byaw0))
        dz = (bpos1[:, 2] - bpos0[:, 2]) * 100
        dxy = torch.norm(bpos1[:, :2] - bpos0[:, :2], dim=-1) * 100
        box_z_env = bpos1[:, 2] - envz
        held = (box_z_env > cfg.box_drop_z_threshold).float().mean() * 100
        print(f"joint{jslot+1:<2d}   {dyaw.mean():6.2f} / {dyaw.std():5.2f}        "
              f"{dz.mean():6.2f}       {dxy.mean():6.2f}        {held:5.0f}")

    print("\n해석: 이상적 수직-yaw 관절 = Δbox_yaw 가 명령(+%.1f°)에 비례하고 "
          "std 작고, Δz≈0, Δxy 작고, held≈100%%." % CMD)
    sys.stdout.flush()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    import os
    os._exit(0)
