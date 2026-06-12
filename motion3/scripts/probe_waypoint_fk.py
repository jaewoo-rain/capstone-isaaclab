"""motion3 — waypoiny.yaml 의 back-low 관절값을 sim(책상 베이스 z=0.30)에 세워
link6 / grip center 의 월드 z 를 측정. layout 의 INSERT_HOVER_Z(grip)/PLACE_Z 를
실제 로봇 waypoint(new6=hover, new7=place)에 맞추기 위함.

실행: ./isaaclab.sh -p source/motion3/scripts/probe_waypoint_fk.py --headless
"""
from __future__ import annotations
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import sys, functools
print = functools.partial(print, flush=True)
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from source.motion3.robot_cfg import OMY_TABLE_MOUNTED_CFG
from source.motion3 import layout

# waypoiny.yaml 관절값 (link0 frame link6 z 도 함께)
WAYPOINTS = {
    "new6 (hover)": (dict(joint1=-2.967929250486271, joint2=0.9940315347748009,
                          joint3=1.3596702366372297, joint4=-0.8081801749425311,
                          joint5=1.5726418974303216, joint6=-0.18600715475603935), 0.154),
    "new7 (place)": (dict(joint1=-2.967593692188921, joint2=1.3215364329883852,
                          joint3=1.210179015167812, joint4=-0.9862298043614137,
                          joint5=1.5726179289805107, joint6=-0.18619890235452505), 0.055),
}


@configclass
class _SceneCfg(InteractiveSceneCfg):
    num_envs: int = 1
    env_spacing: float = 2.0
    robot = OMY_TABLE_MOUNTED_CFG.replace(prim_path="/World/envs/env_.*/Robot")


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/60, device=args_cli.device))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    scene = InteractiveScene(_SceneCfg())
    sim.reset()
    robot: Articulation = scene["robot"]

    arm_ids = [robot.find_joints(f"joint{i}")[0][0] for i in range(1, 7)]
    l_id = robot.find_bodies("rh_p12_rn_l2")[0][0]
    r_id = robot.find_bodies("rh_p12_rn_r2")[0][0]
    link6_id = robot.find_bodies("link6")[0][0]
    base_z = float(robot.data.root_pos_w[0, 2].item())
    print(f"\n로봇 base(link0) world z = {base_z:.3f}  (TABLE_HEIGHT={layout.TABLE_HEIGHT})")
    print(f"현재 layout: INSERT_HOVER_Z(grip)={layout.INSERT_HOVER_Z}  PLACE_Z(box center)={layout.PLACE_Z}")
    print(f"박스: BOX_SIZE={layout.BOX_SIZE} half_z={layout.BOX_SIZE[2]/2:.4f}\n")

    q0 = robot.data.default_joint_pos.clone()
    for name, (jd, wp_z) in WAYPOINTS.items():
        q = q0.clone()
        for jn, jv in jd.items():
            jid = robot.find_joints(jn)[0][0]
            q[0, jid] = jv
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        for _ in range(3):
            robot.set_joint_position_target(q)
            scene.write_data_to_sim(); sim.step(); scene.update(1/60)
        link6_z = float(robot.data.body_pos_w[0, link6_id, 2].item())
        grip_z = float(0.5*(robot.data.body_pos_w[0, l_id, 2] + robot.data.body_pos_w[0, r_id, 2]).item())
        print(f"=== {name} (waypoiny link6 z={wp_z}) ===")
        print(f"  link6 world z = {link6_z:.3f}   (base+wp_z = {base_z+wp_z:.3f} 이면 매핑=단순덧셈)")
        print(f"  grip center world z = {grip_z:.3f}   ← INSERT_HOVER_Z 후보(hover) / place grip")
        print(f"  link6→grip offset = {link6_z-grip_z:.3f}")
        print(f"  (박스 매달림 ~0.16 가정 시 박스밑면 ≈ {grip_z-0.16:.3f}, wall top 0.12 대비 {'위 OK' if grip_z-0.16>0.12 else '아래'})\n")

    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    import os
    os._exit(0)
