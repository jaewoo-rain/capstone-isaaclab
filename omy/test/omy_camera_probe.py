"""
body 이름 확인용
"""
"""
=== Joint names ===
0 joint1
1 joint2
2 joint3
3 joint4
4 joint5
5 joint6
6 rh_l1
7 rh_r1_joint
8 rh_l2
9 rh_r2

=== Body names ===
0 world
1 link1
2 link2
3 link3
4 link4
5 link5
6 link6
7 rh_p12_rn_l1
8 rh_p12_rn_r1
9 rh_p12_rn_l2
10 rh_p12_rn_r2

=== Prim paths containing target keywords ===
/World/Robot/OMY/link6
/World/Robot/OMY/link6/end_effector_flange_link
/World/Robot/OMY/link6/end_effector_flange_link/end_effector_link
/World/Robot/OMY/link6/end_effector_flange_link/rh_p12_rn_base
/World/Robot/OMY/link6/visuals
/World/Robot/OMY/link6/visuals/link6
/World/Robot/OMY/link6/visuals/link6/mesh
/World/Robot/OMY/link6/visuals/flange
/World/Robot/OMY/link6/visuals/flange/mesh
/World/Robot/OMY/link6/visuals/mesh_2
/World/Robot/OMY/link6/visuals/mesh_2/box
/World/Robot/OMY/link6/visuals/base
/World/Robot/OMY/link6/visuals/base/mesh
/World/Robot/OMY/link6/collisions
/World/Robot/OMY/link6/collisions/mesh_0
/World/Robot/OMY/link6/collisions/mesh_0/cylinder
/World/Robot/OMY/link6/collisions/mesh_1
/World/Robot/OMY/link6/collisions/mesh_1/cylinder
/World/Robot/OMY/link6/collisions/base
/World/Robot/OMY/link6/collisions/base/mesh
/World/Robot/OMY/gripper/rh_p12_rn_l1
/World/Robot/OMY/gripper/rh_p12_rn_l1/visuals
/World/Robot/OMY/gripper/rh_p12_rn_l1/visuals/l1
/World/Robot/OMY/gripper/rh_p12_rn_l1/visuals/l1/mesh
/World/Robot/OMY/gripper/rh_p12_rn_l1/collisions
/World/Robot/OMY/gripper/rh_p12_rn_l1/collisions/l1
/World/Robot/OMY/gripper/rh_p12_rn_l1/collisions/l1/mesh
/World/Robot/OMY/gripper/rh_p12_rn_l2
/World/Robot/OMY/gripper/rh_p12_rn_l2/visuals
/World/Robot/OMY/gripper/rh_p12_rn_l2/visuals/l2
/World/Robot/OMY/gripper/rh_p12_rn_l2/visuals/l2/mesh
/World/Robot/OMY/gripper/rh_p12_rn_l2/collisions
/World/Robot/OMY/gripper/rh_p12_rn_l2/collisions/l2
/World/Robot/OMY/gripper/rh_p12_rn_l2/collisions/l2/mesh
/World/Robot/OMY/gripper/rh_p12_rn_r1
/World/Robot/OMY/gripper/rh_p12_rn_r1/visuals
/World/Robot/OMY/gripper/rh_p12_rn_r1/visuals/r1
/World/Robot/OMY/gripper/rh_p12_rn_r1/visuals/r1/mesh
/World/Robot/OMY/gripper/rh_p12_rn_r1/collisions
/World/Robot/OMY/gripper/rh_p12_rn_r1/collisions/r1
/World/Robot/OMY/gripper/rh_p12_rn_r1/collisions/r1/mesh
/World/Robot/OMY/gripper/rh_p12_rn_r2
/World/Robot/OMY/gripper/rh_p12_rn_r2/visuals
/World/Robot/OMY/gripper/rh_p12_rn_r2/visuals/r2
/World/Robot/OMY/gripper/rh_p12_rn_r2/visuals/r2/mesh
/World/Robot/OMY/gripper/rh_p12_rn_r2/collisions
/World/Robot/OMY/gripper/rh_p12_rn_r2/collisions/r2
/World/Robot/OMY/gripper/rh_p12_rn_r2/collisions/r2/mesh
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Probe OMY prim names")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

from source.omy.omy_robot_cfg import OMY_CFG


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = sim_utils.SimulationContext(sim_cfg)

    # 바닥
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/GroundPlane", ground_cfg)

    # 조명
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # 단일 로봇
    robot_cfg = OMY_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.update(sim.get_physics_dt())

    print("\n=== Joint names ===")
    for i, name in enumerate(robot.joint_names):
        print(i, name)

    print("\n=== Body names ===")
    for i, name in enumerate(robot.body_names):
        print(i, name)

    stage = omni.usd.get_context().get_stage()

    keywords = [
        "camera",
        "flange",
        "effector",
        "rh_p12",
        "link6",
    ]

    print("\n=== Prim paths containing target keywords ===")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        lower = path.lower()
        if path.startswith("/World/Robot") and any(k in lower for k in keywords):
            print(path)

    print("\nProbe finished.")
    simulation_app.close()


if __name__ == "__main__":
    main()