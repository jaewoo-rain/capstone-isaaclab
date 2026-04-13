"""
test_spawn_omy.py
로봇만 잘 뜨는지 확인
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn OMY robot only")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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

    # 단일 로봇 spawn용 prim_path로 교체
    robot_cfg = OMY_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()

    print("OMY spawned successfully.")

    while simulation_app.is_running():
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

    simulation_app.close()


if __name__ == "__main__":
    main()