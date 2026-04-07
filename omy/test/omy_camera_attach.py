"""
카메라 붙는지 확인
"""
from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Attach camera to OMY")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera, CameraCfg

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

    # 로봇
    robot_cfg = OMY_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    # 🎯 카메라 붙이기 (핵심)
    camera_cfg = CameraCfg(
        prim_path="/World/Robot/OMY/link6/camera",
        update_period=0.0,
        height=240,
        width=320,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.1, 0.084),
            rot=(0.0, 0.0, 0.7, -0.7),

            convention="ros",
        ),
    )

    camera = Camera(camera_cfg)

    sim.reset()

    print("Camera attached.")

    printed = False

    while simulation_app.is_running():
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        camera.update(sim.get_physics_dt())

        if (not printed) and ("rgb" in camera.data.output):
            rgb = camera.data.output["rgb"]
            print("Camera attached.")
            print("RGB shape:", rgb.shape)
            printed = True
            break

    simulation_app.close()


if __name__ == "__main__":
    main()