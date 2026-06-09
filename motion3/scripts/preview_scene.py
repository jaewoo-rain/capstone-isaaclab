"""motion3 — 초기 scene 미리보기 (정책 불필요).

책상 위 로봇 + 앞쪽 블록 + 뒤쪽 3×3 grid 셀을 띄우고 로봇을 시작 자세로 둔 채 hold.
처음 셋업(좌표/높이/그리드 배치)이 의도대로인지 눈으로 확인용.

실행 (GUI):
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
    ./isaaclab.sh -p source/motion3/scripts/preview_scene.py
    # 뒤돌아선 자세로 보려면:  --pose back
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="motion3 scene preview")
parser.add_argument("--pose", type=str, default="start", choices=["start", "back"],
                    help="start=앞쪽 home, back=뒤돌아선 turn-around 자세")
parser.add_argument("--grid_yaw", type=float, default=0.0, help="그리드 yaw(도) 미리보기")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from source.motion3.robot_cfg import OMY_TABLE_MOUNTED_CFG
from source.motion3 import layout, scene_helpers

_WALL_SPECS = scene_helpers.wall_specs()


def _wall_cfg(name, size, lxy):
    gx, gy = layout.CELL_GRID_CENTER
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{name}",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.35))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(gx + lxy[0], gy + lxy[1], layout.WALL_HEIGHT / 2)),
    )


@configclass
class PreviewSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light",
                         spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)))
    robot = OMY_TABLE_MOUNTED_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    desk: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Desk",
        spawn=sim_utils.CuboidCfg(
            size=(layout.DESK_SIZE_XY[0], layout.DESK_SIZE_XY[1], layout.DESK_TOP_Z),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=50.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.35, 0.25))),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(layout.DESK_CENTER[0], layout.DESK_CENTER[1], layout.DESK_TOP_Z / 2)))
    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Box",
        spawn=sim_utils.CuboidCfg(
            size=layout.BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True, disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.7, 0.72), metallic=0.5)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=layout.BOX_SPAWN))
    def __post_init__(self):
        for _name, _size, _lxy in _WALL_SPECS:
            setattr(self, _name, _wall_cfg(_name, _size, _lxy))


START = {"joint1": 0.0, "joint2": -1.5706, "joint3": 2.653, "joint4": -1.082, "joint5": 1.5707, "joint6": 0.0}
BACK = {"joint1": -2.90, "joint2": 0.73, "joint3": 0.64, "joint4": 0.17, "joint5": 1.571, "joint6": 0.0}


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device))
    scene = InteractiveScene(PreviewSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()
    robot = scene["robot"]
    device = sim.device
    dt = sim.get_physics_dt()

    pose = BACK if args_cli.pose == "back" else START
    q = robot.data.default_joint_pos.clone()
    for n, v in pose.items():
        q[:, robot.find_joints(n)[0][0]] = v
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    robot.set_joint_position_target(q)

    # 그리드 yaw 미리보기 (8벽 회전)
    gyaw = math.radians(args_cli.grid_yaw)
    walls, cells = scene_helpers.grid_world_poses(layout.CELL_GRID_CENTER[0], layout.CELL_GRID_CENTER[1], gyaw)
    env_origin = scene.env_origins[0]
    for name, _sz, (wx, wy, wz), wy_ in walls:
        half = wy_ / 2.0
        qz = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]], device=device)
        pos = torch.tensor([[wx + env_origin[0].item(), wy + env_origin[1].item(), wz + env_origin[2].item()]], device=device)
        scene[name].write_root_pose_to_sim(torch.cat([pos, qz], dim=-1))

    print("\n" + "=" * 55)
    print(f"[preview] pose={args_cli.pose}  robot base z={layout.TABLE_HEIGHT}  grid_center={layout.CELL_GRID_CENTER}")
    print(f"[preview] 책상 top z={layout.DESK_TOP_Z}  박스 z={layout.BOX_SPAWN[2]:.3f}  셀바닥 z=0  hover z={layout.INSERT_HOVER_Z}")
    print(f"[preview] 타깃 셀 9개 (env-rel): {[(round(x,2),round(y,2)) for x,y in cells]}")
    print("[preview] GUI 창 닫으면 종료. 카메라 돌려가며 확인하세요.")
    print("=" * 55 + "\n")

    while simulation_app.is_running():
        scene.write_data_to_sim(); sim.step(); scene.update(dt)
    simulation_app.close()


if __name__ == "__main__":
    main()
