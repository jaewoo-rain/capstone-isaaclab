"""motion3 — 5×2 그리드 10칸에 박스가 모두 놓인 씬 (시연/촬영용).

책상 위 로봇(뒤돌아선 자세) + 뒤쪽 5×2 셀 그리드 + 각 셀에 박스 1개씩(세워서 안착).
정책/모션 없음 — 정적 디스플레이. 카메라 돌려가며 보거나 사용자가 직접 녹화.

실행 (GUI):
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
    ./isaaclab.sh -p source/motion3/scripts/preview_grid_filled.py
    #  로봇 앞쪽 home 자세로:  --pose start
    #  그리드 yaw 미리보기:    --grid_yaw 0
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="motion3 5x2 grid filled preview")
parser.add_argument("--pose", type=str, default="back", choices=["start", "back"])
parser.add_argument("--grid_yaw", type=float, default=0.0, help="그리드 yaw(도)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# 기본 GUI (--headless 미지정 시). smoke test 는 --headless 로 가능.

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
_CELLS_LOCAL = scene_helpers.cell_centers_local()   # 5×2 = 10칸 (grid-local xy)
_N_CELL = len(_CELLS_LOCAL)


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


def _cell_box_cfg(i):
    # 셀에 안착될 박스 — kinematic(고정)으로 깔끔히 세워둠
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/CellBox{i}",
        spawn=sim_utils.CuboidCfg(
            size=layout.BOX_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.55, 0.25), metallic=0.3, roughness=0.5)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, layout.PLACE_Z)),
    )


@configclass
class GridFilledSceneCfg(InteractiveSceneCfg):
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

    def __post_init__(self):
        for _name, _size, _lxy in _WALL_SPECS:
            setattr(self, _name, _wall_cfg(_name, _size, _lxy))
        for _i in range(_N_CELL):
            setattr(self, f"CellBox{_i}", _cell_box_cfg(_i))


START = {"joint1": 0.0, "joint2": -1.5706, "joint3": 2.653, "joint4": -1.082, "joint5": 1.5707, "joint6": 0.0}
BACK = {"joint1": -2.90, "joint2": 0.73, "joint3": 0.64, "joint4": 0.17, "joint5": 1.571, "joint6": 0.0}


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device))
    scene = InteractiveScene(GridFilledSceneCfg(num_envs=1, env_spacing=2.5))
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

    gyaw = math.radians(args_cli.grid_yaw)
    walls, cells = scene_helpers.grid_world_poses(layout.CELL_GRID_CENTER[0], layout.CELL_GRID_CENTER[1], gyaw)
    env_origin = scene.env_origins[0]
    ox, oy, oz = env_origin[0].item(), env_origin[1].item(), env_origin[2].item()

    # 벽 배치
    for name, _sz, (wx, wy, wz), wy_ in walls:
        half = wy_ / 2.0
        qz = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]], device=device)
        pos = torch.tensor([[wx + ox, wy + oy, wz + oz]], device=device)
        scene[name].write_root_pose_to_sim(torch.cat([pos, qz], dim=-1))

    # 각 셀에 박스 안착 (셀 yaw 에 정렬, PLACE_Z 높이)
    half_g = gyaw / 2.0
    box_quat = torch.tensor([[math.cos(half_g), 0.0, 0.0, math.sin(half_g)]], device=device)
    for i, (cx, cy) in enumerate(cells):
        pos = torch.tensor([[cx + ox, cy + oy, layout.PLACE_Z + oz]], device=device)
        scene[f"CellBox{i}"].write_root_pose_to_sim(torch.cat([pos, box_quat], dim=-1))

    print("\n" + "=" * 55)
    print(f"[grid-filled] 5x2 = {_N_CELL}칸, 각 셀에 박스 안착 (box center z={layout.PLACE_Z})")
    print(f"[grid-filled] grid_center={layout.CELL_GRID_CENTER}  grid_yaw={args_cli.grid_yaw}°  robot pose={args_cli.pose}")
    print(f"[grid-filled] 셀 좌표(env-rel): {[(round(x,2),round(y,2)) for x,y in cells]}")
    print("[grid-filled] GUI 창에서 카메라 돌려가며 보세요. 창 닫으면 종료.")
    print("=" * 55 + "\n")

    while simulation_app.is_running():
        scene.write_data_to_sim(); sim.step(); scene.update(dt)
    simulation_app.close()


if __name__ == "__main__":
    main()
