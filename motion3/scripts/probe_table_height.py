"""motion3 S0 — robotis_omy_table.usd 의 실제 상판 높이(H) 측정.

책상 USD를 ground 위에 띄우고 world bounding box를 계산해 상판 top z(=H)와
바닥 bottom z를 출력한다. 이 값으로 layout.TABLE_HEIGHT 를 확정한다.

실행:
    ./isaaclab.sh -p source/motion3/scripts/probe_table_height.py --headless
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="motion3 책상 높이 측정")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------- imports (after app start) --------
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

import omni.usd
from pxr import Usd, UsdGeom

TABLE_USD = "/home/jaewoo/IsaacLab/robotis_lab/source/robotis_lab/data/object/robotis_omy_table.usd"


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cpu"))

    # ground
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    # table (origin at world 0,0,0 — USD 자체 offset 그대로)
    table_cfg = sim_utils.UsdFileCfg(usd_path=TABLE_USD)
    table_cfg.func("/World/Table", table_cfg, translation=(0.0, 0.0, 0.0))

    sim.reset()
    for _ in range(10):
        sim.step()

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath("/World/Table")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bound = bbox_cache.ComputeWorldBound(prim)
    rng = bound.ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()

    print("\n" + "=" * 60)
    print("[S0] robotis_omy_table.usd  (translation (0,0,0) 기준)")
    print(f"[S0] bbox min : x={mn[0]:.4f}  y={mn[1]:.4f}  z={mn[2]:.4f}")
    print(f"[S0] bbox max : x={mx[0]:.4f}  y={mx[1]:.4f}  z={mx[2]:.4f}")
    print(f"[S0] 책상 상판 top z (= H 후보) : {mx[2]:.4f} m")
    print(f"[S0] 책상 바닥 bottom z         : {mn[2]:.4f} m")
    print(f"[S0] 책상 전체 높이             : {mx[2]-mn[2]:.4f} m")
    print(f"[S0] 책상 xy 크기               : {mx[0]-mn[0]:.3f} x {mx[1]-mn[1]:.3f} m")
    print("=" * 60)
    print("→ bottom z 가 0 근처면 USD 원점이 바닥. 그때 H = top z 그대로 사용.")
    print("→ bottom z 가 음수면 USD 원점이 상판. 그때 책상을 z=-bottom 만큼 올려 바닥을 ground(0)에 맞춤.\n")

    simulation_app.close()


if __name__ == "__main__":
    main()
