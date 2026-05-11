"""motion2 — Sim 환경에서 chain 실행 (sim_adapter 사용).

motion1 의 play_motion_chain_with_grasp_insert_camera.py 와 동일 동작.
차이: ChainStateMachine + SimAdapter 분리 구조. real 가려면 RealAdapter 만 바꾸면 됨.

실행:
    ./isaaclab.sh -p source/motion2/scripts/play_chain_sim.py \
        --enable_cameras --repeat 3 --hold_s 5
"""
from __future__ import annotations

import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="motion2 chain runner (sim)")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--hold_s", type=float, default=5.0)
parser.add_argument("--gripper_close", type=float, default=0.8)
parser.add_argument("--yolo_ckpt", type=str,
                    default="/home/jaewoo/IsaacLab/runs/segment/source/motion1/yolo_runs/v2_seg_2class/weights/best.pt",
                    help="YOLO seg .pt path (class 0=box, 1=cell)")
parser.add_argument("--grasp_ckpt", type=str,
                    default="checkpoints/motion1_grasp.zip")
parser.add_argument("--grasp_vecnorm", type=str,
                    default="checkpoints/motion1_grasp_vecnorm.pkl")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from source.motion2.adapters.sim_adapter import SimAdapter, SimSceneCfg
from source.motion2.inference.yolo_box_detector import YoloBoxDetector
from source.motion2.inference.grasp_policy import GraspPolicy
from source.motion2.inference.chain_state_machine import ChainConfig, run_chain_once


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.4, 0.4, 0.9], [0.3, -0.25, 0.10])
    scene = InteractiveScene(SimSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0))
    sim.reset()
    print("[motion2 sim] Sim ready.")

    adapter = SimAdapter(sim, scene, gripper_close=args_cli.gripper_close)
    yolo = YoloBoxDetector(args_cli.yolo_ckpt)
    policy = GraspPolicy(args_cli.grasp_ckpt, args_cli.grasp_vecnorm, device=args_cli.device)
    cfg = ChainConfig(gripper_close=args_cli.gripper_close)

    grasp_n = 0; insert_n = 0
    for rep in range(max(1, args_cli.repeat)):
        print(f"\n========== run {rep+1}/{args_cli.repeat} ==========")
        result = run_chain_once(adapter, yolo, policy, cfg)
        if result["grasp_success"]:
            grasp_n += 1
        if result["insert_success"]:
            insert_n += 1
        print(f"[result] grasp={result['grasp_success']} "
              f"insert={result['insert_success']} "
              f"dist={result['cell_xy_dist_m']*100:.2f}cm")
    print(f"\n[summary] grasp success {grasp_n}/{args_cli.repeat}, "
          f"insert success {insert_n}/{args_cli.repeat}")

    if args_cli.hold_s > 0:
        print(f"[motion2 sim] holding for {args_cli.hold_s:.1f}s.")
        n_hold = int(args_cli.hold_s / sim.get_physics_dt())
        for _ in range(n_hold):
            if not simulation_app.is_running(): break
            scene.write_data_to_sim(); sim.step(); scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    os._exit(0)
