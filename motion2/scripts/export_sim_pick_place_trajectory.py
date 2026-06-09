"""Export motion2 IsaacLab chain commands as a replay dataset.

This script runs the existing motion2 sim chain and records every commanded
end-effector target plus the resulting simulated joint state. It does not
change the chain logic. The output is an NPZ file intended as the first bridge
from IsaacLab success to real-robot replay experiments.

Example:
    ./isaaclab.sh -p source/motion2/scripts/export_sim_pick_place_trajectory.py \
        --enable_cameras \
        --repeat 1 \
        --out motion2/config/sim_exported_pick_place_trajectory.npz
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Export motion2 sim pick/place trajectory.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--hold_s", type=float, default=0.0)
parser.add_argument("--gripper_close", type=float, default=0.8)
parser.add_argument(
    "--scripted",
    action="store_true",
    help=(
        "Bypass camera/YOLO/RL and export a deterministic GT-based chain. "
        "Use this when the grasp RL stage is unstable but a sim-to-real replay "
        "dataset is still needed."
    ),
)
parser.add_argument(
    "--out",
    default="motion2/config/sim_exported_pick_place_trajectory.npz",
    help="Output NPZ path. Relative paths are resolved from the current working directory.",
)
parser.add_argument(
    "--summary-json",
    default="",
    help="Optional summary JSON path. Default: <out>.summary.json",
)
parser.add_argument(
    "--yolo_ckpt",
    type=str,
    default="/home/jaewoo/IsaacLab/runs/segment/source/motion1/yolo_runs/v2_seg_2class/weights/best.pt",
    help="YOLO seg .pt path (class 0=box, 1=cell)",
)
parser.add_argument("--grasp_ckpt", type=str, default="checkpoints/motion1_grasp.zip")
parser.add_argument("--grasp_vecnorm", type=str, default="checkpoints/motion1_grasp_vecnorm.pkl")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from source.motion2.adapters.sim_adapter import SimAdapter, SimSceneCfg
from source.motion2.inference.chain_state_machine import (
    ChainConfig,
    _quat_from_z_yaw,
    _quat_mul,
    _stage_hold,
    _stage_move,
    run_chain_once,
)
from source.motion2.inference.grasp_policy import GraspPolicy
from source.motion2.inference.yolo_box_detector import YoloBoxDetector


ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
GRIPPER_JOINT_NAMES = ["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"]


def _resolve_output_path(path_text: str) -> pathlib.Path:
    path = pathlib.Path(path_text).expanduser()
    if not path.is_absolute():
        path = pathlib.Path.cwd() / path
    return path


class RecordingSimAdapter(SimAdapter):
    """SimAdapter with passive command and joint-state recording."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_index = -1
        self.command_index = -1
        self.sample_index = 0
        self.sim_time_s = 0.0
        self._last_target_pos: np.ndarray | None = None
        self._last_target_quat: np.ndarray | None = None
        self._last_gripper: float | None = None
        self._records: list[dict[str, object]] = []
        self._results: list[dict[str, object]] = []

    def begin_run(self, run_index: int) -> None:
        self.run_index = int(run_index)
        self.command_index = -1
        self._last_target_pos = None
        self._last_target_quat = None
        self._last_gripper = None

    def set_ee_target(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        gripper_value: float,
    ) -> None:
        self.command_index += 1
        self._last_target_pos = np.asarray(target_pos, dtype=np.float32).copy()
        self._last_target_quat = np.asarray(target_quat, dtype=np.float32).copy()
        self._last_gripper = float(gripper_value)
        super().set_ee_target(target_pos, target_quat, gripper_value)

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self._dt)
            self.sim_time_s += float(self._dt)
            if self._last_target_pos is not None:
                self._record_sample()

    def append_result(self, result: dict[str, object]) -> None:
        serializable = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                serializable[key] = value.item()
            else:
                serializable[key] = value
        serializable["run_index"] = self.run_index
        self._results.append(serializable)

    def _record_sample(self) -> None:
        joint_pos = self.robot.data.joint_pos[0].detach().cpu().numpy().astype(np.float32)
        arm_joint_pos = joint_pos[self.arm_ids]
        gripper_joint_pos = joint_pos[self.gripper_ids]
        ee = self.get_ee_pose()
        box = self.get_box_gt()
        cell = self.get_cell_gt()
        self._records.append({
            "run_index": self.run_index,
            "sample_index": self.sample_index,
            "command_index": self.command_index,
            "time_s": self.sim_time_s,
            "target_pos": self._last_target_pos.copy(),
            "target_quat_wxyz": self._last_target_quat.copy(),
            "gripper_command": float(self._last_gripper),
            "ee_pos": ee.pos_w.astype(np.float32),
            "ee_quat_wxyz": ee.quat_w.astype(np.float32),
            "ee_lin_vel": ee.lin_vel.astype(np.float32),
            "ee_ang_vel_z": float(ee.ang_vel_z),
            "arm_joint_pos": arm_joint_pos.astype(np.float32),
            "gripper_joint_pos": gripper_joint_pos.astype(np.float32),
            "all_joint_pos": np.concatenate([arm_joint_pos, gripper_joint_pos]).astype(np.float32),
            "box_xy": np.array(box.xy if box is not None else [np.nan, np.nan], dtype=np.float32),
            "box_yaw": float(box.yaw) if box is not None else float("nan"),
            "cell_xy": np.array(cell.xy if cell is not None else [np.nan, np.nan], dtype=np.float32),
            "cell_yaw": float(cell.yaw) if cell is not None else float("nan"),
        })
        self.sample_index += 1

    def save(self, out_path: pathlib.Path, summary_path: pathlib.Path) -> None:
        if not self._records:
            raise RuntimeError("No trajectory samples were recorded.")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        def stack_array(key: str) -> np.ndarray:
            return np.stack([np.asarray(r[key]) for r in self._records])

        def array_from(key: str, dtype) -> np.ndarray:
            return np.asarray([r[key] for r in self._records], dtype=dtype)

        np.savez(
            out_path,
            schema_version=np.array([1], dtype=np.int32),
            control_dt=np.array([self._dt], dtype=np.float32),
            arm_joint_names=np.array(ARM_JOINT_NAMES),
            gripper_joint_names=np.array(GRIPPER_JOINT_NAMES),
            all_joint_names=np.array(ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES),
            run_index=array_from("run_index", np.int32),
            sample_index=array_from("sample_index", np.int32),
            command_index=array_from("command_index", np.int32),
            time_s=array_from("time_s", np.float32),
            target_pos=stack_array("target_pos").astype(np.float32),
            target_quat_wxyz=stack_array("target_quat_wxyz").astype(np.float32),
            gripper_command=array_from("gripper_command", np.float32),
            ee_pos=stack_array("ee_pos").astype(np.float32),
            ee_quat_wxyz=stack_array("ee_quat_wxyz").astype(np.float32),
            ee_lin_vel=stack_array("ee_lin_vel").astype(np.float32),
            ee_ang_vel_z=array_from("ee_ang_vel_z", np.float32),
            arm_joint_pos=stack_array("arm_joint_pos").astype(np.float32),
            gripper_joint_pos=stack_array("gripper_joint_pos").astype(np.float32),
            all_joint_pos=stack_array("all_joint_pos").astype(np.float32),
            box_xy=stack_array("box_xy").astype(np.float32),
            box_yaw=array_from("box_yaw", np.float32),
            cell_xy=stack_array("cell_xy").astype(np.float32),
            cell_yaw=array_from("cell_yaw", np.float32),
        )

        summary = {
            "schema_version": 1,
            "out": str(out_path),
            "samples": len(self._records),
            "runs": max(0, len(self._results)),
            "control_dt": float(self._dt),
            "arm_joint_names": ARM_JOINT_NAMES,
            "gripper_joint_names": GRIPPER_JOINT_NAMES,
            "result_summary": self._results,
            "fields": [
                "target_pos",
                "target_quat_wxyz",
                "gripper_command",
                "ee_pos",
                "ee_quat_wxyz",
                "arm_joint_pos",
                "gripper_joint_pos",
                "all_joint_pos",
                "box_xy",
                "box_yaw",
                "cell_xy",
                "cell_yaw",
            ],
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def run_scripted_chain_once(adapter: RecordingSimAdapter, cfg: ChainConfig) -> dict[str, object]:
    """GT-based pick/place chain with no camera, YOLO, or RL policy.

    This intentionally mirrors the non-RL stages in run_chain_once(). It is not
    meant to prove policy success. It exports a stable motion chain that can be
    inspected and later converted into a real-robot replay candidate.
    """
    adapter.reset_to_home()
    home_pos, home_quat = adapter.get_home_ee_pose()
    base_ee_quat = adapter.get_base_ee_quat()

    bx, by, byaw = adapter.spawn_random_box()
    cx, cy, cyaw = adapter.spawn_random_cell()
    adapter.step(30)

    box_xy = (bx, by)
    box_yaw = byaw
    cell_xy = (cx, cy)
    cell_yaw = cyaw

    print(
        f"[scripted] box gt={box_xy}, yaw={np.degrees(box_yaw):+.1f}deg  "
        f"cell gt={cell_xy}, yaw={np.degrees(cell_yaw):+.1f}deg"
    )

    box_quat = _quat_mul(_quat_from_z_yaw(box_yaw), base_ee_quat)
    cell_quat = _quat_mul(_quat_from_z_yaw(cell_yaw), base_ee_quat)

    pre_grasp = np.array([bx, by, cfg.pre_grasp_z], dtype=np.float32)
    grasp_pos = np.array([bx, by, cfg.grasp_z], dtype=np.float32)
    lift_pos = np.array([bx, by, cfg.lift_z], dtype=np.float32)
    transport_target = np.array([cx, cy, cfg.transport_z], dtype=np.float32)
    place_pos = np.array([cx, cy, cfg.place_z], dtype=np.float32)
    retract_pos = np.array([cx, cy, cfg.retract_z], dtype=np.float32)

    _stage_move(adapter, cfg, home_pos, pre_grasp, cfg.move_above_box_s,
                cfg.gripper_open, home_quat, box_quat)
    _stage_move(adapter, cfg, pre_grasp, grasp_pos, cfg.descend_s,
                cfg.gripper_open, box_quat, box_quat)
    _stage_hold(adapter, cfg, grasp_pos, cfg.close_s, cfg.gripper_close, box_quat)
    _stage_move(adapter, cfg, grasp_pos, lift_pos, cfg.lift_s,
                cfg.gripper_close, box_quat, box_quat)
    _stage_move(adapter, cfg, lift_pos, transport_target, cfg.transport_s,
                cfg.gripper_close, box_quat, cell_quat)
    _stage_move(adapter, cfg, transport_target, place_pos, cfg.insert_s,
                cfg.gripper_close, cell_quat, cell_quat)
    _stage_hold(adapter, cfg, place_pos, cfg.release_s, cfg.gripper_open, cell_quat)
    _stage_move(adapter, cfg, place_pos, retract_pos, cfg.retract_up_s,
                cfg.gripper_open, cell_quat, base_ee_quat)
    _stage_move(adapter, cfg, retract_pos, home_pos, cfg.retract_home_s,
                cfg.gripper_open, base_ee_quat, home_quat)

    box_gt = adapter.get_box_gt()
    if box_gt is not None:
        dist = float(np.hypot(box_gt.xy[0] - cx, box_gt.xy[1] - cy))
        insert_ok = dist < 0.05
    else:
        dist = float("nan")
        insert_ok = None

    return {
        "mode": "scripted",
        "grasp_success": True,
        "insert_success": insert_ok,
        "cell_xy_dist_m": dist,
        "box_xy_est": box_xy,
        "box_yaw_est": box_yaw,
        "cell_xy_est": cell_xy,
        "cell_yaw_est": cell_yaw,
    }


def main() -> int:
    if args_cli.num_envs != 1:
        raise ValueError("This exporter currently supports --num_envs 1 only.")

    out_path = _resolve_output_path(args_cli.out)
    summary_path = (
        _resolve_output_path(args_cli.summary_json)
        if args_cli.summary_json
        else out_path.with_suffix(out_path.suffix + ".summary.json")
    )

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.4, 0.4, 0.9], [0.3, -0.25, 0.10])
    scene = InteractiveScene(SimSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0))
    sim.reset()
    print("[motion2 export] Sim ready.")

    adapter = RecordingSimAdapter(sim, scene, gripper_close=args_cli.gripper_close)
    cfg = ChainConfig(gripper_close=args_cli.gripper_close)
    yolo = None
    policy = None
    if args_cli.scripted:
        print("[motion2 export] mode=scripted: camera/YOLO/RL are bypassed.")
    else:
        print("[motion2 export] mode=policy: loading YOLO and grasp policy.")
        yolo = YoloBoxDetector(args_cli.yolo_ckpt)
        policy = GraspPolicy(args_cli.grasp_ckpt, args_cli.grasp_vecnorm, device=args_cli.device)

    grasp_n = 0
    insert_n = 0
    repeat = max(1, int(args_cli.repeat))
    for rep in range(repeat):
        print(f"\n========== export run {rep + 1}/{repeat} ==========")
        adapter.begin_run(rep)
        if args_cli.scripted:
            result = run_scripted_chain_once(adapter, cfg)
        else:
            result = run_chain_once(adapter, yolo, policy, cfg)
        adapter.append_result(result)
        if result["grasp_success"]:
            grasp_n += 1
        if result["insert_success"]:
            insert_n += 1
        print(
            f"[result] grasp={result['grasp_success']} "
            f"insert={result['insert_success']} "
            f"dist={result['cell_xy_dist_m'] * 100:.2f}cm"
        )

    adapter.save(out_path, summary_path)
    print(f"\n[motion2 export] saved samples={adapter.sample_index} -> {out_path}")
    print(f"[motion2 export] summary -> {summary_path}")
    print(f"[motion2 export] grasp success {grasp_n}/{repeat}, insert success {insert_n}/{repeat}")

    if args_cli.hold_s > 0:
        print(f"[motion2 export] holding for {args_cli.hold_s:.1f}s.")
        n_hold = int(args_cli.hold_s / sim.get_physics_dt())
        for _ in range(n_hold):
            if not simulation_app.is_running():
                break
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
    os._exit(0)
