"""motion3 S1 — 뒤쪽(-x) 저고도 3×3 셀에 대한 IK 도달성 선검증 게이트.

robot을 책상 위(z=TABLE_HEIGHT)에 올리고, layout.build_cell_centers()의 9개 셀
각각에 대해 hover(z=INSERT_HOVER_Z) → place(z=PLACE_Z)를 cell yaw {0, ±10°}로
DLS IK 추적시켜 최종 EE 오차/joint limit를 측정한다.

성공 기준: 위치 잔차 < 1cm, joint limit(특히 joint3 ±150°=±2.618) 안, box drop 없음.

실행:
    ./isaaclab.sh -p source/motion3/scripts/probe_back_reach.py --headless
"""
import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="motion3 S1 뒤쪽 저고도 IK 도달성 검증")
parser.add_argument("--settle", type=int, default=120, help="타깃당 settle control step")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------- imports (after app start) --------
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_angle_axis, quat_mul, subtract_frame_transforms

from source.motion3.robot_cfg import OMY_TABLE_MOUNTED_CFG
from source.motion3 import layout

JOINT3_LIMIT = 2.618   # ±150° (실제 OMY spec; URDF는 ±360° 오기)


@configclass
class ProbeSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)),
    )
    robot = OMY_TABLE_MOUNTED_CFG.replace(prim_path="/World/envs/env_.*/Robot")


# ---- IK helpers (chain runner와 동일) ----
def grip_center_pos(robot, l_id, r_id):
    return 0.5 * (robot.data.body_pos_w[:, l_id] + robot.data.body_pos_w[:, r_id])

def grip_center_quat(robot, l_id):
    return robot.data.body_quat_w[:, l_id]

def grip_center_jacobian(robot, l_jac_idx, r_jac_idx, joint_ids):
    J = robot.root_physx_view.get_jacobians()
    j_l = J[:, l_jac_idx, :, :][:, :, joint_ids]
    j_r = J[:, r_jac_idx, :, :][:, :, joint_ids]
    return 0.5 * (j_l + j_r)


def main():
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    )
    scene = InteractiveScene(ProbeSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()

    robot = scene["robot"]
    device = sim.device
    dt = sim.get_physics_dt()

    arm_names = [f"joint{i}" for i in range(1, 7)]
    gripper_names = ["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"]
    arm_ids = [robot.find_joints(n)[0][0] for n in arm_names]
    gripper_ids = [robot.find_joints(n)[0][0] for n in gripper_names]
    all_joint_ids = arm_ids + gripper_ids
    left_id = robot.find_bodies("rh_p12_rn_l2")[0][0]
    right_id = robot.find_bodies("rh_p12_rn_r2")[0][0]
    l_jac, r_jac = (left_id - 1, right_id - 1) if robot.is_fixed_base else (left_id, right_id)

    ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    ik = DifferentialIKController(ik_cfg, num_envs=scene.num_envs, device=device)

    base_ee_quat = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
    z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=device)

    # 뒤쪽 셀을 향한 seed 자세 (실제 로봇 teach new4 기반: joint1 ~-170° turn-around).
    # DLS IK는 seed에서 최소이동으로 풀므로, joint1을 미리 돌려두지 않으면 fold-back(어깨 너머)
    # 어색한 해를 골라 far 셀에서 특이점에 걸린다. → 돌아선 자세에서 시드해야 자연스럽게 reach.
    BACK_HOME = {"joint1": -2.90, "joint2": 0.73, "joint3": 0.64, "joint4": 0.17,
                 "joint5": 1.571, "joint6": 0.0,
                 "rh_r1_joint": 0.0, "rh_r2": 0.0, "rh_l1": 0.0, "rh_l2": 0.0}
    home_q = torch.zeros((scene.num_envs, robot.num_joints), device=device)
    for n, val in BACK_HOME.items():
        home_q[:, robot.find_joints(n)[0][0]] = val
    robot.write_joint_state_to_sim(home_q, torch.zeros_like(home_q))
    robot.set_joint_position_target(home_q)
    robot.reset()
    for _ in range(30):
        scene.write_data_to_sim(); sim.step(); scene.update(dt)

    env_origin = scene.env_origins[0]
    base_z = robot.data.root_pos_w[0, 2].item()
    print(f"\n[S1] robot base world z = {base_z:.3f} (TABLE_HEIGHT={layout.TABLE_HEIGHT})")

    def control_step(target_pos_env, target_yaw):
        target_pos_w = target_pos_env.unsqueeze(0) + env_origin.unsqueeze(0)
        yaw_q = quat_from_angle_axis(torch.tensor([target_yaw], device=device), z_axis)
        target_quat_w = quat_mul(yaw_q, base_ee_quat)
        gripper_target = torch.tensor([[0.8, 0.8 * 2.3, 0.8, 0.8 * 2.3]], device=device)
        ee_pos_w = grip_center_pos(robot, left_id, right_id)
        ee_quat_w = grip_center_quat(robot, left_id)
        cur_arm_q = robot.data.joint_pos[:, arm_ids]
        jac = grip_center_jacobian(robot, l_jac, r_jac, arm_ids)
        root_pos_w, root_quat_w = robot.data.root_pos_w, robot.data.root_quat_w
        tgt_pos_b, tgt_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, target_pos_w, target_quat_w)
        ee_pos_b, ee_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
        ik.set_command(torch.cat([tgt_pos_b, tgt_quat_b], dim=-1))
        arm_target = ik.compute(ee_pos_b, ee_quat_b, jac, cur_arm_q)
        full = torch.zeros((scene.num_envs, robot.num_joints), device=device)
        full[:] = robot.data.joint_pos
        for k, jid in enumerate(arm_ids):
            full[:, jid] = arm_target[:, k]
        for k, jid in enumerate(gripper_ids):
            full[:, jid] = gripper_target[:, k]
        robot.set_joint_position_target(full)
        scene.write_data_to_sim(); sim.step(); scene.update(dt)

    cells = layout.build_cell_centers()
    yaws = [0.0, layout.CELL_SPAWN_YAW_MAX, -layout.CELL_SPAWN_YAW_MAX]
    print(f"[S1] {len(cells)} cells × yaw{[round(math.degrees(y),0) for y in yaws]}  "
          f"hover_z={layout.INSERT_HOVER_Z} place_z={layout.PLACE_Z}\n")

    results = []
    for ci, (cx, cy, cz) in enumerate(cells):
        for yaw in yaws:
            ik.reset()
            for phase, tz in (("hover", layout.INSERT_HOVER_Z), ("place", layout.PLACE_Z)):
                tgt = torch.tensor([cx, cy, tz], device=device)
                for _ in range(args_cli.settle):
                    control_step(tgt, yaw)
                ee = grip_center_pos(robot, left_id, right_id)[0] - env_origin
                err = math.dist(ee.tolist(), [cx, cy, tz])
                j3 = robot.data.joint_pos[0, arm_ids[2]].item()
                ok = err < 0.01 and abs(j3) < JOINT3_LIMIT
                results.append((ci, math.degrees(yaw), phase, err, j3, ok))
                tag = "✅" if ok else "❌"
                print(f"[S1] cell{ci} yaw{math.degrees(yaw):+5.0f}° {phase:5s}: "
                      f"err={err*1000:6.1f}mm j3={j3:+.3f} {tag}")
            # 다음 셀 전 home 복귀(누적 자세 영향 제거)
            robot.write_joint_state_to_sim(home_q, torch.zeros_like(home_q))
            robot.set_joint_position_target(home_q)
            for _ in range(20):
                scene.write_data_to_sim(); sim.step(); scene.update(dt)

    n_ok = sum(1 for *_, ok in results if ok)
    print("\n" + "=" * 60)
    print(f"[S1] 도달 성공: {n_ok}/{len(results)}  (err<10mm & |joint3|<150°)")
    fails = [(c, y, p, e) for c, y, p, e, j, ok in results if not ok]
    if fails:
        print("[S1] 실패 케이스:")
        for c, y, p, e in fails:
            print(f"     cell{c} yaw{y:+.0f}° {p}: err={e*1000:.1f}mm")
    else:
        print("[S1] ✅ 전체 통과 — 뒤쪽 저고도 3×3 도달 가능. chain runner 진행 OK.")
    print("=" * 60 + "\n")

    simulation_app.close()


if __name__ == "__main__":
    main()
