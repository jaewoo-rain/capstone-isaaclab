"""motion3 — OMY 관절 수동 조종(teleop). 원하는 자세를 잡고 joint 값을 읽어 보고용.

책상 위(z=TABLE_HEIGHT)에 마운트된 로봇을 GUI에서 키보드로 관절별 조종한다.
원하는 자세를 만든 뒤 P 키로 전체 joint + EE pose를 출력 → 그 값을 알려주면 HOME 등에 반영.

실행 (GUI, headless 아님):
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
    ./isaaclab.sh -p source/motion3/scripts/teleop_joints.py

조작:
    1~6        : 조종할 joint 선택 (joint1~6)
    ↑ / ↓      : 선택 joint +/- (기본 0.05 rad)
    [ / ]      : 스텝 크기 감소/증가
    O / C      : 그리퍼 열기 / 닫기
    P          : 현재 전체 joint + EE(grip center, env-rel) pose 출력
    R          : 실제 로봇 'start' 자세로 리셋
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="motion3 OMY 관절 teleop")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False  # GUI 강제

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import omni.appwindow
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass

from source.motion3.robot_cfg import OMY_TABLE_MOUNTED_CFG
from source.motion3 import layout

ARM = [f"joint{i}" for i in range(1, 7)]
GRIP = ["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"]
# 실제 로봇 'start' 자세 (waypoiny.yaml)
START = {"joint1": 0.0, "joint2": -1.5706, "joint3": 2.653,
         "joint4": -1.082, "joint5": 1.5707, "joint6": 0.0}


@configclass
class TeleopSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light",
                         spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)))
    robot = OMY_TABLE_MOUNTED_CFG.replace(prim_path="/World/envs/env_.*/Robot")


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device))
    scene = InteractiveScene(TeleopSceneCfg(num_envs=1, env_spacing=2.5))
    sim.reset()
    robot = scene["robot"]
    device = sim.device
    dt = sim.get_physics_dt()

    arm_ids = [robot.find_joints(n)[0][0] for n in ARM]
    grip_ids = [robot.find_joints(n)[0][0] for n in GRIP]
    left_id = robot.find_bodies("rh_p12_rn_l2")[0][0]
    right_id = robot.find_bodies("rh_p12_rn_r2")[0][0]

    q = robot.data.default_joint_pos.clone()

    def set_start():
        for n, v in START.items():
            q[:, robot.find_joints(n)[0][0]] = v
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)

    set_start()
    for _ in range(10):
        scene.write_data_to_sim(); sim.step(); scene.update(dt)
    env_origin = scene.env_origins[0]

    st = {"joint": 0, "step": 0.05}

    def apply():
        robot.set_joint_position_target(q)

    def print_pose():
        ee = 0.5 * (robot.data.body_pos_w[0, left_id] + robot.data.body_pos_w[0, right_id]) - env_origin
        eq = robot.data.body_quat_w[0, left_id]
        js = [round(q[0, arm_ids[i]].item(), 4) for i in range(6)]
        print("\n================ 현재 자세 ================")
        for i, n in enumerate(ARM):
            print(f"  {n}: {js[i]:+.4f}")
        print(f"  EE(grip center, env-rel): x={ee[0]:+.4f} y={ee[1]:+.4f} z={ee[2]:+.4f}")
        print(f"  EE quat(wxyz): {[round(v,4) for v in eq.tolist()]}")
        print(f"  (복사용) HOME = {{" + ", ".join(f'\"{n}\": {js[i]:.4f}' for i, n in enumerate(ARM)) + "}")
        print("==========================================\n")

    KI = carb.input.KeyboardInput
    sel = {KI.KEY_1: 0, KI.KEY_2: 1, KI.KEY_3: 2, KI.KEY_4: 3, KI.KEY_5: 4, KI.KEY_6: 5}

    def on_key(e, *a):
        if e.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        k = e.input
        if k in sel:
            st["joint"] = sel[k]
            print(f"[teleop] 선택 → {ARM[st['joint']]} (현재 {q[0, arm_ids[st['joint']]].item():+.4f})")
        elif k == KI.UP:
            q[:, arm_ids[st["joint"]]] += st["step"]; apply()
            print(f"[teleop] {ARM[st['joint']]} = {q[0, arm_ids[st['joint']]].item():+.4f}")
        elif k == KI.DOWN:
            q[:, arm_ids[st["joint"]]] -= st["step"]; apply()
            print(f"[teleop] {ARM[st['joint']]} = {q[0, arm_ids[st['joint']]].item():+.4f}")
        elif k == KI.RIGHT_BRACKET:
            st["step"] = min(0.5, st["step"] * 2); print(f"[teleop] step={st['step']:.4f}")
        elif k == KI.LEFT_BRACKET:
            st["step"] = max(0.005, st["step"] / 2); print(f"[teleop] step={st['step']:.4f}")
        elif k == KI.O:
            for jid in grip_ids: q[:, jid] = 0.0
            apply(); print("[teleop] 그리퍼 열기")
        elif k == KI.C:
            for i, jid in enumerate(grip_ids):
                q[:, jid] = 0.8 * (2.3 if i in (1, 3) else 1.0)
            apply(); print("[teleop] 그리퍼 닫기")
        elif k == KI.P:
            print_pose()
        elif k == KI.R:
            set_start(); print("[teleop] start 자세로 리셋")
        return True

    appwindow = omni.appwindow.get_default_app_window()
    inp = carb.input.acquire_input_interface()
    sub = inp.subscribe_to_keyboard_events(appwindow.get_keyboard(), on_key)  # noqa: F841

    print("\n" + "=" * 50)
    print(" OMY teleop — 1~6 관절선택, ↑/↓ 증감, [ ] 스텝, O/C 그리퍼, P 출력, R 리셋")
    print(f" robot base z = {layout.TABLE_HEIGHT} (책상 위)")
    print("=" * 50 + "\n")

    while simulation_app.is_running():
        scene.write_data_to_sim(); sim.step(); scene.update(dt)

    simulation_app.close()


if __name__ == "__main__":
    main()
