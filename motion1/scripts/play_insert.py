"""motion1 — 학습된 Insert 정책 시각화."""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="motion1 Insert RL — play")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--checkpoint", type=str, default="checkpoints/motion1_insert.zip")
parser.add_argument("--vecnorm", type=str, default="checkpoints/motion1_insert_vecnorm.pkl")
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument("--deterministic", action="store_true", default=True)
parser.add_argument("--render_interval", type=int, default=4,
                    help="sim render interval (높일수록 GUI 가벼움. default=4)")
parser.add_argument("--keep_alive", action="store_true", default=True,
                    help="episodes 끝나도 창 유지 (사용자가 직접 닫을 때까지)")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    try:
        from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper
    except ImportError:
        from isaaclab.envs.wrappers.sb3 import Sb3VecEnvWrapper

import math
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_from_angle_axis

from source.motion1.tasks.insert.insert_env import InsertEnv
from source.motion1.tasks.insert.insert_env_cfg import InsertEnvCfg

# Cell 정의 (play_motion_chain_with_grasp.py 와 동일)
CELL_INNER_X = 0.16
CELL_INNER_Y = 0.065
WALL_THICKNESS = 0.008
WALL_HEIGHT = 0.12
_WALL_Z = WALL_HEIGHT / 2  # ground 위 0.06

_V_WALL_SIZE = (WALL_THICKNESS, CELL_INNER_Y + 2 * WALL_THICKNESS, WALL_HEIGHT)
_H_WALL_SIZE = (CELL_INNER_X + 2 * WALL_THICKNESS, WALL_THICKNESS, WALL_HEIGHT)

# 4 walls 의 cell-local offset + 종류 (v=좌우, h=앞뒤)
_WALL_OFFSETS = [
    (-(CELL_INNER_X / 2 + WALL_THICKNESS / 2), 0.0, "v"),  # V left
    (+(CELL_INNER_X / 2 + WALL_THICKNESS / 2), 0.0, "v"),  # V right
    (0.0, -(CELL_INNER_Y / 2 + WALL_THICKNESS / 2), "h"),  # H front
    (0.0, +(CELL_INNER_Y / 2 + WALL_THICKNESS / 2), "h"),  # H back
]


def main():
    cfg = InsertEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    # GUI lag 줄이기: render fps 낮춤 (sim dt 는 그대로)
    cfg.sim.render_interval = args_cli.render_interval

    raw_env = InsertEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)

    if not os.path.exists(args_cli.checkpoint):
        raise FileNotFoundError(f"checkpoint 없음: {args_cli.checkpoint}")
    if not os.path.exists(args_cli.vecnorm):
        raise FileNotFoundError(f"vecnorm 없음: {args_cli.vecnorm}")

    print(f"🔄 checkpoint: {args_cli.checkpoint}")
    print(f"🔄 vecnorm   : {args_cli.vecnorm}")

    vec_env = VecNormalize.load(args_cli.vecnorm, env)
    vec_env.training = False
    vec_env.norm_reward = False
    model = PPO.load(args_cli.checkpoint, env=vec_env, device="auto")

    # cell 4 walls visual marker (실제 cell 모양 그대로, no collision)
    cell_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/cell_walls",
        markers={
            "v_wall": sim_utils.CuboidCfg(
                size=_V_WALL_SIZE,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.3, 0.3, 0.35),
                ),
            ),
            "h_wall": sim_utils.CuboidCfg(
                size=_H_WALL_SIZE,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.3, 0.3, 0.35),
                ),
            ),
        },
    )
    cell_markers = VisualizationMarkers(cell_marker_cfg)

    device = raw_env.device
    n_envs = args_cli.num_envs

    # cell-local offset / type 을 한 번만 만들기
    _wall_dx = torch.tensor([o[0] for o in _WALL_OFFSETS], device=device)  # (4,)
    _wall_dy = torch.tensor([o[1] for o in _WALL_OFFSETS], device=device)  # (4,)
    _wall_kind = torch.tensor([0 if o[2] == "v" else 1 for o in _WALL_OFFSETS],
                              device=device, dtype=torch.long)  # (4,)

    def update_cell_marker():
        # cell yaw 회전 적용해서 4 walls world pos / orientation 계산
        cell_yaw = raw_env._cell_yaw                    # (N,)
        cos_y = torch.cos(cell_yaw).unsqueeze(-1)       # (N, 1)
        sin_y = torch.sin(cell_yaw).unsqueeze(-1)       # (N, 1)
        # rotated offset: (N, 4)
        wx_local = cos_y * _wall_dx - sin_y * _wall_dy
        wy_local = sin_y * _wall_dx + cos_y * _wall_dy
        # cell world center xy
        env_origin = raw_env.scene.env_origins          # (N, 3)
        cell_x_w = (raw_env._cell_xy[:, 0:1] + env_origin[:, 0:1])  # (N, 1)
        cell_y_w = (raw_env._cell_xy[:, 1:2] + env_origin[:, 1:2])  # (N, 1)
        wall_z_w = _WALL_Z + env_origin[:, 2:3]                     # (N, 1)
        # (N, 4, 3)
        wx_world = cell_x_w + wx_local
        wy_world = cell_y_w + wy_local
        wz_world = wall_z_w.expand_as(wx_world)
        positions = torch.stack([wx_world, wy_world, wz_world], dim=-1)  # (N, 4, 3)
        positions = positions.reshape(-1, 3)                             # (N*4, 3)
        # orientation: 모든 wall 이 cell_yaw 만큼 회전 (wxyz)
        half = cell_yaw / 2.0
        wall_quat = torch.stack([torch.cos(half),
                                 torch.zeros_like(half),
                                 torch.zeros_like(half),
                                 torch.sin(half)], dim=-1)               # (N, 4)
        orientations = wall_quat.unsqueeze(1).expand(n_envs, 4, 4).reshape(-1, 4)
        # marker indices: 매 env 의 4 wall (v, v, h, h)
        indices = _wall_kind.unsqueeze(0).expand(n_envs, 4).reshape(-1)
        cell_markers.visualize(
            translations=positions,
            orientations=orientations,
            marker_indices=indices,
        )

    obs = vec_env.reset()
    update_cell_marker()
    ep_count = 0
    ep_reward = 0.0
    ep_aligned = 0.0
    ep_steps = 0
    success_count = 0
    print("\n========== Play 시작 ==========")
    while simulation_app.is_running() and ep_count < args_cli.episodes:
        action, _ = model.predict(obs, deterministic=args_cli.deterministic)
        obs, reward, dones, infos = vec_env.step(action)
        update_cell_marker()
        ep_reward += float(np.asarray(reward).mean())
        ep_aligned = max(ep_aligned, float(raw_env.reward_log.get("aligned_rate", 0.0)))
        ep_steps += 1

        if bool(np.asarray(dones).any()):
            ep_count += 1
            log = raw_env.reward_log
            success = log.get("aligned_rate", 0.0) > 0.0
            if success:
                success_count += 1
            print(
                f"  ep {ep_count:>3} | steps={ep_steps:>3} | reward={ep_reward:>7.2f} | "
                f"aligned={ep_aligned:.2f} | "
                f"r_xy={log.get('r_xy_align', 0):.3f} r_yaw={log.get('r_yaw_align', 0):.3f} "
                f"is_grasping={log.get('is_grasping_rate', 0):.2f}"
            )
            ep_reward = 0.0
            ep_aligned = 0.0
            ep_steps = 0

    print(f"\n========== Play 끝: {success_count}/{ep_count} 성공 ==========")
    if args_cli.keep_alive:
        print("   창 닫을 때까지 유지 — 추가 episode 자동 반복 (Ctrl+C 또는 창 close 로 종료)")
        # 추가 episodes 무한 반복 (사용자가 충분히 볼 수 있게)
        while simulation_app.is_running():
            action, _ = model.predict(obs, deterministic=args_cli.deterministic)
            obs, reward, dones, infos = vec_env.step(action)
            update_cell_marker()
    vec_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    os._exit(0)
