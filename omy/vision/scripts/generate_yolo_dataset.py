from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description='Generate YOLO dataset from IsaacLab OMY vision env')
parser.add_argument('--out_dir', type=str, default='datasets/omy_yolo')
parser.add_argument('--num_samples', type=int, default=2000)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from source.omy.vision.camera.dataset_generator import IsaacYoloDatasetGenerator
from source.omy.vision.tasks.common.omy_vision_env_cfg import OmyVisionEnvCfg
from source.omy.vision.tasks.grasp.omy_grasp_vision_env import OmyGraspVisionEnv


def main():
    env = None
    try:
        cfg = OmyVisionEnvCfg()
        cfg.scene.num_envs = 1

        cfg.enable_yolo = False
        cfg.dataset_mode = True

        env = OmyGraspVisionEnv(cfg, render_mode=None)
        gen = IsaacYoloDatasetGenerator(
            args_cli.out_dir,
            cfg.camera_width,
            cfg.camera_height,
            cfg.camera_hfov_deg,
            cfg.camera_vfov_deg,
            cfg.object_size_xyz,
            cfg.slot_size_xyz,
        )

        env.reset()
        generated = 0

        while simulation_app.is_running() and generated < args_cli.num_samples:
            obs, rew, terminated, truncated, extras = env.step(
                torch.rand((1, cfg.action_space), device=env.device) * 2.0 - 1.0
            )

            rgb = env._get_camera_rgb()
            if rgb is None:
                continue

            cam_pos_w, cam_quat_w = env._get_camera_pose_for_env(0)

            gen.save_sample(
                'val' if generated % 10 == 0 else 'train',
                f'{generated:06d}',
                rgb[0],
                cam_pos_w,
                cam_quat_w,
                [env.obj_pos_w[0, i].detach().cpu().numpy() for i in range(3)],
                [env.obj_quat_w[0, i].detach().cpu().numpy() for i in range(3)],
                [env.slot_pos_w[0, i].detach().cpu().numpy() for i in range(env.slot_pos_w.shape[1])],
                [env.slot_quat_w[0, i].detach().cpu().numpy() for i in range(env.slot_pos_w.shape[1])],
            )

            generated += 1

            if bool(terminated[0] or truncated[0]):
                env.reset()

        gen.write_yaml()
        print(f'Generated {generated} samples into {args_cli.out_dir}')

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

        try:
            simulation_app.close()
        except Exception:
            pass

        import os
        os._exit(0)


if __name__ == '__main__':
    main()