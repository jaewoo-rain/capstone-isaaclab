"""기존 handoff 데이터셋의 grip 품질 분석

각 handoff 샘플을 env에 적용해서 두 손가락-박스 거리, 대칭성을 측정.
이를 바탕으로 "중심 잡힌" grasp 통과율을 보고함.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Analyze handoff grip quality")
parser.add_argument("--num_envs", type=int, default=64,
                    help="병렬로 분석할 env 수 (각 env에 다른 handoff 샘플 적용)")
parser.add_argument("--handoff_path", type=str,
                    default="checkpoints/handoff_states.npz")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from source.example7_3.tasks.place.place_env import PlaceEnv
from source.example7_3.tasks.place.place_env_cfg import PlaceEnvCfg


def main():
    # 데이터 로드
    data = np.load(args_cli.handoff_path)
    joint_pos = data["joint_pos"]  # (N, num_joints)
    obj_pos_rel = data["obj_pos_rel"]  # (N, 3)
    obj_quat = data["obj_quat"]  # (N, 4)
    n_samples = joint_pos.shape[0]
    print(f"📥 handoff 샘플 {n_samples}개 로드: {args_cli.handoff_path}")

    # env 생성
    cfg = PlaceEnvCfg()
    cfg.scene.num_envs = min(args_cli.num_envs, n_samples)
    raw_env = PlaceEnv(cfg=cfg)
    n_env = raw_env.num_envs

    # 분석 결과
    l_to_obj_all = []
    r_to_obj_all = []
    grip_center_to_obj_all = []
    asymmetry_all = []  # |l - r|

    # 샘플들을 batch로 처리
    n_batches = (n_samples + n_env - 1) // n_env
    for batch_i in range(n_batches):
        start = batch_i * n_env
        end = min(start + n_env, n_samples)
        n_this = end - start

        # 환경 reset (handoff 샘플들이 자동으로 랜덤 적용됨)
        # 강제로 특정 샘플 적용하기 위해 raw_env._reset_idx 호출 후 데이터 덮어쓰기
        env_ids = torch.arange(n_env, device=raw_env.device)
        raw_env._reset_idx(env_ids)

        # 박스 위치/회전 직접 적용
        obj_root_state = raw_env._object.data.default_root_state.clone()
        env_origins = raw_env.scene.env_origins

        for i in range(n_this):
            sample_idx = start + i

            # joint pos 직접 적용
            jp = torch.from_numpy(joint_pos[sample_idx]).to(raw_env.device).float()
            raw_env._robot.write_joint_position_to_sim(
                jp.unsqueeze(0), env_ids=torch.tensor([i], device=raw_env.device)
            )

            # 박스 위치 + quat 적용 (env origin 더해서 world coord)
            obj_pos_w = torch.from_numpy(obj_pos_rel[sample_idx]).to(raw_env.device).float() + env_origins[i]
            obj_q = torch.from_numpy(obj_quat[sample_idx]).to(raw_env.device).float()
            obj_root_state[i, :3] = obj_pos_w
            obj_root_state[i, 3:7] = obj_q

        raw_env._object.write_root_pose_to_sim(obj_root_state[:n_env, :7])
        # vel 0
        zero_vel = torch.zeros((n_env, 6), device=raw_env.device)
        raw_env._object.write_root_velocity_to_sim(zero_vel)

        # 한 step 시뮬레이션 (joint 적용 반영)
        raw_env.sim.step(render=False)
        raw_env.scene.update(dt=raw_env.cfg.sim.dt)

        # 측정
        obj_pos_w_actual = raw_env._object.data.root_pos_w
        l_pos = raw_env._robot.data.body_pos_w[:, raw_env.left_finger_body_id, :]
        r_pos = raw_env._robot.data.body_pos_w[:, raw_env.right_finger_body_id, :]

        l_to_obj = torch.norm(obj_pos_w_actual - l_pos, dim=-1)
        r_to_obj = torch.norm(obj_pos_w_actual - r_pos, dim=-1)
        grip_center = (l_pos + r_pos) * 0.5
        gc_to_obj = torch.norm(obj_pos_w_actual - grip_center, dim=-1)
        asym = torch.abs(l_to_obj - r_to_obj)

        l_to_obj_all.append(l_to_obj[:n_this].cpu().numpy())
        r_to_obj_all.append(r_to_obj[:n_this].cpu().numpy())
        grip_center_to_obj_all.append(gc_to_obj[:n_this].cpu().numpy())
        asymmetry_all.append(asym[:n_this].cpu().numpy())

    l_arr = np.concatenate(l_to_obj_all)
    r_arr = np.concatenate(r_to_obj_all)
    gc_arr = np.concatenate(grip_center_to_obj_all)
    asym_arr = np.concatenate(asymmetry_all)

    print()
    print("=" * 70)
    print(f"📊 Grip 품질 분석 결과 (n={len(l_arr)})")
    print("=" * 70)

    def stats(name, arr, unit="cm"):
        scale = 100 if unit == "cm" else 1
        print(f"  {name}:")
        print(f"    mean  = {arr.mean()*scale:7.2f} {unit}")
        print(f"    median= {np.median(arr)*scale:7.2f} {unit}")
        print(f"    min   = {arr.min()*scale:7.2f} {unit}")
        print(f"    max   = {arr.max()*scale:7.2f} {unit}")
        print(f"    std   = {arr.std()*scale:7.2f} {unit}")

    stats("L finger ↔ obj", l_arr)
    stats("R finger ↔ obj", r_arr)
    stats("grip center ↔ obj", gc_arr)
    stats("asymmetry |L-R|", asym_arr)

    # 통과율 (다양한 임계값)
    print()
    print("📊 필터 통과율 (가능성 시뮬레이션):")
    for tight_th, asym_th in [
        (0.15, 0.10),  # 현재 수준 (느슨)
        (0.10, 0.05),
        (0.07, 0.03),
        (0.05, 0.02),
        (0.04, 0.015),
        (0.03, 0.01),  # 매우 엄격 (깊이/대칭)
    ]:
        passing = (l_arr < tight_th) & (r_arr < tight_th) & (asym_arr < asym_th)
        n_pass = passing.sum()
        pct = 100 * n_pass / len(l_arr)
        print(f"  L,R < {tight_th*100:.0f}cm  AND  |L-R| < {asym_th*100:.0f}cm  →  "
              f"통과 {n_pass}/{len(l_arr)} ({pct:.1f}%)")

    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()
