"""example5 학습된 policy로 handoff 상태 수집

물체가 안정적으로 20cm 근처 높이에 잡혀있는 순간의:
 - 모든 관절 위치 (joint_pos, num_joints)
 - 물체 상대 위치 (obj_pos - env_origin, 3)
 - 물체 회전 쿼터니언 (4)
을 수집해서 checkpoints/handoff_states.npz 로 저장.

example6이 이 파일을 로드해서 reset 시 랜덤 샘플링.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect handoff states from example5")
parser.add_argument("--checkpoint", type=str, default="checkpoints/omy_lift.zip")
parser.add_argument("--vecnorm", type=str, default="checkpoints/omy_lift_vecnorm.pkl")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_samples", type=int, default=200,
                    help="수집할 handoff 상태 개수")
parser.add_argument("--capture_height", type=float, default=0.15,
                    help="물체가 이 높이 이상일 때 캡처 시도")
parser.add_argument("--output", type=str, default="checkpoints/handoff_states.npz")
parser.add_argument("--max_steps", type=int, default=10000,
                    help="안전장치: 이 스텝 넘어도 못 모으면 종료")
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
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

from source.example5.tasks.lift.lift_env_cfg import LiftEnvCfg
from source.example5.tasks.lift.lift_env import LiftEnv


def main():
    cfg = LiftEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs

    raw_env = LiftEnv(cfg=cfg)
    env = Sb3VecEnvWrapper(raw_env)

    if os.path.exists(args_cli.vecnorm):
        env = VecNormalize.load(args_cli.vecnorm, env)
        env.training = False
        env.norm_reward = False
        print(f"✅ VecNormalize 로드: {args_cli.vecnorm}")
    else:
        raise FileNotFoundError(f"VecNormalize 파일 없음: {args_cli.vecnorm}")

    model = PPO.load(args_cli.checkpoint, env=env, device="cuda")
    print(f"✅ 모델 로드: {args_cli.checkpoint}")

    # 수집 버퍼
    collected_joint_pos = []
    collected_obj_pos_rel = []
    collected_obj_quat = []

    # 각 env에서 이번 에피소드에 이미 캡처했는지 (에피소드당 최대 1회)
    captured_this_episode = np.zeros(args_cli.num_envs, dtype=bool)

    obs = env.reset()
    step = 0

    print(f"📥 handoff 상태 {args_cli.num_samples}개 수집 시작...")

    while (
        len(collected_joint_pos) < args_cli.num_samples
        and step < args_cli.max_steps
        and simulation_app.is_running()
    ):
        actions, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = env.step(actions)
        step += 1

        # raw_env 내부 상태 직접 읽기
        obj_pos_w = raw_env._object.data.root_pos_w        # (num_envs, 3)
        obj_vel = raw_env._object.data.root_lin_vel_w      # (num_envs, 3)
        obj_quat = raw_env._object.data.root_quat_w        # (num_envs, 4)
        joint_pos = raw_env._robot.data.joint_pos          # (num_envs, num_joints)
        env_origins = raw_env.scene.env_origins            # (num_envs, 3)

        obj_pos_rel = obj_pos_w - env_origins
        obj_height = obj_pos_rel[:, 2]
        obj_speed = torch.norm(obj_vel, dim=-1)

        # gripper 닫힘 상태
        grip_pos = raw_env._robot.data.joint_pos[:, raw_env.main_gripper_joint_id]
        g_lower = raw_env.robot_dof_lower_limits[raw_env.main_gripper_joint_id]
        g_upper = raw_env.robot_dof_upper_limits[raw_env.main_gripper_joint_id]
        gripper_close = (grip_pos - g_lower) / (g_upper - g_lower + 1e-8)

        # finger-object 거리 (안정적 grasp 확인)
        l_pos = raw_env._robot.data.body_pos_w[:, raw_env.left_finger_body_id, :]
        r_pos = raw_env._robot.data.body_pos_w[:, raw_env.right_finger_body_id, :]
        l_to_obj = torch.norm(obj_pos_w - l_pos, dim=-1)
        r_to_obj = torch.norm(obj_pos_w - r_pos, dim=-1)

        # 캡처 조건:
        # - 물체가 capture_height 이상 올라감
        # - 물체가 거의 정지 (안정적 파지)
        # - 두 손가락 모두 물체 가까이 (실제 잡고 있음)
        # - 그리퍼 충분히 닫힘
        capture_mask = (
            (obj_height > args_cli.capture_height)
            & (obj_speed < 0.25)
            & (l_to_obj < 0.15)
            & (r_to_obj < 0.15)
            & (gripper_close > 0.4)
        ).cpu().numpy()

        # 아직 이번 에피소드에 캡처 안 한 env만 수집
        new_capture_mask = capture_mask & (~captured_this_episode)

        if new_capture_mask.any():
            idxs = np.where(new_capture_mask)[0]
            jp = joint_pos.cpu().numpy()
            opr = obj_pos_rel.cpu().numpy()
            oq = obj_quat.cpu().numpy()

            for i in idxs:
                if len(collected_joint_pos) >= args_cli.num_samples:
                    break
                collected_joint_pos.append(jp[i].copy())
                collected_obj_pos_rel.append(opr[i].copy())
                collected_obj_quat.append(oq[i].copy())
                captured_this_episode[i] = True

        # 에피소드 종료된 env는 캡처 플래그 초기화
        if isinstance(dones, np.ndarray):
            done_mask = dones.astype(bool)
        else:
            done_mask = np.asarray(dones, dtype=bool)
        captured_this_episode[done_mask] = False

        if step % 200 == 0:
            print(f"  step={step} | 수집={len(collected_joint_pos)}/{args_cli.num_samples}")

    if len(collected_joint_pos) == 0:
        print("❌ 하나도 수집 못함. capture 조건을 완화하거나 checkpoint 품질 확인 필요")
        env.close()
        return

    jp_arr = np.stack(collected_joint_pos, axis=0).astype(np.float32)
    opr_arr = np.stack(collected_obj_pos_rel, axis=0).astype(np.float32)
    oq_arr = np.stack(collected_obj_quat, axis=0).astype(np.float32)

    os.makedirs(os.path.dirname(args_cli.output), exist_ok=True)
    np.savez(
        args_cli.output,
        joint_pos=jp_arr,
        obj_pos_rel=opr_arr,
        obj_quat=oq_arr,
    )

    print("=" * 60)
    print(f"✅ 저장 완료: {args_cli.output}")
    print(f"   수집 개수  : {jp_arr.shape[0]}")
    print(f"   joint shape: {jp_arr.shape}")
    print(f"   obj_pos_rel: {opr_arr.shape}, mean={opr_arr.mean(0)}")
    print(f"   obj_quat   : {oq_arr.shape}")
    print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
