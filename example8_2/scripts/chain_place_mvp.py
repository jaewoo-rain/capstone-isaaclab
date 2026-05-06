"""example8_2 chain MVP - place phase only.

PlaceEnv (example7)에서 3개 박스를 3개 셀에 배치하는 시나리오.
- 각 박스 = 하나의 episode
- target cell = box index (0,1,2 고정 매핑)
- handoff_states.npz에서 시작 (박스 잡힌 상태 z=0.17)

결과 측정:
- 각 박스별 place 성공 여부
- 성공률 요약

Usage:
  python source/example8_2/scripts/chain_place_mvp.py --num_boxes 3
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="3-box place chain (example7 단독)")
parser.add_argument("--ckpt", type=str, default="checkpoints/example7.zip")
parser.add_argument("--num_boxes", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--max_steps_per_box", type=int, default=600)  # 10s
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnvWrapper

try:
    from isaaclab_rl.sb3 import Sb3VecEnvWrapper
except ImportError:
    from isaaclab_rl.stable_baselines3 import Sb3VecEnvWrapper

from source.example7.tasks.place.place_env_cfg import PlaceEnvCfg
from source.example7.tasks.place.place_env import PlaceEnv


# Goal Dict wrapper (train.py와 동일)
class GoalEnvVecWrapper(VecEnvWrapper):
    def __init__(self, venv, core_dim=25, goal_dim=3):
        super().__init__(venv)
        self._core_dim = core_dim
        self._goal_dim = goal_dim
        low = np.full(core_dim, -np.inf, dtype=np.float32)
        high = np.full(core_dim, np.inf, dtype=np.float32)
        goal_low = np.full(goal_dim, -np.inf, dtype=np.float32)
        goal_high = np.full(goal_dim, np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=low, high=high, dtype=np.float32),
            "achieved_goal": gym.spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
            "desired_goal": gym.spaces.Box(low=goal_low, high=goal_high, dtype=np.float32),
        })

    def _split(self, obs):
        core = obs[..., :self._core_dim]
        achieved = obs[..., self._core_dim:self._core_dim + self._goal_dim]
        desired = obs[..., self._core_dim + self._goal_dim:]
        return {
            "observation": core.astype(np.float32),
            "achieved_goal": achieved.astype(np.float32),
            "desired_goal": desired.astype(np.float32),
        }

    def reset(self):
        return self._split(self.venv.reset())

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._split(obs), rewards, dones, infos

    def compute_reward(self, achieved_goal, desired_goal, info):
        # placeholder
        return -np.linalg.norm(achieved_goal - desired_goal, axis=-1)


def main():
    print("=" * 70)
    print("example8_2 Place MVP: 3-box place (example7 단독)")
    print("=" * 70)

    if not os.path.exists(args_cli.ckpt):
        print(f"❌ {args_cli.ckpt} 없음")
        return

    cfg = PlaceEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.randomize_target_cell = False  # cell 0 고정 (example7은 1 cell로 학습)
    # grid는 default 1x1 유지

    raw_env = PlaceEnv(cfg=cfg)
    env_isaac = Sb3VecEnvWrapper(raw_env)
    env = GoalEnvVecWrapper(env_isaac)

    model = SAC.load(args_cli.ckpt, env=env, device="cuda")
    print(f"✅ example7 정책: {args_cli.ckpt}")
    print()

    box_results = []
    for box_idx in range(args_cli.num_boxes):
        print(f"━━━ Box {box_idx} (cell 0 고정) ━━━")

        obs = env.reset()
        # Debug: 시작 상태 print
        env_origin = raw_env.scene.env_origins[0]
        obj_pos = raw_env._object.data.root_pos_w[0] - env_origin
        target_pos = raw_env.target_cell_pos_w[0] - env_origin
        print(f"  🔍 start: obj=({obj_pos[0]:.3f},{obj_pos[1]:.3f},{obj_pos[2]:.3f}) tgt=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})")

        success = False
        max_z = 0.0
        step = 0
        done_reason = "?"

        while step < args_cli.max_steps_per_box and simulation_app.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            step += 1

            obj_z = raw_env._object.data.root_pos_w[0, 2].item()
            env_z = raw_env.scene.env_origins[0, 2].item()
            obj_z_rel = obj_z - env_z
            max_z = max(max_z, obj_z_rel)

            # success
            if hasattr(raw_env, '_last_success_now') and raw_env._last_success_now[0].item():
                success = True
                done_reason = "SUCCESS"
                print(f"  ✅ Place 성공 step={step}, obj_z={obj_z_rel:.3f}")
                break

            if done[0]:
                # raw_env _last 속성에서 직접 확인
                tilted = raw_env._last_tilted[0].item() if hasattr(raw_env, '_last_tilted') else False
                sev_tilt = raw_env._last_severely_tilted[0].item() if hasattr(raw_env, '_last_severely_tilted') else False
                aband = raw_env._last_abandoned[0].item() if hasattr(raw_env, '_last_abandoned') else False
                succ = raw_env._last_success_now[0].item() if hasattr(raw_env, '_last_success_now') else False
                inserted = raw_env._last_inserted[0].item() if hasattr(raw_env, '_last_inserted') else False
                xy_loose = raw_env._last_xy_aligned_loose[0].item() if hasattr(raw_env, '_last_xy_aligned_loose') else False
                upright = raw_env._object.data.root_quat_w[0]
                w_up_z = 1.0 - 2.0 * (upright[1].item()**2 + upright[2].item()**2)
                grip_to_obj = torch.norm(raw_env.grip_center_pos[0] - raw_env._object.data.root_pos_w[0]).item()
                obj_pos_rel_now = raw_env._object.data.root_pos_w[0] - env_origin
                print(f"  📋 done detected: success={succ}, tilted={tilted}, sev_tilt={sev_tilt}, aband={aband}, inserted={inserted}, xy_loose={xy_loose}")
                print(f"     obj=({obj_pos_rel_now[0]:.3f},{obj_pos_rel_now[1]:.3f},{obj_pos_rel_now[2]:.3f}) upright_z={w_up_z:.3f} grip_obj={grip_to_obj:.3f}")
                break

        if not success:
            print(f"  ❌ Box {box_idx} 실패 (max_z={max_z:.3f}, step={step}, reason={done_reason})")

        box_results.append({"box": box_idx, "success": success, "max_z": max_z, "steps": step, "reason": done_reason})

    # Summary
    print()
    print("=" * 70)
    print("📊 결과 요약")
    print("=" * 70)
    success_count = sum(1 for r in box_results if r["success"])
    print(f"  Place 성공: {success_count}/{args_cli.num_boxes}")
    for r in box_results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} Box {r['box']}: max_z={r['max_z']:.3f}m, steps={r['steps']}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
