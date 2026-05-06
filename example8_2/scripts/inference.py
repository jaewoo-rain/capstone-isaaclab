"""example8_2: 3-box chain inference using example5_2/7_2/7_3 policies.

학습된 정책 3개를 순차 실행:
1. example5_2 (grasp + N-step hold)
2. example7_2 (lift + transport)
3. example7_3 (align + insert + release)

각 박스 i (i=0,1,2)에 대해 위 사이클 반복.

Note: 이 코드는 SKELETON. 실제 multi-box env가 없어 단일 박스 sequential 시뮬레이션.
example5_2/7_2/7_3 학습 완성 후 multi-box env 추가 필요.
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Chain inference: grasp → lift → place")
parser.add_argument("--grasp_ckpt", type=str,
                    default="checkpoints/example5_2_v47b_40M.zip")
parser.add_argument("--grasp_vecnorm", type=str,
                    default="checkpoints/example5_2_v47b_40M_vecnorm.pkl")
parser.add_argument("--lift_ckpt", type=str,
                    default="checkpoints/example7_2_v14_39M.zip")
parser.add_argument("--lift_replay", type=str,
                    default="checkpoints/example7_2_v14_39M_replay.pkl")
parser.add_argument("--place_ckpt", type=str,
                    default="checkpoints/example7_3.zip")
parser.add_argument("--num_boxes", type=int, default=3)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)

args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import numpy as np
import torch

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecNormalize

print("=" * 70)
print("example8_2: Chain Inference (Skeleton)")
print("=" * 70)
print()
print("⚠️ 현재는 SKELETON 코드. 다음 작업 필요:")
print("  1. example7_2 학습 완성 (lift+transport 성공)")
print("  2. example7_3 학습 완성 (align+insert+release 성공)")
print("  3. Multi-box env 작성 (3개 박스 동시)")
print("  4. 실제 chain 추론 로직 구현")
print()
print("현재 정책 상태:")
print(f"  grasp:  {args_cli.grasp_ckpt}")
print(f"  lift:   {args_cli.lift_ckpt}")
print(f"  place:  {args_cli.place_ckpt} (미학습)")
print()

# State Machine 정의 (의사 코드)
class ChainStateMachine:
    """3-stage chain inference state machine.

    States: GRASPING → LIFTING → PLACING → DONE
    각 박스마다 위 사이클 반복.
    """

    def __init__(self, grasp_policy, lift_policy, place_policy, num_boxes=3):
        self.grasp_policy = grasp_policy
        self.lift_policy = lift_policy
        self.place_policy = place_policy
        self.num_boxes = num_boxes
        self.cell_mapping = list(range(num_boxes))  # 박스 i → 셀 i (고정)
        self.reset()

    def reset(self):
        self.state = "GRASPING"
        self.box_idx = 0
        self.target_cell = self.cell_mapping[0]
        self.step_count = 0

    def step(self, obs):
        """매 step 호출. 현재 state에 맞는 정책으로 action 계산.
        전환 조건 만족 시 다음 state로 이동.
        """
        if self.state == "GRASPING":
            # example5_2 정책 obs 변환 (10 joint + box pos + ...)
            grasp_obs = self._to_grasp_obs(obs)
            action, _ = self.grasp_policy.predict(grasp_obs, deterministic=True)
            # 전환 조건: grasp_streak ≥ 60
            if self._check_grasp_done():
                self.state = "LIFTING"

        elif self.state == "LIFTING":
            lift_obs = self._to_lift_obs(obs)
            action, _ = self.lift_policy.predict(lift_obs, deterministic=True)
            # 전환 조건: 박스 z >= 0.28 + xy align
            if self._check_lift_done():
                self.state = "PLACING"

        elif self.state == "PLACING":
            place_obs = self._to_place_obs(obs)
            action, _ = self.place_policy.predict(place_obs, deterministic=True)
            # 전환 조건: 박스 셀 안 + gripper open + stable
            if self._check_place_done():
                self.box_idx += 1
                if self.box_idx >= self.num_boxes:
                    self.state = "DONE"
                else:
                    self.state = "GRASPING"
                    self.target_cell = self.cell_mapping[self.box_idx]

        elif self.state == "DONE":
            action = self._idle_action()

        self.step_count += 1
        return action

    def _to_grasp_obs(self, obs):
        # TODO: example5_2 obs format으로 변환
        raise NotImplementedError("Multi-box env 작성 후 구현")

    def _to_lift_obs(self, obs):
        # TODO: example7_2 obs format
        raise NotImplementedError("Multi-box env 작성 후 구현")

    def _to_place_obs(self, obs):
        # TODO: example7_3 obs format
        raise NotImplementedError("Multi-box env 작성 후 구현")

    def _check_grasp_done(self):
        # TODO: env에서 grasp_streak 정보 필요
        return False

    def _check_lift_done(self):
        return False

    def _check_place_done(self):
        return False

    def _idle_action(self):
        return torch.zeros(7)


# 정책 로드 시도
def load_policies():
    print("정책 로드 중...")
    policies = {}
    if os.path.exists(args_cli.grasp_ckpt):
        try:
            policies["grasp"] = PPO.load(args_cli.grasp_ckpt, device="cuda")
            print(f"✅ grasp policy 로드: {args_cli.grasp_ckpt}")
        except Exception as e:
            print(f"❌ grasp policy 실패: {e}")
    else:
        print(f"❌ grasp ckpt 없음: {args_cli.grasp_ckpt}")

    if os.path.exists(args_cli.lift_ckpt):
        try:
            policies["lift"] = SAC.load(args_cli.lift_ckpt, device="cuda")
            print(f"✅ lift policy 로드: {args_cli.lift_ckpt}")
        except Exception as e:
            print(f"❌ lift policy 실패: {e}")
    else:
        print(f"❌ lift ckpt 없음: {args_cli.lift_ckpt}")

    if os.path.exists(args_cli.place_ckpt):
        try:
            policies["place"] = SAC.load(args_cli.place_ckpt, device="cuda")
            print(f"✅ place policy 로드: {args_cli.place_ckpt}")
        except Exception as e:
            print(f"❌ place policy 실패: {e}")
    else:
        print(f"⏳ place ckpt 없음 (example7_3 미학습): {args_cli.place_ckpt}")

    return policies


def main():
    policies = load_policies()
    print()
    print("=" * 70)
    print("Chain inference 실행은 multi-box env 작성 후 가능")
    print("=" * 70)


if __name__ == "__main__":
    main()
    simulation_app.close()
