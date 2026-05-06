from __future__ import annotations

import torch

from source.omy.vision.tasks.common.omy_base_vision_env import OmyBaseVisionEnv
from source.omy.vision.tasks.common.omy_vision_env_cfg import OmyVisionEnvCfg


class OmyLiftVisionEnv(OmyBaseVisionEnv):
    cfg: OmyVisionEnvCfg

    def _get_rewards(self) -> torch.Tensor:
        t = self._get_common_terms()
        obj_height = t['obj_pos'][:, 2]
        current_lift = torch.clamp(obj_height - (self.cfg.object_size_xyz[2] * 0.5), min=0.0)
        lift_reward = current_lift * t['is_grasping'].float() * 18.0
        success = (obj_height > self.cfg.lift_height_threshold).float()
        reward = 0.2 * t['approach_reward'] + 2.0 * t['is_grasping'].float() + lift_reward + success * self.cfg.success_bonus + 0.25 * t['vision_ok'].float() - 0.2 * t['vision_stale'].float() - torch.clamp(self.vision_miss_count.float() * 0.01, max=0.25) - torch.sum(self.actions[:, :6] ** 2, dim=-1) * 0.001
        self.reward_log = {'lift_reward': float(lift_reward.mean()), 'success_rate': float(success.mean())}
        self.extras['log'] = dict(self.reward_log)
        return reward
