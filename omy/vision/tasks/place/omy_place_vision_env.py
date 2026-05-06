from __future__ import annotations

import torch

from source.omy.vision.tasks.common.omy_base_vision_env import OmyBaseVisionEnv
from source.omy.vision.tasks.common.omy_vision_env_cfg import OmyVisionEnvCfg


class OmyPlaceVisionEnv(OmyBaseVisionEnv):
    cfg: OmyVisionEnvCfg

    def _get_rewards(self) -> torch.Tensor:
        t = self._get_common_terms()
        slot_pos = self.current_slot_pos_w
        obj_to_slot = slot_pos - t['obj_pos']
        slot_dist_xy = torch.norm(obj_to_slot[:, :2], dim=-1)
        slot_dist_z = torch.abs(obj_to_slot[:, 2])
        over_slot = (slot_dist_xy < 0.05) & (slot_dist_z < 0.10) & self.current_slot_valid
        release_reward = over_slot.float() * (self.actions[:, 6] < -0.1).float() * 4.0
        placed = over_slot & (~t['is_grasping']) & self.current_slot_valid
        reward = 0.8 * torch.exp(-80.0 * slot_dist_xy**2) + 0.8 * torch.exp(-80.0 * slot_dist_z**2) + release_reward + placed.float() * self.cfg.success_bonus + 0.25 * t['vision_ok'].float() - 0.15 * t['vision_stale'].float() - torch.sum(self.actions[:, :6] ** 2, dim=-1) * 0.001
        self.reward_log = {'placed_rate': float(placed.float().mean()), 'slot_valid': float(self.current_slot_valid.float().mean())}
        self.extras['log'] = dict(self.reward_log)
        return reward
