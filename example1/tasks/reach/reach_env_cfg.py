from __future__ import annotations

from isaaclab.utils import configclass

from source.example1.tasks.common.franka_base_env_cfg import FrankaBaseEnvCfg


@configclass
class ReachEnvCfg(FrankaBaseEnvCfg):
    episode_length_s = 3.0
    # episode_length_s = 5.0
    reach_threshold = 0.05