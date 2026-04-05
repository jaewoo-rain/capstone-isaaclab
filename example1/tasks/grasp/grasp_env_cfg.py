from __future__ import annotations

from isaaclab.utils import configclass

from source.example1.tasks.common.franka_base_env_cfg import FrankaBaseEnvCfg


@configclass
class GraspEnvCfg(FrankaBaseEnvCfg):
    episode_length_s = 6.0
    grasp_height_threshold = 0.04