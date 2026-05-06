import gymnasium as gym

from .grasp_env_cfg import GraspEnvCfg

gym.register(
    id="Example1-Grasp-Franka-v0",
    entry_point="source.example1.tasks.grasp.grasp_env:GraspEnv",
    disable_env_checker=True,
    kwargs={"cfg": GraspEnvCfg()},
)