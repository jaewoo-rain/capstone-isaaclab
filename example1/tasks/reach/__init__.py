import gymnasium as gym

from .reach_env_cfg import ReachEnvCfg

gym.register(
    id="Example1-Reach-Franka-v0",
    entry_point="source.example1.tasks.reach.reach_env:ReachEnv",
    disable_env_checker=True,
    kwargs={"cfg": ReachEnvCfg()},
)