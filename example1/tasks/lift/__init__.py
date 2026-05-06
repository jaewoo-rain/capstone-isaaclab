import gymnasium as gym

from .lift_env_cfg import LiftEnvCfg

gym.register(
    id="Example1-Lift-Franka-v0",
    entry_point="source.example1.tasks.lift.lift_env:LiftEnv",
    disable_env_checker=True,
    kwargs={"cfg": LiftEnvCfg()},
)