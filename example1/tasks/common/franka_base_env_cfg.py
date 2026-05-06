from __future__ import annotations

from gymnasium import spaces

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


@configclass
class FrankaBaseEnvCfg(DirectRLEnvCfg):
    """Common config for Franka object manipulation tasks."""

    episode_length_s = 4.0
    decimation = 4

    # action: 7 arm joints + 1 gripper
    num_actions = 8
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(8,),
        dtype=float,
    )

    # obs: 7 joint pos + 7 joint vel + 3 relative object pos
    num_observations = 17
    observation_space = spaces.Box(
        low=-float("inf"),
        high=float("inf"),
        shape=(17,),
        dtype=float,
    )

    num_states = 0
    state_space = 0

    # arm_action_scale = 0.03
    arm_action_scale = 0.1
    gripper_open_pos = 0.04
    gripper_close_pos = 0.0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )

    ground = GroundPlaneCfg()

    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.2, 0.2)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.55, 0.0, 0.025),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    object_pos_noise_x = 0.03
    object_pos_noise_y = 0.03

    reach_threshold = 0.05
    grasp_height_threshold = 0.04
    lift_height_threshold = 0.10