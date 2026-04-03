from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG


@configclass
class GoodRobotFrankaEnvCfg(DirectRLEnvCfg):
    # -------------------------------------------------
    # 기본 설정
    # -------------------------------------------------
    decimation = 2
    episode_length_s = 10.0
    action_space = 8
    observation_space = 42
    state_space = 0

    # -------------------------------------------------
    # simulation
    # -------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
    )

    # -------------------------------------------------
    # scene
    # -------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # -------------------------------------------------
    # robot
    # -------------------------------------------------
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    arm_joint_names = [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ]
    left_finger_joint_name = "panda_finger_joint1"
    right_finger_joint_name = "panda_finger_joint2"
    ee_body_name = "panda_hand"

    # -------------------------------------------------
    # source object: 집을 물체
    # -------------------------------------------------
    source_object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/SourceObject",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.3, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, -0.10, 0.02)),
    )

    # -------------------------------------------------
    # target pad: 목적 위치 표시용
    # -------------------------------------------------
    target_object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TargetObject",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.01),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.01),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.62, 0.12, 0.005)),
    )

    ground = GroundPlaneCfg()

    # -------------------------------------------------
    # action
    # -------------------------------------------------
    arm_action_scale = 0.02
    gripper_open_target = 0.04
    gripper_close_target = 0.0

    # -------------------------------------------------
    # randomization
    # -------------------------------------------------
    source_x_range = (0.46, 0.54)
    source_y_range = (-0.14, -0.06)

    target_x_range = (0.58, 0.66)
    target_y_range = (0.08, 0.16)

    source_object_z = 0.02
    target_place_z = 0.04

    # -------------------------------------------------
    # thresholds
    # -------------------------------------------------
    reach_threshold = 0.05
    lift_height_threshold = 0.06
    transport_xy_threshold = 0.05
    place_xy_threshold = 0.02
    place_z_threshold = 0.02
    stable_steps_required = 12

    # -------------------------------------------------
    # reward scales
    # Good Robot 스타일: 큰 실패 패널티보다
    # progress reward + stage completion reward 위주
    # -------------------------------------------------
    rew_reach = 1.0
    rew_grasp = 3.0
    rew_lift = 3.0
    rew_transport = 2.5
    rew_place = 4.0
    rew_release = 2.0
    rew_stable = 8.0
    rew_stage_bonus = 1.0

    rew_action_penalty = 0.005
    rew_joint_vel_penalty = 0.0005