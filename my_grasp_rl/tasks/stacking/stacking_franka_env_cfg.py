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
class StackingFrankaEnvCfg(DirectRLEnvCfg):
    # -------------------------------------------------
    # 기본 설정
    # -------------------------------------------------
    decimation = 2
    episode_length_s = 8.0
    action_space = 8
    observation_space = 36
    state_space = 0

    # -------------------------------------------------
    # 시뮬레이션
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

    # 일단 유지. 나중에 env.py에서 finger center를 EE로 계산
    ee_body_name = "panda_hand"

    # -------------------------------------------------
    # object (아래 블록)
    # -------------------------------------------------
    base_object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/BaseObject",
        spawn=sim_utils.CuboidCfg(
            size=(0.12, 0.12, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True, # 고정시키기
                max_linear_velocity=10.0,
                max_angular_velocity=10.0,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.60, 0.0, 0.02)),
    )

    # -------------------------------------------------
    # object (위에 쌓을 블록)
    # -------------------------------------------------
    stack_object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/StackObject",
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.50, 0.0, 0.02)),
    )

    ground = GroundPlaneCfg()

    # -------------------------------------------------
    # action
    # -------------------------------------------------
    arm_action_scale = 0.02
    gripper_open_target = 0.04
    gripper_close_target = 0.0

    # -------------------------------------------------
    # reset randomization
    # -------------------------------------------------
    # base_x_range = (0.58, 0.62)
    base_x_range = (0.60, 0.62)
    base_y_range = (-0.03, 0.03)

    # stack_x_range = (0.48, 0.54)
    stack_x_range = (0.48, 0.50)
    stack_y_range = (-0.06, 0.06)

    object_z = 0.02
    cube_size = 0.04

    # target z = base top + half stack cube
    target_stack_height = 0.02 + 0.04

    # -------------------------------------------------
    # success threshold
    # -------------------------------------------------
    success_xy_threshold = 0.015
    success_z_threshold = 0.015
    stable_steps_required = 10

    # -------------------------------------------------
    # reward scale
    # -------------------------------------------------
    rew_reach_obj = 0.5
    rew_grasp = 6.0
    rew_move_target = 2.0
    rew_align_xy = 3.0
    rew_align_z = 2.0
    rew_release = 4.0
    rew_stable = 8.0
    rew_action_penalty = -0.005
    rew_joint_vel_penalty = -0.0005