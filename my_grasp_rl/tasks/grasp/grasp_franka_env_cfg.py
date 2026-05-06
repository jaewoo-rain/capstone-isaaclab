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
class GraspFrankaEnvCfg(DirectRLEnvCfg):
    """Franka 기반 단순 grasp + lift task 설정.

    단순화 방향:
    - State  : robot joint state + gripper state + object relative state
    - Action : 7 arm joint delta + 1 gripper scalar
    - Reward : reach + lift - penalties
    """

    # ---------------------------------------------------------------------
    # 기본 env 설정
    # ---------------------------------------------------------------------
    decimation = 2
    episode_length_s = 6.0

    # action = 7개 팔 관절 + 1개 gripper
    action_space = 8

    # observation =
    # joint_pos(7) + joint_vel(7) + finger_pos(2) + finger_vel(2)
    # + obj_to_ee(3) + obj_lin_vel(3)
    # = 24
    observation_space = 24
    state_space = 0

    # ---------------------------------------------------------------------
    # 시뮬레이션 설정
    # ---------------------------------------------------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
    )

    # ---------------------------------------------------------------------
    # 병렬 환경 개수
    # ---------------------------------------------------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # ---------------------------------------------------------------------
    # 로봇 설정
    # ---------------------------------------------------------------------
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # Franka Panda joint / finger / end-effector 이름
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

    # ---------------------------------------------------------------------
    # 물체 설정
    # ---------------------------------------------------------------------
    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
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
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.2, 0.2)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.02)),
    )

    ground = GroundPlaneCfg()

    # ---------------------------------------------------------------------
    # 액션 스케일
    # ---------------------------------------------------------------------
    # arm joint delta 크기
    arm_action_scale = 0.02

    # gripper 위치 한계
    gripper_open_target = 0.04
    gripper_close_target = 0.0

    # gripper delta 크기 (속도 느낌)
    gripper_action_scale = 0.01

    # ---------------------------------------------------------------------
    # 리셋 랜덤 범위
    # ---------------------------------------------------------------------
    object_x_range = (0.50, 0.56)
    object_y_range = (-0.04, 0.04)
    object_z = 0.02

    # ---------------------------------------------------------------------
    # 성공 조건
    # ---------------------------------------------------------------------
    success_lift_height = 0.08

    # ---------------------------------------------------------------------
    # reward scale
    # 단순 구조: reach + lift - penalties
    # 주의: penalty 계수는 양수로 두고, reward 식에서 빼는 방식 사용
    # ---------------------------------------------------------------------
    rew_reach = 1.0
    rew_lift = 8.0

    rew_action_penalty = 0.005
    rew_joint_vel_penalty = 0.0005