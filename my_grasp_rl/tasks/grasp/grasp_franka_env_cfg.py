
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
    """Franka 기반 grasp + lift task 설정.

    논문 대응:
    - 랜덤 위치 물체 파지 문제 -> object reset randomization
    - continuous control -> joint delta action
    - grasp success / lift success -> staged reward
    """

    # ---------------------------------------------------------------------
    # 기본 env 설정
    # ---------------------------------------------------------------------
    decimation = 2
    episode_length_s = 6 # 에피소드 길이
    action_space = 8  # 7 arm joints + 1 gripper scalar
    observation_space = 30
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
        # num_envs=1,
        env_spacing=3.0,
        replicate_physics=True,
    )
    
    # ---------------------------------------------------------------------
    # 로봇 설정
    # ---------------------------------------------------------------------
    # [실제 로봇 교체 지점 1]
    # OMY 등 다른 로봇을 쓸 때 여기 robot_cfg를 교체하면 된다.
    robot_cfg: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # [실제 로봇 교체 지점 2]
    # joint 이름 / ee link 이름 / gripper joint 이름은 로봇마다 달라진다.
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, 0.0, 0.02)),
    )

    ground = GroundPlaneCfg()


    # ---------------------------------------------------------------------
    # 액션 스케일
    # ---------------------------------------------------------------------
    arm_action_scale = 0.02   # 0.04 -> 0.02, 제어 더 안정적으로
    gripper_open_target = 0.04
    gripper_close_target = 0.0

    # ---------------------------------------------------------------------
    # 리셋 랜덤 범위
    # ---------------------------------------------------------------------
    object_x_range = (0.50, 0.56)   # 범위 축소
    object_y_range = (-0.04, 0.04)  # 범위 축소
    object_z = 0.02

    # ---------------------------------------------------------------------
    # 성공 조건
    # ---------------------------------------------------------------------
    success_lift_height = 0.08      # grasp 단계에서는 lift 성공 높이도 완화
    max_goal_distance = 0.35

    # ---------------------------------------------------------------------
    # reward scale
    # 논문에서의 핵심인 staged reward를 Isaac Lab 식으로 분해
    # ---------------------------------------------------------------------
    # rew_reach = 2.0
    # rew_align = 1.0
    # rew_grasp = 4.0
    # rew_lift = 8.0
    # rew_action_penalty = -0.01
    # rew_joint_vel_penalty = -0.001

    rew_reach = 0.5 # reach 보상 줄임
    rew_align = 0.0
    rew_grasp = 8.0 # grasp 보상 비중 높임
    rew_lift = 2.0
    rew_action_penalty = -0.005
    rew_joint_vel_penalty = -0.0005