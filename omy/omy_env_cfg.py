"""
env 전체 설정
num_envs
action_space / observation_space
카메라 설정
-> 모든 env의 공통 설정
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg

from source.omy.omy_robot_cfg import OMY_CFG


@configclass
class OmyLiftEnvCfg(DirectRLEnvCfg):
    """OMY grasp/lift용 기본 환경 설정"""

    # -------------------------
    # 1. 기본 env 설정
    # -------------------------
    decimation: int = 2
    episode_length_s: float = 8.0

    # action:
    # arm 6축 + gripper 1축(master: rh_r1_joint 기준)
    action_space: int = 7

    # obs:
    # dof_pos(10) + dof_vel(10) + obj_pos_rel(3) + obj_to_ee(3) + gripper_joint(1) + to_target(1)
    observation_space: int = 28
    state_space: int = 0

    # -------------------------
    # 2. 시뮬레이션 설정
    # -------------------------
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # -------------------------
    # 3. Scene 설정
    # -------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # -------------------------
    # 4. 로봇
    # -------------------------
    robot = OMY_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # -------------------------
    # 5. 물체
    # -------------------------
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.7, 0.2)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, -0.10, 0.02),
        ),
    )

    # -------------------------
    # 6. 카메라
    # 실제 probe 결과상 USD 안에는 camera prim이 없었으므로
    # end_effector_flange_link 아래에 새로 붙임
    # -------------------------
    camera: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/OMY/link6/camera",
        update_period=0.0,
        # height=240,
        # width=320,
        # 화질 줄임
        height=84,
        width=84,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, -0.1, 0.084),
            rot=(0.0, 0.0, 0.7, -0.7),
            convention="ros",
        ),
    )

    # -------------------------
    # 7. 바닥
    # -------------------------
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # -------------------------
    # 8. task 파라미터
    # -------------------------
    action_scale: float = 2.0
    dof_velocity_scale: float = 1.0

    object_pos_noise: float = 0.05
    lift_height_threshold: float = 0.12
    success_bonus: float = 10.0

    # 기준 링크 이름
    ee_body_name: str = "link6"
    camera_body_name: str = "camera"

    left_finger_body_name: str = "rh_p12_rn_l1"
    right_finger_body_name: str = "rh_p12_rn_r1"

    left_tip_body_name: str = "rh_p12_rn_l2"
    right_tip_body_name: str = "rh_p12_rn_r2"

    gripper_master_joint_name: str = "rh_r1_joint"

    # -------------------------
    # 9. PPO 기본값
    # -------------------------
    n_steps: int = 512
    batch_size: int = 4096
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    n_epochs: int = 5