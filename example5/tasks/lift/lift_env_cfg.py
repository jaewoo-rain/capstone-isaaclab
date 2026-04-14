from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from source.omy.omy_robot_cfg import OMY_CFG


@configclass
class LiftEnvCfg(DirectRLEnvCfg):
    """OMY Lift Env 설정"""

    # -------------------------
    # 1. 기본 env 설정
    # -------------------------
    decimation: int = 2
    episode_length_s: float = 8.0

    # Arm(6) + Gripper(1)
    action_space: int = 7

    # 실제 obs dim은 env에서 다시 맞춰서 사용
    observation_space: int = 22
    state_space: int = 0

    # -------------------------
    # 2. 시뮬레이션
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
    # 3. Scene
    # -------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=256,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # -------------------------
    # 4. Robot
    # -------------------------
    robot = OMY_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # -------------------------
    # 5. Object
    # -------------------------
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.139, 0.044, 0.118),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3), # 0.3kg
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.667, 0.686, 0.706), # (R,G,B)색
                metallic=0.8, # 높을수록 금속 느낌
                roughness=0.4, # 낮을수록 반짝임
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.45, -0.10, 0.06),
        ),
    )

    # -------------------------
    # 6. Ground
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
    # 7. 태스크 파라미터
    # -------------------------
    action_scale: float = 1.0
    dof_velocity_scale: float = 1.0

    # OMY가 들기 현실적인 높이
    lift_height_threshold: float = 0.2

    success_bonus: float = 10.0
    object_pos_noise: float = 0.02

    # 카메라 바닥 접근 패널티
    camera_min_height: float = 0.03

    # -------------------------
    # 8. PPO 학습 파라미터
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