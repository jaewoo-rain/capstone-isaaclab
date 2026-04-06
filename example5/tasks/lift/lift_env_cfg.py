from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@configclass
class LiftEnvCfg(DirectRLEnvCfg):
    """Franka Lift Env 설정"""

    # -------------------------
    # 1. 기본 env 설정
    # -------------------------
    decimation: int = 2
    episode_length_s: float = 8.0
    action_space: int = 8       # Arm(7) + Gripper(1, 미러링)
    # dof_pos(9)+dof_vel(9)+obj_rel(3)+obj_to_grip(3)+gripper_width(1)+to_target_height(1) = 26
    observation_space: int = 26
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
    # 4. 에셋 — 공식 코드와 동일하게 필드명을 'robot', 'object' 로 통일
    # -------------------------
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.785,   # -45 deg
                "panda_joint3": 0.0,
                "panda_joint4": -2.356,   # -135 deg
                "panda_joint5": 0.0,
                "panda_joint6": 1.571,    # 90 deg
                "panda_joint7": 0.785,    # 45 deg
                "panda_finger_joint.*": 0.04,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "panda_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_forearm": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                stiffness=2e3,
                damping=1e2,
            ),
        },
    )

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.7, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.0, 0.02),
        ),
    )

    # Ground — 공식 코드와 동일하게 TerrainImporterCfg 사용
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
    # 5. 태스크 파라미터
    # -------------------------
    action_scale: float = 2.0 # 움직임 속도
    dof_velocity_scale: float = 2.0 # dof_velocity_scale ↓ → 속도 정보 영향 줄임, dof_velocity_scale ↑ → 속도 정보 강조
    lift_height_threshold: float = 0.5
    success_bonus: float = 10.0
    object_pos_noise: float = 0.05 # 조금 넓게 → 일반화 향상

    # -------------------------
    # 6. PPO 학습 파라미터
    # -------------------------
    # n_steps * num_envs = 512 * 256 = 131072 (총 배치)
    # batch_size = 4096 → 미니배치 32개 → 충분한 gradient update
    n_steps: int = 512
    batch_size: int = 4096
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    n_epochs: int = 5                  # train.py에서도 PPO에 넘겨줄 것