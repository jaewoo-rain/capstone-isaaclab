"""Multi-box env 설정 (example8_2 — 3개 박스 동시 spawn).

PlaceEnv를 확장해서 3개 박스 동시 존재.
Chain inference에서 활용:
- 각 박스에 대해 example5 (grasp+lift) → example7 (place) 순환
- 박스 i 처리 후 박스 i+1로 진행 (다른 박스들은 그대로 유지)

설계:
- 박스 3개: y축으로 10cm 간격 (충돌 방지)
- 셀 3개 (3x1 or 9 중 3개)
- box ↔ cell 고정 매핑
- reset 시 active box만 초기 위치로 (다른 박스는 그대로)
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from source.omy.omy_robot_cfg import OMY_OFF_SELF_COLLISION_CFG


@configclass
class MultiBoxEnvCfg(DirectRLEnvCfg):
    """3-box, 3-cell deployment env."""

    # 기본 env
    decimation: int = 2
    episode_length_s: float = 30.0  # 3 박스 처리 시간 (각 10초)
    action_space: int = 7
    observation_space: int = 31  # PlaceEnv와 동일 형식
    state_space: int = 0

    # 시뮬레이션
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

    # Scene (낮은 num_envs, 추론 전용)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1,  # 추론은 1 env로
        env_spacing=4.0,
        replicate_physics=True,
    )

    # Robot
    robot = OMY_OFF_SELF_COLLISION_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # 3개 박스 (각각 다른 prim_path)
    box_size: tuple[float, float, float] = (0.139, 0.044, 0.118)
    box_mass: float = 0.3

    # 박스 0
    object_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object0",
        spawn=sim_utils.CuboidCfg(
            size=(0.139, 0.044, 0.118),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.3, 0.3),  # 빨강
                metallic=0.6,
                roughness=0.4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.20, 0.06)),
    )
    # 박스 1
    object_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object1",
        spawn=sim_utils.CuboidCfg(
            size=(0.139, 0.044, 0.118),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.8, 0.3),  # 녹색
                metallic=0.6,
                roughness=0.4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, -0.10, 0.06)),
    )
    # 박스 2
    object_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object2",
        spawn=sim_utils.CuboidCfg(
            size=(0.139, 0.044, 0.118),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.8),  # 파랑
                metallic=0.6,
                roughness=0.4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.00, 0.06)),
    )

    # Ground
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

    # 그리드 (3개 셀, 1x3 row)
    grid_num_x: int = 3
    grid_num_y: int = 1
    grid_center_x: float = 0.25
    grid_center_y: float = -0.45
    cell_inner_x: float = 0.16
    cell_inner_y: float = 0.065
    wall_thickness: float = 0.008
    wall_height: float = 0.12

    @property
    def cell_pitch_x(self) -> float:
        return self.cell_inner_x + self.wall_thickness

    @property
    def cell_pitch_y(self) -> float:
        return self.cell_inner_y + self.wall_thickness

    # 박스 ↔ 셀 매핑 (고정)
    # box i → cell box_to_cell[i]
    box_to_cell: tuple[int, ...] = (0, 1, 2)

    # 액션 / 태스크
    action_scale: float = 1.0
    dof_velocity_scale: float = 1.0

    cell_tolerance: float = 0.025
    stable_vel_threshold: float = 0.02
    gripper_open_threshold: float = 0.2
    success_hold_steps: int = 1
    on_floor_touch_threshold: float = 0.005
    object_fall_z: float = -0.05
    tilt_upright_threshold: float = 0.85
    bonus_upright_threshold: float = 0.985
    on_floor_z_threshold: float = 0.10
    abandoned_dist_threshold: float = 0.20

    # 이름 매핑
    left_finger_body_name: str = "rh_p12_rn_l2"
    right_finger_body_name: str = "rh_p12_rn_r2"
    gripper_joint_names: tuple[str, ...] = (
        "rh_r1_joint", "rh_r2", "rh_l1", "rh_l2",
    )

    # 박스 z=lift_threshold 도달하면 example5 → example7 전환
    lift_phase_threshold: float = 0.15
