"""OMY Hierarchy Env 설정

example5 (Lift PPO) + example6 (Place SAC+HER)을 계층적으로 실행하는 환경.

특징:
- 그리드는 example6에서 그대로 가져옴 (수직 4 + 수평 4 벽)
- 시작 상태는 example5 스타일: 물체 바닥, 로봇 기본 자세
- Lift policy → 물체를 0.2m까지 올림 → Place policy 전환

play 스크립트가 obj_height > 0.2 시점에 policy를 바꿔치움.
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
class HierEnvCfg(DirectRLEnvCfg):
    """OMY Hierarchy Env — Lift + Place 통합"""

    # -------------------------
    # 1. 기본 env 설정
    # -------------------------
    decimation: int = 2
    # lift 12s + place 10s 합쳐서 22s 정도 필요
    episode_length_s: float = 10.0

    # action: arm 6 + gripper 1 = 7 (둘 다 동일)
    action_space: int = 7

    # 기본 observation: place 형식 (31차원)
    # lift용 34차원은 별도 메서드로 노출
    observation_space: int = 31

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
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # -------------------------
    # 4. Robot
    # -------------------------
    robot = OMY_OFF_SELF_COLLISION_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # -------------------------
    # 5. Object — example5 lift와 동일한 위치에서 시작
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.667, 0.686, 0.706),
                metallic=0.8,
                roughness=0.4,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # lift 환경과 동일: 로봇 앞 45cm, 오른쪽 10cm, 바닥에 놓임
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
    # 7. 그리드 (example6에서 그대로 가져옴)
    # -------------------------
    grid_num_x: int = 1
    grid_num_y: int = 1
    grid_center_x: float = 0.25
    grid_center_y: float = -0.45
    cell_inner_x: float = 0.17
    cell_inner_y: float = 0.06
    wall_thickness: float = 0.008
    wall_height: float = 0.12

    @property
    def cell_pitch_x(self) -> float:
        return self.cell_inner_x + self.wall_thickness

    @property
    def cell_pitch_y(self) -> float:
        return self.cell_inner_y + self.wall_thickness

    # -------------------------
    # 8. 태스크 파라미터
    # -------------------------
    action_scale: float = 1.0
    dof_velocity_scale: float = 1.0

    # lift → place 전환 임계값
    lift_to_place_threshold: float = 0.20

    # 물체 초기 위치 노이즈 (lift cfg와 동일)
    object_pos_noise: float = 0.02

    # lift obs용 — grasp_target = obj 중심 + z offset
    grasp_target_z_offset: float = 0.04

    # place 관련 파라미터 (place env와 동일)
    cell_tolerance: float = 0.015
    stable_vel_threshold: float = 0.02
    gripper_open_threshold: float = 0.2
    success_hold_steps: int = 50
    object_fall_z: float = -0.05
    tilt_upright_threshold: float = 0.3
    on_floor_z_threshold: float = 0.10
    abandoned_dist_threshold: float = 0.15

    randomize_target_cell: bool = True

    # -------------------------
    # 9. 이름 매핑 (둘 다 동일)
    # -------------------------
    left_finger_body_name: str = "rh_p12_rn_l2"
    right_finger_body_name: str = "rh_p12_rn_r2"

    gripper_joint_names: tuple[str, ...] = (
        "rh_r1_joint",
        "rh_r2",
        "rh_l1",
        "rh_l2",
    )
