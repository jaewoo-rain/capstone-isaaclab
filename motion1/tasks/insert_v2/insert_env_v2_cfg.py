"""motion1 — Insert RL env cfg (v2: coded reset).

Task: 박스 잡힌 채 셀 위에서 xy/yaw 미세 정렬 (RL).
- Action (3): Δx, Δy, Δyaw (cartesian, relative)
- State (7): slot_rel_x, slot_rel_y, slot_yaw_err, is_grasping, ee_vel_x, ee_vel_y, yaw_vel
- ee_z, gripper: 고정 (학습 안 함)
- **시작 상태**: handoff dataset X. 코드로 직접 매핑.
  - robot 을 cell + ee_dxy 위로 IK 이동
  - 박스를 grip center 에 강제 매핑 (yaw = cell_yaw)

v1 (insert_env) 대비 변경:
- handoff dataset 사용 X → 손상 sample 비율 0
- 박스 정확히 손에 매핑 → is_grasping rate 보장
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
class InsertEnvV2Cfg(DirectRLEnvCfg):
    """OMY Insert 미세 정렬 RL Env cfg (v2: coded reset)."""

    # =========================
    # 1. 기본 RL env 설정
    # =========================
    decimation: int = 2
    episode_length_s: float = 5.0

    action_space: int = 3       # Δx, Δy, Δyaw
    observation_space: int = 7  # slot_rel_x/y, slot_yaw_err, is_grasping, ee_vel_x/y, yaw_vel
    state_space: int = 0

    # =========================
    # 2. Sim
    # =========================
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # =========================
    # 3. Scene
    # =========================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # =========================
    # 4. Robot (motion-only / grasp 와 동일)
    # =========================
    robot = OMY_OFF_SELF_COLLISION_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # =========================
    # 5. Object — 박스
    # =========================
    box: RigidObjectCfg = RigidObjectCfg(
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
                diffuse_color=(0.7, 0.7, 0.72), metallic=0.5, roughness=0.4,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="max",
                static_friction=3.0,
                dynamic_friction=3.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, -0.10, 0.07)),
    )

    # =========================
    # 6. Ground
    # =========================
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

    # =========================
    # 7. Insert task 파라미터
    # =========================
    # ee 가 고정으로 유지하는 z (= TRANSPORT_Z, 박스 spawn 위 19cm)
    ee_fixed_z: float = 0.26

    # 박스 매핑 z offset (grip center 에서 박스 center 까지)
    # box thickness 0.118 / 2 = 0.059 — 박스 short edge Y 잡힘 → ee z 와 박스 center z 차이 약 0.06
    box_z_offset: float = 0.06    # box_z = ee_fixed_z - 0.06 = 0.20

    # ee yaw clip
    ee_yaw_min: float = -1.5708       # -π/2
    ee_yaw_max: float =  1.5708       # +π/2

    # =========================
    # 8. Reset 분포 (코드로 직접 random)
    # =========================
    # cell base
    cell_center_x: float = 0.30
    cell_center_y: float = -0.30

    # cell xy noise (매 episode)
    cell_xy_noise: float = 0.05       # ±5cm

    # cell yaw range
    cell_yaw_max: float = 1.396       # ±80°

    # ee 시작점 noise (cell 기준 random distance + direction)
    ee_offset_min: float = 0.03       # 3cm
    ee_offset_max: float = 0.05       # 5cm

    # reset 시 IK 이동 step 수 (sim_dt = 1/120, 60 step = 0.5초)
    reset_ik_settle_steps: int = 60

    # =========================
    # 9. Action scale
    # =========================
    action_scale_xy: float = 0.01     # 10mm/step
    action_scale_yaw: float = 0.05    # ~2.86°/step

    # =========================
    # 10. Reward 가중치
    # all-positive (drop 은 termination 으로 자연 페널티)
    # =========================
    reward_xy_align_gain: float = 80.0
    reward_yaw_align_gain: float = 5.0
    reward_smooth_w: float = 0.01
    reward_success_bonus: float = 50.0
    reward_success_lump: float = 5000.0

    # is_grasping 판정
    grasping_dist_threshold: float = 0.07
    box_drop_z_threshold: float = 0.12

    # =========================
    # 11. 종료 조건
    # =========================
    align_xy_threshold: float = 0.010     # 10mm
    align_yaw_threshold: float = 0.087    # ~5°
    success_hold_steps: int = 15          # 0.25초
    fail_xy_threshold: float = 0.30

    # =========================
    # 12. 이름 매핑
    # =========================
    left_finger_body_name: str = "rh_p12_rn_l2"
    right_finger_body_name: str = "rh_p12_rn_r2"
    gripper_joint_names: tuple[str, ...] = (
        "rh_r1_joint", "rh_r2", "rh_l1", "rh_l2",
    )

    # gripper close cmd (박스 잡고 있는 상태 유지용)
    gripper_close_cmd: float = 0.8
    gripper_tip_ratio: float = 2.3

    # =========================
    # 13. PPO 하이퍼파라미터
    # =========================
    n_steps: int = 1024
    batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.97
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    n_epochs: int = 5
