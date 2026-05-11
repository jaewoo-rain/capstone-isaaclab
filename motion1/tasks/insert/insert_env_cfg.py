"""motion1 — Insert RL env cfg.

Task: 박스 잡힌 채 셀 위에서 xy/yaw 미세 정렬 (RL).
- Action (3): Δx, Δy, Δyaw (cartesian, relative)
- State (7): slot_rel_x, slot_rel_y, slot_yaw_err, is_grasping, ee_vel_x, ee_vel_y, yaw_vel
- ee_z, gripper: 고정 (학습 안 함)
- 시작 상태: handoff dataset 에서 random sample + 추가 noise

motion-only chain 의 단계 4 (insert 미세조정) 만 RL 로 학습.
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
class InsertEnvCfg(DirectRLEnvCfg):
    """OMY Insert 미세 정렬 RL Env cfg."""

    # =========================
    # 1. 기본 RL env 설정
    # =========================
    decimation: int = 1
    episode_length_s: float = 5.0

    action_space: int = 3       # Δx, Δy, Δyaw
    observation_space: int = 7  # slot_rel_x/y, slot_yaw_err, is_grasping, ee_vel_x/y, yaw_vel
    state_space: int = 0

    # =========================
    # 2. Sim
    # =========================
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 60.0,
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
    # 5. Object — 박스 (motion-only 와 동일)
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.40, -0.10, 0.07)),
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

    # handoff dataset 경로
    handoff_dataset_path: str = "checkpoints/insert_handoff_states.npz"

    # ee yaw clip (학습 시 사용)
    ee_yaw_min: float = -1.5708       # -π/2
    ee_yaw_max: float =  1.5708       # +π/2
    # noise 는 handoff dataset 자체에 포함됨 (collect_insert_handoff.py 의 transport stage 에서)

    # =========================
    # 8. Action scale
    # =========================
    action_scale_xy: float = 0.005    # 5mm/step (overshoot 감소, fine 정렬용)
    action_scale_yaw: float = 0.05    # ~2.86°/step

    # =========================
    # 9. Reward 가중치
    # all-positive (drop 은 termination 으로 자연 페널티)
    # =========================
    reward_xy_align_gain: float = 80.0        # exp(-gain * xy_dist²) — 멀리 가도 작은 신호 (exploration)
    reward_xy_align_gain_close: float = 200.0  # exp(-gain * xy_dist²) — 가까이 sharp (정밀 정렬)
    reward_yaw_align_gain: float = 5.0    # exp(-gain * yaw_err²)
    reward_smooth_w: float = 0.01         # -w * (vel² 합)
    reward_success_bonus: float = 50.0    # aligned 매 step (정렬 유지 + holding)
    reward_success_lump: float = 5000.0   # success terminate 시 한 번에

    # is_grasping 판정 (finger ↔ box 거리 + box z 임계)
    grasping_dist_threshold: float = 0.07   # finger center ↔ box 거리 7cm 이내
    box_drop_z_threshold: float = 0.12      # box z < 12cm 면 떨어뜨림 (박스가 long edge 로 누워도 검출 가능)

    # =========================
    # 10. 종료 조건
    # =========================
    align_xy_threshold: float = 0.010     # 10mm (이전 5mm 빡셈)
    align_yaw_threshold: float = 0.087    # ~5° (이전 2.86° 빡셈)
    success_hold_steps: int = 15          # 0.25초 (이전 30=0.5초)
    fail_xy_threshold: float = 0.30       # ee 가 셀에서 30cm 이상 멀어지면 실패

    # =========================
    # 11. 이름 매핑 (motion-only 와 동일)
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
    # 12. PPO 하이퍼파라미터 (grasp 와 동일)
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
