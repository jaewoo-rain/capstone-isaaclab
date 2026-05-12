"""motion1 — Grasp RL env cfg.

Task: 박스 위 PRE_GRASP_Z 에서 xy/yaw 미세 정렬 (RL).
- Action (3): Δx, Δy, Δyaw (cartesian, relative)
- State (6): obj_rel_x, obj_rel_y, obj_yaw_err, ee_vel_x, ee_vel_y, yaw_vel
- ee_z, gripper: 고정 (학습 안 함)
- 박스 spawn: xy ±2cm, yaw ±80°
- ee 시작: 박스 + 3~5cm random offset (xy 방향 random), yaw=0 고정

motion-only pipeline 의 단계 2 (grasp 미세조정) 만 RL 로 학습.
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
class GraspEnvCfg(DirectRLEnvCfg):
    """OMY Grasp 미세 정렬 RL Env cfg."""

    # =========================
    # 1. 기본 RL env 설정
    # =========================
    decimation: int = 2          # 물리 120Hz, 제어 60Hz
    episode_length_s: float = 5.0  # 정렬은 짧음 -> 이후 3초로 줄이기

    # action: Δx, Δy, Δyaw — 3차원 [-1, 1]
    action_space: int = 3

    # state: obj_rel_x, obj_rel_y, obj_yaw_err, ee_vel_x, ee_vel_y, yaw_vel — 6차원
    observation_space: int = 6
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
    # 4. Robot (motion-only 와 동일)
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.45, 0.10, 0.07)),
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
    # 7. Grasp 정렬 task 파라미터
    # =========================
    # ee 가 고정으로 유지하는 z (motion-only 의 PRE_GRASP_Z 와 동일)
    ee_fixed_z: float = 0.17

    # 박스 spawn (env-rel) — chain runner 와 동기화 (y +0.10)
    box_spawn_xy: tuple[float, float] = (0.45, 0.10)
    box_spawn_z: float = 0.07
    box_spawn_xy_noise: float = 0.1      # 박스 생성 노이즈 ±10cm
    box_spawn_yaw_max: float = 1.396     # 박스 생성 노이즈 ±80° (= 1.396 rad)

    # ee 시작 (박스 + offset, motion-only 의 pre_grasp_offset 와 동일 발상)
    ee_offset_min_m: float = 0.03         # 그립 생성 노이즈 박스로부터 최소 3cm
    ee_offset_max_m: float = 0.05         # 그립 생성 노이즈 박스로부터 최대 5cm

    # ee yaw clip — clip 안 하면 wrist 끝없이 회전
    ee_yaw_min: float = -1.5708           # -π/2
    ee_yaw_max: float = 1.5708            # +π/2

    # =========================
    # 8. Action scale (RL action [-1,1] → 실제 delta)
    # 나중에 튜닝 가능 (5mm/step + 3°/step)
    # =========================
    action_scale_xy: float = 0.01        # 10 mm / step
    action_scale_yaw: float = 0.05        # ~2.86° / step

    # =========================
    # 9. Reward 가중치
    # 가까이 갔을 때 신호 강하도록 exp gain 부드럽게.
    # 학습 안 되면 가중치 / gain 조정.
    # =========================
    reward_xy_align_gain: float = 80.0     # exp(-gain * xy_dist²)
    reward_yaw_align_gain: float = 5.0       # exp(-gain * yaw_err²)
    reward_smooth_w: float = 0.01            # -w * (vel² 합) — 부드러움 유도
    reward_success_bonus: float = 50.0       # aligned 매 step 시 보너스 (정렬 유지 인센티브)
    # success terminate (success_hold_steps 도달) 시 한 번에 받는 압도적 보너스.
    # 목적: timeout 까지 헤매다 reward 누적하는 것보다 빨리 success terminate 가 이득이 되도록.
    reward_success_lump: float = 5000.0

    # =========================
    # 10. 종료 조건
    # =========================
    align_xy_threshold: float = 0.005     # 5 mm 이내 → aligned
    align_yaw_threshold: float = 0.05     # ~2.86° 이내 (3° 근사)
    # aligned 상태로 N step 유지 시에만 success → terminate (잠깐 스쳐가는 케이스 방지)
    success_hold_steps: int = 30          # 30 step = 0.5초 (60Hz)
    # NOTE: 학습 초기엔 ee 시작 위치가 박스에서 멀어 (fallback pose 기준 ~20cm),
    # fail_xy_threshold 가 너무 작으면 episode 가 1~3 step 에 끝나서 학습 안 됨.
    # 우선 0.30 (30cm) 으로 크게 → 학습 진행 시 점차 줄이는 curriculum 고려.
    fail_xy_threshold: float = 0.30       # ee 가 박스에서 30cm 이상 멀어지면 실패

    # =========================
    # 11. 이름 매핑 (motion-only 와 동일)
    # =========================
    left_finger_body_name: str = "rh_p12_rn_l2"
    right_finger_body_name: str = "rh_p12_rn_r2"
    gripper_joint_names: tuple[str, ...] = (
        "rh_r1_joint", "rh_r2", "rh_l1", "rh_l2",
    )

    # =========================
    # 12. PPO 하이퍼파라미터 (간단 task 에 맞춤)
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
