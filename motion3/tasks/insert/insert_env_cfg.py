"""motion3 — Insert RL env cfg.

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

from source.motion3.robot_cfg import OMY_TABLE_MOUNTED_CFG
from source.motion3 import layout


@configclass
class InsertEnvCfg(DirectRLEnvCfg):
    """OMY Insert 미세 정렬 RL Env cfg."""

    # =========================
    # 1. 기본 RL env 설정
    # =========================
    decimation: int = 1
    episode_length_s: float = 15.0   # 5→15 테스트: 높은 자세 yaw 제어가 느린(시간부족)지 vs 못하는(제어)지 구분용

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
    robot = OMY_TABLE_MOUNTED_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # =========================
    # 5. Object — 박스 (motion-only 와 동일)
    # =========================
    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CuboidCfg(
            size=layout.BOX_SIZE,   # 서있는 박스 (layout 단일출처)
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
        # 박스는 책상 위 앞쪽. 실제 reset 값은 handoff dataset이 덮어씀(의미만 일치).
        init_state=RigidObjectCfg.InitialStateCfg(pos=layout.BOX_SPAWN),
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
    # ee 가 고정으로 유지하는 z — 뒤쪽 저고도 셀 위 hover (ground 기준, wall top 0.12 위).
    # z 하강(0.20→0.065)은 chain runner의 motion planning이 전담.
    ee_fixed_z: float = layout.INSERT_HOVER_Z   # 0.20

    # handoff dataset 경로 (책상/뒤쪽 분포 v15)
    handoff_dataset_path: str = "checkpoints/insert_handoff_states_v16.npz"  # v16: 하강X, high(turn+lift) 위치 handoff

    # ★ wrap-safe yaw (v25 reference-anchored unwrap) — 절대 ±π clip 폐기.
    #   배경: turn 후 그리퍼 yaw ~±180°(=wrap 경계). 절대 setpoint 를 ±π 로 hard clamp 하면
    #   working point 가 경계에 물려 정책이 yaw 조정을 포기(v24: yaw_err 8.5° plateau, success 0.05).
    #   해결: reset 시 setpoint 기준값 _yaw_ref(=handoff 그리퍼 yaw, ~±180°) 저장 → 누적 setpoint 를
    #   _yaw_ref ± yaw_margin 로만 제한. quat_from_angle_axis 는 |yaw|>π 도 연속 → wrap 경계 자체가 사라짐.
    #   margin 은 자연 yaw 오차(fold 후 최대 ~90°)를 넉넉히 덮어야 정상 보정을 안 막음.
    yaw_margin: float = 1.75          # ~100° (fold 한계 90° + 여유). _yaw_ref 중심 상대 제한.
    # (deprecated) ee_yaw_min/max — 절대 clip 은 더 이상 안 씀. 호환용으로만 남김.
    ee_yaw_min: float = -3.14159      # -π
    ee_yaw_max: float =  3.14159      # +π
    # xy noise 는 handoff dataset 자체에 포함됨 (collect transport stage).
    # ★ yaw noise: reset 에서 _ee_target_yaw 를 handoff(=cell정렬)에서 ±이만큼 흔들어
    #   실제 yaw 오차를 만든다(누적 IK 가 박스를 그리로 돌림) → 정책이 yaw 보정 학습(sim2real 강건성).
    reset_yaw_noise: float = 0.0      # 0: 방식 B 의 자연 yaw 오차(mean 45°, max 83°)만 사용 — 주입 노이즈 불필요(이미 넓음)
    disable_xy_noise: bool = False
    # ★ yaw-only 모드: xy 는 IK 가 cell 에 고정 holding(정책 제어X), 정책/보상/success 는 yaw 만.
    #   누적 yaw 제어가 xy 를 끌어내는 coupling 때문에 yaw+xy 동시 학습 불가 → xy 는 motion/IK 가 잡고 RL 은 yaw 만.
    yaw_only: bool = True

    # =========================
    # 8. Action scale
    # =========================
    action_scale_xy: float = 0.005    # 5mm/step (overshoot 감소, fine 정렬용)
    action_scale_yaw: float = 0.05    # ~2.86°/step

    # =========================
    # 9. Reward 가중치
    # all-positive (drop 은 termination 으로 자연 페널티)
    # =========================
    reward_xy_align_gain: float = 40.0        # exp(-gain * xy_dist²) — 멀리 가도 작은 신호 (exploration). 80→40: 정책이 갇힌 12cm에서 gain↑는 far gradient를 죽임 → 낮춰서 펼침
    reward_xy_align_gain_close: float = 200.0  # exp(-gain * xy_dist²) — 가까이 sharp (정밀 정렬)
    reward_yaw_align_gain: float = 50.0   # exp(-gain * yaw_err²) — sharp, 0° 근처 fine-lock 전용
    # ★ 비포화 linear yaw cone: w*(1 - |yaw_err|/(π/2)). 80°(1.4rad)에서도 일정한 복구 gradient.
    #   Gaussian 단독은 80°에서 gradient≈0(saturation)이라 정책이 yaw 망가뜨려도 못 돌아옴 → cone 추가.
    reward_yaw_lin_w: float = 0.4   # 2.0→0.4: yaw 제어가 누적IK로 robust해진 뒤 cone(w2.0)이 yaw 평형보상을 부풀려 xy를 압살 → 1/5로 (mid-range yaw 신호는 남김)
    reward_smooth_w: float = 0.01         # -w * (vel² 합)
    reward_success_bonus: float = 50.0    # aligned 매 step (정렬 유지 + holding)
    reward_success_lump: float = 5000.0   # success terminate 시 한 번에

    # is_grasping 판정 — 서있는 박스를 위쪽(GRASP_Z=0.455)에서 잡아 그립보다 ~8.5cm 아래 매달림.
    #   handoff dataset의 box 중심 z ≈ 0.11 (max 0.128), finger↔box 거리 ≈ 0.09.
    #   → 매달린 박스를 "잡힘"으로 인정하도록 임계 완화 (떨어진 박스 z≈0.07과는 구분).
    grasping_dist_threshold: float = 0.12   # finger center ↔ box 거리 (매달린 ~9cm 수용)
    box_drop_z_threshold: float = 0.085     # box 중심 < 8.5cm 면 drop (잡힘 0.11 vs 떨어짐 0.07 구분)

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
