"""OMY Place+Insert Env 설정 (example7)

example5(Lift) 이후 계층 구조로 동작하는 Place+Insert 태스크.
물체가 그리퍼에 잡힌 채로 20cm 높이에서 시작 → 3×3 그리드 셀 위로 들어올림 →
yaw 정렬 → 셀 안으로 8cm 삽입 후 그리퍼 release.

example6 대비 변경점: 보상함수만 새로 작성 (3-phase: lift→align→insert).
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
class PlaceEnvCfg(DirectRLEnvCfg):
    """OMY Place Env — 3×3 그리드 셀에 물체 내려놓기"""

    # -------------------------
    # 1. 기본 env 설정
    # -------------------------
    decimation: int = 2
    episode_length_s: float = 10.0

    # action: arm 6 + gripper 1 = 7 (example5와 동일)
    action_space: int = 7

    # observation: 31 (25 obs + 3 achieved_goal + 3 desired_goal, flat tensor)
    # GoalEnvVecWrapper가 split해서 SB3 HER의 Dict 형식으로 변환
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
    # SAC는 off-policy라 적은 환경 수로도 동작, PPO보다 env 수 줄여도 OK
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=3.0,
        replicate_physics=True,
    )

    # -------------------------
    # 4. Robot (example5와 동일)
    # -------------------------
    robot = OMY_OFF_SELF_COLLISION_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # -------------------------
    # 5. Object (example5와 동일 박스)
    # -------------------------
    # init_state.pos는 reset에서 덮어씌움 (그리퍼에 잡힌 위치로)
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
            # TODO: validate in sim — 그리퍼가 20cm 높이에서 물체를 잡고 있는 상태
            pos=(0.25, -0.10, 0.20),
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
    # 7. 그리드 파라미터
    # 3×3 셀, 중심 (0.25, -0.45), 수직 4개 + 수평 4개 벽 (corner overlap 허용)
    # -------------------------
    grid_num_x: int = 1
    grid_num_y: int = 1

    # 그리드 중심 (환경 원점 기준) — TODO: validate in sim
    grid_center_x: float = 0.25
    grid_center_y: float = -0.45

    # 셀 내부 치수 — 물체(13.9×4.4)에 약 2cm 여유, 격벽 환경 복원
    cell_inner_x: float = 0.16   # 13.9 + 2.1cm 여유
    cell_inner_y: float = 0.065  # 4.4 + 2.1cm 여유

    # 벽
    wall_thickness: float = 0.008
    wall_height: float = 0.12    # 물체 높이(11.8cm)와 거의 동일

    # 셀 피치 (중심 간 거리)
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

    # 셀 내부 허용 오차 (XY 거리) — 이보다 가까우면 타겟 셀 안으로 간주
    # 0.015 → 0.025: 박스(13.9cm) ↔ 셀(17cm) 여유 3.1cm 고려, 학습 진입 허용
    cell_tolerance: float = 0.025

    # 물체 속도가 이 값 이하이면 "정지" 판정
    stable_vel_threshold: float = 0.02

    # gripper가 이 값 미만이면 "열림" 판정
    gripper_open_threshold: float = 0.2

    # 성공 판정 최소 유지 스텝 (60Hz × 1s = 60; v7: 120→60 으로 완화)
    success_hold_steps: int = 60
    # 바닥 접촉 임계 — obj_bottom_z 이하면 "바닥 닿음"으로 판정
    on_floor_touch_threshold: float = 0.005

    # 물체가 이 z 아래로 떨어지면 실패
    object_fall_z: float = -0.05

    # 종료 조건 (tilt + on_floor + abandoned)
    tilt_upright_threshold: float = 0.85    # upright_score 이 값 이하면 기울어짐 (≈30도) — 페널티/리셋용
    bonus_upright_threshold: float = 0.985  # 보너스용 임계 (≈10도) — T8 bonus, 더 엄격
    on_floor_z_threshold: float = 0.10      # 물체 중심 z가 이 이하면 바닥 판정
    abandoned_dist_threshold: float = 0.20  # 13→20cm: 격벽 충돌 시 튕김 허용 (그리퍼 떨어뜨림 reset 폭증 방지)

    # -------------------------
    # 3-Phase 보상 파라미터 (example7 신규)
    # phase1: lift over cell, phase2: yaw align, phase3: insert+release
    # -------------------------
    # phase1: 셀 top + 물체 높이 + 여유. 에피소드 절대 z 기준 (env-relative).
    lift_clearance: float = 0.01            # 0.05 → 0.01 (학습 curriculum, 진입 후 0.05로 복원)

    # phase1 XY 정렬 허용 (느슨) — 셀 위에 대략 올라간 상태
    phase1_xy_tolerance: float = 0.07  # 0.05 → 0.07 (handoff 노이즈 흡수, 정체 해소)

    # phase2 yaw 정렬 허용 — 끝점 거리(2cm 이내)
    yaw_tolerance: float = 0.04  # 0.02 → 0.04 (yaw 학습 진입장벽 완화)

    # phase3 삽입 목표 깊이 — stage1=0.005 (작동 설정), stage2=0.04, final=0.06
    insertion_target_depth: float = 0.005

    # 카메라 달려있는 쪽 끝점 — 물체 local x 부호 (+1 / -1, 필요시 flip)
    camera_side_sign: float = 1.0
    # 적재 위치(셀)의 대응 끝점 — 셀 local x 부호 (+1 / -1)
    target_endpoint_sign: float = 1.0

    # 잠겨있던 stage 관련 변수 (legacy, 사용 안 함 — 호환성 위해 남겨둠)
    curriculum_stage: int = 1
    stage1_xy_tolerance: float = 0.10
    stage1_hover_z_min: float = 0.15
    stage1_hover_z_max: float = 0.35

    # -------------------------
    # 9. Handoff randomization (sim2real 핵심)
    # example5가 실제로 건네주는 물체/팔 자세에 노이즈를 미리 주어
    # example7이 다양한 시작 상태에서도 동작하도록 학습
    # -------------------------
    handoff_obj_pos_noise_xy: float = 0.02   # 그리퍼 내 물체 XY 위치 노이즈 (±2cm)
    handoff_obj_pos_noise_z: float = 0.01    # Z 노이즈 (±1cm)
    handoff_joint_noise: float = 0.05        # 팔 관절 노이즈 (±0.05 rad)

    # 에피소드마다 타겟 셀 번호 랜덤 (0 ~ grid_num_x*grid_num_y-1)
    # False로 두면 순차적(row-major) 고정 순서
    randomize_target_cell: bool = True

    # -------------------------
    # 10. 이름 매핑 (example5와 동일)
    # -------------------------
    left_finger_body_name: str = "rh_p12_rn_l2"
    right_finger_body_name: str = "rh_p12_rn_r2"

    gripper_joint_names: tuple[str, ...] = (
        "rh_r1_joint",
        "rh_r2",
        "rh_l1",
        "rh_l2",
    )

    # Handoff 상태 데이터셋 경로 (collect_handoff.py로 미리 생성)
    # 존재하면 reset 시 이 파일에서 랜덤 샘플링
    # 존재하지 않으면 fallback_holding_joint_pos 사용
    handoff_dataset_path: str = "checkpoints/handoff_states.npz"

    # Fallback용 추정 자세 (handoff 데이터 없을 때만 사용)
    fallback_holding_joint_pos: dict[str, float] = {
        "joint1": 0.0,
        "joint2": 0.06,
        "joint3": 1.98,
        "joint4": -1.02,
        "joint5": 1.26,
        "joint6": -0.13,
        "rh_r1_joint": 0.40,
        "rh_r2": 0.86,
        "rh_l1": 0.52,
        "rh_l2": 1.32,
    }

    # Fallback 사용 시 물체 초기 위치 (환경 원점 기준)
    fallback_obj_pos_rel: tuple[float, float, float] = (0.457, -0.091, 0.155)

    # -------------------------
    # 11. SAC + HER 하이퍼파라미터
    # -------------------------
    # SAC
    learning_rate: float = 3e-4  # 1e-4 → 3e-4 (정책 변경 강제, hover trajectory 탈출)
    buffer_size: int = 1_000_000
    batch_size: int = 512
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    learning_starts: int = 10_000
    ent_coef: str = "auto"

    # HER
    n_sampled_goal: int = 4
    goal_selection_strategy: str = "future"
