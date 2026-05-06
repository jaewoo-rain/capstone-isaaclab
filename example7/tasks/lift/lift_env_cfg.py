from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from source.omy.omy_robot_cfg import OMY_CFG, OMY_OFF_SELF_COLLISION_CFG

@configclass
class LiftEnvCfg(DirectRLEnvCfg):
    """OMY Lift Env 설정"""

    # -------------------------
    # 1. 기본 env 설정
    # -------------------------

    # 물리 스텝을 몇 번 돌고 나서 RL에 observation을 줄지
    # 2 = 물리를 2번 계산한 뒤 한 번 action을 받음 → 실질적으로 60Hz로 제어하는 것
    decimation: int = 2

    # 한 에피소드가 최대 몇 초인지 → 12초 안에 물체를 못 들면 강제 종료
    episode_length_s: float = 12.0
    # episode_length_s: float = 20.0

    # 네트워크가 출력하는 action 벡터의 크기
    # joint1~6 (arm 6개) + gripper 1개 = 7
    action_space: int = 7

    # 네트워크가 입력받는 observation 벡터의 크기 = 34
    # (joint_pos 10 + joint_vel 10 + obj_pos_rel 3 + obj_to_grip 3
    #  + left_to_obj 3 + right_to_obj 3 + gripper_close 1 + to_lift_target 1)
    # joint_pos/vel이 10인 이유: all_obs_joint_ids = arm 6개 + gripper 4개
    observation_space: int = 34

    # 중앙집중식 critic에 쓰이는 global state 크기
    # 0 = 사용 안 함 (actor와 critic이 같은 obs 사용)
    state_space: int = 0

    # -------------------------
    # 2. 시뮬레이션
    # -------------------------
    sim: SimulationCfg = SimulationCfg(
        # 1스텝당 시간 = 1/120초 → 물리 엔진이 1초에 120번 계산
        # decimation=2 이므로 RL 제어 주기는 1/60초(60Hz)
        dt=1.0 / 120.0,

        # 렌더링을 몇 스텝마다 할지 → decimation과 같게 맞춤
        render_interval=decimation,

        # 기본 물리 재질 (로봇/물체에 별도 설정 없으면 이게 적용됨)
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # 두 물체가 접촉할 때 마찰계수를 어떻게 합칠지
            # "multiply" = 두 마찰계수를 곱함 (1.0 × 1.0 = 1.0)
            friction_combine_mode="multiply",

            # 반발계수(튕김)를 어떻게 합칠지
            restitution_combine_mode="multiply",

            # 정지마찰: 물체가 멈춰있을 때 버티는 힘. 1.0 = 미끄럼 없음
            static_friction=1.0,

            # 동적마찰: 물체가 움직이는 중 저항. 1.0 = 강한 저항
            dynamic_friction=1.0,

            # 반발계수: 0.0 = 전혀 안 튕김 (완전 비탄성)
            restitution=0.0,
        ),
    )

    # -------------------------
    # 3. Scene
    # -------------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        # GPU에서 동시에 돌릴 병렬 환경 수 → 256개 로봇이 동시에 학습
        num_envs=256,

        # 각 환경 사이의 거리(m) → 환경끼리 겹치지 않도록 2.5m 간격
        env_spacing=2.5,

        # True = 모든 환경이 동일한 물리 설정을 공유 → 메모리 절약, 속도 향상
        replicate_physics=True,
    )

    # -------------------------
    # 4. Robot
    # -------------------------

    # OMY 로봇 설정에서 prim_path만 바꿔서 사용
    # "/World/envs/env_.*/Robot" 에서 ".*" 는 정규표현식으로
    # env_0, env_1, env_2 ... 모든 환경의 로봇을 한번에 가리킴
    robot = OMY_OFF_SELF_COLLISION_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # -------------------------
    # 5. Object
    # -------------------------
    object: RigidObjectCfg = RigidObjectCfg(
        # 모든 환경의 물체 경로 (env_.*가 각 환경 번호로 치환됨)
        prim_path="/World/envs/env_.*/Object",

        spawn=sim_utils.CuboidCfg(
            # 박스 크기 (x, y, z) 단위: 미터
            # 13.9cm × 4.4cm × 11.8cm → 얇고 긴 직육면체 (예: 작은 상자)
            size=(0.139, 0.044, 0.118),

            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                # True = 이 물체가 물리 시뮬레이션에 참여함
                rigid_body_enabled=True,

                # False = 중력 적용함 (True면 공중에 떠 있음)
                disable_gravity=False,

                # 물체가 다른 물체에 파고들었을 때 빠져나오는 최대 속도(m/s)
                # 너무 크면 튀어나가는 버그 발생, 5.0은 안전한 값
                max_depenetration_velocity=5.0,
            ),

            # 물체 질량 = 0.3kg (실제 작은 상자 정도의 무게)
            mass_props=sim_utils.MassPropertiesCfg(mass=0.3),

            collision_props=sim_utils.CollisionPropertiesCfg(
                # True = 충돌 계산 활성화 (False면 로봇이 통과해 버림)
                collision_enabled=True
            ),

            visual_material=sim_utils.PreviewSurfaceCfg(
                # RGB 색상값 (각각 0~1 범위)
                # (0.667, 0.686, 0.706) ≈ 회청색 금속 느낌
                diffuse_color=(0.667, 0.686, 0.706),

                # 0~1, 높을수록 금속처럼 반짝임. 0.8 = 거의 금속
                metallic=0.8,

                # 0~1, 낮을수록 표면이 매끄럽고 반짝임. 0.4 = 약간 반짝
                roughness=0.4,
            ),
        ),

        init_state=RigidObjectCfg.InitialStateCfg(
            # 물체의 초기 위치 (x, y, z) 단위: 미터, 환경 원점 기준
            # x=0.45: 로봇 앞으로 45cm
            # y=-0.10: 로봇 오른쪽으로 10cm
            # z=0.06: 바닥에서 6cm 위 (박스 높이의 절반 = 5.9cm이므로 바닥에 닿아있음)
            pos=(0.45, -0.10, 0.06),
        ),
    )

    # -------------------------
    # 6. Ground
    # -------------------------
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        # 바닥 지형의 USD 경로
        prim_path="/World/ground",

        # "plane" = 단순 평면 바닥 (계단, 경사 등 없음)
        terrain_type="plane",

        # -1 = 모든 환경과 충돌 처리. 0 이상이면 특정 그룹만 충돌
        collision_group=-1,

        # 바닥의 물리 재질 (로봇/물체와 같은 값 → 마찰 계수가 1.0×1.0=1.0)
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

    # action에 곱해지는 배율. 1.0 = 네트워크 출력값을 그대로 사용
    action_scale: float = 1.0

    # joint 속도 observation에 곱해지는 배율. 1.0 = 그대로 사용
    dof_velocity_scale: float = 1.0

    # 물체 높이가 이 값(0.2m = 20cm)을 넘으면 lift 성공으로 판정
    lift_height_threshold: float = 0.2

    # 성공 시 추가 보상 (현재 reward 함수에서 직접 사용되진 않음)
    success_bonus: float = 10.0

    # 에피소드 시작 시 물체 위치에 추가하는 랜덤 노이즈의 최대값(m)
    # ±0.02m = ±2cm 범위에서 랜덤하게 물체 위치를 변경 → 일반화 성능 향상
    object_pos_noise: float = 0.02

    # 그리퍼가 목표로 삼는 위치가 물체 중심보다 얼마나 위인지(m)
    # 0.05 = 물체 중심에서 5cm 위를 잡으려 함 → 박스 윗면 근처를 집도록 유도
    grasp_target_z_offset: float = 0.04

    # -------------------------
    # 8. 이름 매핑
    # 모델 구조가 바뀌면 여기만 수정
    # -------------------------

    # 왼쪽 손가락 끝 body의 USD 이름
    # lift_env.py에서 find_bodies()로 이 이름을 검색해서 body ID를 얻음
    left_finger_body_name: str = "rh_p12_rn_l2"

    # 오른쪽 손가락 끝 body의 USD 이름
    right_finger_body_name: str = "rh_p12_rn_r2"

    # gripper를 구성하는 joint 이름 목록
    # mimic 구조 = 하나의 명령으로 4개 관절이 연동해서 움직임
    # rh_r1_joint, rh_r2 = 오른쪽 손가락 관절 2개
    # rh_l1, rh_l2        = 왼쪽 손가락 관절 2개
    gripper_joint_names: tuple[str, ...] = (
        "rh_r1_joint",
        "rh_r2",
        "rh_l1",
        "rh_l2",
    )

    # -------------------------
    # 9. PPO 학습 파라미터
    # -------------------------

    # 한 번 업데이트하기 전에 각 환경에서 몇 스텝을 모을지
    # 총 데이터량 = n_steps(512) × num_envs(256) = 131,072 스텝
    n_steps: int = 512

    # 한 번에 gradient를 계산할 데이터 수
    # 131,072 스텝을 4096씩 나눠서 업데이트 → 32번 미니배치
    batch_size: int = 4096

    # 학습률. 3e-4 = 0.0003. 너무 크면 발산, 너무 작으면 느림
    learning_rate: float = 3e-4

    # 미래 보상을 얼마나 중요하게 볼지 (할인율)
    # 0.99 = 100스텝 뒤 보상도 0.99^100 ≈ 37% 가치로 인정
    gamma: float = 0.99

    # GAE에서 bias-variance 트레이드오프 조절
    # 1.0에 가까울수록 variance 높고 bias 낮음
    gae_lambda: float = 0.95

    # PPO clip 범위. 정책이 한 번에 이 비율 이상 바뀌지 못하게 제한
    # 0.2 = 이전 정책 대비 ±20% 이상 변하면 clipping
    clip_range: float = 0.2

    # 엔트로피 보너스 계수. 높을수록 더 다양한 행동을 탐색
    # 0.005 = 약한 탐색 장려
    ent_coef: float = 0.005

    # value function loss의 가중치
    # 0.5 = policy loss와 value loss를 1:0.5 비율로 합산
    vf_coef: float = 0.5

    # 같은 데이터로 몇 번 반복 학습할지
    # 5 = 수집한 131,072 스텝을 5번 돌려서 학습
    n_epochs: int = 5
