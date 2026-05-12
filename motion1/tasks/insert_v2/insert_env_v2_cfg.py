"""motion1 — Insert RL v2 cfg (SAC + HER).

Task: 박스 잡힌 채 셀 위에서 xy/yaw 미세 정렬.

핵심 변경 (v1 → v2):
- 알고리즘: PPO → SAC + HER (sparse reward relabeling)
- Reset: handoff dataset → 코드 reset (cell + ee offset random + IK)
- Obs (flat): core(5) + achieved_goal(4) + desired_goal(4) = 13
- Action (3): Δx, Δy, Δyaw (cartesian, relative)
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
    """OMY Insert 미세 정렬 RL Env cfg (v2: SAC + HER, coded reset)."""

    # =========================
    # 1. 기본 RL env
    # =========================
    decimation: int = 1
    episode_length_s: float = 5.0

    # Obs flat: core(5) + achieved_goal(4) + desired_goal(4) = 13
    # train.py 의 GoalEnvVecWrapper 가 자동 split
    action_space: int = 3
    observation_space: int = 13
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
    # 3. Scene — SAC 는 off-policy 라 env 수 적어도 OK
    # =========================
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # =========================
    # 4. Robot
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, -0.30, 0.20)),
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
    # 7. Task 좌표
    # =========================
    ee_fixed_z: float = 0.26                  # TRANSPORT_Z (학습/chain 일치)
    box_z_offset: float = 0.06                # ee_z - box_center_z

    ee_yaw_min: float = -1.5708               # -π/2
    ee_yaw_max: float =  1.5708               # +π/2

    # =========================
    # 8. Reset 분포 (코드 reset)
    # =========================
    cell_center_x: float = 0.30
    cell_center_y: float = -0.30
    cell_xy_noise: float = 0.05               # ±5cm
    cell_yaw_max: float = 1.396               # ±80°

    ee_offset_min: float = 0.03               # 3cm
    ee_offset_max: float = 0.05               # 5cm

    reset_ik_settle_steps: int = 30           # reset 후 IK settle (0.5초)

    # =========================
    # 9. Action scale
    # =========================
    action_scale_xy: float = 0.005            # 5mm/step
    action_scale_yaw: float = 0.05            # ~2.86°/step

    # =========================
    # 10. Reward (dense, 최소 3 terms)
    # =========================
    reward_xy_align_gain: float = 80.0        # exp(-gain × xy_dist²)
    reward_yaw_align_gain: float = 15.0       # exp(-gain × yaw_err²)
    reward_success_bonus: float = 50.0        # aligned 매 step bonus

    # =========================
    # 11. is_grasping 판정
    # =========================
    grasping_dist_threshold: float = 0.07
    box_drop_z_threshold: float = 0.12

    # =========================
    # 12. Sparse reward 판정 (HER compute_reward)
    # =========================
    align_xy_threshold: float = 0.010         # 10mm
    align_yaw_threshold: float = 0.087        # ~5° (≈ cos 0.9962)
    success_hold_steps: int = 15
    fail_xy_threshold: float = 0.20

    # =========================
    # 13. 이름 매핑
    # =========================
    left_finger_body_name: str = "rh_p12_rn_l2"
    right_finger_body_name: str = "rh_p12_rn_r2"
    gripper_joint_names: tuple[str, ...] = (
        "rh_r1_joint", "rh_r2", "rh_l1", "rh_l2",
    )

    gripper_close_cmd: float = 0.8
    gripper_tip_ratio: float = 2.3

    # =========================
    # 14. SAC 하이퍼파라미터
    # =========================
    learning_rate: float = 3e-4
    buffer_size: int = 1_000_000
    batch_size: int = 512
    tau: float = 0.005
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    learning_starts: int = 5_000
    ent_coef: str = "auto"

    # =========================
    # 15. HER 하이퍼파라미터
    # =========================
    n_sampled_goal: int = 4
    goal_selection_strategy: str = "future"
