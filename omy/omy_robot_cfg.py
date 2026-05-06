
"""
OMY 로봇 정의
USD 경로
joint / actuator 설정
-> 로봇 자체 정의
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


# OMY USD 경로
OMY_USD_PATH = "/home/jaewoo/IsaacLab/source/omy_f3m_urdf/OMY.usd"


OMY_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=OMY_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # 먼저 자기충돌 끄고 테스트
            enabled_self_collisions=False,

            # position은 그대로 두고
            solver_position_iteration_count=32,

            # velocity iteration은 1 -> 4로 올림
            solver_velocity_iteration_count=2,
        ),
        activate_contact_sensors=False,
    ),

    # 초기 자세
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -1.55,
            "joint3": 2.66,
            "joint4": -1.1,
            "joint5": 1.6,
            "joint6": 0.0,

            "rh_r1_joint": 0.0,
            "rh_r2": 0.0,
            "rh_l1": 0.0,
            "rh_l2": 0.0,
        },
    ),

    actuators={
        # 어깨/기반부 쪽은 더 강하게
        "arm_group_1": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            velocity_limit_sim=6.0,
            effort_limit_sim=100.0,
            stiffness=350.0,
            damping=25.0,
        ),

        # 나머지 팔은 조금 낮게
        "arm_group_2": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-6]"],
            velocity_limit_sim=6.0,
            effort_limit_sim=80.0,
            stiffness=300.0,
            damping=20.0,
        ),

        # 그리퍼 4개 관절 통일
        # 원래 300이었는데 tip이 물체에 밀리는 문제 → 500으로 올림
        # base/tip 스티프니스 차이가 크면 따로 노는 것처럼 보임 → 동일하게 유지
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"],
            velocity_limit_sim=2.2,
            effort_limit_sim=150.0,
            stiffness=500.0,
            damping=30.0,
        ),
    },
)


OMY_OFF_SELF_COLLISION_CFG = OMY_CFG.replace(
    spawn=OMY_CFG.spawn.replace(
        articulation_props=OMY_CFG.spawn.articulation_props.replace(
            enabled_self_collisions=False,
        )
    )
)