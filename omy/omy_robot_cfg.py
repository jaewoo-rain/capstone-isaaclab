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


# 네가 source 폴더 안에 둔 OMY.usd 경로 기준
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
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": -1.55,
            "joint3": 2.66,
            "joint4": -1.1,
            "joint5": 1.6,
            "joint6": 0.0,
            "rh_r1_joint": 0.0,
        },
    ),
    actuators={
        "arm_group_1": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-2]"],
            velocity_limit_sim=6.0,
            effort_limit_sim=61.4,
            stiffness=120.0,
            damping=4.0,
        ),
        "arm_group_2": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-6]"],
            velocity_limit_sim=6.0,
            effort_limit_sim=31.7,
            stiffness=120.0,
            damping=4.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["rh_r1_joint", "rh_r2", "rh_l1", "rh_l2"],
            velocity_limit_sim=2.2,
            effort_limit_sim=30.0,
            stiffness=100.0,
            damping=4.0,
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