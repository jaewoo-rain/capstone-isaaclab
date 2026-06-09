"""motion3 — 책상 위에 마운트된 OMY 로봇 cfg.

공유 `source/omy/omy_robot_cfg.py`(frozen)를 건드리지 않고, 여기서 base를
책상 높이(layout.TABLE_HEIGHT)로 올린 변형만 정의한다.

⚠️ isaaclab을 import하므로 반드시 Isaac Sim app 런칭 후에 import할 것
   (train_insert.py / chain runner / collect 의 기존 import 패턴과 동일).
"""
from __future__ import annotations

from source.omy.omy_robot_cfg import OMY_OFF_SELF_COLLISION_CFG
from source.motion3.layout import TABLE_HEIGHT

# init_state.joint_pos는 그대로 보존하고 base pos만 (0,0,H)로 덮어쓴다.
# disable_gravity=True 라 로봇이 z=H에 떠 있는 상태로 고정된다(책상 위 마운트).
OMY_TABLE_MOUNTED_CFG = OMY_OFF_SELF_COLLISION_CFG.replace(
    init_state=OMY_OFF_SELF_COLLISION_CFG.init_state.replace(
        pos=(0.0, 0.0, TABLE_HEIGHT),
    ),
)
