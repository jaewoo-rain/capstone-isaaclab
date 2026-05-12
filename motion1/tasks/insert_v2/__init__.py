"""motion1 — Insert RL v2 (SAC + HER).

박스 잡힌 채 셀 위에서 xy/yaw 미세 정렬.
v1 (insert) 와 차이:
- 알고리즘: PPO → SAC + HER (off-policy + relabeling)
- Reset: handoff dataset → 코드 reset (cell + ee offset random + IK)
- Obs: flat (core + achieved_goal + desired_goal)
"""
from .insert_env_v2_cfg import InsertEnvV2Cfg
from .insert_env_v2 import InsertEnvV2

__all__ = ["InsertEnvV2Cfg", "InsertEnvV2"]
