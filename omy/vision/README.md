# OMY Vision package

- YOLO + tracking + depth
- nearest object picking among 3 objects
- 3x3 place slot detection
- separate vision envs for grasp / lift / place
- A-mode: GT RL training stays separate, vision env is used in play/eval

Important:
1. `source/omy/omy_robot_cfg.py` must export `OMY_CFG`.
2. `checkpoints/yolo/best.pt` must exist.
3. Camera field names can differ between IsaacLab versions, so `_get_camera_pose_for_env()` includes a fallback.
4. Place slots are logic-level targets and dataset labels. If you want visible meshes, add visual marker prims later.
