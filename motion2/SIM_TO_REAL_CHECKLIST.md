# motion2 sim-to-real checklist for OMY-F3M

This document records the current real-robot deployment assumptions and safety
checks for `motion2`.

## Current hardware state

- Robot: OMY-F3M, ROS2 Jazzy bringup works with:
  - `ros2 launch open_manipulator_bringup omy_ai.launch.py`
- Wrist camera: D405 is expected to be mounted on the wrist.
- Top camera: D435 exists, but is not mounted/calibrated as a top-view camera yet.

Because the D435 top camera is not ready, the original fully automatic
`top_cam_scan()` flow must not be used on the real robot yet.

## Confirmed robot interfaces

- Joint state topic:
  - `/joint_states`
- Controlled joints:
  - `joint1`
  - `joint2`
  - `joint3`
  - `joint4`
  - `joint5`
  - `joint6`
  - `rh_r1_joint`
- Trajectory action:
  - `/arm_controller/follow_joint_trajectory`
- Active controllers:
  - `joint_state_broadcaster`
  - `arm_controller`
- Command interface:
  - `position`

## Confirmed TF frames

- Robot base frame for real deployment:
  - `link0`
- End-effector frame:
  - `end_effector_link`
- Wrist camera frame:
  - `camera_link`
- Do not assume `base_link` exists in the real robot TF tree.

## Important MoveIt/controller mismatch

MoveIt config declares:

- `arm_controller`: `joint1` to `joint6`
- `gripper_controller`: `rh_r1_joint`

But the real bringup exposes:

- `arm_controller`: `joint1` to `joint6` plus `rh_r1_joint`
- no separate `gripper_controller`

Real execution code should account for this mismatch. For the current bringup,
the gripper joint should be treated as part of the same trajectory controller
unless the bringup/config is changed.

## Current recommended deployment path: manual target mode

Since D435 top-view detection is not available, use manual target poses first.

Recommended flow:

1. Move to home.
2. Use manually configured `box_xy` and `box_yaw`.
3. Move above the manually configured box pose.
4. Use D405 wrist camera only for grasp alignment.
5. Grasp, lift, and transport to manually configured `cell_xy` and `cell_yaw`.
6. Insert/release only after dry-run and planning checks pass.

The original top-view flow:

```text
get_top_cam() -> YOLO -> top_cam_scan() -> box/cell xy/yaw
```

must remain disabled for real tests until the D435 is mounted and calibrated.

## Manual target config draft

Use a config file before real execution, for example:

```yaml
frame_id: link0
box:
  x: 0.30
  y: -0.10
  yaw: 0.0
cell:
  x: 0.30
  y: -0.30
  yaw: 0.0
heights:
  pre_grasp_z: 0.17
  grasp_z: 0.115
  lift_z: 0.26
  transport_z: 0.26
  place_z: 0.165
  retract_z: 0.26
```

All values must be treated as unverified until checked against the real
workspace, object dimensions, and collision environment.

## Safety requirements before any real motion

- Keep using `DryRunAdapter` until target poses and frame conventions are
  verified.
- Do not use hard-coded home EE pose from sim. Read home pose from TF after
  the robot is at home.
- Enforce workspace limits in `link0`.
- Enforce conservative z limits before descend/insert.
- Enforce joint limits, especially `joint3` near the real +/-150 degree range.
- Verify `rh_r1_joint` open/close range on the real gripper before grasp tests.
- Use slow trajectory timing for any future `FollowJointTrajectory` command.
- Add an emergency stop procedure before real execution.
- Do not run `run_chain_once()` with `RealAdapter` until the adapter has safety
  checks and a manual target mode.

## Next implementation steps

1. Add a manual target config loader.
2. Add a dry-run chain script that skips `top_cam_scan()` and uses manual
   `box_xy`, `box_yaw`, `cell_xy`, and `cell_yaw`.
3. Validate generated stage targets with `DryRunAdapter`.
4. Add MoveIt planning-only checks for each waypoint.
5. Only after the above, implement a guarded real trajectory sender.
