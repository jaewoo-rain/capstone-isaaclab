# motion2 MVP demo guide

This guide records the current demo path for the OMY-F3M capstone project.

The MVP goal is not full autonomous pick-and-place execution yet. The current
demo shows the complete object-to-slot pipeline shape in dry-run/plan-only mode
and validates real robot command paths with guarded smoke motions.

## Terminal setup

Use three robot-container terminals.

Terminal A, regular robot bringup:

```bash
ssh root@omy-SNPR44B1021.local
cd /data/docker/open_manipulator
./docker/container.sh enter
ros2 launch open_manipulator_bringup omy_f3m.launch.py
```

Terminal B, MoveIt:

```bash
ssh root@omy-SNPR44B1021.local
cd /data/docker/open_manipulator
./docker/container.sh enter
ros2 launch open_manipulator_moveit_config omy_f3m_moveit.launch.py start_rviz:=false
```

Terminal C, commands:

```bash
ssh root@omy-SNPR44B1021.local
cd /data/docker/open_manipulator
./docker/container.sh enter
cd /root/ros2_ws/src/open_manipulator
```

Do not use `omy_ai.launch.py` for MoveIt execution tests. That mode is only for
leader/follower teleoperation and manual pose recovery.

## Preflight

Run in Terminal C:

```bash
ros2 control list_controllers
ros2 action list
ros2 topic echo /joint_states --once
```

Expected controllers:

- `joint_state_broadcaster` active
- `arm_controller` active
- `gripper_controller` active

Expected actions:

- `/arm_controller/follow_joint_trajectory`
- `/gripper_controller/gripper_cmd`
- `/move_action`
- `/execute_trajectory`

## Demo 1: full MVP chain dry-run

Capstone-level pipeline summary:

```bash
python3 motion2/scripts/run_capstone_mvp_pipeline.py
```

Fast safe presentation mode:

```bash
python3 motion2/scripts/run_pick_place_mvp.py --allow-unverified --demo-safe
```

This prints the object-to-slot chain without moving the robot:

```bash
python3 motion2/scripts/run_pick_place_mvp.py \
  --allow-unverified \
  --use-current-ee-orientation
```

Use this as the presentation-level pipeline:

```text
object pose
-> pre_grasp
-> grasp
-> close_gripper
-> lift
-> transport
-> insert
-> open_gripper
-> retract
```

## Demo 2: MoveIt plan-only checks

High stages that have planned successfully:

```bash
python3 motion2/scripts/plan_manual_targets_moveit.py \
  --config motion2/config/manual_targets_safe_dryrun.yaml \
  --allow-unverified \
  --use-current-ee-orientation \
  --only pre_grasp lift transport insert retract \
  --planning-time 10.0 \
  --attempts 10
```

Known blocked stage:

```bash
python3 motion2/scripts/plan_manual_targets_moveit.py \
  --config motion2/config/manual_targets_safe_dryrun.yaml \
  --allow-unverified \
  --use-current-ee-orientation \
  --only grasp \
  --planning-time 10.0 \
  --attempts 10
```

`grasp_z=0.115` is not approved for execution in the current setup.

## Demo 3: real robot smoke sequences

Dry-run:

```bash
python3 motion2/scripts/run_smoke_sequence.py --name full_basic_smoke
```

Execute only after the robot area is clear:

```bash
python3 motion2/scripts/run_smoke_sequence.py --name full_basic_smoke --execute
```

This validates tiny joint-space and gripper command paths.

## Demo 4: guarded relative pose smoke

Plan guard only:

```bash
python3 motion2/scripts/run_pick_place_mvp.py \
  --allow-unverified \
  --plan-guard \
  --relative-z 0.002 \
  --planning-time 10.0 \
  --attempts 10
```

Execute only if plan guard passes immediately before execution:

```bash
python3 motion2/scripts/run_pick_place_mvp.py \
  --allow-unverified \
  --execute \
  --confirm EXECUTE_PICK_PLACE_MVP \
  --relative-z 0.002 \
  --planning-time 10.0 \
  --attempts 10
```

## Current real-robot boundary

Allowed for demo:

- full MVP chain dry-run
- MoveIt plan-only checks
- tiny joint/gripper smoke sequence
- single relative pose stage only after immediate `--plan-guard` pass

Not allowed yet:

- full pick-and-place real execution
- `grasp_z=0.115` execution
- absolute `transport`/`insert` execution
- camera-driven real execution
- RL policy output driving the real robot

Reason: MoveIt may generate large continuous-joint wraparound trajectories even
for small or reachable pose targets. The MVP runner therefore rejects
trajectories whose joint span exceeds the configured guard.
