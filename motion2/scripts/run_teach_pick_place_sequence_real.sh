#!/usr/bin/env bash
set -euo pipefail

# Execute the taught OMY-F3M waypoint sequence recorded in:
#   motion2/config/teach_pick_place_waypoints.yaml
#
# Run inside the robot Docker container after bringup/controller is active.
# This wrapper asks for one global confirmation, then disables per-step prompts.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="motion2/scripts/run_teach_pick_place.py"
CONFIRM="EXECUTE_TEACH_PICK_PLACE"
ARM_MAX_DELTA="${ARM_MAX_DELTA:-3.0}"
GRIPPER_MAX_DELTA="${GRIPPER_MAX_DELTA:-2.5}"
DURATION="${DURATION:-8.0}"
CLOSE_GRIPPER="${CLOSE_GRIPPER:-1.05}"
OPEN_GRIPPER="${OPEN_GRIPPER:-0.0}"

run_arm_step() {
  local step_name="$1"
  python3 "${RUNNER}" \
    --sequence "${step_name}" \
    --max-joint-delta "${ARM_MAX_DELTA}" \
    --duration "${DURATION}" \
    --execute \
    --no-step-prompts \
    --confirm "${CONFIRM}"
}

run_gripper_step() {
  local position="$1"
  python3 "${RUNNER}" \
    --sequence close_gripper \
    --close-gripper "${position}" \
    --max-joint-delta "${GRIPPER_MAX_DELTA}" \
    --duration "${DURATION}" \
    --execute \
    --no-step-prompts \
    --allow-gripper-partial \
    --confirm "${CONFIRM}"
}

echo "[teach-sequence] This will execute the real robot."
echo "[teach-sequence] Sequence: start -> 1 -> 2 -> close(waypoint) -> 3 -> 4 -> close_gripper(${OPEN_GRIPPER})"
echo "[teach-sequence] Arm duration: ${DURATION}s, arm max delta: ${ARM_MAX_DELTA} rad"
echo "[teach-sequence] Close waypoint: arm waypoint named close; final gripper command: ${OPEN_GRIPPER}"
read -r -p "[teach-sequence] Type RUN_TEACH_SEQUENCE to continue: " typed
if [[ "${typed}" != "RUN_TEACH_SEQUENCE" ]]; then
  echo "[teach-sequence] Refusing to execute."
  exit 2
fi

run_arm_step start
run_arm_step 1
run_arm_step 2
run_arm_step close
run_arm_step 3
run_arm_step 4
run_gripper_step "${OPEN_GRIPPER}"

echo "[teach-sequence] Complete."
