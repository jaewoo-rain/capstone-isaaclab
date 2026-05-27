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
DURATION="${DURATION:-8.0}"
OPEN_GRIPPER="${OPEN_GRIPPER:-0.0}"
GRIPPER_TIMEOUT="${GRIPPER_TIMEOUT:-3.0}"
SEQUENCE="${SEQUENCE:-start,1,2,close,3,4,close_gripper}"

echo "[teach-sequence] This will execute the real robot."
echo "[teach-sequence] Sequence: ${SEQUENCE}"
echo "[teach-sequence] Arm duration: ${DURATION}s, arm max delta: ${ARM_MAX_DELTA} rad"
echo "[teach-sequence] Close waypoint: arm waypoint named close; final gripper command: ${OPEN_GRIPPER}"
echo "[teach-sequence] Gripper result timeout: ${GRIPPER_TIMEOUT}s"
read -r -p "[teach-sequence] Type RUN_TEACH_SEQUENCE to continue: " typed
if [[ "${typed}" != "RUN_TEACH_SEQUENCE" ]]; then
  echo "[teach-sequence] Refusing to execute."
  exit 2
fi

python3 "${RUNNER}" \
  --sequence "${SEQUENCE}" \
  --close-gripper "${OPEN_GRIPPER}" \
  --max-joint-delta "${ARM_MAX_DELTA}" \
  --duration "${DURATION}" \
  --execute \
  --no-step-prompts \
  --allow-gripper-partial \
  --gripper-result-timeout "${GRIPPER_TIMEOUT}" \
  --confirm "${CONFIRM}"

echo "[teach-sequence] Complete."
