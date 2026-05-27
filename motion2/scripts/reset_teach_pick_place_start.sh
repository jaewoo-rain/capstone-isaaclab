#!/usr/bin/env bash
set -euo pipefail

# Return the OMY-F3M to the taught start state via init -> start.
# Run inside the robot Docker container after bringup/controllers are active.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RUNNER="motion2/scripts/run_teach_pick_place.py"
CONFIRM="EXECUTE_TEACH_PICK_PLACE"
ARM_MAX_DELTA="${ARM_MAX_DELTA:-3.0}"
DURATION="${DURATION:-8.0}"

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

echo "[teach-reset] This will move the real robot: init -> start"
echo "[teach-reset] Arm duration: ${DURATION}s, arm max delta: ${ARM_MAX_DELTA} rad"
read -r -p "[teach-reset] Type RESET_TO_START to continue: " typed
if [[ "${typed}" != "RESET_TO_START" ]]; then
  echo "[teach-reset] Refusing to execute."
  exit 2
fi

run_arm_step init
run_arm_step start

echo "[teach-reset] Complete."
