#!/usr/bin/env bash
set -Ee -o pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +u
source /opt/ros/jazzy/setup.bash
source "${SCRIPT_DIR}/../install/force_position/local_setup.bash"
set -u
exec /usr/bin/python3 "${SCRIPT_DIR}/research/force_position_control/runtime.py" \
  --config "${SCRIPT_DIR}/research/force_position_control/calibrated_config.json" \
  --active --select-servo "$@"
