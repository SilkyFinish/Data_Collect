set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}/../submodule/r3kit:${PYTHONPATH:-}"

python3 calib.py \
  --robot_id Rizon4s-063586 \
  --camera_id 327322062498 \
  --tool_name Flange \
  --save_path ./data \
  --jog \
  --jog_step_m 0.01 \
  --jog_step_deg 3