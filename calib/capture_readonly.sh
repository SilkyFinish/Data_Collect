#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="${SCRIPT_DIR}/../submodule/r3kit:${PYTHONPATH:-}"

python3 calib_readonly.py \
  --robot_id Rizon4s-063586 \
  --camera_id 327322062498 \
  --save_path ./data \
  --warmup_frames 30
