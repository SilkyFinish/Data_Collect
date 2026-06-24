#!/usr/bin/env sh

# Edit parameters here, then run:
#   sh collect/run_dual_collect.sh
#
# Runtime keys:
#   r: activate teleoperation
#   s: deactivate teleoperation
#   c: start recording one trajectory
#   v: stop current trajectory recording
#   q: quit

PYTHON_BIN="python3"

# Robot serial numbers
FIRST_SN="Rizon4s-063652"
SECOND_SN="Rizon4s-063586"

# Data collection
SAVE_ROOT="/home/xense/flexiv_hizon4s/Data/test"
SESSION_NAME=""
FPS="30"

# Gripper collection: true or false
USE_GRIPPER="true"
MASTER_GRIPPER_ID="master_xense_id"
SLAVE_GRIPPER_ID="slave_xense_id"

# Optional LAN interface whitelist. Leave empty to let TDK try all interfaces.
# Multiple addresses can be separated by spaces, for example:
# NETWORK_INTERFACES="192.168.2.102 10.42.0.1"
NETWORK_INTERFACES=""

# Runtime tuning
GRIPPER_EPS="0.0001"
GRIPPER_WAIT_TIME="0.1"
NULL_SPACE_PERIOD="0.1"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

set -- \
  "$SCRIPT_DIR/dual_collect.py" \
  -1 "$FIRST_SN" \
  -2 "$SECOND_SN" \
  --save-root "$SAVE_ROOT" \
  --fps "$FPS" \
  --use-gripper "$USE_GRIPPER" \
  --gripper-eps "$GRIPPER_EPS" \
  --gripper-wait-time "$GRIPPER_WAIT_TIME" \
  --null-space-period "$NULL_SPACE_PERIOD"

if [ -n "$SESSION_NAME" ]; then
  set -- "$@" --session-name "$SESSION_NAME"
fi

if [ "$USE_GRIPPER" = "true" ]; then
  set -- "$@" \
    --master-gripper-id "$MASTER_GRIPPER_ID" \
    --slave-gripper-id "$SLAVE_GRIPPER_ID"
fi

for interface in $NETWORK_INTERFACES; do
  set -- "$@" --network-interface "$interface"
done

echo "Running:"
printf '  %s' "$PYTHON_BIN"
for arg in "$@"; do
  printf ' %s' "$arg"
done
printf '\n\n'

exec "$PYTHON_BIN" "$@"
