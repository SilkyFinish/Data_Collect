#!/usr/bin/env sh

# Edit parameters here, then run:
#   sh postprocess/run_convert_hdf5.sh

PYTHON_BIN="python3"

# Input can be one session directory, or a root containing multiple sessions.
# Multiple paths can be separated by spaces.
SESSIONS="/home/xense/flexiv_hizon4s/Data/test"

# Output HDF5 file for MaskACT-3D training.
OUTPUT_HDF5="/home/xense/flexiv_hizon4s/Data/train_data.hdf5"

# Camera folder to export. Required when each session has multiple cameras.
CAMERA_NAME="cam_327322062498"

# Calibration. These defaults point to Data_Collect/calib/data.
INTRINSICS=""
CAMERA_C2W=""
DEPTH_SCALE="0.001"

# Point cloud and HDF5 settings.
NUM_POINTS="10000"
DOWNSAMPLE_SEED="42"
DEPTH_INVALID_MAX="100.0"
MASK_VALUE="0"
FPS="30"
DEMO_PREFIX="demo"
FRAME_STRIDE="1"
MAX_FRAMES=""
COMPRESSION="lzf"
R3KIT_ROOT=""
LOG_EVERY="50"
FORCE="true"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

set -- \
  "$SCRIPT_DIR/convert_hdf5.py" \
  -o "$OUTPUT_HDF5" \
  --num-points "$NUM_POINTS" \
  --downsample-seed "$DOWNSAMPLE_SEED" \
  --depth-invalid-max "$DEPTH_INVALID_MAX" \
  --mask-value "$MASK_VALUE" \
  --fps "$FPS" \
  --demo-prefix "$DEMO_PREFIX" \
  --frame-stride "$FRAME_STRIDE" \
  --compression "$COMPRESSION" \
  --log-every "$LOG_EVERY"

if [ -n "$CAMERA_NAME" ]; then
  set -- "$@" --camera-name "$CAMERA_NAME"
fi

if [ -n "$INTRINSICS" ]; then
  set -- "$@" --intrinsics "$INTRINSICS"
fi

if [ -n "$CAMERA_C2W" ]; then
  set -- "$@" --camera-c2w "$CAMERA_C2W"
fi

if [ -n "$DEPTH_SCALE" ]; then
  set -- "$@" --depth-scale "$DEPTH_SCALE"
fi

if [ -n "$MAX_FRAMES" ]; then
  set -- "$@" --max-frames "$MAX_FRAMES"
fi

if [ -n "$R3KIT_ROOT" ]; then
  set -- "$@" --r3kit-root "$R3KIT_ROOT"
fi

if [ "$FORCE" = "true" ]; then
  set -- "$@" --force
fi

for session in $SESSIONS; do
  set -- "$@" "$session"
done

echo "Running:"
printf '  %s' "$PYTHON_BIN"
for arg in "$@"; do
  printf ' %s' "$arg"
done
printf '\n\n'

exec "$PYTHON_BIN" "$@"
