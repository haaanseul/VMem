#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-test_samples/oxford.jpg}"
FPS="${FPS:-12}"
OUT_ROOT="${OUT_ROOT:-revisit_outputs/failure_suite}"

mkdir -p "$OUT_ROOT"

python scripts/revisit_test.py \
  --image "$IMAGE" \
  --preset yaw-return \
  --interp-frames "${YAW_INTERP_FRAMES:-7}" \
  --fps "$FPS" \
  --out "$OUT_ROOT/01_yaw_return"

python scripts/revisit_test.py \
  --image "$IMAGE" \
  --preset wide-yaw-return \
  --interp-frames "${WIDE_YAW_INTERP_FRAMES:-7}" \
  --fps "$FPS" \
  --out "$OUT_ROOT/02_wide_yaw_return"

python scripts/revisit_test.py \
  --image "$IMAGE" \
  --preset forward-back \
  --interp-frames "${FORWARD_BACK_INTERP_FRAMES:-7}" \
  --fps "$FPS" \
  --out "$OUT_ROOT/03_forward_back"

python scripts/revisit_test.py \
  --image "$IMAGE" \
  --preset box-return \
  --interp-frames "${BOX_INTERP_FRAMES:-4}" \
  --fps "$FPS" \
  --out "$OUT_ROOT/04_box_return"

echo "Failure suite complete: $OUT_ROOT"
