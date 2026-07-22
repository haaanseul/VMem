#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-test_samples/oxford.jpg}"
FPS="${FPS:-13}"
INTERP_FRAMES="${INTERP_FRAMES:-4}"
IMAGE_NAME="$(basename "$IMAGE")"
IMAGE_NAME="${IMAGE_NAME%.*}"
OUT_ROOT="${OUT_ROOT:-revisit_outputs/${IMAGE_NAME}_revisit_original}"

mkdir -p "$OUT_ROOT"

# Original app's 10-degree button effectively rotates about 5 degrees.
python scripts/revisit_test.py \
  --image "$IMAGE" \
  --commands \
    yaw:5 yaw:5 yaw:5 yaw:5 yaw:5 yaw:5 yaw:5 yaw:5 yaw:5 \
    yaw:-5 yaw:-5 yaw:-5 yaw:-5 yaw:-5 yaw:-5 yaw:-5 yaw:-5 yaw:-5 \
  --interp-frames "$INTERP_FRAMES" \
  --fps "$FPS" \
  --out "$OUT_ROOT/01_yaw45_return"

# Original app's 20-degree button effectively rotates about 10 degrees.
python scripts/revisit_test.py \
  --image "$IMAGE" \
  --commands \
    yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 \
    yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 \
  --interp-frames "$INTERP_FRAMES" \
  --fps "$FPS" \
  --out "$OUT_ROOT/02_yaw90_return"

# A longer repeated revisit using the same original-ish yaw step.
python scripts/revisit_test.py \
  --image "$IMAGE" \
  --commands \
    yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 \
    yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 \
    yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 \
    yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 \
  --interp-frames "$INTERP_FRAMES" \
  --fps "$FPS" \
  --out "$OUT_ROOT/03_yaw90_return_twice"

# A full rotation revisit, still using the original-ish 10-degree yaw step.
python scripts/revisit_test.py \
  --image "$IMAGE" \
  --commands \
    yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 \
    yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 \
    yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 \
    yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 \
  --interp-frames "$INTERP_FRAMES" \
  --fps "$FPS" \
  --out "$OUT_ROOT/04_yaw360_return"

echo "Original-setting revisit suite complete: $OUT_ROOT"
