#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-test_samples/oxford.jpg}"
FPS="${FPS:-13}"
INTERP_FRAMES="${INTERP_FRAMES:-4}"
OUT_ROOT="${OUT_ROOT:-revisit_outputs/originalish_suite}"

mkdir -p "$OUT_ROOT"

python scripts/revisit_test.py \
  --image "$IMAGE" \
  --commands \
    yaw:5 yaw:5 yaw:5 yaw:5 yaw:5 yaw:5 yaw:5 yaw:5 yaw:5 \
    yaw:-5 yaw:-5 yaw:-5 yaw:-5 yaw:-5 yaw:-5 yaw:-5 yaw:-5 yaw:-5 \
  --interp-frames "$INTERP_FRAMES" \
  --fps "$FPS" \
  --out "$OUT_ROOT/01_yaw5_return"

python scripts/revisit_test.py \
  --image "$IMAGE" \
  --commands \
    yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 yaw:10 \
    yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 yaw:-10 \
  --interp-frames "$INTERP_FRAMES" \
  --fps "$FPS" \
  --out "$OUT_ROOT/02_yaw10_return"

python scripts/revisit_test.py \
  --image "$IMAGE" \
  --commands \
    forward:1 forward:1 forward:1 forward:1 forward:1 forward:1 forward:1 forward:1 \
    backward:1 backward:1 backward:1 backward:1 backward:1 backward:1 backward:1 backward:1 \
  --interp-frames "$INTERP_FRAMES" \
  --fps "$FPS" \
  --out "$OUT_ROOT/03_forward_back"

echo "Original-ish sanity suite complete: $OUT_ROOT"
