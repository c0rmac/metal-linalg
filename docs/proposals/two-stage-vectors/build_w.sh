#!/bin/sh
# Builds q2w (the pipelined Q2 kernel) next to build.sh's output.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-${TMPDIR:-/tmp}/metal-linalg-two-stage-vectors}
mkdir -p "$OUT"
cd "$OUT"
xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -fno-fast-math -c "$HERE/q2w.metal" -o q2w.air
xcrun -sdk macosx metallib q2w.air -o q2w.metallib
clang++ -std=c++17 -O2 -fobjc-arc "$HERE/q2w_gpu.mm" "$HERE/chase_record.cpp" -framework Metal -framework Foundation \
    -framework Accelerate -o q2w_gpu
echo "built in $OUT"
