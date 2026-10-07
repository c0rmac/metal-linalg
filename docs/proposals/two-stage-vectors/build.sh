#!/bin/sh
# Builds the two-stage-with-vectors prototype (../two-stage-vectors.md).
#   docs/proposals/two-stage-vectors/build.sh [output dir]
# Needs the Metal toolchain. Run the binaries from the output dir, since
# q2_gpu loads q2.metallib by relative path:
#   check_q2 517 16    the recorded reflectors and the grouped order, on the CPU
#   q2_gpu 4096        Q2 on the GPU (checked against the CPU up to n = 2048)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-${TMPDIR:-/tmp}/metal-linalg-two-stage-vectors}
mkdir -p "$OUT"
cd "$OUT"
xcrun -sdk macosx metal -mmacosx-version-min=14.0 -fno-fast-math -c "$HERE/q2.metal" -o q2.air
xcrun -sdk macosx metallib q2.air -o q2.metallib
CXX="clang++ -std=c++17 -O2"
$CXX "$HERE/check_q2.cpp" "$HERE/chase_record.cpp" -framework Accelerate -o check_q2
$CXX -fobjc-arc "$HERE/q2_gpu.mm" "$HERE/chase_record.cpp" -framework Metal -framework Foundation \
    -framework Accelerate -o q2_gpu
echo "built in $OUT"
