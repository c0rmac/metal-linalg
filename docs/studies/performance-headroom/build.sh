#!/bin/sh
# Builds the experiments behind ../performance-headroom-apple-m5-pro.md.
#   docs/studies/performance-headroom/build.sh [library build dir] [output dir]
# Needs the Metal toolchain (xcodebuild -downloadComponent MetalToolchain) and a
# built library (the default is ./build). Binaries go to the output dir, outside
# the repository; run them from there, since they load the .metallib files by
# relative path. Each prints one table; see the report for what each measures.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
BUILD=$(cd "${1:-$ROOT/build}" && pwd)
OUT=${2:-${TMPDIR:-/tmp}/metal-linalg-headroom}
mkdir -p "$OUT"
cd "$OUT"

xcrun -sdk macosx metal -std=metal4.0 -O3 -c "$HERE/gemm.metal" -o gemm.air
xcrun -sdk macosx metallib gemm.air -o gemm.metallib
for n in 64 32; do
    xcrun -sdk macosx metal -std=metal3.1 -O3 -DTDQL_NMAX=$n -c "$HERE/tdql.metal" -o tdql$n.air
    xcrun -sdk macosx metallib tdql$n.air -o tdql$n.metallib
done
cp tdql64.metallib tdql.metallib

LIB="-I$ROOT/include -L$BUILD -lmetal_linalg -Wl,-rpath,$BUILD"
FW="-framework Metal -framework MetalPerformanceShaders -framework Accelerate -framework Foundation"
CXX="clang++ -std=c++17 -O2 -fobjc-arc -DACCELERATE_NEW_LAPACK"
for x in exp_batch exp_large exp_svdlarge exp_tdql exp_routes; do
    src="$HERE/$x.mm"; [ -f "$src" ] || src="$HERE/$x.cpp"
    $CXX "$src" $LIB $FW -o $x
done
for x in exp_gemm exp_stage exp_mrrr exp_bisect exp_stedc_threads; do
    $CXX "$HERE/$x.mm" $FW -o $x
done
echo "built in $OUT"
