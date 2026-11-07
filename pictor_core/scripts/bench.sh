#!/usr/bin/env bash
# Build and run the comparison benchmarks. These are NOT tests and are not wired into ctest:
# they exist so we can compare pictor primitives against their CUDA/CUB references.
#
# Usage:
#   scripts/bench.sh                      configure + build, then run every benchmark (defaults)
#   scripts/bench.sh scan                 scan: pictor::Scan vs cub::DeviceScan
#   scripts/bench.sh scan-vulkan --sizes 1048576,4194304
#   scripts/bench.sh scan-cuda   --bits 64 --kinds inclusive
#   scripts/bench.sh --build              configure + build only
#
# Environment: PICTOR_BUILD_DIR (default: build).
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

build_dir="${PICTOR_BUILD_DIR:-build}"

cmake -S . -B "$build_dir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPICTOR_BUILD_BENCHMARKS=ON
cmake --build "$build_dir" -j "$(nproc)"

if [[ "${1:-}" == "--build" ]]; then
    echo "benchmarks built in $build_dir"
    exit 0
fi

target="${1:-all}"
if [[ $# -gt 0 ]]; then
    shift
fi

run() {
    local name="$1"
    shift
    local exe="$build_dir/bench/primitives/$name"
    echo "+ $exe $*"
    "$exe" "$@"
}

case "$target" in
    all)
        run pictor_bench_scan_vulkan "$@"
        run pictor_bench_scan_cuda "$@"
        run pictor_bench_sort_vulkan "$@"
        run pictor_bench_sort_cuda "$@"
        ;;
    scan)
        run pictor_bench_scan_vulkan "$@"
        run pictor_bench_scan_cuda "$@"
        ;;
    scan-vulkan)
        run pictor_bench_scan_vulkan "$@"
        ;;
    scan-cuda)
        run pictor_bench_scan_cuda "$@"
        ;;
    sort)
        run pictor_bench_sort_vulkan "$@"
        run pictor_bench_sort_cuda "$@"
        ;;
    sort-vulkan)
        run pictor_bench_sort_vulkan "$@"
        ;;
    sort-cuda)
        run pictor_bench_sort_cuda "$@"
        ;;
    *)
        echo "unknown benchmark: $target (try: all, scan, sort, scan-vulkan, scan-cuda, sort-vulkan, sort-cuda)" >&2
        exit 1
        ;;
esac
