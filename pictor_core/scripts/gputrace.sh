#!/usr/bin/env bash
# GPU Trace a headless pictor_core app with Nsight Graphics.
#
# Nsight Graphics' own launcher (`ngfx --exe`) can crash the target during Vulkan ICD
# negotiation on some drivers; this wrapper starts the app itself with the GPU Trace
# layer enabled and then attaches the host, scoping the trace by submits. Counter
# metrics require the R610+ capability grant (see docs/profiling/gputrace.md); without it the
# wrapper degrades to an event/label trace.
#
# Usage:
#   scripts/gputrace.sh <app> [app args...]
#       [--start-after-submits N] [--limit-to-submits N] [--max-duration-ms D]
#       [--out DIR] [--metrics] [--no-metrics] [--rts] [--sdk]
#
# Environment overrides: NG (ngfx binary), NGFX_INSTALL (install root).
set -euo pipefail

usage()
{
    sed -n '2,12p' "$0" | cut -c3-
}

NG=${NG:-/opt/nsight-graphics/latest/host/linux-desktop-nomad-x64/ngfx}
INST=${NGFX_INSTALL:-/opt/nsight-graphics/latest}
LAYER_DIR="$INST/target/linux-desktop-nomad-x64/GPUTraceVulkanLayer"

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 2
fi

APP=$1
shift

START_SUBMITS=0
LIMIT_SUBMITS=8
MAX_MS=2000
OUT=""
METRICS=auto
RTS=0
SDK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-after-submits) START_SUBMITS=$2; shift 2 ;;
        --limit-to-submits) LIMIT_SUBMITS=$2; shift 2 ;;
        --max-duration-ms) MAX_MS=$2; shift 2 ;;
        --out) OUT=$2; shift 2 ;;
        --metrics) METRICS=on; shift ;;
        --no-metrics) METRICS=off; shift ;;
        --rts) RTS=1; shift ;;
        --sdk) SDK=1; shift ;;
        *) break ;;
    esac
done

OUT=${OUT:-scratch/gputrace/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT"

# Counter permissions: R610+ capability nodes, or the legacy admin-only regkey (pre-R610).
counters_ok=1
if grep -q profiler-device /proc/driver/nvidia-caps/sys-minors 2>/dev/null; then
    for cap in profiler-device profiler-context trace-device; do
        minor=$(awk -v c="$cap" '$1==c{print $2}' /proc/driver/nvidia-caps/sys-minors 2>/dev/null || true)
        [[ -n "$minor" && -r "/dev/nvidia-caps/nvidia-cap$minor" ]] || counters_ok=0
    done
else
    admin_only=$(awk '$1=="RmProfilingAdminOnly:"{print $2}' /proc/driver/nvidia/params 2>/dev/null || echo 1)
    [[ $admin_only == 0 || $EUID -eq 0 ]] || counters_ok=0
fi

if [[ $METRICS == on && $counters_ok == 0 ]]; then
    echo "gputrace: counters are not permitted for this user; see docs/profiling/gputrace.md" >&2
    exit 1
fi

ENABLE_VAR=$(grep -ohE 'ENABLE_[A-Za-z0-9_]+' "$LAYER_DIR"/vulkan/implicit_layer.d/*.json 2>/dev/null | head -1 || true)
ENABLE_VAR=${ENABLE_VAR:-ENABLE_VK_LAYER_NV_GPU_Trace_release_public_2026_2_0}

env_args=(
    "$ENABLE_VAR=1"
    "XDG_DATA_DIRS=/usr/local/share:/usr/share:$LAYER_DIR"
    "NV_TPS_LAUNCH_TOKEN=GPU Trace Profiler"
    "GPUTRACE_START_EVENT=$([[ $SDK == 1 ]] && echo 'NGFX SDK Start' || echo 'Submit Count')"
    "GPUTRACE_STOP_EVENT=$([[ $SDK == 1 ]] && echo 'NGFX SDK Stop' || echo 'Max Submits')"
    "GPUTRACE_START_SUBMIT_COUNT=0"
    "GPUTRACE_STOP_SUBMIT_COUNT=$LIMIT_SUBMITS"
    "GPUTRACE_STOP_DURATION_IN_MS=$MAX_MS"
    "GPUTRACE_LOCKCLOCKS=Lock to Base"
    "GPUTRACE_COLLECT_SHADERS=1"
    "GPUTRACE_COLLECT_EXTERNAL_DEBUG_INFO=1"
    "GPUTRACE_ENABLE_NVTX_RANGES=1"
    "GPUTRACE_ENABLE_SCREENSHOT=0"
    "GPUTRACE_TIMESTAMP_COUNT=100000"
    "GPUTRACE_EVENT_BUFFER_MEMORY_KB=20000"
)
if [[ $counters_ok == 0 || $METRICS == off ]]; then
    echo "gputrace: tracing events/labels only (no counter metrics)" >&2
    env_args+=("GPUTRACE_DISABLE_METRICS=1")
fi
if [[ $SDK == 1 ]]; then
    env_args+=("PICTOR_NGFX_SDK=1")
fi

env "${env_args[@]}" "$APP" "$@" &
APP_PID=$!
trap 'kill $APP_PID 2>/dev/null || true' EXIT
sleep 1

host_args=(
    --activity="GPU Trace Profiler"
    --platform="Linux (x86_64)"
    --attach-pid="$APP_PID"
    --max-duration-ms="$MAX_MS"
    --auto-export
    --output-dir="$OUT"
)
if [[ $SDK == 1 ]]; then
    host_args+=(--start-with-ngfx-sdk --stop-with-ngfx-sdk)
else
    host_args+=(--start-after-submits="$START_SUBMITS" --limit-to-submits="$LIMIT_SUBMITS")
fi
[[ $RTS == 1 ]] && host_args+=(--real-time-shader-profiler)

"$NG" "${host_args[@]}"
wait "$APP_PID" || true

echo "gputrace: report  -> $(ls "$OUT"/*.ngfx-gputrace 2>/dev/null | tail -1)"
echo "gputrace: exports -> $OUT/BASE*/"
