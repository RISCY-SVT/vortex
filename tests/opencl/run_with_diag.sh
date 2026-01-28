#!/usr/bin/env bash
set -euo pipefail

TOOL=""
PROJECT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tool)
      TOOL="$2"; shift 2;;
    --project)
      PROJECT="$2"; shift 2;;
    --)
      shift; break;;
    *)
      break;;
  esac
done

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [--tool PATH] [--project NAME] -- <command> [args...]" >&2
  exit 2
fi

CMD=("$@")

if [[ -z "$PROJECT" ]]; then
  PROJECT=$(basename "${CMD[0]}")
fi

if [[ -z "$TOOL" ]]; then
  SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
  TOOL="$SCRIPT_DIR/ocl_diag"
fi

diag() {
  echo "[opencl-diag] $*" >&2
}

child_pid=0
on_signal() {
  local sig="$1"
  diag "caught $sig, terminating child..."
  if [[ $child_pid -ne 0 ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  exit 128
}

trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

run_cmd() {
  "${CMD[@]}" &
  child_pid=$!
  wait "$child_pid"
  local rc=$?
  child_pid=0
  return $rc
}

start_ts=$(date +%s)

diag "project=$PROJECT driver=${VORTEX_DRIVER:-} xlen=${POCL_VORTEX_XLEN:-} llvm_prefix=${LLVM_PREFIX:-}"

diag_level=$(echo "${OPENCL_DIAG_LEVEL:-brief}" | tr '[:upper:]' '[:lower:]')
tool_args=()
case "$diag_level" in
  full|2)
    tool_args=(--full)
    ;;
  brief|1|"")
    tool_args=()
    ;;
  0|off|none)
    tool_args=()
    ;;
  *)
    tool_args=()
    ;;
esac

if [[ -x "$TOOL" && "$diag_level" != "0" && "$diag_level" != "off" && "$diag_level" != "none" ]]; then
  "$TOOL" "${tool_args[@]}" || true
fi

if run_cmd; then
  end_ts=$(date +%s)
  diag "status=PASS elapsed=$((end_ts-start_ts))s"
  exit 0
else
  status=$?
fi
end_ts=$(date +%s)
diag "status=FAIL exit=$status elapsed=$((end_ts-start_ts))s"

# Rerun-on-fail uses documented PoCL envs:
#   POCL_DEBUG=err,warn
#   POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES=1
#   POCL_CACHE_DIR=/tmp/vortex-pocl-cache-<project>-<timestamp>
# It also enables VX_OPENCL_DIAG for richer OpenCL error reporting in tests.
diag "rerun with diagnostics..."
export VX_OPENCL_DIAG=1
export POCL_DEBUG=${POCL_DEBUG:-err,warn}
export POCL_LEAVE_KERNEL_COMPILER_TEMP_FILES=1
cache_ts=$(date +%Y%m%d_%H%M%S)
export POCL_CACHE_DIR=${POCL_CACHE_DIR:-/tmp/vortex-pocl-cache-${PROJECT}-${cache_ts}}
mkdir -p "$POCL_CACHE_DIR"
diag "POCL_CACHE_DIR=$POCL_CACHE_DIR"

if [[ -x "$TOOL" ]]; then
  "$TOOL" --full || true
fi

run_cmd || true
exit $status
