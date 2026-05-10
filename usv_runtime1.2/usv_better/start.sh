#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BIN_DIR="${BIN_DIR:-${BUILD_DIR}/bin}"
PROCESSOR_PORT="${PROCESSOR_PORT:-19521}"
GATEWAY_HOST="${GATEWAY_HOST:-127.0.0.1}"
PROCESSOR_TIMEOUT_MS="${PROCESSOR_TIMEOUT_MS:-8000}"
CLIENT_PORT_MSG="19520"

MOCK_D435I="${MOCK_D435I:-0}"
USV_THRUSTER_SYSFS="${USV_THRUSTER_SYSFS:-1}"

usage() {
  cat <<'USAGE'
Usage: ./start.sh [--virtual|--mock] [--mock-d435i] [--no-thruster] [--help]

Default mode uses real thruster sysfs output and a real D435i camera.

Options:
  --virtual, --mock   Use mock D435i data and disable thruster sysfs output.
  --mock-d435i        Use mock D435i data only.
  --no-thruster       Disable thruster sysfs output only.
  --help              Show this help text.

Environment overrides:
  BUILD_DIR, BIN_DIR, PROCESSOR_PORT, GATEWAY_HOST, PROCESSOR_TIMEOUT_MS,
  MOCK_D435I, USV_THRUSTER_SYSFS
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --virtual|--mock)
      MOCK_D435I=1
      USV_THRUSTER_SYSFS=0
      ;;
    --mock-d435i)
      MOCK_D435I=1
      ;;
    --no-thruster)
      USV_THRUSTER_SYSFS=0
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

PROCESSOR_BIN="${BIN_DIR}/MainProcessor"
GATEWAY_BIN="${BIN_DIR}/CommunicationLayer"

if [[ ! -x "${PROCESSOR_BIN}" || ! -x "${GATEWAY_BIN}" ]]; then
  cat <<MISSING >&2
Required executables were not found in ${BIN_DIR}.
Run this first:

  bash scripts/build.sh
MISSING
  exit 2
fi

truthy() {
  case "${1,,}" in
    1|true|on|yes) return 0 ;;
    *) return 1 ;;
  esac
}

if truthy "${MOCK_D435I}"; then
  D435I_LABEL="MOCK"
else
  D435I_LABEL="REAL"
fi

case "${USV_THRUSTER_SYSFS,,}" in
  0|false|off|no) THRUSTER_LABEL="DISABLED / LOG-ONLY" ;;
  *) THRUSTER_LABEL="REAL SYSFS PWM" ;;
esac

cat <<BANNER
USV startup mode:
  Thruster output: ${THRUSTER_LABEL}
  D435i mode: ${D435I_LABEL}
  Processor port: ${PROCESSOR_PORT}
  Gateway processor target: ${GATEWAY_HOST}:${PROCESSOR_PORT}
BANNER

cleanup() {
  if [[ -n "${PROCESSOR_PID:-}" ]] && kill -0 "${PROCESSOR_PID}" 2>/dev/null; then
    kill "${PROCESSOR_PID}" 2>/dev/null || true
    wait "${PROCESSOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

processor_args=("--tcp" "--port" "${PROCESSOR_PORT}")
if truthy "${MOCK_D435I}"; then
  processor_args+=("--mock-d435i")
fi

USV_THRUSTER_SYSFS="${USV_THRUSTER_SYSFS}" "${PROCESSOR_BIN}" "${processor_args[@]}" &
PROCESSOR_PID=$!
sleep 1

cat <<READY
[READY] Starting gateway on client port ${CLIENT_PORT_MSG}; forwarding to ${GATEWAY_HOST}:${PROCESSOR_PORT}
[READY] Test client example: python3 client_test.py --host 127.0.0.1 --port ${CLIENT_PORT_MSG} --actions F,L,R,S --print-full-detail
READY

"${GATEWAY_BIN}" \
  --processor-host "${GATEWAY_HOST}" \
  --processor-port "${PROCESSOR_PORT}" \
  --processor-timeout-ms "${PROCESSOR_TIMEOUT_MS}"
