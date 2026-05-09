#!/usr/bin/env bash
# Interactive helper for the ThrusterActuator executable.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="${BIN_DIR:-${ROOT_DIR}/build/bin}"

CANDIDATES=(
  "${BIN_DIR}/ThrusterActuator"
  "${ROOT_DIR}/ThrusterActuator"
  "${ROOT_DIR}/build/ThrusterActuator"
)
TA_BIN=""
for candidate in "${CANDIDATES[@]}"; do
  if [[ -x "${candidate}" ]]; then
    TA_BIN="${candidate}"
    break
  fi
done

if [[ -z "${TA_BIN}" ]] && command -v ThrusterActuator >/dev/null 2>&1; then
  TA_BIN="$(command -v ThrusterActuator)"
fi

if [[ -z "${TA_BIN}" ]]; then
  cat <<EOF_MISSING >&2
ERROR: ThrusterActuator executable was not found.
Build the project first:

  bash scripts/build.sh

Expected executable path:

  build/bin/ThrusterActuator

For direct PWM sysfs validation, use:

  bash tools/thruster_test_runner.sh
EOF_MISSING
  exit 2
fi

echo "Using ThrusterActuator binary: ${TA_BIN}"
echo "Interactive mode: enter F, L, R, or S, then press Enter. Enter q to quit."
echo "Binary output rule: S -> 0 ns; F/L/R -> ON (period - 1)."
echo "Safety: S is a hard stop with duty 0."

while true; do
  printf "> "
  if ! IFS= read -r line; then
    echo
    break
  fi
  line="$(echo "${line}" | tr -d '\r')"
  case "${line}" in
    q|Q)
      echo "Exiting."
      break
      ;;
    "")
      continue
      ;;
  esac

  cmd="$(echo "${line}" | awk '{print toupper($1)}')"
  case "${cmd}" in
    F|L|R|S)
      set +e
      "${TA_BIN}" "${cmd}"
      rc=$?
      set -e
      ;;
    *)
      echo "Unknown command: use F/L/R/S or q."
      continue
      ;;
  esac

  if [[ ${rc} -ne 0 ]]; then
    echo "ThrusterActuator returned non-zero status: ${rc}" >&2
  fi
done

echo "ThrusterActuator helper finished."
