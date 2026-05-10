#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
if [[ -n "${BUILD_JOBS:-}" ]]; then
  cmake --build "${BUILD_DIR}" --parallel "${BUILD_JOBS}"
else
  cmake --build "${BUILD_DIR}"
fi

cat <<MSG

Build complete.
Executables are in: ${BUILD_DIR}/bin
  - CommunicationLayer
  - MainProcessor
  - ThrusterActuator
MSG
