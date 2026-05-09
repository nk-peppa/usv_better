#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
BIN_DIR="${BIN_DIR:-${BUILD_DIR}/bin}"
TEST_BIN="${BIN_DIR}/SelD435iSmokeTest"

if [[ ! -x "${TEST_BIN}" \
      || "${ROOT_DIR}/tests/SelD435iSmokeTest.cc" -nt "${TEST_BIN}" \
      || "${ROOT_DIR}/src/SlamExecutionLayer.cc" -nt "${TEST_BIN}" \
      || "${ROOT_DIR}/include/usv/SlamExecutionLayer.h" -nt "${TEST_BIN}" ]]; then
  mkdir -p "${BIN_DIR}"

  REALSENSE_INCLUDE="${REALSENSE_INCLUDE:-}"
  REALSENSE_LIB="${REALSENSE_LIB:-}"

  include_flag=()
  lib_flag=(-lrealsense2)
  if [[ -n "${REALSENSE_INCLUDE}" ]]; then
    include_flag=(-I"${REALSENSE_INCLUDE}")
  fi
  if [[ -n "${REALSENSE_LIB}" ]]; then
    lib_flag=(-L"${REALSENSE_LIB}" -lrealsense2)
  fi

  echo "SelD435iSmokeTest executable was not found; compiling it now."
  "${CXX:-g++}" ${CXXFLAGS:--std=c++17 -O2 -Wall -Wextra -pedantic} \
    -I"${ROOT_DIR}/include" "${include_flag[@]}" \
    "${ROOT_DIR}/tests/SelD435iSmokeTest.cc" \
    "${ROOT_DIR}/src/SlamExecutionLayer.cc" \
    "${lib_flag[@]}" \
    -pthread \
    -o "${TEST_BIN}"
fi

"${TEST_BIN}" "$@"
