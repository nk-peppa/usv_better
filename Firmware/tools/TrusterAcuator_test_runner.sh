#!/usr/bin/env bash
# Virtual core test helper for TrusterAcuator
# - Single-file tool placed in tools/
# - If TA binary not found, exits with clear compile instructions
# - Interactive: 输入两个 0..100 的百分比 (left right)，或输入 q 退出

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Candidate binary names to check (workspace-relative and PATH)
CANDIDATES=("$ROOT_DIR/truster_actuator_demo" "$ROOT_DIR/TrusterAcuator" "$ROOT_DIR/truster_actuator" "$ROOT_DIR/truster")
TA_BIN=""
for c in "${CANDIDATES[@]}"; do
  if [ -x "$c" ]; then
    TA_BIN="$c"
    break
  fi
done
if [ -z "$TA_BIN" ]; then
  if command -v truster_actuator_demo >/dev/null 2>&1; then
    TA_BIN="$(command -v truster_actuator_demo)"
  elif command -v TrusterAcuator >/dev/null 2>&1; then
    TA_BIN="$(command -v TrusterAcuator)"
  fi
fi

if [ -z "$TA_BIN" ]; then
  cat <<EOF >&2
ERROR: 没有在仓库或 PATH 中找到 TrusterAcuator 可执行文件。
请在仓库根目录编译它，示例命令（在 /root/usv_better）:

  g++ -std=c++17 -O2 -Wall TrusterAcuator.cc -o truster_actuator_demo

或根据项目的构建系统调整编译命令。编译完成后将可执行文件放在仓库根目录或 PATH 中，脚本即会自动使用它。

此脚本不会修改现有核心代码；如需直接控制 PWM，可使用 tools/thruster_test_runner.sh。
EOF
  exit 2
fi

echo "Using TrusterAcuator binary: $TA_BIN"
echo "交互式模式：输入两列 0..100 的百分比（left right），回车发送；输入 q 退出。"
echo "Safety: 0 0 = hard stop (duty 0)."

print_help() {
  cat <<EOF
示例：
  30 40    # 左推进器 30%, 右推进器 40%
  0 0      # 停止
  q        # 退出
EOF
}

print_help

while true; do
  printf "> "
  if ! IFS= read -r line; then
    echo
    break
  fi
  line="$(echo "$line" | tr -d '\r')"
  if [ "$line" = "q" ] || [ "$line" = "Q" ]; then
    echo "退出。"
    break
  fi
  if [ -z "$line" ]; then
    continue
  fi
  # 允许用空格或逗号分隔
  left=$(echo "$line" | awk -F'[ ,]+' '{print $1}')
  right=$(echo "$line" | awk -F'[ ,]+' '{print $2}')
  if [ -z "$right" ]; then
    echo "需要两个值：left right（0..100）。输入 'h' 查看帮助。"
    continue
  fi

  # 验证是数字且在 0..100
  re='^[0-9]+([.][0-9]+)?$'
  if ! [[ $left =~ $re ]] || ! [[ $right =~ $re ]]; then
    echo "输入错误：请使用数字，范围 0..100。"
    continue
  fi
  # 强制为整数（或者保留小数）
  # bounds
  left_val=$(awk "BEGIN{printf \"%.0f\", ($left<0?0:($left>100?100:$left))}")
  right_val=$(awk "BEGIN{printf \"%.0f\", ($right<0?0:($right>100?100:$right))}")

  echo "发送到 TrusterAcuator: left=$left_val% right=$right_val%"

  # 调用 TA，可根据 TA 的实际参数接口修改
  set +e
  "$TA_BIN" "$left_val" "$right_val"
  rc=$?
  set -e
  if [ $rc -ne 0 ]; then
    echo "TrusterAcuator 返回非零状态: $rc" >&2
  fi
done

echo "虚拟核心脚本结束。"
