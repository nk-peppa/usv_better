#!/bin/bash
set -euo pipefail

PWM_CHIP_DIR="/sys/class/pwm/pwmchip0"
PWM1_DIR="$PWM_CHIP_DIR/pwm1"
PWM2_DIR="$PWM_CHIP_DIR/pwm2"
PERIOD_NS=20000000
NEUTRAL_NS=0
SPAN_NS=5000000


sysfs_write() {
  local path="$1"
  local value="$2"
  local retries="${3:-3}"
  local i
  for ((i=1; i<=retries; i++)); do
    if printf '%s\n' "$value" > "$path"; then
      return 0
    fi
    sleep 0.01
  done
  echo "ERR: failed to write ${value} -> ${path} after ${retries} attempts" >&2
  return 1
}

ensure_pwm_exported() {
  local channel="$1"
  local pwm_dir="$2"
  local retries="${3:-5}"
  local i
  if [[ -d "$pwm_dir" ]]; then
    return 0
  fi

  # Export may transiently fail (e.g. EBUSY race); retry and re-check directory.
  for ((i=1; i<=retries; i++)); do
    sysfs_write "$PWM_CHIP_DIR/export" "$channel" || true
    sleep 0.1
    if [[ -d "$pwm_dir" ]]; then
      return 0
    fi
  done
  return 1
}

clamp_duty_to_period() {
  local duty="$1"
  awk -v d="$duty" -v p="$PERIOD_NS" 'BEGIN {
    max = p - 1;
    if (max < 0) max = 0;
    if (d < 0) d = 0;
    if (d > max) d = max;
    printf "%.0f", d;
  }'
}

HW_MODE=0
if [[ "${1:-}" == "--hw" ]]; then
  HW_MODE=1
fi

percent_to_duty_ns() {
  local percent="$1"
  awk -v p="$percent" -v n="$NEUTRAL_NS" -v s="$SPAN_NS" 'BEGIN {
    if (p < 0) p = 0;
    if (p > 100) p = 100;
    duty = n + (p / 100.0) * s;
    min = n;
    max = n + s;
    if (duty < min) duty = min;
    if (duty > max) duty = max;
    printf "%.0f", duty;
  }'
}

write_duty_pair() {
  local left_percent="$1"
  local right_percent="$2"
  local left_duty right_duty

  left_duty="$(percent_to_duty_ns "$left_percent")"
  right_duty="$(percent_to_duty_ns "$right_percent")"

  left_duty="$(clamp_duty_to_period "$left_duty")"
  right_duty="$(clamp_duty_to_period "$right_duty")"

  if [[ $HW_MODE -eq 1 ]]; then
    sysfs_write "$PWM1_DIR/duty_cycle" "$left_duty"
    sysfs_write "$PWM2_DIR/duty_cycle" "$right_duty"
  fi

  echo "left=${left_percent}% -> duty_ns=${left_duty}, right=${right_percent}% -> duty_ns=${right_duty}"
}

cleanup() {
  if [[ $HW_MODE -eq 1 ]]; then
    sysfs_write "$PWM1_DIR/duty_cycle" "0" || true
    sysfs_write "$PWM2_DIR/duty_cycle" "0" || true
    sysfs_write "$PWM1_DIR/enable" "0" || true
    sysfs_write "$PWM2_DIR/enable" "0" || true
    if [[ -w "$PWM_CHIP_DIR/unexport" ]]; then
      sysfs_write "$PWM_CHIP_DIR/unexport" "1" || true
      sysfs_write "$PWM_CHIP_DIR/unexport" "2" || true
    fi
  fi
}

trap cleanup EXIT

if [[ $HW_MODE -eq 1 ]]; then
  if [[ $EUID -ne 0 ]]; then
    echo "HW mode requires root access to write PWM sysfs." >&2
    exit 2
  fi
  if [[ ! -d "$PWM_CHIP_DIR" ]]; then
    echo "PWM chip directory not found: $PWM_CHIP_DIR" >&2
    exit 3
  fi
  ensure_pwm_exported "1" "$PWM1_DIR" || { echo "Failed to export pwm1." >&2; exit 4; }
  ensure_pwm_exported "2" "$PWM2_DIR" || { echo "Failed to export pwm2." >&2; exit 4; }
  if [[ ! -d "$PWM1_DIR" || ! -d "$PWM2_DIR" ]]; then
    echo "Failed to export pwm channels or they do not appear." >&2
    ls -la "$PWM_CHIP_DIR"
    exit 4
  fi
  # Some PWM drivers require disable before period updates.
  sysfs_write "$PWM1_DIR/enable" "0" || true
  sysfs_write "$PWM1_DIR/duty_cycle" "0"
  sysfs_write "$PWM1_DIR/period" "$PERIOD_NS"
  sysfs_write "$PWM1_DIR/enable" "1"

  sysfs_write "$PWM2_DIR/enable" "0" || true
  sysfs_write "$PWM2_DIR/duty_cycle" "0"
  sysfs_write "$PWM2_DIR/period" "$PERIOD_NS"
  sysfs_write "$PWM2_DIR/enable" "1"
fi

echo "Running in $([[ $HW_MODE -eq 1 ]] && echo HW || echo dry-run) mode"
echo "Input format: <left_percent> <right_percent>"
echo "Percent range: 0 to 100 only, mapped from neutral duty ${NEUTRAL_NS}ns up to ${NEUTRAL_NS}ns + ${SPAN_NS}ns"
echo "Type q to quit."
echo "Safety: 0% always maps to duty_ns=0 (hard stop)."
echo "Hardware pin mapping: pwm1 -> pin8 (PH3), pwm2 -> pin10 (PH2)"

step=0
while true; do
  printf 'thruster> '
  if ! IFS= read -r line; then
    break
  fi
  if [[ -z "$line" ]]; then
    continue
  fi
  if [[ "$line" == "q" || "$line" == "Q" ]]; then
    break
  fi

  read -r left_percent right_percent <<< "$line" || true
  if [[ -z "${left_percent:-}" || -z "${right_percent:-}" ]]; then
    echo "WARN: expected two numbers, for example: 30 25"
    continue
  fi

  if ! [[ "$left_percent" =~ ^-?[0-9]+([.][0-9]+)?$ && "$right_percent" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
    echo "WARN: expected numeric percentages"
    continue
  fi

  if awk -v l="$left_percent" -v r="$right_percent" 'BEGIN { exit !((l >= 0 && l <= 100) && (r >= 0 && r <= 100)) }'; then
    :
  else
    echo "WARN: negative values are not allowed; use 0..100 only"
    continue
  fi

  step=$((step + 1))
  echo "step=${step}"
  write_duty_pair "$left_percent" "$right_percent"
done

echo "done"
