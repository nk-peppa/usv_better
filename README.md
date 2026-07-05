# USV Edge Demo

## Overview
This workspace is a lightweight three-layer demo for USV edge validation. It focuses on the gateway/processor/SLAM-executor contract and provides runnable TCP flows, SLAM ingest simulation, and thruster PWM validation.

Core layers:
- Communication layer (gateway): TCP listener, line protocol parsing, forwarding, and gateway management commands.
- Main processor: control, SLAM ingest gating, D435i capture orchestration, fusion semantics, health, and rollback signals.
- SLAM executor bridge: Intel RealSense D435i capture by default with an explicit mock D435i mode for local smoke testing.

## Repository Layout

```
usv_better/
	Firmware/
		CommunicationLayer.cc
		MainProcessor.cc
		SlamExecutionLayer.cc
		SlamExecutionLayer.h
		SelD435iSmokeTest.cc
		ThrusterActuator.cc
		tools/
			D435iImuProbe.cc
			run_d435i_smoke_test.sh
			run_tcp_stack.sh
			thruster_test_runner.sh
			ThrusterActuator_test_runner.sh
			usv_tcp_full_flow_client.py
```

The sibling workspace folder `usv_test/` is currently empty and not used by this demo.

## Key Files
- `CommunicationLayer.cc`: TCP gateway adapter demo with line-oriented protocol handling and gateway management commands.
- `MainProcessor.cc`: processor parser/dispatcher demo, D435i capture orchestration, health output, `--d435i-selftest` entrypoint, and TCP backend mode.
- `SlamExecutionLayer.h/.cc`: hub-to-SLAM execution bridge, RealSense D435i capture bridge, mock D435i bridge, RGB/depth/gyro row feature enrichment.
- `SelD435iSmokeTest.cc`: 10-second D435i smoke test with `--mock`/`+mock` support.
- `ThrusterActuator.cc`: actuator mapping/output demo and sysfs PWM writer.
- `tools/thruster_test_runner.sh`: PWM validation entrypoint for dry-run and hardware sysfs writeout.
- `tools/ThrusterActuator_test_runner.sh`: legacy interactive helper that invokes `ThrusterActuator` if present.
- `tools/run_tcp_stack.sh`: helper to run gateway + processor together in TCP mode.
- `tools/usv_tcp_full_flow_client.py`: TCP client for full-flow tests (interactive on Windows).
- `tools/D435iImuProbe.cc`: D435i IMU diagnostics probe — validates gyro/accelerometer streams, checks sensor validity flags, and reports timestamp health.
- `tools/run_d435i_smoke_test.sh`: One-shot wrapper for the D435i smoke test binary (`sel_d435i_smoke`).

## Dependencies
- RealSense: targets that include `SlamExecutionLayer.cc` or `SlamExecutionLayer.h` link against Intel RealSense (`librealsense2`) and include `<librealsense2/rs.hpp>`.
- No RealSense needed: targets that only build `CommunicationLayer.cc` or `ThrusterActuator.cc`.

If your RealSense installation is not in the default search path, add the appropriate `-I`, `-L`, and runtime library path flags.

## Build
Run these commands from `Firmware/`:

```bash
g++ -std=c++17 -Wall -Wextra -pedantic CommunicationLayer.cc -o communication_layer_demo
g++ -std=c++17 -Wall -Wextra -pedantic MainProcessor.cc SlamExecutionLayer.cc -lrealsense2 -o main_processor_demo
g++ -std=c++17 -Wall -Wextra -pedantic SelD435iSmokeTest.cc SlamExecutionLayer.cc -lrealsense2 -o sel_d435i_smoke
g++ -std=c++17 -Wall -Wextra -pedantic ThrusterActuator.cc -o thruster_actuator_demo
```

## Runtime Entry Points

### Communication gateway
```bash
./communication_layer_demo
```

Listens on `0.0.0.0`. Default port is `19520`; fallback order is `9773`, `11514`, `23758`, `52019`.

Environment variables:
- `USV_PROCESSOR_HOST` (default `127.0.0.1`)
- `USV_PROCESSOR_PORT` (default `19521`)

CLI overrides:
```bash
./communication_layer_demo --processor-host 127.0.0.1 --processor-port 19521 --processor-timeout-ms 1000
```

### Main processor
Stdin/stdout mode:

```bash
./main_processor_demo
```

TCP backend mode:

```bash
./main_processor_demo --tcp --port 19521
```

By default, realtime commands (F/L/R/S) write PWM duty cycles to sysfs for thruster control. To disable hardware output and log-only mode:

```bash
USV_THRUSTER_SYSFS=0 ./main_processor_demo --tcp --port 19521
```

Mock D435i (no hardware required):

```bash
./main_processor_demo --tcp --port 19521 --mock-d435i
```

Combine both:

```bash
USV_THRUSTER_SYSFS=0 ./main_processor_demo --tcp --port 19521 --mock-d435i
```

The `SLI` processing path attempts D435i capture through the SLAM execution bridge. On a host without a D435i device and RealSense runtime, `SLI` commands can return capture errors unless mock mode is enabled.

### D435i self-test and smoke test
Processor self-test:

```bash
./main_processor_demo --d435i-selftest
```

Dedicated 10-second smoke test:

```bash
./sel_d435i_smoke
./sel_d435i_smoke --mock
```

`--mock`/`+mock` enables the mock D435i bridge for local validation without a camera.

## TCP Stack Helper
Run processor in TCP mode and start the gateway in one script:

```bash
bash tools/run_tcp_stack.sh
```

Default behavior: real thruster sysfs output + real D435i. Script prints mode banners at startup.

For lab testing with mock D435i and real sysfs output:

```bash
MOCK_D435I=1 bash tools/run_tcp_stack.sh
```

For log-only mode (no hardware output, real D435i):

```bash
USV_THRUSTER_SYSFS=0 bash tools/run_tcp_stack.sh
```

For lab simulation (log-only + mock D435i):

```bash
USV_THRUSTER_SYSFS=0 MOCK_D435I=1 bash tools/run_tcp_stack.sh
```

The script prints a ready banner and forwards the gateway to the processor. Follow the on-screen instructions to run the client.

## TCP Client
The Python client sends a start config, realtime commands, and SLAM input:

```bash
python3 tools/usv_tcp_full_flow_client.py --host 127.0.0.1 --port 19520 --actions F,L,R,S --print-full-detail
```

Windows interactive mode uses WASD (requires `msvcrt`):

```bash
python tools/usv_tcp_full_flow_client.py --host <device-ip> --interactive --print-full-detail
```

## Thruster PWM Validation
The current thruster validation flow is consolidated into one script:

```bash
bash tools/thruster_test_runner.sh
sudo bash tools/thruster_test_runner.sh --hw
```

Dry-run mode (no hardware write):

```bash
bash tools/thruster_test_runner.sh
```

Hardware mode (writes to sysfs; requires sudo):

```bash
sudo bash tools/thruster_test_runner.sh --hw
```

The script writes PWM duty cycles directly to sysfs and no longer compiles or invokes a temporary C++ runner. No log file is generated by default. It only accepts positive thrust percentages in the range `0..100` for both channels.

### 常见“几步后不响应”故障排查（PWM sysfs）
如果你把 `duty_cycle` 文件句柄长期保持打开（例如在循环外 `std::ofstream duty(...);`，循环内持续 `<<`），在部分内核/驱动上会出现“前几次有效，随后看起来不再响应”的现象。

建议：
- 每次写入都重新打开并写入（脚本 `>` 重定向就是这种方式）。
- 或在每次写前明确 `seekp(0)` 并检查 `good()/fail()`，任何一次失败都立刻告警。
- 在写入路径上保留错误检查，避免静默失败。

此外需要确认：
- `duty_cycle` 不应超过 `period`（有些驱动对等于 `period` 也会拒绝或行为不稳定）。
- `export` 已存在时应容错处理（避免“设备忙”导致初始化分支未完整执行）。
- `enable`、`period`、`duty_cycle` 的写入顺序和返回状态都要检查。

### PWM topology
Orange Pi Zero 3 pin topology used by this workspace:
- Processor PWM sysfs path for channel 1: `/sys/class/pwm/pwmchip0/pwm1`
- Processor PWM sysfs path for channel 2: `/sys/class/pwm/pwmchip0/pwm2`
- Channel 1 duty file: `/sys/class/pwm/pwmchip0/pwm1/duty_cycle`
- Channel 2 duty file: `/sys/class/pwm/pwmchip0/pwm2/duty_cycle`
- Channel 1 hardware pin: `PH3`
- Channel 2 hardware pin: `PH2`
- 40-pin header mapping: `PH3 -> pin8`, `PH2 -> pin10`

Runtime mapping used by the script and current actuator demo:
- Neutral duty: `0 ns`
- Period: `20000000 ns` (`50 Hz`)
- Input format: left/right percentage values in the range `0` to `100`
- Negative values are rejected by the script; there is no reverse thrust path in the validation flow
- `0%` maps to duty `0 ns` (hard stop)
- Any positive value maps to duty `20000000 ns` in the current binary validation flow

`ThrusterActuator.cc` still contains shaping configuration fields such as `duty_span_ns` for future hardware behavior, but current output behavior is binary: non-positive input maps to `0 ns`; positive input maps to the full PWM period.

## Communication Frames
All communication frames are newline-delimited text sent over TCP to `127.0.0.1:<bound_port>` or the host IP that runs the gateway. The default bound port is usually `19520` unless fallback binding was needed.

### 1) Realtime short frame
Gateway input:

```text
R <seq> <F|L|R|S> [client_ts_ms]
RT <seq> <F|L|R|S> [client_ts_ms]
```

Processor-normalized format:

```text
R <seq> <tx_ms> <F|L|R|S>
```

- Used for high-rate action control.
- Requires active session started by `C START`.
- Legacy alias `RT` is accepted by the gateway while `legacy_alias=on` and normalized to `R`.
- Realtime commands keep FLRS semantics: `F` forward, `L` turn-left, `R` turn-right, `S` runtime stop.
- `C STOP` remains a session stop command and is separate from realtime `S`.

### 2) SLAM image ingest frame
```text
SLI <seq> <tx_ms> frame_id=<n> width=<w> height=<h> pixel_fmt=<GRAY8|RGB24|NV12> keyframe=<0|1> quality_hint=<0..100> payload_ref=<id>
SL <seq> <tx_ms> frame_id=<n> width=<w> height=<h> pixel_fmt=<GRAY8|RGB24|NV12> keyframe=<0|1> quality_hint=<0..100> payload_ref=<id>
```

- Sent to the processor for SLAM ingest gating and D435i-backed executor handling.
- `payload_ref` is currently an opaque correlation/reference field in this demo. The present processing path captures from the D435i bridge instead of dereferencing the payload data.
- Legacy alias `SL` is accepted by the gateway while `legacy_alias=on` and normalized to `SLI`.

### 3) Row-feature SLI frame (processor-supported)
The processor also accepts already-enriched row-feature `SLI` frames:

```text
SLI <seq> <tx_ms> frame_id=<n> width=<w> height=<h> pixel_fmt=ROW1 keyframe=<0|1> quality_hint=<0..100> payload_ref=<id> feature=row row_index=<n> channel_mode=<R|G|B|GRAY> stride=<n> sample_count=<n> payload_len=<n> payload_crc32=<n>
```

Required row metadata fields are `feature=row`, `row_index`, `channel_mode`, `stride`, `sample_count`, `payload_len`, and `payload_crc32`.

### 4) Config long frame
```text
C START seq=<n> ts=<ms> soft_hz=<1..100> max_power=<v> left_gain=<v> right_gain=<v> left_trim=<v> right_trim=<v> slam_max_fps=<1..30> slam_timeout_ms=<1..200> slam_max_groups=<1..64> slam_min_quality=<0..100> slam_drop_policy=<reject|oldest|newest> row_ratio=<0..1> channel_mode=<R|G|B|GRAY> sample_stride=<1..64> max_rows=<1..8> pack_mode=<bin|hex>
C STOP seq=<n> ts=<ms>
```

- Legacy aliases `CS` and `CE` are accepted by the gateway while `legacy_alias=on` and normalized to `C START` and `C STOP`.

### Unsupported / deferred frame
`SR <seq> ...` row extraction requests are not implemented in the gateway. Send processor-supported row-feature data as `SLI feature=row ...` instead, or use normal `SLI` to trigger the current D435i-backed path.

## ACK Format
Processor ACK format:

```text
ACK <OK|ERR> seq=<n> up_ms=<n> down_ms=<n> tag=<code> detail=<code_or_payload>
```

Gateway ACK format after forwarding or gateway-local management/error handling:

```text
ACK <OK|ERR> seq=<n> up_ms=<n> down_ms=<n> tag=<code> detail=<code_or_payload> gw_trace=<id> gw_route=<result>
```

- Processor `tag` examples: `cfg_start`, `rt_apply`, `sli_ok`, `sli_fail`, `slam_row_ok`, `slam_row_fail`.
- Gateway `tag` examples: `gw_bad_msg`, `gw_route_timeout`, `gw_switch`, `gw_rollback`.
- `up_ms` is computed from the command transmit timestamp when possible; otherwise it can be `0`.
- `down_ms` is local processing/downlink time measured by the processor or parsed from processor ACK by the gateway.
- `gw_trace` is the gateway-side trace identifier used for cross-layer correlation.
- `gw_route` describes gateway routing outcome such as `forwarded`, `parse_reject`, `route_timeout`, or `mgmt`.

## Runtime Management
Gateway management commands:

```text
GW HEALTH
GW SWITCH legacy_alias=<on|off> sli_enabled=<on|off> route_timeout_ms=<1..5000>
GW ROLLBACK
```

Processor health command:

```text
HEALTH
```

Notes:
- `GW SWITCH` is the compatibility/traffic governance switch entry used by canary migration.
- `GW ROLLBACK` applies conservative defaults immediately: `legacy_alias=on`, `sli_enabled=on`, `route_timeout_ms=80`.
- `GW HEALTH` exposes gateway rollout SLO signals including `ack_p95_ms`, `ack_p99_ms`, timeout/drop counts, and rollback recommendation.
- `HEALTH` exposes processor state, ACK counters/latency percentiles, SLI drops, executor timeouts, and rollback recommendation.

## D435i IMU Probe Tool (`v1.2.0.000`)
The IMU probe binary provides stand-alone diagnostics for the Intel RealSense D435i internal IMU:

```bash
g++ -std=c++17 -Wall -Wextra -pedantic tools/D435iImuProbe.cc SlamExecutionLayer.cc -lrealsense2 -o d435i_imu_probe
./d435i_imu_probe
```

Output includes:
- Gyro and accelerometer stream health (frame rate, drop count)
- Per-axis validity flags with sustained-invalidity warnings
- Timestamp delta diagnostics (drift, gaps, jitter histogram)
- Summary pass/fail verdict suitable for CI gating

### IMU Enhancement (`v1.2.0.000`)
`SlamExecutionLayer` now tracks gyroscope sensor validity on every captured frame:
- Invalid gyro frames are flagged and counted; sustained invalidity triggers a processor health warning.
- Extended capture timeout range supports longer RealSense pipeline startup on constrained hardware.
- Processor health output (`HEALTH`) includes an `imu_validity_pct` metric and gyro invalidity counters.

## Release Gates
Block release if any condition is true:
- Protocol consistency checks fail for `C START/C STOP/R/RT/SL/SLI`.
- `rollback_recommended=1` in either gateway or processor health output for a sustained window.
- `ack_p95_ms` exceeds 80 ms in canary gateway traffic or 100 ms in sustained processor traffic.
- Route timeout, parse failure, processor ACK error, or executor timeout rate exceeds the configured threshold.
- D435i smoke test fails in a hardware-required deployment environment.

## Canary and Rollback Flow
1. Stage A (lab): keep `legacy_alias=on`, verify full regression.
2. Stage B (single vessel): disable alias gradually with `GW SWITCH legacy_alias=off`.
3. Stage C (fleet): keep alias off, monitor health, and roll forward only when stable.
4. Rollback trigger: when timeout/error SLO is violated, execute `GW ROLLBACK` and re-enable compatible path.

## Quick Test
Processor stdin/stdout flow. This command uses `SLI`, so it requires a working D435i path; on a non-camera host, expect a capture error for the `SLI` command.

```bash
printf 'C START seq=1 ts=100 soft_hz=20 max_power=60 left_gain=1 right_gain=1 left_trim=0 right_trim=0 slam_max_fps=8 slam_timeout_ms=60 slam_max_groups=4 slam_min_quality=15 slam_drop_policy=newest row_ratio=0.333333 channel_mode=G sample_stride=2 max_rows=1 pack_mode=bin\nR 2 101 F\nSLI 3 102 frame_id=11 width=640 height=480 pixel_fmt=RGB24 keyframe=1 quality_hint=60 payload_ref=buf_11\nC STOP seq=4 ts=120\nq\n' | ./main_processor_demo
```

Mock D435i smoke test for local validation without hardware:

```bash
./sel_d435i_smoke --mock
```

Gateway process:

```bash
./communication_layer_demo
```

In another terminal, send commands to the bound port shown by `communication_layer_demo`:

```bash
printf 'C START seq=1 ts=100 soft_hz=20 max_power=60 left_gain=1 right_gain=1 left_trim=0 right_trim=0 slam_max_fps=8 slam_timeout_ms=60 slam_max_groups=4 slam_min_quality=15 slam_drop_policy=newest row_ratio=0.333333 channel_mode=G sample_stride=2 max_rows=1 pack_mode=bin\nR 2 F 101\nSLI 3 102 frame_id=11 width=640 height=480 pixel_fmt=RGB24 keyframe=1 quality_hint=70 payload_ref=buf_11\nC STOP seq=4 ts=120\nq\n' | nc 127.0.0.1 19520
```

Gateway management:

```bash
printf 'GW HEALTH\nGW SWITCH legacy_alias=off route_timeout_ms=40\nGW HEALTH\nGW ROLLBACK\nGW HEALTH\nq\n' | nc 127.0.0.1 19520
```

Processor health through gateway forwarding:

```bash
printf 'HEALTH\nq\n' | nc 127.0.0.1 19520
```

Direct processor row-feature example:

```bash
printf 'C START seq=1 ts=100 soft_hz=20 max_power=60 left_gain=1 right_gain=1 left_trim=0 right_trim=0 slam_max_fps=8 slam_timeout_ms=60 slam_max_groups=4 slam_min_quality=15 slam_drop_policy=newest row_ratio=0.333333 channel_mode=G sample_stride=2 max_rows=1 pack_mode=bin\nSLI 2 101 frame_id=11 width=64 height=1 pixel_fmt=ROW1 keyframe=1 quality_hint=70 payload_ref=row_11 feature=row row_index=16 channel_mode=G stride=2 sample_count=32 payload_len=32 payload_crc32=12345\nC STOP seq=3 ts=120\nq\n' | ./main_processor_demo
```

## Processor-Executor Reserved Interface
Core session/execution interface:
- `PushConfig(session_id, config_version, slam_cfg)`
- `ProcessFrame(session_id, frame_meta, timeout_ms)`
- `StopSession(session_id)`
- `GetHealth()`

D435i capture interface:
- `InitializeD435i(error)`
- `EnableMockD435i(enabled)`
- `ShutdownD435i()`
- `CaptureD435iFrame(timeout_ms, slam_cfg, out, error)`
- `IsD435iReady()`

## Notes
- Communication layer no longer decides SLAM business semantics.
- Processing layer controls realtime rate limiting, SLAM ingest gating, drop policy, timeout classification, D435i capture orchestration, and fusion output.
- The current executor bridge performs D435i-backed row feature enrichment and simulated SLAM result generation; mock D435i is available only when explicitly enabled.
- Gateway `SR` row extraction is deferred and should not be used as a documented runnable path until implemented.
