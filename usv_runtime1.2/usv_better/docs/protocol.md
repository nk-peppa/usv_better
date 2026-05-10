# Communication Protocol

All communication frames are newline-delimited text sent over TCP to the gateway
host. The default gateway client port is usually `19520` unless fallback binding
is needed.

## Realtime Short Frame

Gateway input:

```text
R <seq> <F|L|R|S> [client_ts_ms]
RT <seq> <F|L|R|S> [client_ts_ms]
```

Processor-normalized format:

```text
R <seq> <tx_ms> <F|L|R|S>
```

Notes:

- Used for high-rate action control.
- Requires an active session started by `C START`.
- Legacy alias `RT` is accepted by the gateway while `legacy_alias=on` and normalized to `R`.
- Realtime actions keep FLRS semantics: `F` forward, `L` turn-left, `R` turn-right, `S` runtime stop.
- `C STOP` remains a session stop command and is separate from realtime `S`.

## SLAM Image Ingest Frame

```text
SLI <seq> <tx_ms> frame_id=<n> width=<w> height=<h> pixel_fmt=<GRAY8|RGB24|NV12> keyframe=<0|1> quality_hint=<0..100> payload_ref=<id>
SL <seq> <tx_ms> frame_id=<n> width=<w> height=<h> pixel_fmt=<GRAY8|RGB24|NV12> keyframe=<0|1> quality_hint=<0..100> payload_ref=<id>
```

Notes:

- Sent to the processor for SLAM ingest gating and D435i-backed executor handling.
- `payload_ref` is an opaque correlation/reference field in this demo.
- The current processing path captures from the D435i bridge instead of dereferencing payload data.
- Legacy alias `SL` is accepted by the gateway while `legacy_alias=on` and normalized to `SLI`.

## Row-Feature SLI Frame

The processor also accepts already-enriched row-feature `SLI` frames:

```text
SLI <seq> <tx_ms> frame_id=<n> width=<w> height=<h> pixel_fmt=ROW1 keyframe=<0|1> quality_hint=<0..100> payload_ref=<id> feature=row row_index=<n> channel_mode=<R|G|B|GRAY> stride=<n> sample_count=<n> payload_len=<n> payload_crc32=<n>
```

Row-feature frames are useful for validating the processor path without sending
full image payloads through the text protocol.

## Local Processor Smoke Input

After starting `build/bin/MainProcessor` in stdin/stdout mode, you can pipe a
small flow into it:

```bash
printf 'C START seq=1 ts=100 soft_hz=20 max_power=60 left_gain=1 right_gain=1 left_trim=0 right_trim=0 slam_max_fps=8 slam_timeout_ms=60 slam_max_groups=4 slam_min_quality=15 slam_drop_policy=newest row_ratio=0.333333 channel_mode=G sample_stride=2 max_rows=1 pack_mode=bin\nR 2 101 F\nSLI 3 102 frame_id=11 width=640 height=480 pixel_fmt=RGB24 keyframe=1 quality_hint=60 payload_ref=buf_11\nC STOP seq=4 ts=120\nq\n' | build/bin/MainProcessor
```
