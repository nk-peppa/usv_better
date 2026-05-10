#!/usr/bin/env python3
"""Real D435i observation adapter for the semantic PPO actor.

This script connects to the USV gateway, requests fused D435i ACK details, turns
raw `r=` mask values and `depth=` Z16 values into the 195-D observation expected
by PPO_USV_Hunter_Semantic3_gpu.py, then prints actor action probabilities.

By default it sends action `S` for every probe frame so the boat stays stopped.
It does not autonomously control the boat unless `--send-policy-action` is set.
"""

from __future__ import annotations

import argparse
import socket
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


ACTION_LABELS = ("F", "S", "L", "R")


class Actor(nn.Module):
    def __init__(self, input_dim: int = 195, hidden: int = 256, n_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_ack_fields(ack: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    parts = ack.strip().split()
    if len(parts) >= 2:
        fields["_kind"] = parts[0]
        fields["_status"] = parts[1]
    for part in parts[2:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def parse_number_list(value: str, cast) -> List:
    if not value:
        return []
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            out.append(cast(item))
        except ValueError:
            continue
    return out


def parse_fused_detail(detail: str) -> Dict[str, str]:
    if "d435i_ok|" not in detail:
        raise ValueError(f"ACK detail does not contain d435i_ok payload: {detail}")
    _, slam = detail.split("d435i_ok|", 1)
    tokens = slam.split("_")
    kv: Dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        kv[key] = value
    return kv


class UsvGatewayClient:
    def __init__(self, host: str, port: int, timeout: float):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.seq = 1
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        print(f"[CONNECTED] gateway={self.host}:{self.port}")

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def next_seq(self) -> int:
        seq = self.seq
        self.seq += 1
        return seq

    def send_line(self, line: str) -> str:
        if self.sock is None:
            raise RuntimeError("socket is not connected")
        self.sock.sendall((line + "\n").encode("utf-8"))
        chunks = []
        while True:
            ch = self.sock.recv(1)
            if not ch:
                raise ConnectionError("gateway closed connection before ACK")
            if ch == b"\n":
                break
            if ch != b"\r":
                chunks.append(ch)
        return b"".join(chunks).decode("utf-8", errors="replace")

    def gw_switch(self, route_timeout_ms: int) -> str:
        return self.send_line(
            f"GW SWITCH legacy_alias=on sli_enabled=on route_timeout_ms={route_timeout_ms}"
        )

    def c_start(self, args: argparse.Namespace) -> str:
        seq = self.next_seq()
        return self.send_line(
            f"C START seq={seq} ts={now_ms()} "
            f"soft_hz={args.soft_hz} "
            f"max_power={args.max_power} "
            "left_gain=1 right_gain=1 left_trim=0 right_trim=0 "
            f"slam_max_fps={args.slam_max_fps} "
            f"slam_timeout_ms={args.slam_timeout_ms} "
            "slam_max_groups=8 slam_min_quality=10 slam_drop_policy=newest "
            f"row_ratio={args.row_ratio} "
            "channel_mode=R "
            f"sample_stride={args.sample_stride} "
            f"max_rows={args.max_rows} "
            "pack_mode=bin"
        )

    def rt(self, action: str) -> str:
        return self.send_line(f"R {self.next_seq()} {action} {now_ms()}")

    def c_stop(self) -> str:
        return self.send_line(f"C STOP seq={self.next_seq()} ts={now_ms()}")


class RealObsAdapter:
    def __init__(
        self,
        bins: int,
        stack_frames: int,
        max_range_mm: float,
        r_threshold: float,
        depth_stat: str,
    ):
        self.bins = bins
        self.stack_frames = stack_frames
        self.max_range_mm = max_range_mm
        self.r_threshold = r_threshold
        self.depth_stat = depth_stat
        self.mem_left = 0.0
        self.mem_right = 0.0
        self.stack: deque[np.ndarray] = deque(maxlen=stack_frames)

    def _bin_ranges(self, cols: int) -> Iterable[Tuple[int, int]]:
        edges = np.linspace(0, cols, self.bins + 1, dtype=np.int32)
        for i in range(self.bins):
            start = int(edges[i])
            end = int(edges[i + 1])
            if end <= start:
                end = min(cols, start + 1)
            yield start, end

    def _depth_value(self, values: np.ndarray) -> float:
        valid = values[(values > 0) & (values <= self.max_range_mm)]
        if valid.size == 0:
            return self.max_range_mm
        if self.depth_stat == "min":
            return float(np.min(valid))
        return float(np.median(valid))

    def _close_single_bin_gaps(self, semantic: np.ndarray) -> np.ndarray:
        closed = semantic.copy()
        for i in range(1, self.bins - 1):
            if semantic[i] == 0.0 and semantic[i - 1] > 0.0 and semantic[i + 1] > 0.0:
                closed[i] = 1.0
        return closed

    def _keep_largest_semantic_component(self, semantic: np.ndarray) -> np.ndarray:
        best_start = -1
        best_end = -1
        best_len = 0
        start = -1

        for i, value in enumerate(semantic):
            if value > 0.0 and start < 0:
                start = i
            is_component_end = value == 0.0 and start >= 0
            is_last_bin = i == self.bins - 1 and start >= 0
            if is_component_end or is_last_bin:
                end = i if is_last_bin and value > 0.0 else i - 1
                length = end - start + 1
                if length > best_len:
                    best_start = start
                    best_end = end
                    best_len = length
                start = -1

        filtered = np.zeros_like(semantic)
        if best_start >= 0:
            filtered[best_start:best_end + 1] = 1.0
        return filtered

    def update_from_kv(self, kv: Dict[str, str]) -> Tuple[np.ndarray, Dict[str, float]]:
        r_values = np.asarray(parse_number_list(kv.get("r", ""), int), dtype=np.float32)
        depth_values = np.asarray(parse_number_list(kv.get("depth", ""), int), dtype=np.float32)
        if r_values.size == 0 or depth_values.size == 0:
            raise ValueError("missing r/depth values in fused detail")
        count = int(min(r_values.size, depth_values.size))
        r_values = r_values[:count]
        depth_values = depth_values[:count]

        rows = int(kv.get("rows", "1") or "1")
        rows = max(1, rows)
        cols = max(1, count // rows)
        usable = rows * cols
        r_grid = r_values[:usable].reshape(rows, cols)
        depth_grid = depth_values[:usable].reshape(rows, cols)

        semantic = np.zeros(self.bins, dtype=np.float32)
        depth = np.zeros(self.bins, dtype=np.float32)
        for i, (start, end) in enumerate(self._bin_ranges(cols)):
            r_bin = r_grid[:, start:end]
            d_bin = depth_grid[:, start:end]
            semantic[i] = 1.0 if np.any(r_bin > self.r_threshold) else 0.0
            depth[i] = self._depth_value(d_bin) / self.max_range_mm

        raw_semantic_hits = float(np.sum(semantic))
        semantic = self._close_single_bin_gaps(semantic)
        semantic = self._keep_largest_semantic_component(semantic)

        gyro = parse_number_list(kv.get("gyro", ""), float)
        gyro_z = float(gyro[2]) if len(gyro) >= 3 else 0.0

        sees_target = float(np.sum(semantic) > 0)
        if sees_target:
            idxs = np.arange(self.bins, dtype=np.float32)
            center_idx = float(np.sum(semantic * idxs) / (np.sum(semantic) + 1e-6))
            mid = (self.bins - 1) / 2.0
            self.mem_left = 1.0 if center_idx < mid else 0.0
            self.mem_right = 1.0 if center_idx > mid else 0.0
        else:
            center_idx = -1.0

        single_obs = np.concatenate(
            [
                depth.astype(np.float32),
                semantic.astype(np.float32),
                np.asarray([gyro_z, self.mem_left, self.mem_right], dtype=np.float32),
            ]
        )
        if len(self.stack) == 0:
            for _ in range(self.stack_frames):
                self.stack.append(single_obs.copy())
        else:
            self.stack.append(single_obs.copy())
        stacked = np.concatenate(list(self.stack)).astype(np.float32)

        diagnostics = {
            "raw_count": float(count),
            "rows": float(rows),
            "cols": float(cols),
            "raw_semantic_hits": raw_semantic_hits,
            "semantic_hits": float(np.sum(semantic)),
            "depth_valid": float(np.sum((depth_values[:usable] > 0) & (depth_values[:usable] <= self.max_range_mm))),
            "center_idx": center_idx,
            "gyro_z": gyro_z,
        }
        return stacked, diagnostics


def load_actor(model_path: Path, device: torch.device) -> Actor:
    actor = Actor().to(device)
    state = torch.load(model_path, map_location=device)
    actor.load_state_dict(state)
    actor.eval()
    return actor


def format_probs(probs: np.ndarray) -> str:
    return " ".join(f"{label}={probs[i]:.3f}" for i, label in enumerate(ACTION_LABELS))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real USV observation adapter for PPO actor")
    parser.add_argument("--host", default="192.168.31.31")
    parser.add_argument("--port", type=int, default=19520)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--route-timeout-ms", type=int, default=1000)
    parser.add_argument("--model", default="models/ppo_usv_sim2real_aligned.pth")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--frames", type=int, default=0, help="0 means run until Ctrl+C")
    parser.add_argument("--probe-action", default="S", choices=ACTION_LABELS)
    parser.add_argument("--send-policy-action", action="store_true")
    parser.add_argument("--bins", type=int, default=31)
    parser.add_argument("--stack-frames", type=int, default=3)
    parser.add_argument("--max-range-mm", type=float, default=1500.0)
    parser.add_argument("--r-threshold", type=float, default=80.0)
    parser.add_argument("--depth-stat", default="median", choices=("median", "min"))
    parser.add_argument("--sample-stride", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--row-ratio", type=float, default=0.333333)
    parser.add_argument("--soft-hz", type=float, default=5.0)
    parser.add_argument("--max-power", type=float, default=20.0)
    parser.add_argument("--slam-max-fps", type=int, default=10)
    parser.add_argument("--slam-timeout-ms", type=int, default=120)
    parser.add_argument("--print-raw-ack", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = Path(__file__).resolve().parent
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = workspace / model_path

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    actor = load_actor(model_path, device)
    adapter = RealObsAdapter(
        bins=args.bins,
        stack_frames=args.stack_frames,
        max_range_mm=args.max_range_mm,
        r_threshold=args.r_threshold,
        depth_stat=args.depth_stat,
    )

    client = UsvGatewayClient(args.host, args.port, args.timeout)
    frame_idx = 0
    try:
        client.connect()
        print(f"<< {client.gw_switch(args.route_timeout_ms)}")
        start_ack = client.c_start(args)
        print(f"<< {start_ack}")
        if "ACK OK" not in start_ack:
            print("[ERROR] C START failed; aborting")
            return 2

        while args.frames <= 0 or frame_idx < args.frames:
            action_to_send = args.probe_action
            ack = client.rt(action_to_send)
            if args.print_raw_ack:
                print(f"<< {ack}")
            fields = parse_ack_fields(ack)
            detail = fields.get("detail", "")
            try:
                kv = parse_fused_detail(detail)
                obs, diag = adapter.update_from_kv(kv)
            except ValueError as exc:
                print(f"[SKIP] {exc}")
                time.sleep(args.interval)
                continue

            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = actor(obs_t)
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            policy_action = ACTION_LABELS[int(np.argmax(probs))]

            frame_idx += 1
            print(
                f"[{frame_idx:04d}] {format_probs(probs)} "
                f"policy={policy_action} sent={action_to_send} "
                f"hits={diag['semantic_hits']:.0f}/{args.bins} raw_hits={diag['raw_semantic_hits']:.0f}/{args.bins} "
                f"depth_valid={diag['depth_valid']:.0f}/{diag['raw_count']:.0f} "
                f"center={diag['center_idx']:.1f} gyro_z={diag['gyro_z']:+.4f}"
            )

            if args.send_policy_action and policy_action != action_to_send:
                client.rt(policy_action)
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C received")
    finally:
        try:
            print(f"<< {client.c_stop()}")
        except Exception:
            pass
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

