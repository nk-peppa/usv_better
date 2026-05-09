#!/usr/bin/env python3
"""USV TCP full-flow client: C START -> realtime RT commands -> C STOP.

Default C START config asks the processor to return 64 interval-picked D435i rows,
using R-channel and depth pairs in each accepted realtime ACK detail.
"""

import argparse
import socket
import sys
import time

try:
    import msvcrt  # Windows interactive keyboard
except ImportError:  # pragma: no cover - Linux fallback for manual tests
    msvcrt = None


def now_ms() -> int:
    return int(time.time() * 1000)


def parse_ack_fields(ack: str) -> dict:
    fields = {}
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


def summarize_fused_detail(detail: str, print_full: bool) -> None:
    if not detail:
        print("[FUSED] missing detail")
        return

    if "d435i_ok|" not in detail:
        print(f"[FUSED] {detail}")
        return

    # Example detail after ACK sanitizing:
    # action_sent|d435i_ok|SL_12_ctrl=1_..._IMU_gyro=0.0100,0.0200,0.0300_rows=64_samples=640_r=1,65,..._depth=807,999,...
    try:
        _, slam = detail.split("d435i_ok|", 1)
    except ValueError:
        slam = detail

    tokens = slam.split("_")
    kv = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        kv[key] = value

    rows = kv.get("rows", "?")
    samples = kv.get("samples", "?")
    imu = kv.get("gyro", "")  # token comes from IMU_gyro after '_' splitting
    r_values = [x for x in kv.get("r", "").split(",") if x]
    depth_values = [x for x in kv.get("depth", "").split(",") if x]

    print(f"[FUSED] rows={rows} samples={samples} r_count={len(r_values)} depth_count={len(depth_values)}")
    if imu:
        print(f"[IMU] gyro_xyz={imu}")
    if r_values:
        print(f"[COLOR_R] first_values={','.join(r_values[:16])}")
    if depth_values:
        print(f"[DEPTH] first_values={','.join(depth_values[:16])}")
    if print_full:
        print(f"[FUSED-DETAIL] {detail}")


class UsvClient:
    def __init__(self, host: str, port: int, timeout: float):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.seq = 1
        self.sock = None

    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        print(f"[CONNECTED] gateway={self.host}:{self.port}")

    def close(self):
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
        ack = b"".join(chunks).decode("utf-8", errors="replace")
        print(f">> {line}")
        print(f"<< {ack}")
        return ack

    def gw_switch(self, route_timeout_ms: int):
        return self.send_line(
            f"GW SWITCH legacy_alias=on sli_enabled=on route_timeout_ms={route_timeout_ms}"
        )

    def c_start(self, args):
        seq = self.next_seq()
        ts = now_ms()
        line = (
            f"C START seq={seq} ts={ts} "
            f"soft_hz={args.soft_hz} "
            f"max_power={args.max_power} "
            f"left_gain={args.left_gain} "
            f"right_gain={args.right_gain} "
            f"left_trim={args.left_trim} "
            f"right_trim={args.right_trim} "
            f"slam_max_fps={args.slam_max_fps} "
            f"slam_timeout_ms={args.slam_timeout_ms} "
            f"slam_max_groups={args.slam_max_groups} "
            f"slam_min_quality={args.slam_min_quality} "
            f"slam_drop_policy={args.slam_drop_policy} "
            f"row_ratio={args.row_ratio} "
            f"channel_mode={args.channel_mode} "
            f"sample_stride={args.sample_stride} "
            f"max_rows={args.max_rows} "
            f"pack_mode={args.pack_mode}"
        )
        return self.send_line(line)

    def rt(self, action: str, print_full_detail: bool):
        seq = self.next_seq()
        ts = now_ms()
        ack = self.send_line(f"R {seq} {action} {ts}")
        fields = parse_ack_fields(ack)
        if fields.get("tag") == "rt_apply":
            summarize_fused_detail(fields.get("detail", ""), print_full_detail)
        return ack

    def c_stop(self):
        seq = self.next_seq()
        ts = now_ms()
        return self.send_line(f"C STOP seq={seq} ts={ts}")


def build_parser():
    parser = argparse.ArgumentParser(description="USV gateway full-flow client")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19520)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--route-timeout-ms", type=int, default=8000)
    parser.add_argument("--actions", default="", help="Comma separated actions for non-interactive mode; if omitted, Windows uses WASD interactive mode")
    parser.add_argument("--interactive", action="store_true", help="Use WASD keyboard loop on Windows")
    parser.add_argument("--print-full-detail", action="store_true", help="Print full fused ACK detail")

    # C START config. Defaults request 64 interval-picked rows and R/depth pairs.
    parser.add_argument("--soft-hz", type=float, default=20)
    parser.add_argument("--max-power", type=float, default=70)
    parser.add_argument("--left-gain", type=float, default=1)
    parser.add_argument("--right-gain", type=float, default=1)
    parser.add_argument("--left-trim", type=float, default=0)
    parser.add_argument("--right-trim", type=float, default=0)
    parser.add_argument("--slam-max-fps", type=int, default=10)
    parser.add_argument("--slam-timeout-ms", type=int, default=1000)
    parser.add_argument("--slam-max-groups", type=int, default=8)
    parser.add_argument("--slam-min-quality", type=int, default=10)
    parser.add_argument("--slam-drop-policy", default="newest", choices=("reject", "oldest", "newest"))
    parser.add_argument("--row-ratio", type=float, default=0.333333)
    parser.add_argument("--channel-mode", default="R", choices=("R", "G", "B", "GRAY"))
    parser.add_argument("--sample-stride", type=int, default=64, help="640px/64 gives 10 samples per selected row")
    parser.add_argument("--max-rows", type=int, default=64, help="Default asks core for 64 interval-picked D435i rows")
    parser.add_argument("--pack-mode", default="bin", choices=("bin", "hex"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    key_to_action = {"w": "F", "a": "L", "d": "R", "s": "S"}
    client = UsvClient(args.host, args.port, args.timeout)
    try:
        client.connect()
        client.gw_switch(args.route_timeout_ms)
        start_ack = client.c_start(args)
        if "ACK OK" not in start_ack:
            print("[ERROR] C START failed; aborting RT flow")
            return 2

        use_interactive = args.interactive or not args.actions
        if use_interactive:
            if msvcrt is None:
                print("[ERROR] interactive mode requires Windows msvcrt; pass --actions F,L,R,S for non-interactive mode")
                return 2
            print("[READY] W=F A=L D=R S=Stop Q=C_STOP+quit")
            while True:
                ch = msvcrt.getch()
                if ch in (b"\x00", b"\xe0"):
                    msvcrt.getch()
                    continue
                key = ch.decode("utf-8", errors="ignore").lower()
                if key == "q":
                    break
                if key in key_to_action:
                    client.rt(key_to_action[key], args.print_full_detail)
        else:
            for action in [x.strip().upper() for x in args.actions.split(",") if x.strip()]:
                if action not in ("F", "L", "R", "S"):
                    raise ValueError(f"bad action: {action}")
                client.rt(action, args.print_full_detail)
                time.sleep(0.2)

        client.c_stop()
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
