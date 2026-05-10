#!/usr/bin/env python3
"""USV TCP full-flow client: C START -> realtime RT commands -> C STOP.

Enhanced with:
  - Sensor data extraction from ACK detail (D435i depth, R-channel, IMU gyro)
  - Observation building to 65-dim (31 depth + 31 semantic + 1 gyro_z + 2 memory)
  - 3-frame stacking to 195-dim PPO input
  - PPO Actor network inference for AI auto-pilot (reference: PPO_USV_Hunter_Semantic3_gpu.py)
  - Keyboard toggle between manual WASD and AI auto-pilot modes
"""

import argparse
import os
from pathlib import Path
import socket
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import msvcrt
except ImportError:
    msvcrt = None


# ======================== CONFIGURATION ========================
class AutoPilotConfig:
    single_obs_dim = 65       # 31 depth + 31 semantic + 1 gyro_z + 2 memory
    stack_frames = 3          # number of stacked frames
    obs_dim = single_obs_dim * stack_frames  # 195
    action_dim = 4            # F(0), L(1), R(2), S(3)
    hidden_dim = 256
    max_range = 1.5           # meters, matches PPO training
    n_ray_points = 31         # number of horizontal ray samples


# ======================== UTILITY ========================
def now_ms() -> int:
    return int(time.time() * 1000)


# ======================== ACK PARSING ========================
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


# ======================== SENSOR DATA ========================
class SensorData:
    """Parsed D435i sensor data from an ACK detail string."""

    __slots__ = (
        "r_values", "depth_values", "rows", "samples", "gyro_x", "gyro_y",
        "gyro_z", "gyro_valid", "gyro_frames", "raw_detail", "has_data",
    )

    def __init__(self):
        self.r_values: List[float] = []
        self.depth_values: List[float] = []
        self.rows: int = 0
        self.samples: int = 0
        self.gyro_x: float = 0.0
        self.gyro_y: float = 0.0
        self.gyro_z: float = 0.0
        self.gyro_valid: bool = False
        self.gyro_frames: int = 0
        self.raw_detail: str = ""
        self.has_data: bool = False


def extract_sensor_data(detail: str) -> SensorData:
    """Parse the fused ACK detail string into structured SensorData."""
    sd = SensorData()
    sd.raw_detail = detail
    if not detail or "d435i_ok|" not in detail:
        return sd

    try:
        _, slam = detail.split("d435i_ok|", 1)
    except ValueError:
        return sd

    tokens = slam.split("_")
    kv = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        kv[key] = value

    sd.rows = int(kv.get("rows", 0))
    sd.samples = int(kv.get("samples", 0))

    r_str = kv.get("r", "")
    depth_str = kv.get("depth", "")

    sd.r_values = [float(x) for x in r_str.split(",") if x]
    sd.depth_values = [float(x) for x in depth_str.split(",") if x]

    gyro_str = kv.get("gyro", "")
    sd.gyro_valid = kv.get("valid", "0") == "1"
    try:
        sd.gyro_frames = int(kv.get("frames", "0"))
    except ValueError:
        sd.gyro_frames = 0
    if gyro_str:
        parts = gyro_str.split(",")
        if len(parts) >= 3:
            sd.gyro_x = float(parts[0])
            sd.gyro_y = float(parts[1])
            sd.gyro_z = float(parts[2])

    sd.has_data = len(sd.depth_values) > 0 or len(sd.r_values) > 0
    return sd


def _profile_preview(values: List[float], width: int, max_value: float, invert: bool = False) -> str:
    if not values:
        return ""
    blocks = " .:-=+*#%@"
    if len(values) <= width:
        samples = values
    else:
        step = len(values) / width
        samples = [values[min(int(i * step), len(values) - 1)] for i in range(width)]
    chars = []
    for value in samples:
        norm = max(0.0, min(float(value) / max_value, 1.0))
        if invert:
            norm = 1.0 - norm
        chars.append(blocks[int(norm * (len(blocks) - 1))])
    return "".join(chars)


def summarize_fused_detail(detail: str, print_full: bool) -> Optional[SensorData]:
    if not detail:
        print("[FUSED] missing detail")
        return None

    if "d435i_ok|" not in detail:
        print(f"[FUSED] {detail}")
        return None

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
    imu = kv.get("gyro", "")
    imu_valid = kv.get("valid", "0")
    imu_frames = kv.get("frames", "0")
    r_values = [x for x in kv.get("r", "").split(",") if x]
    depth_values = [x for x in kv.get("depth", "").split(",") if x]

    print(f"[FUSED] rows={rows} samples={samples} r_count={len(r_values)} depth_count={len(depth_values)}")
    if imu:
        print(f"[IMU] valid={imu_valid} frames={imu_frames} gyro_xyz={imu}")
        if imu_valid != "1":
            print("[IMU-WARN] Motion Module has not delivered gyro frames; gyro values are the startup fallback 0.0000.")
    if r_values:
        print(f"[COLOR_R] first_values={','.join(r_values[:16])}")
        print(f"[R-VIS]     {_profile_preview([float(x) for x in r_values], 64, 255.0)}")
    if depth_values:
        print(f"[DEPTH] first_values={','.join(depth_values[:16])}")
        print(f"[DEPTH-VIS] {_profile_preview([float(x) for x in depth_values], 64, 6000.0, invert=True)}")
    if print_full:
        print(f"[FUSED-DETAIL] {detail}")

    return extract_sensor_data(detail)


# ======================== OBSERVATION BUILDER ========================
class ObservationBuilder:
    """Converts SensorData into the 65-dim observation expected by the PPO model.

    Model input (65 dims):
      [0:31]   - 31 depth values, normalized by max_range (1.5m)
      [31:62]  - 31 semantic (binary target presence from R-channel)
      [62]     - angular velocity r (gyro_z)
      [63]     - memory left  (was target seen on left side?)
      [64]     - memory right (was target seen on right side?)
    """

    def __init__(self, config: AutoPilotConfig):
        self.n_points = config.n_ray_points
        self.max_range = config.max_range
        self.mem_left = 0.0
        self.mem_right = 0.0
        self.r_threshold = 80.0  # R-channel threshold for target detection

    def _subsample_horizontal(self, values: List[float], target_n: int) -> np.ndarray:
        """Subsample/interpolate a 1D horizontal profile to `target_n` points."""
        if not values:
            return np.zeros(target_n, dtype=np.float32)
        src = np.array(values, dtype=np.float32)
        src_indices = np.linspace(0, len(src) - 1, len(src))
        tgt_indices = np.linspace(0, len(src) - 1, target_n)
        return np.interp(tgt_indices, src_indices, src)

    def build(self, sd: SensorData) -> np.ndarray:
        """Build a 65-dim observation from SensorData. Returns zeros if no data."""
        obs = np.zeros(65, dtype=np.float32)

        if not sd.has_data or sd.rows == 0:
            return obs

        samples_per_row = max(1, sd.samples // max(1, sd.rows))
        total_expected = sd.rows * samples_per_row

        depth_arr = np.zeros(total_expected, dtype=np.float32)
        if len(sd.depth_values) > 0:
            n_copy = min(len(sd.depth_values), total_expected)
            depth_arr[:n_copy] = sd.depth_values[:n_copy]
        depth_2d = depth_arr.reshape(sd.rows, samples_per_row)
        depth_profile = depth_2d.mean(axis=0)  # (samples_per_row,)
        depth_31 = self._subsample_horizontal(depth_profile.tolist(), self.n_points)
        obs[0:31] = np.clip(depth_31 / (self.max_range * 1000.0), 0.0, 1.0)

        r_arr = np.zeros(total_expected, dtype=np.float32)
        if len(sd.r_values) > 0:
            n_copy = min(len(sd.r_values), total_expected)
            r_arr[:n_copy] = sd.r_values[:n_copy]
        r_2d = r_arr.reshape(sd.rows, samples_per_row)
        r_profile = r_2d.mean(axis=0)  # (samples_per_row,)
        r_31 = self._subsample_horizontal(r_profile.tolist(), self.n_points)
        obs[31:62] = (r_31 > self.r_threshold).astype(np.float32)

        obs[62] = sd.gyro_z

        sees_target = obs[31:62].sum() > 0
        if sees_target:
            center_idx = (obs[31:62] * np.arange(31)).sum() / max(obs[31:62].sum(), 1e-6)
            self.mem_left = 1.0 if center_idx < 15.0 else 0.0
            self.mem_right = 1.0 if center_idx > 15.0 else 0.0
        obs[63] = self.mem_left
        obs[64] = self.mem_right

        return obs

    def reset_memory(self):
        self.mem_left = 0.0
        self.mem_right = 0.0


# ======================== PPO ACTOR NETWORK ========================
class ActorNetwork(nn.Module):
    """Mirrors the Actor from PPO_USV_Hunter_Semantic3_gpu.py."""

    def __init__(self, input_dim=195, hidden=256, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ======================== AI AUTO-PILOT ========================
class AutoPilot:
    """Manages frame stacking, observation accumulation, and model inference.

    Maintains a sliding window of `stack_frames` observations and runs
    the PPO Actor to select discrete actions (0=F, 1=L, 2=R, 3=S).
    """

    ACTION_MAP = {0: "F", 1: "L", 2: "R", 3: "S"}

    def __init__(self, config: AutoPilotConfig, model_path: str, device: str = "cpu"):
        self.config = config
        self.device = device
        self.obs_builder = ObservationBuilder(config)

        self.obs_stack: np.ndarray = np.zeros(config.obs_dim, dtype=np.float32)

        self.actor: Optional[ActorNetwork] = None
        self.model_loaded = False

        if HAS_TORCH:
            self.actor = ActorNetwork(
                input_dim=config.obs_dim,
                hidden=config.hidden_dim,
                n_actions=config.action_dim,
            ).to(device)
            if model_path and os.path.exists(model_path):
                self._load_model(model_path)
        else:
            print("[WARN] torch not available; auto-pilot cannot run inference")

    def _load_model(self, model_path: str):
        try:
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
            except TypeError:
                state = torch.load(model_path, map_location=self.device)
            self.actor.load_state_dict(state)
            self.actor.eval()
            self.model_loaded = True
            print(f"[MODEL] loaded {model_path}")
        except Exception as e:
            print(f"[MODEL] failed to load {model_path}: {e}")
            self.model_loaded = False

    def feed_sensor(self, sd: SensorData):
        single_obs = self.obs_builder.build(sd)
        self.obs_stack = np.roll(self.obs_stack, -self.config.single_obs_dim)
        self.obs_stack[-self.config.single_obs_dim:] = single_obs

    def get_action(self, deterministic: bool = True) -> str:
        if not self.model_loaded or self.actor is None:
            return "S"

        with torch.no_grad():
            tensor = torch.from_numpy(self.obs_stack).float().unsqueeze(0).to(self.device)
            logits = self.actor(tensor)
            if deterministic:
                action_idx = int(logits.argmax(dim=-1).item())
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action_idx = int(dist.sample().item())
        return self.ACTION_MAP.get(action_idx, "S")

    def reset(self):
        self.obs_stack = np.zeros(self.config.obs_dim, dtype=np.float32)
        self.obs_builder.reset_memory()


def resolve_model_path(model_path: str) -> str:
    if not model_path:
        return model_path

    path = Path(model_path).expanduser()
    if path.exists():
        return str(path)

    script_dir_path = Path(__file__).resolve().parent / model_path
    if script_dir_path.exists():
        return str(script_dir_path)

    desktop_rl_path = Path.home() / "Desktop" / "final_pro_git" / "rl_training" / model_path
    if desktop_rl_path.exists():
        return str(desktop_rl_path)

    return model_path


# ======================== TCP CLIENT ========================
class UsvClient:
    def __init__(self, host: str, port: int, timeout: float):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.seq = 1
        self.sock: Optional[socket.socket] = None

    def connect(self):
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        except ConnectionRefusedError as exc:
            raise ConnectionError(
                f"gateway refused {self.host}:{self.port}; check that CommunicationLayer is running "
                f"and listening on this IP/port"
            ) from exc
        except TimeoutError as exc:
            raise TimeoutError(
                f"timed out connecting to gateway {self.host}:{self.port}; check device IP, network, "
                f"firewall, and that the gateway is reachable"
            ) from exc
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
        cmd = "GW SWITCH legacy_alias=on sli_enabled=on"
        if route_timeout_ms > 0:
            cmd += f" route_timeout_ms={route_timeout_ms}"
        return self.send_line(cmd)

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

    def rt(self, action: str, print_full_detail: bool) -> Tuple[str, Optional[SensorData]]:
        seq = self.next_seq()
        ts = now_ms()
        ack = self.send_line(f"R {seq} {action} {ts}")
        fields = parse_ack_fields(ack)
        sd = None
        if fields.get("tag") == "rt_apply":
            sd = summarize_fused_detail(fields.get("detail", ""), print_full_detail)
        return ack, sd

    def c_stop(self):
        seq = self.next_seq()
        ts = now_ms()
        return self.send_line(f"C STOP seq={seq} ts={ts}")


# ======================== CLI ========================
def build_parser():
    parser = argparse.ArgumentParser(description="USV gateway full-flow client with PPO auto-pilot")
    parser.add_argument("--host", default="192.168.31.31")
    parser.add_argument("--port", type=int, default=19520)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--route-timeout-ms", type=int, default=5000, help="Gateway route timeout ms, 1..5000; 0 to omit from GW SWITCH")
    parser.add_argument("--actions", default="", help="Comma separated actions for non-interactive mode")
    parser.add_argument("--interactive", action="store_true", help="Use WASD keyboard loop on Windows")
    parser.add_argument("--print-full-detail", action="store_true", help="Print full fused ACK detail")
    parser.add_argument("--auto", action="store_true", help="Start directly in AI auto-pilot mode")
    parser.add_argument("--model", default="ppo_usv_sim2real_aligned.pth", help="Path to trained PPO actor weights")
    parser.add_argument("--model-device", default="cpu", choices=("cpu", "cuda"), help="Device for model inference")

    # C START config
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
    parser.add_argument("--sample-stride", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--pack-mode", default="bin", choices=("bin", "hex"))
    parser.add_argument("--r-threshold", type=float, default=80.0, help="R-channel threshold for target detection (0-255)")
    return parser


# ======================== MAIN ========================
def main() -> int:
    args = build_parser().parse_args()

    KEY_TO_ACTION = {"w": "F", "a": "L", "d": "R", "s": "S"}

    ap_config = AutoPilotConfig()
    args.model = resolve_model_path(args.model)
    auto_pilot = AutoPilot(ap_config, args.model, args.model_device)
    auto_pilot.obs_builder.r_threshold = args.r_threshold

    client = UsvClient(args.host, args.port, args.timeout)

    try:
        client.connect()
        gw_ack = client.gw_switch(args.route_timeout_ms)
        if "ACK OK" not in gw_ack:
            print(f"[WARN] GW SWITCH returned error, continuing anyway (was: {gw_ack.strip()})")
        start_ack = client.c_start(args)
        if "ACK OK" not in start_ack:
            fields = parse_ack_fields(start_ack)
            if fields.get("tag") == "gw_route_timeout":
                print(
                    "[HINT] C START reached the gateway, but the backend processor did not ACK before "
                    f"route_timeout_ms={args.route_timeout_ms}. Use --route-timeout-ms 5000, or check "
                    "that MainProcessor/D435i/SLAM startup is healthy on the boat."
                )
            elif fields.get("tag") == "gw_route_fail":
                print(
                    "[HINT] Gateway could not forward to MainProcessor. Check that MainProcessor is "
                    "running and that CommunicationLayer --processor-host/--processor-port are correct."
                )
            print("[ERROR] C START failed; aborting RT flow")
            return 2

        use_interactive = args.interactive or not args.actions
        if not use_interactive:
            for action in [x.strip().upper() for x in args.actions.split(",") if x.strip()]:
                if action not in ("F", "L", "R", "S"):
                    raise ValueError(f"bad action: {action}")
                ack, sd = client.rt(action, args.print_full_detail)
                if sd and auto_pilot.model_loaded:
                    auto_pilot.feed_sensor(sd)
                time.sleep(0.2)
            client.c_stop()
            return 0

        if msvcrt is None:
            print("[ERROR] interactive mode requires Windows msvcrt; pass --actions F,L,R,S for non-interactive mode")
            return 2

        ai_mode = args.auto
        if ai_mode and not auto_pilot.model_loaded:
            print("[WARN] model not loaded; falling back to manual WASD mode")
            ai_mode = False

        print("=" * 60)
        print("  USV Control Client")
        print("  W=F  A=L  D=R  S=Stop  Q=Quit  T=Toggle AI")
        if auto_pilot.model_loaded:
            print(f"  AI model: {args.model}  (device={args.model_device})")
        print(f"  Mode: {'[AI AUTO-PILOT]' if ai_mode else '[MANUAL WASD]'}")
        print("=" * 60)

        while True:
            if ai_mode:
                # ── AI auto-pilot mode: model selects action, keyboard can override ──
                action = auto_pilot.get_action(deterministic=True)
                ack, sd = client.rt(action, args.print_full_detail)
                if sd:
                    auto_pilot.feed_sensor(sd)

                # non-blocking keyboard check for toggle/quit within the control interval
                key = ""
                waited = 0.0
                poll_interval = 0.05
                interval = 1.0 / max(args.soft_hz, 1)
                while waited < interval:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b"\x00", b"\xe0"):
                            if msvcrt.kbhit():
                                msvcrt.getch()
                            continue
                        key = ch.decode("utf-8", errors="ignore").lower()
                        if key == "q":
                            break
                        if key == "t":
                            auto_pilot.reset()
                            print("[MODE] switched to MANUAL WASD")
                            break
                    else:
                        time.sleep(poll_interval)
                        waited += poll_interval

                if key == "q":
                    break
                if key == "t":
                    ai_mode = False
                    continue
            else:
                # ── Manual WASD mode: keyboard controls thrusters ──
                ch = msvcrt.getch()
                if ch in (b"\x00", b"\xe0"):
                    msvcrt.getch()
                    continue
                key = ch.decode("utf-8", errors="ignore").lower()
                if key == "q":
                    break
                if key == "t":
                    if auto_pilot.model_loaded:
                        ai_mode = True
                        auto_pilot.reset()
                        print("[MODE] switched to AI AUTO-PILOT")
                    else:
                        print("[WARN] model not loaded; cannot enter AI mode")
                    continue
                if key in KEY_TO_ACTION:
                    ack, sd = client.rt(KEY_TO_ACTION[key], args.print_full_detail)
                    if sd and auto_pilot.model_loaded:
                        auto_pilot.feed_sensor(sd)

        client.c_stop()
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
