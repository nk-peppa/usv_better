import builtins
import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from matplotlib.pyplot import *
from numpy import *
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# ── Windows / macOS / Linux matplotlib 适配 ──────────────────────────
_mp_config_dir = os.environ.get("MPLCONFIGDIR")
if not _mp_config_dir:
    if sys.platform == "win32":
        _mp_config_dir = os.path.join(tempfile.gettempdir(), "matplotlib_cache")
    else:
        _mp_config_dir = "/tmp/matplotlib_cache"
os.environ.setdefault("MPLCONFIGDIR", _mp_config_dir)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
_backend = os.environ.get("MPLBACKEND")
if not _backend:
    _backend = "TkAgg" if sys.platform == "win32" else "MacOSX"
matplotlib.use(_backend)


SEGMENT_LENGTH = 10   # 控制步窗口（匹配 1.6s 步长数据）
SEGMENT_STEP = 5      # 滑动步长（50%重叠，数据增强）
RAW_SEGMENT_LENGTH = 400  # 原始20Hz数据窗口（20秒）
RAW_SEGMENT_STEP = 200    # 滑动步长（50%重叠）
CSV_NUM = 23  # count csv files
TEST_NUM = 5  # count test trajectories
TIME_LENGTH = -1  # steps used, -1 all
START_POINT = 2  # count first point
CONTROL_DT = 1.6  # s control step  1.6
START_TIME = 0.0  # s discard start
N_FITTING = 3  # 滑动窗口内多项式阶数
SG_WINDOW = 4   # Savitzky-Golay 滑动窗口大小（4点拟合3阶）
R_EPS = 0.001  # rad/s turn threshold
HEADING_OFFSET_RAD = 0  # math.pi / 2.0  # rad frame offset
EPOCHS = 10000  # count training
HIDDEN = 64  # count neurons
LR = 1e-3  # 1/step learning
WEIGHT_DECAY = 1e-6  # 1/step decay
GRAD_CLIP = 1.0  # gradient clipping
PRINT_EVERY = 20  # 每20 epoch 更新一次图像（减少绘图开销）
PLOT_WINDOW = 200  # 只显示最后200个 epoch
ACTION_TEST_NUM = 50  # count test actions

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "all_data 2", "all_data")  # folder csv
SPECIFIC_CSV = "20260706_154456_keyboard-latched-WASD_interval-1p6s_until-wall_20hz.csv"  # file plot
TIMESTAMP = time.strftime("%m%d_%H%M")
OUTPUT_DIR = Path("fit_outputs") / "uvr_next_fit_v33"  # v33 固定输出目录

POSE_COLUMNS = ["pos_x", "pos_y", "yaw"]
STATE_COLUMNS = ["u", "v", "r"]
STATE_SCALE = np.array([0.02, 0.002, 0.05], dtype=np.float32)  # u,v,r scale
DELTA_SCALE = np.array([0.005, 0.001, 0.02], dtype=np.float32)  # u,v,r change
ACTIONS = ["S", "W", "A", "D"]
FIRST_MODEL_IDX = 2  # count first measured uvr (= N_PRE, 确保有足够历史)
N_PRE = 2  # count previous
N_PRED = 1  # count next
INPUT_SIZE = 3 * (N_PRE + 1) + len(ACTIONS)  # count input
OUTPUT_SIZE = 3  # count output
OUTPUT_SCALE = DELTA_SCALE.astype(np.float32)  # u,v,r change


# ── GPU 自动检测：CUDA (NVIDIA) / ROCm (AMD) / MPS (Apple) / CPU ────
def get_device():
    """Detect best available accelerator: ROCm → CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        # ROCm 和 CUDA 都通过 torch.cuda 暴露，ROCm 版本号含 'hip'
        v = torch.version.cuda or ""
        tag = "ROCm" if ("hip" in v.lower() or torch.cuda.get_device_properties(0).name.lower().startswith("amd")) else "CUDA"
        device = torch.device("cuda")
        print(f"[GPU] {tag} detected: {torch.cuda.get_device_name(0)} (device={device})")
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[GPU] Apple MPS detected (device={device})")
        return device
    print("[GPU] No accelerator found, using CPU.")
    return torch.device("cpu")

DEVICE = get_device()


def action_name(value):
    if pd.isna(value):
        return "S"
    text = str(value).strip().upper()
    aliases = {
        "0": "S", "1": "W", "2": "A", "3": "D",
        "STOP": "S", "NONE": "S", "IDLE": "S",
        "F": "W", "FORWARD": "W", "LEFT": "A", "RIGHT": "D",
    }
    return aliases.get(text, text if text in ["S", "W", "A", "D"] else "S")


def find_action_column(df):
    for col in ("action_key", "action", "tcp_action", "action_a"):
        if col in df.columns:
            return col
    raise ValueError("CSV must contain action_key, action, tcp_action, or action_a")


def yaw_from_dataframe(df):
    if "yaw" in df.columns:
        return df["yaw"].astype(float).to_numpy()
    if "yaw_rad" in df.columns:
        return df["yaw_rad"].astype(float).to_numpy()
    if "yaw_deg" in df.columns:
        return np.deg2rad(df["yaw_deg"].astype(float).to_numpy())
    raise ValueError("CSV must contain yaw, yaw_rad, or yaw_deg")


def load_raw_csv(path):
    df = pd.read_csv(path)
    missing = {"time", "pos_x", "pos_y"}.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = pd.DataFrame()
    out["time"] = df["time"].astype(float)
    out["action"] = df[find_action_column(df)].map(action_name)
    out["pos_x"] = df["pos_x"].astype(float)
    out["pos_y"] = df["pos_y"].astype(float)
    out["yaw"] = np.unwrap(yaw_from_dataframe(df))
    return out.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def resample_to_control_steps(df):
    df = df.sort_values("time").reset_index(drop=True)
    first_time = float(df["time"].iloc[0])
    df = df[df["time"] >= first_time + START_TIME].reset_index(drop=True)
    start = float(df["time"].iloc[0])
    end = float(df["time"].iloc[-1])
    count = int(math.floor((end - start) / CONTROL_DT)) + 1
    rows = []
    for step in range(count):
        sample_time = start + step * CONTROL_DT
        idx = int((df["time"] - sample_time).abs().idxmin())
        row = df.loc[idx].copy()
        row["original_time"] = row["time"]
        row["time"] = step
        rows.append(row)
    out = pd.DataFrame(rows).reset_index(drop=True)
    return out


def cut_start_and_length(df):
    start_idx = builtins.max(0, START_POINT - 1)
    out = df.iloc[start_idx:].reset_index(drop=True)
    if TIME_LENGTH > 0:
        out = out.iloc[:TIME_LENGTH].reset_index(drop=True)
    return out


def inverse_interval_uvr(dx_body, dy_body, delta_yaw):
    r = delta_yaw / CONTROL_DT
    if abs(r) <= R_EPS:
        return dx_body / CONTROL_DT, dy_body / CONTROL_DT, r
    turn = r * CONTROL_DT
    transform = np.array([
        [math.sin(turn) / r, (math.cos(turn) - 1.0) / r],
        [(1.0 - math.cos(turn)) / r, math.sin(turn) / r],
    ])
    u, v = np.linalg.solve(transform, np.array([dx_body, dy_body]))
    return float(u), float(v), float(r)


def add_control_rate_uvr(df):
    x = df["pos_x"].to_numpy(dtype=np.float64)
    y = df["pos_y"].to_numpy(dtype=np.float64)
    yaw = df["yaw"].to_numpy(dtype=np.float64)
    u = np.zeros(len(df), dtype=np.float64)
    v = np.zeros(len(df), dtype=np.float64)
    r = np.zeros(len(df), dtype=np.float64)
    for idx in range(len(df) - 1):
        dx_global = x[idx + 1] - x[idx]
        dy_global = y[idx + 1] - y[idx]
        heading = yaw[idx] + HEADING_OFFSET_RAD
        cos_heading = math.cos(heading)
        sin_heading = math.sin(heading)
        dx_body = dx_global * cos_heading + dy_global * sin_heading
        dy_body = -dx_global * sin_heading + dy_global * cos_heading
        u[idx + 1], v[idx + 1], r[idx + 1] = inverse_interval_uvr(dx_body, dy_body, yaw[idx + 1] - yaw[idx])
    u[0], v[0], r[0] = 2*u[1]-u[2], 2*v[1]-v[2], 2*r[1]-r[2]
    df = df.copy()
    df["u"] = u
    df["v"] = v
    df["r"] = r
    return df


def action_one_hot(actions):
    out = np.zeros((len(actions), len(ACTIONS)), dtype=np.float32)
    for idx, action in enumerate(actions):
        out[idx, ACTIONS.index(action)] = 1.0
    return out


def integrate_pose_step(pos_x, pos_y, yaw, u_next, v_next, r_next):
    yaw_next = yaw + r_next * CONTROL_DT
    turn = r_next * CONTROL_DT
    if abs(r_next) > R_EPS:
        dx_body = (u_next / r_next) * math.sin(turn) + (v_next / r_next) * (math.cos(turn) - 1.0)
        dy_body = (u_next / r_next) * (1.0 - math.cos(turn)) + (v_next / r_next) * math.sin(turn)
    else:
        dx_body = u_next * CONTROL_DT
        dy_body = v_next * CONTROL_DT
    heading = yaw + HEADING_OFFSET_RAD
    pos_x_next = pos_x + dx_body * math.cos(heading) - dy_body * math.sin(heading)
    pos_y_next = pos_y + dx_body * math.sin(heading) + dy_body * math.cos(heading)
    return pos_x_next, pos_y_next, yaw_next


def integrate_pose_series(initial_pose, uvr):
    pose = np.zeros((len(uvr), 3), dtype=np.float64)
    pose[0] = initial_pose
    for idx in range(len(uvr) - 1):
        pose[idx + 1] = integrate_pose_step(pose[idx, 0], pose[idx, 1], pose[idx, 2], *uvr[idx + 1])
    return pose


def savgol_smooth(y, window, order):
    """Savitzky-Golay 局部滑动拟合：每 window 点拟合 order 阶，加权平均."""
    n = len(y)
    if n < window:
        return y.copy()
    accum = np.zeros(n, dtype=np.float64)
    weights = np.zeros(n, dtype=np.float64)
    x_local = np.arange(window, dtype=np.float64)
    A = np.vander(x_local, order + 1, increasing=True)
    for start in range(n - window + 1):
        y_win = y[start:start + window]
        coef = np.linalg.lstsq(A, y_win, rcond=None)[0]
        y_fit = A @ coef
        accum[start:start + window] += y_fit
        weights[start:start + window] += 1.0
    return accum / weights


def process_single_segment(segment_df, segment_id, orig_path):
    """Savitzky-Golay 局部4点3阶拟合 + 原始位姿差分算uvr."""
    df_prime = cut_start_and_length(segment_df)
    if len(df_prime) < 3:
        return None

    # 对有足够点的轨迹做 SG 局部拟合
    for col in POSE_COLUMNS:
        y_raw = df_prime[col].to_numpy(np.float64)
        df_prime[col] = savgol_smooth(y_raw, SG_WINDOW, N_FITTING)

    # 从平滑后的位姿计算 uvr
    df_prime = add_control_rate_uvr(df_prime)

    # 修正 yaw 初始方向
    dx0 = df_prime["pos_x"].iloc[1] - df_prime["pos_x"].iloc[0]
    dy0 = df_prime["pos_y"].iloc[1] - df_prime["pos_y"].iloc[0]
    df_prime["yaw"] = df_prime["yaw"] - df_prime["yaw"].iloc[0] + math.atan2(dy0, dx0)

    df_prime["trajectory_id"] = orig_path.stem
    return (segment_id, df_prime)


def load_trajectories():
    """先对20Hz原始数据滑动窗口，再重采样到控制步 → 数据量暴增."""
    trajectories = []
    csv_paths = sorted(Path(DATA_DIR).glob("*.csv"))[:CSV_NUM]
    total_segments = 0
    for path in csv_paths:
        raw_df = load_raw_csv(path)
        n_raw = len(raw_df)

        if n_raw < RAW_SEGMENT_LENGTH:
            # 短轨迹：直接重采样后处理
            ctrl_df = resample_to_control_steps(raw_df)
            n_ctrl = len(ctrl_df)
            if n_ctrl < SEGMENT_LENGTH:
                result = process_single_segment(ctrl_df, f"{path.stem}", path)
                if result is not None:
                    trajectories.append(result)
                    total_segments += 1
                    plot(ctrl_df["pos_x"], ctrl_df["pos_y"], 'b')
                    plot(result[1]["pos_x"], result[1]["pos_y"], 'r')
                print(f"{path} ({n_ctrl} ctrl pts, 1 seg)")
            else:
                seg_count = 0
                for start in range(0, n_ctrl - SEGMENT_LENGTH + 1, SEGMENT_STEP):
                    seg = ctrl_df.iloc[start:start + SEGMENT_LENGTH].reset_index(drop=True)
                    r = process_single_segment(seg, f"{path.stem}_s{seg_count:02d}", path)
                    if r: trajectories.append(r); seg_count += 1
                total_segments += seg_count
                if seg_count > 0:
                    plot(ctrl_df.iloc[0:SEGMENT_LENGTH]["pos_x"], ctrl_df.iloc[0:SEGMENT_LENGTH]["pos_y"], 'b')
                    plot(trajectories[-seg_count][1]["pos_x"], trajectories[-seg_count][1]["pos_y"], 'r')
                print(f"{path} ({n_ctrl} ctrl pts, {seg_count} segs)")
        else:
            # 长原始轨迹：20Hz 窗口 → 每个窗口重采样
            seg_count = 0
            last_start = 0
            for start in range(0, n_raw - RAW_SEGMENT_LENGTH + 1, RAW_SEGMENT_STEP):
                raw_seg = raw_df.iloc[start:start + RAW_SEGMENT_LENGTH].reset_index(drop=True)
                ctrl_seg = resample_to_control_steps(raw_seg)
                seg_id = f"{path.stem}_s{seg_count:02d}"
                result = process_single_segment(ctrl_seg, seg_id, path)
                if result is not None:
                    trajectories.append(result)
                    seg_count += 1
                last_start = start
            if last_start + RAW_SEGMENT_LENGTH < n_raw:
                raw_seg = raw_df.iloc[n_raw - RAW_SEGMENT_LENGTH:n_raw].reset_index(drop=True)
                ctrl_seg = resample_to_control_steps(raw_seg)
                seg_id = f"{path.stem}_s{seg_count:02d}"
                result = process_single_segment(ctrl_seg, seg_id, path)
                if result is not None:
                    trajectories.append(result)
                    seg_count += 1
            total_segments += seg_count
            if seg_count > 0:
                pass  # skip interactive plot to avoid OpenGL issues
            print(f"{path} ({n_raw} raw pts → {seg_count} segs)")
        # pause(1e-5)  # disabled to avoid GUI freeze

    print(f"\nTotal: {len(csv_paths)} CSV files -> {total_segments} trajectory segments")
    if len(trajectories) == 0:
        raise ValueError("No valid csv trajectories found.")
    return trajectories


def simulate_uvr(model, real_uvr, actions):
    pred = np.zeros_like(real_uvr)
    first_idx = builtins.max(FIRST_MODEL_IDX, N_PRE)
    pred[:first_idx + 1] = real_uvr[:first_idx + 1]
    action_input = action_one_hot(actions)
    model.eval()
    with torch.no_grad():
        for idx in range(first_idx, len(real_uvr) - 1):
            history = pred[idx - N_PRE:idx + 1] / STATE_SCALE
            x = np.r_[history.ravel(), action_input[idx]][None]
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
            delta = model(x).cpu().numpy()[0] * DELTA_SCALE
            pred[idx + 1] = pred[idx] + delta
    return pred


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
figure()
axis('equal')
trajectories = load_trajectories()
title(f"{len(trajectories)} segments (blue=raw, red=processed)")
savefig(OUTPUT_DIR / f"all_trajectories_{TIMESTAMP}.png", dpi=150)
close()  # 关闭概览图，避免干扰后续显示
if len(trajectories) <= TEST_NUM:
    raise ValueError("Need more trajectories than TEST_NUM.")

# 按 CSV 文件分 train/test（同一 CSV 的段不会跨分）
import re
csv_stems = sorted(set(re.sub(r"_s\d+$", "", seg_id) for seg_id, _ in trajectories))
if len(csv_stems) <= TEST_NUM:
    raise ValueError(f"Need more CSV files than TEST_NUM ({TEST_NUM}), got {len(csv_stems)}")
test_stems = set(csv_stems[-TEST_NUM:])
train_trajectories = [(sid, df) for sid, df in trajectories if re.sub(r"_s\d+$", "", sid) not in test_stems]
test_trajectories = [(sid, df) for sid, df in trajectories if re.sub(r"_s\d+$", "", sid) in test_stems]
print(f"Train: {len(train_trajectories)} segs from {len(csv_stems) - TEST_NUM} CSVs, Test: {len(test_trajectories)} segs from {TEST_NUM} CSVs")
train_names = [name for name, _ in train_trajectories]
test_names = [name for name, _ in test_trajectories]

train_states = []
train_targets = []
for _, df in train_trajectories:
    uvr = df[STATE_COLUMNS].to_numpy(np.float32)
    actions = action_one_hot(df["action"].to_numpy())
    first_idx = builtins.max(FIRST_MODEL_IDX, N_PRE)
    for i in range(first_idx, len(uvr) - 1):
        history = uvr[i - N_PRE:i + 1] / STATE_SCALE
        delta = uvr[i + 1] - uvr[i]
        train_states.append(np.r_[history.ravel(), actions[i]])
        train_targets.append(delta)

test_states = []
test_targets = []
for _, df in test_trajectories:
    uvr = df[STATE_COLUMNS].to_numpy(np.float32)
    actions = action_one_hot(df["action"].to_numpy())
    first_idx = builtins.max(FIRST_MODEL_IDX, N_PRE)
    for i in range(first_idx, len(uvr) - 1):
        history = uvr[i - N_PRE:i + 1] / STATE_SCALE
        delta = uvr[i + 1] - uvr[i]
        test_states.append(np.r_[history.ravel(), actions[i]])
        test_targets.append(delta)

x_train = np.vstack(train_states).astype(np.float32)
y_train = np.vstack(train_targets).astype(np.float32)
x_test = np.vstack(test_states).astype(np.float32)
y_test = np.vstack(test_targets).astype(np.float32)
x = torch.tensor(x_train, dtype=torch.float32, device=DEVICE)
y = torch.tensor(y_train / OUTPUT_SCALE, dtype=torch.float32, device=DEVICE)
x_test_torch = torch.tensor(x_test, dtype=torch.float32, device=DEVICE)
y_test_torch = torch.tensor(y_test / OUTPUT_SCALE, dtype=torch.float32, device=DEVICE)

model = nn.Sequential(
    nn.Linear(INPUT_SIZE, HIDDEN),
    nn.Tanh(),
    nn.Linear(HIDDEN, HIDDEN),
    nn.Tanh(),
    nn.Linear(HIDDEN, OUTPUT_SIZE),
).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
train_loss_history = []
test_loss_history = []
best_test_loss = float("inf")
best_state = None
best_epoch = 0

ioff()  # 非交互模式，避免 OpenGL/GUI 问题
figure()
clf()
xlabel("epoch")
ylabel("log10 loss [scaled]")
plot([], [], "k.", label="train")
plot([], [], "r.", label="test")
legend()

for epoch in tqdm(range(1, EPOCHS + 1), desc="analytic uvr (fit+deriv)"):
    pred = model(x)
    err = pred - y
    loss = (err ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()
    train_loss_history.append(float(loss.item()))
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test_torch)
        test_err = test_pred - y_test_torch
        test_loss = (test_err ** 2).mean()
    model.train()
    test_loss_history.append(float(test_loss.item()))
    # 最佳模型追踪
    if test_loss_history[-1] < best_test_loss:
        best_test_loss = test_loss_history[-1]
        best_epoch = epoch
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if epoch % PRINT_EVERY == 0 or epoch == 1:
        cla()
        start_epoch = builtins.max(0, epoch - PLOT_WINDOW)
        x_vals = range(start_epoch + 1, epoch + 1)
        y_train = [log10(v) for v in train_loss_history[start_epoch:]]
        y_test = [log10(v) for v in test_loss_history[start_epoch:]]
        plot(x_vals, y_train, "k.")
        plot(x_vals, y_test, "r.")
        xlabel("epoch")
        ylabel("log10 loss [scaled]")
        legend(["train", "test"])
        xlim(start_epoch + 1, epoch + 1)
        ax = gca()
        ax.text(0.98, 0.95, f"train={train_loss_history[-1]:.6f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="black")
        ax.text(0.98, 0.85, f"test={test_loss_history[-1]:.6f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="red")
        pause(1e-5)

# 恢复最佳模型
if best_state is not None:
    model.load_state_dict(best_state)
    print(f"Loaded best model from epoch {best_epoch} (test_loss={best_test_loss:.6f})")

policy_path = OUTPUT_DIR / f"uvr_next_policy_{TIMESTAMP}.pt"
torch.save({
    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
    "state_scale": STATE_SCALE.tolist(),
    "delta_scale": DELTA_SCALE.tolist(),
    "control_dt": CONTROL_DT,
    "start_time": START_TIME,
    "start_point": START_POINT,
    "n_fitting": N_FITTING,
    "end_second_derivative": 0.0,
    "r_eps": R_EPS,
    "heading_offset_rad": HEADING_OFFSET_RAD,
    "hidden": HIDDEN,
    "actions": ACTIONS,
    "first_model_idx": FIRST_MODEL_IDX,
    "n_pre": N_PRE,
    "n_pred": N_PRED,
    "train_trajectories": train_names,
    "test_trajectories": test_names,
    "input": "u,v,r at i-1 and i plus action at i",
    "output": "delta u,v,r from i to i+1",
}, policy_path)

loss_path = OUTPUT_DIR / f"loss_train_test_{TIMESTAMP}.png"
savefig(loss_path, dpi=200)

plot_dir = OUTPUT_DIR / "trajectory_plots"
plot_dir.mkdir(parents=True, exist_ok=True)
trajectory_paths = []
only_once = 1
for plot_name, plot_df in trajectories:
    if only_once == 1:
        only_once = 0  # 仅第一条轨迹生成详情图
        real_uvr = plot_df[STATE_COLUMNS].to_numpy(np.float64)
        pred_uvr = simulate_uvr(model, real_uvr, plot_df["action"].to_numpy())
        real_pose = plot_df[POSE_COLUMNS].to_numpy(np.float64)
        data_integrated_pose = integrate_pose_series(real_pose[0], real_uvr)
        pred_pose = integrate_pose_series(real_pose[0], pred_uvr)
        t = plot_df["original_time"].to_numpy() - plot_df["original_time"].iloc[0]

        figure(figsize=(9, 7))
        subplot(3, 1, 1)
        plot(real_pose[:, 0], real_pose[:, 1], "k-", label="data")
        plot(data_integrated_pose[:, 0], data_integrated_pose[:, 1], "m:", label="data integrated")
        plot(pred_pose[:, 0], pred_pose[:, 1], "r--", label="rollout sim")
        xlabel("X [m]")
        ylabel("Y [m]")
        axis("equal")
        legend(fontsize=8)

        subplot(3, 1, 2)
        plot(t, np.rad2deg(real_pose[:, 2]), "k-", label="data")
        plot(t, np.rad2deg(data_integrated_pose[:, 2]), "b:", label="data integrated")
        plot(t, np.rad2deg(pred_pose[:, 2]), "r--", label="rollout sim")
        xlabel("Time [s]")
        ylabel(r"Yaw [$^\circ$]")
        legend(fontsize=8)

        subplot2grid((3, 3), (2, 0), colspan=2)
        plot(t, np.zeros_like(t), 'm--')
        plot(t, real_uvr[:, 0], "k-", label="data u")
        plot(t, pred_uvr[:, 0], "k--", label="sim u")
        ylabel("u [m/s]")
        plot(t, np.zeros_like(t), 'm--')
        plot(t, real_uvr[:, 1], "b-", label="data v")
        plot(t, pred_uvr[:, 1], "b--", label="sim v")
        ylabel("v [m/s]")

        subplot(3, 3, 9)
        plot(t, np.zeros_like(t), 'm--')
        plot(t, real_uvr[:, 2], "r-", label="data r")
        plot(t, pred_uvr[:, 2], "r--", label="sim r")
        xlabel("Time [s]")
        ylabel("r [rad/s]")
        legend(fontsize=8, ncol=3)

        tight_layout()
        trajectory_path = plot_dir / f"{plot_name}_trajectory_fit_{TIMESTAMP}.png"
        savefig(trajectory_path, dpi=200)
        close()  # 仅保存，不显示
        trajectory_paths.append(trajectory_path)




plot_cut = 10
action_colors = {
    "A": "b",   # left
    "D": "r",   # right
    "W": "k",   # forward
    "S": "0.6", # optional: stop / no action
}

figure(figsize=(8, 4))
first_data = True
first_pred = True
for plot_name, plot_df in trajectories:
    if plot_name in test_names:
        subplot(122)
        real_uvr = plot_df[STATE_COLUMNS].to_numpy(np.float64).copy()
        actions = plot_df["action"].to_numpy()
        pred_uvr = simulate_uvr(model, real_uvr, actions)
        real_pose = plot_df[POSE_COLUMNS].to_numpy(np.float64).copy()
        pred_pose = integrate_pose_series(real_pose[0], pred_uvr)
        real_pose -= real_pose[0]
        pred_pose -= pred_pose[0]
        n = builtins.min(plot_cut, len(real_pose))

        for i in range(n - 1):
            c = action_colors.get(actions[i], "0.6")
            if first_data and i == 0:
                plot(real_pose[i:i + 2, 0], real_pose[i:i + 2, 1], color=c, linestyle="", marker=".", label="data")
            else:
                plot(real_pose[i:i + 2, 0], real_pose[i:i + 2, 1], color=c, linestyle="", marker=".")

            if first_pred and i == 0:
                plot(pred_pose[i:i + 2, 0], pred_pose[i:i + 2, 1], color=c, linestyle=":", label="rollout sim")
            else:
                plot(pred_pose[i:i + 2, 0], pred_pose[i:i + 2, 1], color=c, linestyle=":")
        first_data = False
        first_pred = False
        text(real_pose[n - 1, 0], real_pose[n - 1, 1], f"test {test_names.index(plot_name) + 1}", fontsize=9)
        axis('equal')
    else:
        subplot(121)
        real_uvr = plot_df[STATE_COLUMNS].to_numpy(np.float64).copy()
        actions = plot_df["action"].to_numpy()
        pred_uvr = simulate_uvr(model, real_uvr, actions)
        real_pose = plot_df[POSE_COLUMNS].to_numpy(np.float64).copy()
        pred_pose = integrate_pose_series(real_pose[0], pred_uvr)
        real_pose -= real_pose[0]
        pred_pose -= pred_pose[0]
        n = builtins.min(plot_cut, len(real_pose))

        for i in range(n - 1):
            c = action_colors.get(actions[i], "0.6")
            if first_data and i == 0:
                plot(real_pose[i:i + 2, 0], real_pose[i:i + 2, 1], color=c, linestyle="", marker=".", label="data")
            else:
                plot(real_pose[i:i + 2, 0], real_pose[i:i + 2, 1], color=c, linestyle="", marker=".")

            if first_pred and i == 0:
                plot(pred_pose[i:i + 2, 0], pred_pose[i:i + 2, 1], color=c, linestyle=":", label="rollout sim")
            else:
                plot(pred_pose[i:i + 2, 0], pred_pose[i:i + 2, 1], color=c, linestyle=":")
        first_data = False
        first_pred = False
        axis('equal')
xlabel("X [m]")
ylabel("Y [m]")
tight_layout()
train_test_path = OUTPUT_DIR / f"train_test_comparison_{TIMESTAMP}.png"
savefig(train_test_path, dpi=200)

figure(figsize=(5, 4))
first_idx = builtins.max(FIRST_MODEL_IDX, N_PRE)
for test_action in ["A", "D", "W"]:
    real_uvr = np.zeros((ACTION_TEST_NUM + first_idx + 1, 3), dtype=np.float64)
    actions = np.array([test_action] * len(real_uvr))
    pred_uvr = simulate_uvr(model, real_uvr, actions)
    pred_pose = integrate_pose_series(np.zeros(3), pred_uvr)
    pred_pose = pred_pose[first_idx:] - pred_pose[first_idx]
    plot(pred_pose[:, 0], pred_pose[:, 1], color=action_colors[test_action], label=f"{ACTION_TEST_NUM} {test_action}")

xlabel("X [m]")
ylabel("Y [m]")
axis("equal")
legend()
tight_layout()
action_test_path = OUTPUT_DIR / f"action_tests_{ACTION_TEST_NUM}_{TIMESTAMP}.png"
savefig(action_test_path, dpi=200)

print(f"\n[SAVED] {loss_path}")
print(f"[SAVED] {trajectory_path}")
print(f"[SAVED] {train_test_path}")
print(f"[SAVED] {action_test_path}")
