import builtins
import csv
import html
import json
import math
import os
import pickle
import sys
import time
import webbrowser
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm
os.environ.setdefault("QT_API", "PyQt6")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from numpy import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Enable TF32 tensor cores for ~2x matmul throughput on AMD/NVIDIA GPUs
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.pyplot import *
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.patches import Wedge, Circle, Rectangle, Polygon
from matplotlib.animation import PillowWriter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent  # 项目根目录
DEFAULT_DDM_PATH = str(PROJECT_DIR / "fit_outputs" / "uvr_next_fit_v33" / "uvr_next_policy_0718_2040.pt")
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "ppo_v33_sg_4096"
BEST_ARTIFACT_STEM = "train_test_teacher_student_best"


# ============================================================
# RUN SETTINGS
# ============================================================
# Edit these constants directly instead of passing command-line arguments.
RUN_MODE = "train"  # "config", "train", "test", "both", or "distill"
RUN_TRAINING = True
RUN_TESTING = True
PURE_RUN_TEST = 0
TEST_NUM = 1

DDM_PATH = str(DEFAULT_DDM_PATH)
OUTPUT_DIR = str(DEFAULT_OUTPUT_DIR)
DEVICE = "cuda"
AGENTS = 16384
MAX_STEP = 60
EPOCHS = 4096
HIDDEN = 256
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
UPDATE_EPOCHS = 1
BATCH_SIZE = 65536
DISTILL_EPOCHS = 1
DISTILL_BATCH_SIZE = 65536
DISTILL_LR = 3e-4
DISTILL_TEMPERATURE = 2.0
NO_OBSTACLE_PRIVILEGED = False
STUDENT_REFINE_EPOCHS = 5
TEACHER_POLICY_PATH = "auto"
STUDENT_POLICY_PATH = "auto"
BASELINE_TRAINING_LOGS = "auto"
WORLD_SIZE = 1.2
MAX_RANGE = 1.2
TARGET_RADIUS = 0.05
OBSTACLE_RADIUS = 0.10
SUCCESS_RADIUS = 0.02
CRASH_MARGIN = 0.05
U_REWARD_SCALE = 0.006
TARGET_DISTANCE_LEVELS = "0.35:0.45,0.45:0.60,0.60:0.75,0.75:0.90"
TARGET_ANGLE_DEG = 12.0
HEADING_JITTER_DEG = 10.0
RESUME_ACTOR = ""
RESUME_CRITIC = ""
SAVE_EVERY = 100
SAVE_BEST = False
SEED = random.randint(0, 2147483647)
TORCH_THREADS = 16
EMPTY_CACHE_EACH_EPOCH = False
PRINT_EVERY = 1
PLOT_EVENT_INTERVAL = 1000000
TRAINING_PLOT_EVERY = 0
SAVE_TRAINING_PLOTS = False
NO_REALTIME_PLOT = True
PROFILE = False

TEST_EPISODES = 16
TEST_DISTANCE_LEVEL = 3
TEST_POLICY_PATH = "auto"
TEST_DETERMINISTIC = True
TEST_REALTIME_PLOT = False
TEST_FRAME_PAUSE = 0.05
TEST_SHOW_PLOT = False
TEST_SAVE_PLOTS = False
EVAL_STUDENT_DISTILLATION = False

PUBLISH_PROGRESS_PAGE = True
OPEN_PROGRESS_PAGE = True
INITIAL_PROGRESS = {
    "avg_reward": 0.3343,
    "success_rate": 39.9,
    "distill_loss": 0.4023,
}


def args_from_constants():
    return SimpleNamespace(
        mode=RUN_MODE,
        ddm_path=DDM_PATH,
        output_dir=OUTPUT_DIR,
        device=DEVICE,
        agents=AGENTS,
        max_step=MAX_STEP,
        epochs=EPOCHS,
        hidden=HIDDEN,
        gamma=GAMMA,
        lam=LAM,
        clip=CLIP,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        update_epochs=UPDATE_EPOCHS,
        batch_size=BATCH_SIZE,
        distill_epochs=DISTILL_EPOCHS,
        distill_batch_size=DISTILL_BATCH_SIZE,
        distill_lr=DISTILL_LR,
        distill_temperature=DISTILL_TEMPERATURE,
        no_obstacle_privileged=NO_OBSTACLE_PRIVILEGED,
        student_refine_epochs=STUDENT_REFINE_EPOCHS,
        teacher_policy_path=TEACHER_POLICY_PATH,
        student_policy_path=STUDENT_POLICY_PATH,
        baseline_training_logs=BASELINE_TRAINING_LOGS,
        world_size=WORLD_SIZE,
        max_range=MAX_RANGE,
        target_radius=TARGET_RADIUS,
        obstacle_radius=OBSTACLE_RADIUS,
        success_radius=SUCCESS_RADIUS,
        crash_margin=CRASH_MARGIN,
        u_reward_scale=U_REWARD_SCALE,
        target_distance_levels=TARGET_DISTANCE_LEVELS,
        target_angle_deg=TARGET_ANGLE_DEG,
        heading_jitter_deg=HEADING_JITTER_DEG,
        resume_actor=RESUME_ACTOR,
        resume_critic=RESUME_CRITIC,
        save_every=SAVE_EVERY,
        save_best=SAVE_BEST,
        seed=SEED,
        torch_threads=TORCH_THREADS,
        empty_cache_each_epoch=EMPTY_CACHE_EACH_EPOCH,
        print_every=PRINT_EVERY,
        plot_event_interval=PLOT_EVENT_INTERVAL,
        training_plot_every=TRAINING_PLOT_EVERY,
        save_training_plots=SAVE_TRAINING_PLOTS,
        no_realtime_plot=NO_REALTIME_PLOT,
        profile=PROFILE,
        test_episodes=TEST_EPISODES,
        test_distance_level=TEST_DISTANCE_LEVEL,
        test_policy_path=TEST_POLICY_PATH,
        test_deterministic=TEST_DETERMINISTIC,
        test_realtime_plot=TEST_REALTIME_PLOT,
        test_frame_pause=TEST_FRAME_PAUSE,
        no_test_plot=not TEST_SHOW_PLOT,
        show_test_plot=TEST_SHOW_PLOT,
        save_test_plots=TEST_SAVE_PLOTS,
        eval_student_distillation=EVAL_STUDENT_DISTILLATION,
    )


def progress_paths(output_dir):
    output_dir = Path(output_dir)
    return output_dir / "training_progress.html", output_dir / "training_progress.json"


def json_safe(value):
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, (builtins.bool, np.bool_)):
        return builtins.bool(value)
    if isinstance(value, np.integer):
        return builtins.int(value)
    if isinstance(value, np.floating):
        return builtins.float(value)
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        if value.numel() == 1:
            return json_safe(value.detach().cpu().item())
        return [json_safe(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def publish_training_progress(output_dir, payload):
    if not PUBLISH_PROGRESS_PAGE:
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html_path, json_path = progress_paths(output_dir)
    payload = dict(payload)
    payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = json_safe(payload)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def metric(name, suffix="", precision=4):
        value = payload.get(name)
        if value is None:
            return "--"
        if isinstance(value, float):
            text = f"{value:.{precision}f}"
        else:
            text = str(value)
        return f"{html.escape(text)}{suffix}"

    def trend_svg(title, epochs, values, color, suffix="", precision=4):
        clean = [
            (float(x), float(y))
            for x, y in zip(epochs, values)
            if x is not None and y is not None
        ]
        title = html.escape(title)
        if len(clean) < 2:
            if clean:
                current = html.escape(f"{clean[-1][1]:.{precision}f}{suffix}")
            else:
                current = "--"
            return (
                f'<div class="chart-card"><div class="chart-head"><span>{title}</span>'
                f'<strong>{current}</strong></div><div class="empty-chart">waiting for more epochs</div></div>'
            )
        xs = [float(p[0]) for p in clean]
        ys = [float(p[1]) for p in clean]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if abs(x_max - x_min) < 1e-9:
            x_max = x_min + 1.0
        if abs(y_max - y_min) < 1e-9:
            pad = builtins.max(1.0, abs(y_min) * 0.1)
            y_min -= pad
            y_max += pad
        else:
            pad = (y_max - y_min) * 0.12
            y_min -= pad
            y_max += pad
        width, height = 620, 230
        left, right, top, bottom = 52, 18, 18, 36
        plot_w = width - left - right
        plot_h = height - top - bottom

        def sx(x):
            return left + (x - x_min) / (x_max - x_min) * plot_w

        def sy(y):
            return top + (y_max - y) / (y_max - y_min) * plot_h

        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in clean)
        last_value = clean[-1][1]
        last_text = f"{last_value:.{precision}f}{suffix}"
        y_ticks = [y_min, (y_min + y_max) / 2.0, y_max]
        tick_markup = "\n".join(
            f'<text x="44" y="{sy(v) + 4:.1f}" class="axis" text-anchor="end">{v:.2f}</text>'
            f'<line x1="{left}" y1="{sy(v):.1f}" x2="{width - right}" y2="{sy(v):.1f}" class="gridline" />'
            for v in y_ticks
        )
        return f"""
        <div class="chart-card">
          <div class="chart-head"><span>{title}</span><strong>{html.escape(last_text)}</strong></div>
          <svg viewBox="0 0 {width} {height}" role="img" aria-label="{title} trend over epochs">
            <rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" class="plot-bg" />
            {tick_markup}
            <text x="{left}" y="{height - 8}" class="axis">epoch {int(x_min)}</text>
            <text x="{width - right}" y="{height - 8}" class="axis" text-anchor="end">epoch {int(x_max)}</text>
            <polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />
            <circle cx="{sx(clean[-1][0]):.1f}" cy="{sy(clean[-1][1]):.1f}" r="4.5" fill="{color}" />
          </svg>
        </div>
        """

    status = html.escape(str(payload.get("status", "training")))
    epoch = html.escape(str(payload.get("epoch", "--")))
    total_epochs = html.escape(str(payload.get("total_epochs", "--")))
    updated = html.escape(str(payload["updated_at"]))
    history = payload.get("history") or {}
    history_epochs = history.get("epoch") or []
    reward_values = history.get("avg_reward") or []
    success_values = history.get("success_rate") or []
    distill_values = history.get("distill_loss") or []
    reward_chart = trend_svg("avg_reward", history_epochs, reward_values, "#65a9ff")
    success_chart = trend_svg("success_rate", history_epochs, success_values, "#4fd28b", "%", 1)
    distill_chart = trend_svg("distill_loss", history_epochs, distill_values, "#ffbe55")
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="2">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>teacher_student.py Progress</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Segoe UI, Arial, sans-serif;
      background: #111318;
      color: #f2f4f8;
    }}
    body {{
      margin: 0;
      min-height: 100vh;
      display: grid;
      place-items: center;
    }}
    main {{
      width: min(860px, calc(100vw - 40px));
    }}
    .top {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      border-bottom: 1px solid #2d3340;
      padding-bottom: 18px;
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      letter-spacing: 0;
    }}
    .status {{
      color: #9fb0c7;
      font-size: 14px;
      text-align: right;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .metric {{
      background: #1a1f29;
      border: 1px solid #2d3340;
      border-radius: 8px;
      padding: 18px;
    }}
    .label {{
      color: #9fb0c7;
      font-size: 13px;
      margin-bottom: 10px;
    }}
    .value {{
      font-size: 26px;
      line-height: 1.1;
      font-weight: 700;
    }}
    .charts {{
      display: grid;
      gap: 14px;
    }}
    .chart-card {{
      background: #1a1f29;
      border: 1px solid #2d3340;
      border-radius: 8px;
      padding: 16px;
    }}
    .chart-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
      color: #c8d3e5;
      font-size: 14px;
    }}
    .chart-head strong {{
      color: #f2f4f8;
      font-size: 18px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .plot-bg {{
      fill: #121722;
      stroke: #2d3340;
    }}
    .gridline {{
      stroke: #2d3340;
      stroke-width: 1;
    }}
    .axis {{
      fill: #9fb0c7;
      font-size: 12px;
    }}
    .empty-chart {{
      height: 160px;
      display: grid;
      place-items: center;
      color: #9fb0c7;
      background: #121722;
      border: 1px solid #2d3340;
      border-radius: 6px;
    }}
    .footer {{
      color: #9fb0c7;
      font-size: 13px;
      margin-top: 18px;
    }}
    @media (max-width: 720px) {{
      .top, .summary {{
        display: block;
      }}
      .status {{
        text-align: left;
        margin-top: 10px;
      }}
      .metric {{
        margin-bottom: 12px;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="top">
      <div>
        <h1>teacher_student.py Progress</h1>
        <div class="footer">Epoch {epoch} / {total_epochs}</div>
      </div>
      <div class="status">{status}<br>Updated {updated}</div>
    </div>
    <section class="summary">
      <div class="metric">
        <div class="label">avg_reward</div>
        <div class="value">{metric("avg_reward")}</div>
      </div>
      <div class="metric">
        <div class="label">success_rate</div>
        <div class="value">{metric("success_rate", "%", 1)}</div>
      </div>
      <div class="metric">
        <div class="label">distill_loss</div>
        <div class="value">{metric("distill_loss")}</div>
      </div>
    </section>
    <section class="charts">
      {reward_chart}
      {success_chart}
      {distill_chart}
    </section>
    <div class="footer">This page auto-refreshes every 2 seconds.</div>
  </main>
</body>
</html>
"""
    html_path.write_text(page, encoding="utf-8")
    return html_path


def open_training_progress_page(output_dir):
    if not (PUBLISH_PROGRESS_PAGE and OPEN_PROGRESS_PAGE):
        return None
    html_path, _ = progress_paths(output_dir)
    if html_path.exists():
        webbrowser.open(html_path.resolve().as_uri(), new=2)
    return html_path


class Config:
    def __init__(self, args):
        self.device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.agents_num = args.agents
        self.max_step = args.max_step
        self.train_epochs = args.epochs
        self.ddm_path = Path(args.ddm_path)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.gamma = args.gamma
        self.lam = args.lam
        self.clip = args.clip
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.update_epochs = args.update_epochs
        self.batch_size = args.batch_size

        self.policy_history_len = 5
        self.depth_dim = 31
        self.target_mask_dim = 31
        self.policy_velocity_dim = 2
        self.action_history_dim = 4  # v16: S/W/A/D 全部4个动作
        self.include_obstacle_privileged = not getattr(args, "no_obstacle_privileged", False)
        self.privileged_dim = 2 + (2 if self.include_obstacle_privileged else 0) + 3 + 3
        self.raw_teacher_feature_dim = (
            self.depth_dim
            + self.target_mask_dim
            + self.policy_velocity_dim
            + self.privileged_dim
        )
        self.raw_teacher_obs_dim = self.raw_teacher_feature_dim + self.action_history_dim
        self.raw_student_obs_dim = self.depth_dim + self.target_mask_dim
        self.teacher_obs_dim = self.raw_teacher_obs_dim * self.policy_history_len
        self.student_obs_dim = self.raw_student_obs_dim * self.policy_history_len
        self.obs_dim = self.teacher_obs_dim
        self.action_dim = 4  # v16: S/W/A/D 全部4个动作
        self.hidden = args.hidden
        self.distill_epochs = getattr(args, "distill_epochs", 1)
        self.distill_batch_size = getattr(args, "distill_batch_size", self.batch_size)
        self.distill_lr = getattr(args, "distill_lr", self.actor_lr)
        self.distill_temperature = getattr(args, "distill_temperature", 2.0)

        self.world_size = args.world_size
        self.max_range = args.max_range
        self.target_radius = args.target_radius
        self.obstacle_radius = args.obstacle_radius
        self.success_radius = args.success_radius
        self.crash_margin = args.crash_margin
        self.u_reward_scale = args.u_reward_scale
        self.target_distance_levels = [
            tuple(float(v) for v in item.split(":"))
            for item in args.target_distance_levels.split(",")
        ]
        self.target_angle_deg = args.target_angle_deg
        self.heading_jitter_deg = args.heading_jitter_deg


# ============================================================
# DDM wrapper V16 — N_PRED=1 model with N_PRE=3-step uvr history
# Input:  (n_pre+1)-step uvr history ((n_pre+1)*3) + action one-hot (4)
# Output: delta uvr (3) for next step
# Actions: S/W/A/D (4-动作，包含右转D)
# ============================================================

class FrozenUVRDDM:
    """Wrapper for uvr-delta model (n_pre+1-step history + 1 action → delta)."""
    def __init__(self, checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.device = device
        sd = checkpoint["model_state_dict"]
        w0_shape = sd["0.weight"].shape

        self.dt = float(checkpoint["control_dt"])
        self.r_eps = float(checkpoint.get("r_eps", 0.001))
        self.heading_offset = float(checkpoint.get("heading_offset_rad", 0.0))
        actions = checkpoint.get("actions", ["S", "W", "A", "D"])
        self.action_to_id = {a: i for i, a in enumerate(actions)}
        self.state_columns = ["u", "v", "r"]

        state_scale = np.array(checkpoint["state_scale"], dtype=np.float32)
        delta_scale = np.array(checkpoint["delta_scale"], dtype=np.float32)
        self.state_scale = torch.tensor(state_scale, device=device).view(1, 3)
        self.delta_scale = torch.tensor(delta_scale, device=device).view(1, 3)
        self.mean = torch.zeros(1, 3, device=device)
        self.std = self.state_scale.clone()

        # Model uses (n_pre+1)-step history; read n_pre from checkpoint, default 1 for ex-new code
        self.n_pre = checkpoint.get("n_pre", 1)
        self.history_len = self.n_pre + 1

        self.model = nn.Sequential(
            nn.Linear(w0_shape[1], checkpoint["hidden"]), nn.Tanh(),
            nn.Linear(checkpoint["hidden"], checkpoint["hidden"]), nn.Tanh(),
            nn.Linear(checkpoint["hidden"], sd["4.weight"].shape[0]),
        ).to(device)
        self.model.load_state_dict(sd)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        print(f"[DDM V16] Loaded (in={w0_shape[1]} hid={checkpoint['hidden']} out={sd['4.weight'].shape[0]} n_pre={self.n_pre} history_len={self.history_len}): {checkpoint_path}")

    def normalize(self, uvr):
        return uvr / self.state_scale

    def denormalize(self, uvr_norm):
        return uvr_norm * self.state_scale

    @torch.no_grad()
    def predict_next(self, uvr_history_norm, action_history, action):
        batch = uvr_history_norm.shape[0]
        next_action_history = torch.cat(
            [action_history[:, 1:], action.view(-1, 1).long()], dim=1
        )

        # Model uses ALL (n_pre+1) uvr history steps + current action
        uvr_hist = uvr_history_norm[:, -self.history_len:, :]   # (B, history_len, 3)
        action_one_hot = F.one_hot(action.long(), num_classes=4).float()  # (B, 4)
        x = torch.cat([uvr_hist.reshape(batch, -1), action_one_hot], dim=1)  # (B, (n_pre+1)*3+4)
        delta_norm = self.model(x)                              # (B, 3)

        scale_ratio = self.delta_scale / self.state_scale
        next_uvr_norm = uvr_hist[:, -1, :] + delta_norm * scale_ratio
        next_uvr = next_uvr_norm * self.state_scale
        next_uvr_history_norm = torch.cat(
            [uvr_history_norm[:, 1:, :], next_uvr_norm.unsqueeze(1)], dim=1
        )
        return next_uvr, next_uvr_norm, next_uvr_history_norm, next_action_history


# ============================================================


class AgentStateBuffer:
    def __init__(self, obs_dim, agent_num, max_step, device, student_obs_dim=None):
        self.obs_buffer = torch.zeros((max_step, agent_num, obs_dim), device=device)
        self.student_obs_buffer = None
        if student_obs_dim is not None:
            self.student_obs_buffer = torch.zeros((max_step, agent_num, student_obs_dim), device=device)
        self.action_buffer = torch.zeros((max_step, agent_num, 1), device=device, dtype=torch.long)
        self.logp_buffer = torch.zeros((max_step, agent_num, 1), device=device)
        self.reward_buffer = torch.zeros((max_step, agent_num, 1), device=device)
        self.over_buffer = torch.zeros((max_step, agent_num, 1), device=device)
        self.value_buffer = torch.zeros((max_step, agent_num, 1), device=device)
        self.adv_buffer = torch.zeros((max_step, agent_num, 1), device=device)
        self.return_buffer = torch.zeros((max_step, agent_num, 1), device=device)

    def compute_gae(self, next_value, gamma, lam):
        advantage = torch.zeros_like(self.reward_buffer[0])
        for t in reversed(range(self.reward_buffer.shape[0])):
            next_nonterminal = 1.0 - self.over_buffer[t]
            next_values = next_value if t == self.reward_buffer.shape[0] - 1 else self.value_buffer[t + 1]
            delta = self.reward_buffer[t] + gamma * next_values * next_nonterminal - self.value_buffer[t]
            advantage = delta + gamma * lam * next_nonterminal * advantage
            self.adv_buffer[t] = advantage
        self.return_buffer = self.adv_buffer + self.value_buffer
        std = self.adv_buffer.std()
        if std > 1e-6:
            self.adv_buffer = (self.adv_buffer - self.adv_buffer.mean()) / (std + 1e-8)


class Actor(nn.Module):
    def __init__(self, input_dim=66, hidden=256, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, input_dim=66, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


class USVHunterEnvDDM:
    def __init__(self, config):
        self.agent_num = config.agents_num
        self.device = config.device
        self.max_steps = config.max_step
        self.world_size = config.world_size
        self.max_range = config.max_range
        self.target_radius = config.target_radius
        self.obstacle_radius = config.obstacle_radius
        self.success_radius = config.success_radius
        self.crash_margin = config.crash_margin
        self.u_reward_scale = config.u_reward_scale
        self.ddm = FrozenUVRDDM(config.ddm_path, self.device)
        self.dt = self.ddm.dt

        self.pos = torch.zeros((self.agent_num, 2), device=self.device)
        self.psi = torch.zeros((self.agent_num, 1), device=self.device)
        self.vel = torch.zeros((self.agent_num, 3), device=self.device)
        self.target_pos = torch.zeros((self.agent_num, 2), device=self.device)
        self.target_speed = torch.zeros((self.agent_num, 1), device=self.device)
        self.target_heading = torch.zeros((self.agent_num, 1), device=self.device)
        self.target_w = torch.zeros((self.agent_num, 1), device=self.device)
        self.prev_target_dist = torch.zeros((self.agent_num, 1), device=self.device)
        self.step_count = torch.zeros((self.agent_num, 1), device=self.device)
        self.uvr_history_norm = torch.zeros((self.agent_num, self.ddm.history_len, 3), device=self.device)
        self.action_history = torch.zeros((self.agent_num, self.ddm.history_len), dtype=torch.long, device=self.device)

        self.n_points = 31
        self.raw_teacher_obs_dim = config.raw_teacher_obs_dim
        self.raw_teacher_feature_dim = config.raw_teacher_feature_dim
        self.raw_student_obs_dim = config.raw_student_obs_dim
        self.teacher_obs_dim = config.teacher_obs_dim
        self.student_obs_dim = config.student_obs_dim
        self.policy_history_len = config.policy_history_len
        self.include_obstacle_privileged = config.include_obstacle_privileged
        self.teacher_obs_history = torch.zeros(
            (self.agent_num, self.policy_history_len, self.raw_teacher_obs_dim),
            device=self.device,
        )
        self.student_obs_history = torch.zeros(
            (self.agent_num, self.policy_history_len, self.raw_student_obs_dim),
            device=self.device,
        )
        self.last_teacher_obs = torch.zeros((self.agent_num, self.teacher_obs_dim), device=self.device)
        self.last_student_obs = torch.zeros((self.agent_num, self.student_obs_dim), device=self.device)
        self.angles_body = torch.linspace(-math.radians(60), math.radians(60), self.n_points, device=self.device)
        self.angles_body_cos = torch.cos(self.angles_body).view(1, -1)
        self.angles_body_sin = torch.sin(self.angles_body).view(1, -1)
        self.ray_idx = torch.arange(self.n_points, device=self.device).float()
        self.obstacle_pos = torch.zeros((self.agent_num, 2), device=self.device)
        self.target_distance_levels = config.target_distance_levels
        self.distance_level = 0
        self.target_distance_min, self.target_distance_max = self.target_distance_levels[self.distance_level]
        self.target_angle_deg = config.target_angle_deg
        self.heading_jitter_deg = config.heading_jitter_deg

        # ── 预计算所有障碍线段 + 圆形参数（广播射线检测用）──
        sq = pure_square_vertices()
        tri = pure_triangle_vertices()
        seg_s = torch.tensor([sq[0], sq[1], sq[2], sq[3], tri[0], tri[1], tri[2]],
                            dtype=torch.float32, device=self.device)  # (7,2)
        seg_e = torch.tensor([sq[1], sq[2], sq[3], sq[0], tri[1], tri[2], tri[0]],
                            dtype=torch.float32, device=self.device)  # (7,2)
        self.obs_seg_s = seg_s.view(1, 1, 7, 2)   # (1,1,7,2) 广播用
        self.obs_seg_v = (seg_e - seg_s).view(1, 1, 7, 2)
        self.obs_circ_c = torch.tensor(PURE_CIRCLE_CENTER, dtype=torch.float32, device=self.device)
        self.obs_circ_r = PURE_CIRCLE_RADIUS

    def _ada_pattern(self):
        """Return initial action history pattern [A, D, A...] for warm-up."""
        a_id = self.ddm.action_to_id.get("A", 2)
        d_id = self.ddm.action_to_id.get("D", 3)
        s_id = self.ddm.action_to_id.get("S", 0)
        return torch.tensor([a_id, d_id, a_id] + [s_id] * (self.ddm.history_len - 3),
                            dtype=torch.long, device=self.device)

    def _make_history_obs(self, indices=None):
        if indices is None:
            teacher_history = self.teacher_obs_history
            student_history = self.student_obs_history
        else:
            teacher_history = self.teacher_obs_history[indices]
            student_history = self.student_obs_history[indices]
        return (
            teacher_history.reshape(teacher_history.shape[0], -1),
            student_history.reshape(student_history.shape[0], -1),
        )

    def _reset_obs_history(self, indices, teacher_obs, student_obs):
        zero_action = torch.zeros((teacher_obs.shape[0], 4), device=self.device)  # v16: 4 actions S/W/A/D
        teacher_obs_with_action = torch.cat((teacher_obs, zero_action), dim=1)
        self.teacher_obs_history[indices] = teacher_obs_with_action.unsqueeze(1).repeat(1, self.policy_history_len, 1)
        self.student_obs_history[indices] = student_obs.unsqueeze(1).repeat(1, self.policy_history_len, 1)
        return self._make_history_obs(indices)

    def _append_obs_history(self, teacher_obs, student_obs, action_idx):
        action_one_hot = F.one_hot(action_idx.view(-1).long(), num_classes=4).float()  # v16: 4 actions S/W/A/D
        teacher_obs_with_action = torch.cat((teacher_obs, action_one_hot), dim=1)
        self.teacher_obs_history = torch.cat(
            (self.teacher_obs_history[:, 1:, :], teacher_obs_with_action.unsqueeze(1)),
            dim=1,
        )
        self.student_obs_history = torch.cat((self.student_obs_history[:, 1:, :], student_obs.unsqueeze(1)), dim=1)
        return self._make_history_obs()

    def _target_velocity_world(self, indices):
        return torch.cat(
            (
                self.target_speed[indices] * torch.cos(self.target_heading[indices]),
                self.target_speed[indices] * torch.sin(self.target_heading[indices]),
            ),
            dim=1,
        )

    def _target_velocity_usv_frame(self, indices, heading):
        target_velocity_world = self._target_velocity_world(indices)
        cos_h = torch.cos(heading)
        sin_h = torch.sin(heading)
        vx_world = target_velocity_world[:, 0:1]
        vy_world = target_velocity_world[:, 1:2]
        vx_body = cos_h * vx_world + sin_h * vy_world
        vy_body = -sin_h * vx_world + cos_h * vy_world
        return torch.cat((vx_body, vy_body, self.target_w[indices]), dim=1)

    def _build_observations(self, indices, sensor_depth, target_mask):
        depth_obs = sensor_depth / self.max_range
        target_vec = self.target_pos[indices] - self.pos[indices]
        target_dist = torch.norm(target_vec, dim=1, keepdim=True).clamp_min(1e-6)
        target_angle = torch.atan2(target_vec[:, 1:2], target_vec[:, 0:1])
        heading = self.psi[indices] + self.ddm.heading_offset
        rel_bearing = torch.atan2(torch.sin(target_angle - heading), torch.cos(target_angle - heading))
        rel_state = torch.cat((target_dist / self.max_range, rel_bearing / math.pi), dim=1)
        obstacle_vec = self._nearest_obstacle_vectors(self.pos[indices])
        obstacle_dist = torch.norm(obstacle_vec, dim=1, keepdim=True).clamp_min(1e-6)
        obstacle_angle = torch.atan2(obstacle_vec[:, 1:2], obstacle_vec[:, 0:1])
        obstacle_bearing = torch.atan2(torch.sin(obstacle_angle - heading), torch.cos(obstacle_angle - heading))
        obstacle_state = torch.cat((obstacle_dist / self.max_range, obstacle_bearing / math.pi), dim=1)
        usv_velocity = self.vel[indices]
        policy_velocity = self.vel[indices][:, [0, 2]]
        target_velocity = self._target_velocity_usv_frame(indices, heading)
        closest_target_ray = torch.argmin(
            torch.abs(torch.atan2(torch.sin(rel_bearing - self.angles_body), torch.cos(rel_bearing - self.angles_body))),
            dim=1,
        )
        target_color_obs = F.one_hot(closest_target_ray, num_classes=self.n_points).float()
        teacher_parts = [depth_obs, target_mask, policy_velocity, rel_state]
        if self.include_obstacle_privileged:
            teacher_parts.append(obstacle_state)
        teacher_parts.extend((usv_velocity, target_velocity))
        teacher_obs = torch.cat(teacher_parts, dim=1)
        student_obs = torch.cat((depth_obs, target_color_obs), dim=1)
        return teacher_obs, student_obs

    def set_distance_level(self, level):
        self.distance_level = int(builtins.max(0, builtins.min(level, len(self.target_distance_levels) - 1)))
        self.target_distance_min, self.target_distance_max = self.target_distance_levels[self.distance_level]

    def _fixed_obstacle_centers(self):
        return torch.tensor(
            [PURE_SQUARE_CENTER, PURE_CIRCLE_CENTER, PURE_TRIANGLE_CENTER],
            dtype=torch.float32,
            device=self.device,
        )

    def _nearest_obstacle_vectors(self, positions):
        centers = self._fixed_obstacle_centers().view(1, 3, 2)
        deltas = centers - positions.unsqueeze(1)
        distances = torch.norm(deltas, dim=2)
        nearest = torch.argmin(distances, dim=1)
        return deltas[torch.arange(positions.shape[0], device=self.device), nearest]

    def _nearest_obstacle_centers(self, positions):
        centers = self._fixed_obstacle_centers()
        deltas = centers.view(1, 3, 2) - positions.unsqueeze(1)
        nearest = torch.argmin(torch.norm(deltas, dim=2), dim=1)
        return centers[nearest]

    def _ray_circle_distance(self, pos, dir_x, dir_y, center, radius):
        center = torch.tensor(center, dtype=torch.float32, device=self.device).view(1, 2)
        oc_x = pos[:, 0:1] - center[:, 0:1]
        oc_y = pos[:, 1:2] - center[:, 1:2]
        b = oc_x * dir_x + oc_y * dir_y
        c = oc_x ** 2 + oc_y ** 2 - radius ** 2
        disc = b ** 2 - c
        t = -b - torch.sqrt(torch.relu(disc))
        return torch.where((disc > 0) & (t > 0), t, torch.full_like(dir_x, float("inf")))

    @staticmethod
    def _seg_dist(pos, dir_x, dir_y, sx, sy, ex, ey):
        """内联线段距离（无 Python 循环，单次向量化调用）."""
        seg_x = ex - sx; seg_y = ey - sy
        rel_x = sx - pos[:, 0:1]; rel_y = sy - pos[:, 1:2]
        denom = dir_x * seg_y - dir_y * seg_x
        safe_denom = torch.where(torch.abs(denom) > 1e-8, denom, torch.ones_like(denom))
        t = (rel_x * seg_y - rel_y * seg_x) / safe_denom
        u = (rel_x * dir_y - rel_y * dir_x) / safe_denom
        valid = (torch.abs(denom) > 1e-8) & (t > 0) & (u >= 0) & (u <= 1)
        return torch.where(valid, t, torch.full_like(dir_x, float("inf")))

    def _ray_fixed_obstacle_distance(self, pos, dir_x, dir_y):
        # 硬编码正方形4边（消除 Python for 循环）
        vs = pure_square_vertices()
        d_sq = self._seg_dist(pos, dir_x, dir_y, vs[0][0], vs[0][1], vs[1][0], vs[1][1])
        for i in range(1, 4):
            d_sq = torch.min(d_sq, self._seg_dist(pos, dir_x, dir_y, vs[i][0], vs[i][1], vs[(i+1)%4][0], vs[(i+1)%4][1]))
        d_circ = self._ray_circle_distance(pos, dir_x, dir_y, PURE_CIRCLE_CENTER, PURE_CIRCLE_RADIUS)
        # 硬编码三角形3边
        vt = pure_triangle_vertices()
        d_tri = self._seg_dist(pos, dir_x, dir_y, vt[0][0], vt[0][1], vt[1][0], vt[1][1])
        d_tri = torch.min(d_tri, self._seg_dist(pos, dir_x, dir_y, vt[1][0], vt[1][1], vt[2][0], vt[2][1]))
        d_tri = torch.min(d_tri, self._seg_dist(pos, dir_x, dir_y, vt[2][0], vt[2][1], vt[0][0], vt[0][1]))
        return torch.min(torch.min(d_sq, d_circ), d_tri)

    @staticmethod
    def _point_in_poly(x, y, vertices):
        """硬编码多边形点在内部测试（消除 Python for 循环）."""
        inside = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=x.device)
        n = len(vertices)
        for i in range(n):
            xi, yi = vertices[i]
            xj, yj = vertices[(i + 1) % n]
            crosses = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            inside = inside ^ crosses
        return inside

    def _points_in_fixed_obstacles(self, points):
        x, y = points[:, 0:1], points[:, 1:2]
        sq = self._point_in_poly(x, y, pure_square_vertices())
        cc = torch.tensor(PURE_CIRCLE_CENTER, dtype=torch.float32, device=self.device).view(1, 2)
        circ = torch.norm(points - cc, dim=1, keepdim=True) <= PURE_CIRCLE_RADIUS
        tri = self._point_in_poly(x, y, pure_triangle_vertices())
        return sq | circ | tri

    def _segment_clearance_to_center(self, start, end, center):
        segment = end - start
        denom = torch.sum(segment * segment, dim=1, keepdim=True).clamp_min(1e-6)
        t = torch.sum((center - start) * segment, dim=1, keepdim=True) / denom
        t = torch.clamp(t, 0.0, 1.0)
        closest = start + t * segment
        return torch.norm(closest - center, dim=1)

    def _target_path_valid(self, path_clearance):
        if self.distance_level <= 1:
            return path_clearance > self.obstacle_radius + 0.08
        if self.distance_level == 2:
            return path_clearance > self.obstacle_radius - 0.02
        return torch.ones_like(path_clearance, dtype=torch.bool)

    def _sample_target_angle(self, boat_pos, center_subset):
        m = boat_pos.shape[0]
        target_vec = boat_pos - center_subset
        outward = torch.atan2(target_vec[:, 1:2], target_vec[:, 0:1])
        if self.distance_level <= 1:
            tangent_sign = torch.where(torch.rand((m, 1), device=self.device) > 0.5, 1.0, -1.0)
            base_angle = outward + tangent_sign * math.pi / 2.0
        elif self.distance_level == 2:
            tangent_sign = torch.where(torch.rand((m, 1), device=self.device) > 0.5, 1.0, -1.0)
            tangent_angle = outward + tangent_sign * math.pi / 2.0
            inward_angle = outward + math.pi
            use_inward = torch.rand((m, 1), device=self.device) < 0.35
            base_angle = torch.where(use_inward, inward_angle, tangent_angle)
        else:
            tangent_sign = torch.where(torch.rand((m, 1), device=self.device) > 0.5, 1.0, -1.0)
            tangent_angle = outward + tangent_sign * math.pi / 2.0
            inward_angle = outward + math.pi
            use_inward = torch.rand((m, 1), device=self.device) < 0.70
            base_angle = torch.where(use_inward, inward_angle, tangent_angle)
        angle_jitter = (torch.rand((m, 1), device=self.device) - 0.5) * math.radians(2.0 * self.target_angle_deg)
        return base_angle + angle_jitter

    def _sample_obstacle_positions(self, n):
        center = torch.tensor(PURE_CIRCLE_CENTER, dtype=torch.float32, device=self.device).view(1, 2)
        return center.repeat(n, 1)

    def _sample_boat_positions(self, n, obstacle_pos):
        x = PURE_USV_X_RANGE[0] + torch.rand((n, 1), device=self.device) * (PURE_USV_X_RANGE[1] - PURE_USV_X_RANGE[0])
        y = PURE_USV_Y_RANGE[0] + torch.rand((n, 1), device=self.device) * (PURE_USV_Y_RANGE[1] - PURE_USV_Y_RANGE[0])
        return torch.cat((x, y), dim=1)

    def _sample_target_positions(self, boat_pos, obstacle_pos):
        n = boat_pos.shape[0]
        x = PURE_TARGET_X_RANGE[0] + torch.rand((n, 1), device=self.device) * (PURE_TARGET_X_RANGE[1] - PURE_TARGET_X_RANGE[0])
        y = PURE_TARGET_Y_RANGE[0] + torch.rand((n, 1), device=self.device) * (PURE_TARGET_Y_RANGE[1] - PURE_TARGET_Y_RANGE[0])
        return torch.cat((x, y), dim=1)

    def _target_motion_is_safe(self, target_pos, target_speed, target_heading, obstacle_pos, max_steps=None):
        if max_steps is None:
            max_steps = self.max_steps
        steps = torch.arange(max_steps + 1, device=self.device, dtype=torch.float32).view(-1, 1, 1)
        step_delta = torch.cat(
            (
                target_speed * torch.cos(target_heading) * self.dt,
                target_speed * torch.sin(target_heading) * self.dt,
            ),
            dim=1,
        ).unsqueeze(0)
        trajectory = target_pos.unsqueeze(0) + steps * step_delta

        inside_wall = (
            (trajectory[:, :, 0:1] >= self.target_radius)
            & (trajectory[:, :, 0:1] <= self.world_size - self.target_radius)
            & (trajectory[:, :, 1:2] >= self.target_radius)
            & (trajectory[:, :, 1:2] <= self.world_size - self.target_radius)
        ).all(dim=0)
        flat_trajectory = trajectory.reshape(-1, 2)
        outside_obstacle = (~self._points_in_fixed_obstacles(flat_trajectory)).view(
            trajectory.shape[0],
            trajectory.shape[1],
            1,
        ).all(dim=0)
        return inside_wall & outside_obstacle

    def _sample_world_safe_target_candidates(self, n):
        pos = self._sample_target_positions(torch.empty((n, 2), device=self.device), None)
        speed = torch.zeros((n, 1), device=self.device)
        heading = torch.rand((n, 1), device=self.device) * math.pi / 2.0
        return pos, speed, heading

    def _sample_safe_target_states(self, boat_pos, obstacle_pos, max_attempts=64):
        n = boat_pos.shape[0]
        target_pos = torch.zeros((n, 2), device=self.device)
        target_speed = torch.zeros((n, 1), device=self.device)
        target_heading = torch.zeros((n, 1), device=self.device)
        remaining = torch.ones((n, 1), dtype=torch.bool, device=self.device)
        attempts = 0

        while remaining.any():
            attempts += 1
            idx = torch.nonzero(remaining.flatten(), as_tuple=False).flatten()
            if attempts <= max_attempts:
                born_pos = self._sample_target_positions(boat_pos[idx], obstacle_pos[idx])
                born_speed = torch.rand((idx.numel(), 1), device=self.device) * PURE_TARGET_SPEED_RANGE[1]
                born_heading = torch.rand((idx.numel(), 1), device=self.device) * math.pi / 2.0
            else:
                born_pos, born_speed, born_heading = self._sample_world_safe_target_candidates(idx.numel())

            safe = self._target_motion_is_safe(
                born_pos,
                born_speed,
                born_heading,
                obstacle_pos[idx],
            ).flatten()

            safe_idx = idx[safe]
            if safe_idx.numel() > 0:
                target_pos[safe_idx] = born_pos[safe]
                target_speed[safe_idx] = born_speed[safe]
                target_heading[safe_idx] = born_heading[safe]
                remaining[safe_idx] = False

        return target_pos, target_speed, target_heading

    def reset(self, indices=None):
        if indices is None:
            return self.reset_mask()
        return self.reset_indices(indices)

    def reset_indices(self, indices):
        n = indices.shape[0]
        if n == 0:
            return torch.empty((0, self.teacher_obs_dim), device=self.device)

        self.obstacle_pos[indices] = self._sample_obstacle_positions(n)
        new_pos = self._sample_boat_positions(n, self.obstacle_pos[indices])
        self.pos[indices] = new_pos
        self.obstacle_pos[indices] = self._nearest_obstacle_centers(self.pos[indices])

        new_target_pos, new_target_speed, new_target_heading = self._sample_safe_target_states(
            self.pos[indices],
            self.obstacle_pos[indices],
        )
        self.target_pos[indices] = new_target_pos

        start_heading = -torch.rand((n, 1), device=self.device) * math.pi / 2.0
        self.psi[indices] = start_heading - self.ddm.heading_offset

        self.vel[indices] = 0.0
        self.step_count[indices] = 0.0
        if not hasattr(self, 'total_rotation'):
            self.total_rotation = torch.zeros((self.agent_num, 1), device=self.device)

        self.total_rotation[indices] = 0.0
        zero_uvr_norm = self.ddm.normalize(torch.zeros((n, 3), device=self.device))
        self.uvr_history_norm[indices] = zero_uvr_norm.unsqueeze(1).repeat(1, self.ddm.history_len, 1)
        # v16: 初始动作为 ADA 模式，适配 N_PRE=3 的历史输入
        self.action_history[indices] = self._ada_pattern().unsqueeze(0).repeat(n, 1)

        self.target_speed[indices] = new_target_speed
        self.target_heading[indices] = new_target_heading
        self.target_w[indices] = 0.0
        self.prev_target_dist[indices] = torch.norm(self.target_pos[indices] - self.pos[indices], dim=1, keepdim=True)
        raw_teacher_obs, raw_student_obs, _, _ = self._update_perception(indices)
        teacher_obs, student_obs = self._reset_obs_history(indices, raw_teacher_obs, raw_student_obs)
        self.last_teacher_obs[indices] = teacher_obs
        self.last_student_obs[indices] = student_obs
        return teacher_obs

    def reset_mask(self, reset_mask=None):
        if reset_mask is None:
            reset_mask = torch.ones((self.agent_num, 1), dtype=torch.bool, device=self.device)
        reset_mask = reset_mask.view(self.agent_num, 1).bool()
        reset_flat = reset_mask.squeeze(1)
        new_obstacle_pos = self._sample_obstacle_positions(self.agent_num)
        self.obstacle_pos = torch.where(reset_mask.expand_as(self.obstacle_pos), new_obstacle_pos, self.obstacle_pos)
        new_pos = self._sample_boat_positions(self.agent_num, self.obstacle_pos)
        self.pos = torch.where(reset_mask, new_pos, self.pos)
        nearest_obstacle_pos = self._nearest_obstacle_centers(self.pos)
        self.obstacle_pos = torch.where(reset_mask.expand_as(self.obstacle_pos), nearest_obstacle_pos, self.obstacle_pos)

        new_target_pos, new_target_speed, new_target_heading = self._sample_safe_target_states(
            self.pos,
            self.obstacle_pos,
        )
        self.target_pos = torch.where(reset_mask, new_target_pos, self.target_pos)

        start_heading = -torch.rand((self.agent_num, 1), device=self.device) * math.pi / 2.0
        self.psi = torch.where(reset_mask, start_heading - self.ddm.heading_offset, self.psi)

        self.vel = torch.where(reset_mask.expand_as(self.vel), torch.zeros_like(self.vel), self.vel)
        self.step_count = torch.where(reset_mask, torch.zeros_like(self.step_count), self.step_count)
        if not hasattr(self, 'total_rotation'):
            self.total_rotation = torch.zeros((self.agent_num, 1), device=self.device)

        self.total_rotation = torch.where(
            reset_mask,
            torch.zeros_like(self.total_rotation),
            self.total_rotation
        )
        zero_uvr_norm = self.ddm.normalize(torch.zeros((self.agent_num, 3), device=self.device))
        zero_history = zero_uvr_norm.unsqueeze(1).repeat(1, self.ddm.history_len, 1)
        self.uvr_history_norm = torch.where(reset_mask.view(self.agent_num, 1, 1), zero_history, self.uvr_history_norm)
        self.action_history = torch.where(
            reset_mask.expand_as(self.action_history),
            self._ada_pattern().unsqueeze(0).repeat(self.agent_num, 1),
            self.action_history,
        )
        self.target_speed = torch.where(reset_mask, new_target_speed, self.target_speed)
        self.target_heading = torch.where(reset_mask, new_target_heading, self.target_heading)
        self.target_w = torch.where(reset_mask, torch.zeros_like(self.target_w), self.target_w)
        new_prev_target_dist = torch.norm(self.target_pos - self.pos, dim=1, keepdim=True)
        self.prev_target_dist = torch.where(reset_mask, new_prev_target_dist, self.prev_target_dist)
        raw_teacher_obs, raw_student_obs, _, _ = self._update_perception()
        reset_indices = torch.nonzero(reset_flat, as_tuple=False).flatten()
        if reset_indices.numel() > 0:
            teacher_obs, student_obs = self._reset_obs_history(
                reset_indices,
                raw_teacher_obs[reset_indices],
                raw_student_obs[reset_indices],
            )
            self.last_teacher_obs[reset_indices] = teacher_obs
            self.last_student_obs[reset_indices] = student_obs
        return self.last_teacher_obs

    def _batched_raycast_semantic(self, indices):
        N = indices.shape[0]
        pos = self.pos[indices]
        psi = self.psi[indices] + self.ddm.heading_offset
        target_pos = self.target_pos[indices]
        cos_p = torch.cos(psi)
        sin_p = torch.sin(psi)
        dir_x = cos_p * self.angles_body_cos - sin_p * self.angles_body_sin + 1e-8  # (N,31)
        dir_y = sin_p * self.angles_body_cos + cos_p * self.angles_body_sin + 1e-8

        # 墙壁距离
        ws = self.world_size
        tx0 = torch.where(dir_x < 0, -pos[:, 0:1] / dir_x, torch.full_like(dir_x, float('inf')))
        tx1 = torch.where(dir_x > 0, (ws - pos[:, 0:1]) / dir_x, torch.full_like(dir_x, float('inf')))
        ty0 = torch.where(dir_y < 0, -pos[:, 1:2] / dir_y, torch.full_like(dir_y, float('inf')))
        ty1 = torch.where(dir_y > 0, (ws - pos[:, 1:2]) / dir_y, torch.full_like(dir_y, float('inf')))
        dist = torch.min(torch.min(tx0, tx1), torch.min(ty0, ty1))

        # 障碍线段: 广播 (N,31,7) 一次算完
        dir4x = dir_x.view(N, 31, 1); dir4y = dir_y.view(N, 31, 1)
        oc_x = (self.obs_seg_s[..., 0] - pos[:, 0:1].view(N, 1, 1)).expand(N, 31, 7)
        oc_y = (self.obs_seg_s[..., 1] - pos[:, 1:2].view(N, 1, 1)).expand(N, 31, 7)
        denom = dir4x * self.obs_seg_v[..., 0] + dir4y * self.obs_seg_v[..., 1]
        safe_d = torch.where(torch.abs(denom) > 1e-8, denom, torch.ones_like(denom))
        t_seg = (oc_x * self.obs_seg_v[..., 1] - oc_y * self.obs_seg_v[..., 0]) / safe_d
        u_seg = (oc_x * dir4y - oc_y * dir4x) / safe_d
        valid = (torch.abs(denom) > 1e-8) & (t_seg > 0) & (u_seg >= 0) & (u_seg <= 1)
        t_seg = torch.where(valid, t_seg, torch.full_like(t_seg, float('inf')))
        dist = torch.min(dist, t_seg.min(dim=2).values)

        # 圆形障碍
        cx, cy = self.obs_circ_c[0], self.obs_circ_c[1]
        ocx, ocy = pos[:, 0:1] - cx, pos[:, 1:2] - cy
        b = ocx * dir_x + ocy * dir_y; c = ocx*ocx + ocy*ocy - self.obs_circ_r*self.obs_circ_r
        disc = b*b - c
        t_circ = -b - torch.sqrt(torch.relu(disc))
        t_circ = torch.where((disc > 0) & (t_circ > 0), t_circ, torch.full_like(t_circ, float('inf')))
        dist = torch.min(dist, t_circ)

        # 目标检测
        tox, toy = pos[:, 0:1] - target_pos[:, 0:1], pos[:, 1:2] - target_pos[:, 1:2]
        tb = tox * dir_x + toy * dir_y; tc = tox*tox + toy*toy - self.target_radius*self.target_radius
        tdisc = tb*tb - tc
        t_tgt = torch.where(tdisc > 0, -tb - torch.sqrt(torch.relu(tdisc)), torch.full_like(tdisc, float('inf')))
        t_tgt = torch.where(t_tgt > 0, t_tgt, torch.full_like(t_tgt, float('inf')))
        hit_target = (t_tgt < dist) & (t_tgt < self.max_range)

        sensor_depth = torch.clamp(dist, 0.02, self.max_range)
        return sensor_depth, hit_target.float(), dist

    def _update_perception(self, indices=None):
        if indices is None:
            indices = torch.arange(self.agent_num, device=self.device)
        sensor_depth, target_mask, physical_dist = self._batched_raycast_semantic(indices)
        teacher_obs, student_obs = self._build_observations(indices, sensor_depth, target_mask)
        return teacher_obs, student_obs, target_mask, physical_dist

    def _integrate_pose(self, next_uvr):
        u = next_uvr[:, 0:1]
        v = next_uvr[:, 1:2]
        r = next_uvr[:, 2:3]
        turn = r * self.dt
        abs_r = torch.abs(r)
        r_safe = torch.where(abs_r > self.ddm.r_eps, r, torch.ones_like(r))
        dx_arc = (u / r_safe) * torch.sin(turn) + (v / r_safe) * (torch.cos(turn) - 1.0)
        dy_arc = (u / r_safe) * (1.0 - torch.cos(turn)) + (v / r_safe) * torch.sin(turn)
        dx_line = u * self.dt
        dy_line = v * self.dt
        use_arc = abs_r > self.ddm.r_eps
        dx_body = torch.where(use_arc, dx_arc, dx_line)
        dy_body = torch.where(use_arc, dy_arc, dy_line)
        heading_current = self.psi + self.ddm.heading_offset
        self.pos[:, 0:1] += dx_body * torch.cos(heading_current) - dy_body * torch.sin(heading_current)
        self.pos[:, 1:2] += dx_body * torch.sin(heading_current) + dy_body * torch.cos(heading_current)
        self.psi += r * self.dt

    def _update_target(self):
        self.target_w.zero_()
        self.target_pos[:, 0:1] += self.target_speed * torch.cos(self.target_heading) * self.dt
        self.target_pos[:, 1:2] += self.target_speed * torch.sin(self.target_heading) * self.dt

    def step(self, action_idx):
        action_flat = action_idx.view(-1).long()
        next_uvr, _, self.uvr_history_norm, self.action_history = self.ddm.predict_next(
            self.uvr_history_norm, self.action_history, action_flat
        )
        self.vel = next_uvr
        self._integrate_pose(next_uvr)
        self._update_target()
        self.step_count += 1
        # total accumulated rotation angle
        if not hasattr(self, 'total_rotation'):
            self.total_rotation = torch.zeros((self.agent_num, 1), device=self.device)

        self.total_rotation += torch.abs(next_uvr[:, 2:3] * self.dt)

        raw_teacher_obs, raw_student_obs, target_mask, physical_dist = self._update_perception()
        dist_to_target = torch.norm(self.target_pos - self.pos, dim=1, keepdim=True)
        crash_wall = (
            (self.pos[:, 0:1] <= 0.0)
            | (self.pos[:, 0:1] >= self.world_size)
            | (self.pos[:, 1:2] <= 0.0)
            | (self.pos[:, 1:2] >= self.world_size)
        )
        crash_obs = self._points_in_fixed_obstacles(self.pos)
        # fail if accumulated rotation exceeds 360 deg
        spin_fail = self.total_rotation > (2.0 * math.pi)

        crash = crash_wall | crash_obs | spin_fail
        success = dist_to_target < self.success_radius
        valid_success = success & ~crash

        sees_target = (target_mask.sum(dim=1, keepdim=True) > 0).float()
        progress = self.prev_target_dist - dist_to_target
        shaped_progress = torch.clamp(2.0 * progress, -0.05, 0.08)
        reward = -0.10 + shaped_progress
        timeout = self.step_count >= self.max_steps
        reward = torch.where(timeout & ~valid_success & ~crash, torch.full_like(reward, -10.0), reward)
        reward = torch.where(crash, torch.full_like(reward, -50.0), reward)
        success_reward = 80.0 + 20.0 * sees_target
        reward = torch.where(valid_success, success_reward, reward)
        self.prev_target_dist = torch.where(crash | valid_success | timeout, self.prev_target_dist, dist_to_target)

        over = crash | timeout | valid_success
        teacher_obs, student_obs = self._append_obs_history(raw_teacher_obs, raw_student_obs, action_flat)
        self.last_teacher_obs = teacher_obs
        self.last_student_obs = student_obs
        return teacher_obs, reward, over.float(), valid_success, crash


class TrainingLogger:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.history = {"epoch": [], "reward": [], "success": [], "crash": [], "timeout": []}

    def update(self, epoch, reward, success_rate, crash_rate, timeout_rate):
        self.history["epoch"].append(epoch)
        self.history["reward"].append(reward)
        self.history["success"].append(success_rate)
        self.history["crash"].append(crash_rate)
        self.history["timeout"].append(timeout_rate)
        pd = __import__("pandas")
        pd.DataFrame(self.history).to_csv(self.output_dir / "ppo_training_log.csv", index=False)

    def save_copy(self, path):
        pd = __import__("pandas")
        pd.DataFrame(self.history).to_csv(path, index=False)


def pump_gui_events():
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents()


def sync_if_cuda(device):
    if isinstance(device, str) and device.startswith("cuda"):
        torch.cuda.synchronize()


def load_baseline_training_figure(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def load_training_log_rows(path):
    path = Path(path)
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows or None


def row_value(row, *names):
    for name in names:
        if name in row and row[name] != "":
            return float(row[name])
    raise KeyError(names[0])


def resolve_baseline_training_logs(args, cfg):
    setting = getattr(args, "baseline_training_logs", "auto")
    if not setting or str(setting).lower() == "none":
        return []
    if str(setting).lower() != "auto":
        return [Path(item.strip()) for item in str(setting).split(",") if item.strip()]

    paired_roots = [
        PROJECT_DIR / "compare_outputs_last_working",
        PROJECT_DIR / "compare_outputs_sequential_smoke_best",
        PROJECT_DIR / "compare_outputs_sequential_smoke",
        PROJECT_DIR / "compare_outputs_smoke2",
        PROJECT_DIR / "compare_outputs_smoke",
    ]
    for root in paired_roots:
        teacher_log = root / "teacher_student_current" / "ppo_training_log.csv"
        history_log = root / "history_perfect" / "ppo_training_log.csv"
        if teacher_log.exists() and history_log.exists():
            return [teacher_log, history_log]

    fallback_pairs = [
        (
            PROJECT_DIR / "compare_outputs_sequential_20ep" / "teacher_student_current" / "ppo_training_log.csv",
            PROJECT_DIR / "compare_outputs_sequential_smoke_best" / "history_perfect" / "ppo_training_log.csv",
        ),
        (
            PROJECT_DIR / "compare_outputs" / "teacher_student_current" / "ppo_training_log.csv",
            PROJECT_DIR / "compare_outputs_smoke" / "history_perfect" / "ppo_training_log.csv",
        ),
    ]
    for teacher_log, history_log in fallback_pairs:
        if teacher_log.exists() and history_log.exists():
            return [teacher_log, history_log]
    return []


def plot_training_log_references(log_paths):
    loaded = []
    for path in log_paths:
        rows = load_training_log_rows(path)
        if rows:
            loaded.append((Path(path), rows))
    if not loaded:
        return None, []

    fig = figure("PPO training", figsize=(10, 4))
    fig.clf()
    fig.suptitle("PPO training: previous logs solid, current dashed")
    specs = [
        (("reward", "avg_reward"), "Reward", "tab:blue"),
        (("success", "success_rate"), "Success rate", "tab:green"),
        (("crash", "crash_rate"), "Crash", "tab:red"),
    ]
    styles = ["-", "-."]
    for idx, (keys, ylabel, color) in enumerate(specs, start=1):
        ax = fig.add_subplot(1, 3, idx)
        for log_idx, (path, rows) in enumerate(loaded):
            available_keys = tuple(key for key in keys if key in rows[0])
            if not available_keys:
                continue
            x = np.array([row_value(row, "epoch") if "epoch" in row else i + 1 for i, row in enumerate(rows)])
            y = np.array([row_value(row, *available_keys) for row in rows])
            label = path.parent.name
            if path.parent.name == "teacher_student_current":
                label = "previous_teacher_student"
            elif path.parent.name == "history_perfect":
                label = "previous_history_perfect"
            marker = "o" if len(x) <= 50 else None
            markevery = None if len(x) <= 50 else builtins.max(1, len(x) // 25)
            ax.plot(
                x,
                y,
                color=color,
                linestyle=styles[log_idx % len(styles)],
                marker=marker,
                markevery=markevery,
                markersize=4,
                linewidth=1.8,
                label=label,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        ax.margins(x=0.08, y=0.15)
    fig.tight_layout()
    return fig, [path for path, _ in loaded]


def plot_training_curves(plot_fig, avg_rewards, success_rates, crash_rates, baseline_loaded):
    if baseline_loaded and len(plot_fig.axes) >= 3:
        series = [avg_rewards, success_rates, crash_rates]
        for ax, values in zip(plot_fig.axes[:3], series):
            for line in list(ax.lines):
                if line.get_label() == "_current_training":
                    line.remove()
            x = np.arange(1, len(values) + 1)
            ax.plot(
                x,
                values,
                color="black",
                linestyle="--",
                marker="x",
                markersize=5,
                linewidth=2.0,
                label="_current_training",
            )
            handles, labels = ax.get_legend_handles_labels()
            visible_labels = ["current_training" if label == "_current_training" else label for label in labels]
            ax.legend(handles, visible_labels, fontsize=7)
        return

    figure(plot_fig.number)
    clf()
    subplot(131)
    plot(np.arange(1, len(avg_rewards) + 1), avg_rewards, color="black", linestyle="--", marker="x", label="current_training")
    xlabel('Epoch')
    ylabel('Avg Reward')
    legend(fontsize=7)

    subplot(132)
    plot(np.arange(1, len(success_rates) + 1), success_rates, color="black", linestyle="--", marker="x", label="current_training")
    ylabel('Success rate')
    legend(fontsize=7)
    subplot(133)
    plot(np.arange(1, len(crash_rates) + 1), crash_rates, color="black", linestyle="--", marker="x", label="current_training")
    xlabel('Epoch')
    ylabel('Crash')
    legend(fontsize=7)
    savefig('result.pdf')


def best_policy_artifact_paths(output_dir):
    output_dir = Path(output_dir)
    return {
        "policy": output_dir / f"{BEST_ARTIFACT_STEM}.pth",
        "checkpoint": output_dir / f"{BEST_ARTIFACT_STEM}_checkpoint.pt",
        "performance_index": output_dir / f"{BEST_ARTIFACT_STEM}_performance_index.json",
        "performance_plot": output_dir / f"{BEST_ARTIFACT_STEM}_performance_index.png",
        "performance_figure": output_dir / f"{BEST_ARTIFACT_STEM}_performance_index.fig.pkl",
    }


def build_performance_index(history, best_snapshot):
    return {
        "selection_metric": "highest success rate, then highest average reward, then lowest crash rate",
        "best_epoch": best_snapshot["epoch"],
        "best_avg_reward": best_snapshot["avg_reward"],
        "best_success_rate": best_snapshot["success_rate"],
        "best_crash_rate": best_snapshot["crash_rate"],
        "best_timeout_rate": best_snapshot["timeout_rate"],
        "history": {
            "epoch": history["epoch"],
            "avg_reward": history["avg_reward"],
            "success_rate": history["success_rate"],
            "crash_rate": history["crash_rate"],
            "timeout_rate": history["timeout_rate"],
            "distill_loss": history.get("distill_loss", []),
        },
    }


def save_best_policy_and_performance_index(cfg, best_snapshot, performance_index):
    paths = best_policy_artifact_paths(cfg.output_dir)
    if best_snapshot["policy_path"] != paths["policy"]:
        torch.save(torch.load(best_snapshot["policy_path"], map_location="cpu"), paths["policy"])
    torch.save(
        {
            "teacher_policy_path": str(paths["policy"]),
            "performance_index": performance_index,
        },
        paths["checkpoint"],
    )
    paths["performance_index"].write_text(json.dumps(performance_index, indent=2), encoding="utf-8")
    return paths


def load_best_policy_and_performance_index(output_dir):
    paths = best_policy_artifact_paths(output_dir)
    policy_state_dict = torch.load(paths["policy"], map_location="cpu")
    performance_index = json.loads(paths["performance_index"].read_text(encoding="utf-8"))
    return policy_state_dict, performance_index, paths


def plot_loaded_performance_index(performance_index, output_path=None):
    history = performance_index["history"]
    fig = figure("Reloaded best performance index", figsize=(9, 4))
    fig.clf()

    subplot(131)
    plot(history["epoch"], history["avg_reward"], "b")
    axvline(performance_index["best_epoch"], color="k", linestyle="--", linewidth=1)
    xlabel("Epoch")
    ylabel("Avg Reward")

    subplot(132)
    plot(history["epoch"], history["success_rate"], "g", label="Success")
    axvline(performance_index["best_epoch"], color="k", linestyle="--", linewidth=1)
    xlabel("Epoch")
    ylabel("Success rate")

    subplot(133)
    plot(history["epoch"], history["crash_rate"], "r", label="Crash")
    axvline(performance_index["best_epoch"], color="k", linestyle="--", linewidth=1)
    xlabel("Epoch")
    ylabel("Crash")

    fig.suptitle(f"Reloaded best policy: epoch {performance_index['best_epoch']}")
    tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150)
    return fig


def distill_student(student, teacher, optimizer, student_obs, teacher_obs, cfg):
    student.train()
    teacher.eval()
    dataset_size = student_obs.shape[0]
    temperature = cfg.distill_temperature
    total_loss = 0.0
    total_batches = 0
    for _ in range(cfg.distill_epochs):
        idxs = torch.randperm(dataset_size, device=cfg.device)
        for start in range(0, dataset_size, cfg.distill_batch_size):
            batch_idx = idxs[start: start + cfg.distill_batch_size]
            with torch.no_grad():
                teacher_logits = teacher(teacher_obs[batch_idx])
                teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            student_logits = student(student_obs[batch_idx])
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().item())
            total_batches += 1
    return total_loss / builtins.max(1, total_batches)


def train(args):
    cfg = Config(args)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = USVHunterEnvDDM(cfg)
    actor = Actor(cfg.obs_dim, cfg.hidden, cfg.action_dim).to(cfg.device)
    critic = Critic(cfg.obs_dim, cfg.hidden).to(cfg.device)
    student_actor = Actor(cfg.student_obs_dim, cfg.hidden, cfg.action_dim).to(cfg.device)
    if args.resume_actor:
        actor.load_state_dict(torch.load(args.resume_actor, map_location=cfg.device))
        print(f"Loaded resume actor: {args.resume_actor}")
    if args.resume_critic:
        critic.load_state_dict(torch.load(args.resume_critic, map_location=cfg.device))
        print(f"Loaded resume critic: {args.resume_critic}")
    optimizer_a = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    optimizer_c = optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    optimizer_student = optim.Adam(student_actor.parameters(), lr=cfg.distill_lr)
    buffer = AgentStateBuffer(cfg.obs_dim, cfg.agents_num, cfg.max_step, cfg.device, cfg.student_obs_dim)
    logger = TrainingLogger(cfg.output_dir)

    metadata = {
        "ddm_path": str(cfg.ddm_path),
        "ddm_dt": env.dt,
        "ddm_action_to_id": env.ddm.action_to_id,
        "teacher_obs_dim": cfg.teacher_obs_dim,
        "student_obs_dim": cfg.student_obs_dim,
        "policy_history_len": cfg.policy_history_len,
        "raw_teacher_feature_dim": cfg.raw_teacher_feature_dim,
        "raw_teacher_obs_dim": cfg.raw_teacher_obs_dim,
        "raw_student_obs_dim": cfg.raw_student_obs_dim,
        "teacher_input": (
            "5-frame history of: 31 depth + 31 target mask + USV u/r velocity "
            "+ relative target distance/bearing"
            + (" + relative obstacle distance/bearing" if cfg.include_obstacle_privileged else "")
            + " + USV u/v/r velocity + target 2D velocity in USV frame/angular velocity + previous action one-hot"
        ),
        "student_input": "5-frame history of: 31 depth rays + 31 target color one-hot rays",
        "distill_epochs": cfg.distill_epochs,
        "distill_temperature": cfg.distill_temperature,
        "include_obstacle_privileged": cfg.include_obstacle_privileged,
        "action_dim": cfg.action_dim,
        "agents_num": cfg.agents_num,
        "max_step": cfg.max_step,
        "device": cfg.device,
        "target_distance_levels": cfg.target_distance_levels,
        "target_angle_deg": cfg.target_angle_deg,
        "heading_jitter_deg": cfg.heading_jitter_deg,
        "curriculum": "disabled",
        "training_distance_level": len(cfg.target_distance_levels) - 1,
        "training_distance_range": cfg.target_distance_levels[-1],
        "resume_actor": args.resume_actor,
        "resume_critic": args.resume_critic,
        "baseline_training_logs": str(getattr(args, "baseline_training_logs", "auto")),
        "environment_layout": "pure_run_test_default",
        "usv_birth_x_range": list(PURE_USV_X_RANGE),
        "usv_birth_y_range": list(PURE_USV_Y_RANGE),
        "usv_heading_deg_range": list(PURE_USV_HEADING_DEG_RANGE),
        "target_birth_x_range": list(PURE_TARGET_X_RANGE),
        "target_birth_y_range": list(PURE_TARGET_Y_RANGE),
        "target_heading_deg_range": list(PURE_TARGET_HEADING_DEG_RANGE),
        "target_speed_range": list(PURE_TARGET_SPEED_RANGE),
        "fixed_obstacles": pure_obstacles(),
    }
    (cfg.output_dir / "ppo_with_ddm_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"PPO with frozen UVR-DDM on {cfg.device}. DDM dt={env.dt}, agents={cfg.agents_num}")
    progress_page = publish_training_progress(
        cfg.output_dir,
        {
            "status": "training",
            "epoch": 0,
            "total_epochs": cfg.train_epochs,
            "history": {
                "epoch": [0],
                "avg_reward": [INITIAL_PROGRESS["avg_reward"]],
                "success_rate": [INITIAL_PROGRESS["success_rate"]],
                "distill_loss": [INITIAL_PROGRESS["distill_loss"]],
            },
            **INITIAL_PROGRESS,
        },
    )
    if progress_page is not None:
        open_training_progress_page(cfg.output_dir)
        print(f"Published training progress page: {progress_page}")
    best_reward = -float("inf")
    best_success = -float("inf")
    distance_level = len(cfg.target_distance_levels) - 1
    print(f"Curriculum disabled. Training starts at hardest level {distance_level}: {cfg.target_distance_levels[distance_level]}")

    realtime_plot = (
        not args.no_realtime_plot
        and args.training_plot_every > 0
        and matplotlib.get_backend().lower() not in {"agg", "pdf", "ps", "svg", "template"}
    )
    training_plots_enabled = bool(args.save_training_plots or realtime_plot)
    baseline_loaded = False
    plot_fig = None
    if training_plots_enabled:
        baseline_log_paths = resolve_baseline_training_logs(args, cfg)
        try:
            plot_fig, loaded_log_paths = plot_training_log_references(baseline_log_paths)
            baseline_loaded = plot_fig is not None
            if baseline_loaded:
                print("Loaded baseline training logs:")
                for path in loaded_log_paths:
                    print(f"  {path}")
        except Exception as exc:
            print(f"Could not load baseline training logs {baseline_log_paths}: {exc}")

        baseline_fig_path = PROJECT_DIR / "baseline.fig.pkl"
        try:
            if plot_fig is None:
                plot_fig = load_baseline_training_figure(baseline_fig_path)
                baseline_loaded = plot_fig is not None
            if baseline_loaded and not baseline_log_paths:
                plot_fig.suptitle("PPO training: baseline solid, current dashed")
                print(f"Loaded baseline figure: {baseline_fig_path}")
        except Exception as exc:
            print(f"Could not load baseline figure {baseline_fig_path}: {exc}")

    if realtime_plot:
        ion()
        if plot_fig is None:
            plot_fig = figure("PPO training", figsize=(9, 4))
            plot_fig.suptitle("PPO training starting...")
        show(block=False)
        pump_gui_events()
    else:
        if training_plots_enabled and plot_fig is None:
            plot_fig = figure(figsize=(9, 4))
    avg_rewards = []
    success_rates = []
    crash_rates = []
    timeout_rates = []
    performance_history = {
        "epoch": [],
        "avg_reward": [],
        "success_rate": [],
        "crash_rate": [],
        "timeout_rate": [],
        "distill_loss": [],
    }
    best_policy_score = None
    best_policy_snapshot = None
    best_policy_paths = best_policy_artifact_paths(cfg.output_dir)
    for epoch in tqdm(range(cfg.train_epochs)):
        epoch_times = {}
        sync_if_cuda(cfg.device)
        t_epoch = time.perf_counter()
        env.set_distance_level(distance_level)
        t0 = time.perf_counter()
        obs = env.reset()
        sync_if_cuda(cfg.device)
        epoch_times["reset"] = time.perf_counter() - t0
        total_success = torch.zeros((), device=cfg.device)
        total_crashes = torch.zeros((), device=cfg.device)
        total_timeouts = torch.zeros((), device=cfg.device)
        total_episodes = torch.zeros((), device=cfg.device)

        t0 = time.perf_counter()
        for step in range(cfg.max_step):
            student_obs = env.last_student_obs
            with torch.no_grad():
                logits = actor(obs)
                log_probs = F.log_softmax(logits, dim=-1)
                action = torch.multinomial(log_probs.exp(), 1)
                value = critic(obs)

            next_obs, reward, over, success, crash = env.step(action)

            buffer.obs_buffer[step] = obs
            buffer.student_obs_buffer[step] = student_obs
            buffer.action_buffer[step] = action
            buffer.reward_buffer[step] = reward
            buffer.over_buffer[step] = over
            buffer.value_buffer[step] = value
            buffer.logp_buffer[step] = log_probs.gather(1, action)

            done_mask = over.bool()
            total_success += success.float().sum()
            total_crashes += crash.float().sum()
            total_timeouts += (done_mask & ~(success | crash)).float().sum()
            total_episodes += done_mask.float().sum()
            obs = next_obs
            done_idx = torch.nonzero(done_mask.flatten(), as_tuple=False).flatten()
            if done_idx.numel() > 0:
                obs[done_idx] = env.reset_indices(done_idx)
            if realtime_plot and (step + 1) % args.plot_event_interval == 0:
                plot_fig.canvas.flush_events()
                pump_gui_events()
        sync_if_cuda(cfg.device)
        epoch_times["rollout"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.no_grad():
            next_value = critic(obs)
        buffer.compute_gae(next_value, cfg.gamma, cfg.lam)
        sync_if_cuda(cfg.device)
        epoch_times["gae"] = time.perf_counter() - t0

        b_obs = buffer.obs_buffer.view(-1, cfg.obs_dim)
        b_student_obs = buffer.student_obs_buffer.view(-1, cfg.student_obs_dim)
        b_act = buffer.action_buffer.view(-1)
        b_logp = buffer.logp_buffer.view(-1)
        b_adv = buffer.adv_buffer.view(-1)
        b_ret = buffer.return_buffer.view(-1)
        dataset_size = b_obs.shape[0]

        t0 = time.perf_counter()
        for _ in range(cfg.update_epochs):
            idxs = torch.randperm(dataset_size, device=cfg.device)
            for start in range(0, dataset_size, cfg.batch_size):
                batch_idx = idxs[start: start + cfg.batch_size]
                logits = actor(b_obs[batch_idx])
                new_logp = F.log_softmax(logits, dim=-1).gather(1, b_act[batch_idx].unsqueeze(1)).squeeze(1)
                ratio = torch.exp(new_logp - b_logp[batch_idx])
                surr1 = ratio * b_adv[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * b_adv[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean()

                optimizer_a.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
                optimizer_a.step()

                value_pred = critic(b_obs[batch_idx]).squeeze(-1)
                critic_loss = (b_ret[batch_idx] - value_pred).pow(2).mean()
                optimizer_c.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer_c.step()
                if realtime_plot and (start // cfg.batch_size + 1) % args.plot_event_interval == 0:
                    plot_fig.canvas.flush_events()
                    pump_gui_events()
        sync_if_cuda(cfg.device)
        epoch_times["update"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        distill_loss = distill_student(student_actor, actor, optimizer_student, b_student_obs, b_obs, cfg)
        sync_if_cuda(cfg.device)
        epoch_times["distill"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        avg_reward = buffer.reward_buffer.mean().item()
        episodes_run = torch.clamp(total_episodes, min=1.0)
        success_rate = float(((total_success / episodes_run) * 100.0).item())
        crash_rate = float(((total_crashes / episodes_run) * 100.0).item())
        timeout_rate = float(((total_timeouts / episodes_run) * 100.0).item())
        avg_rewards.append(avg_reward)
        success_rates.append(success_rate)
        crash_rates.append(crash_rate)
        timeout_rates.append(timeout_rate)
        performance_history["epoch"].append(epoch + 1)
        performance_history["avg_reward"].append(builtins.float(avg_reward))
        performance_history["success_rate"].append(builtins.float(success_rate))
        performance_history["crash_rate"].append(builtins.float(crash_rate))
        performance_history["timeout_rate"].append(builtins.float(timeout_rate))
        performance_history["distill_loss"].append(builtins.float(distill_loss))

        should_update_training_plot = (
            realtime_plot
            and args.training_plot_every > 0
            and ((epoch + 1) % args.training_plot_every == 0 or epoch + 1 == cfg.train_epochs)
        )
        if should_update_training_plot:
            plot_training_curves(plot_fig, avg_rewards, success_rates, crash_rates, baseline_loaded)
        # subplot(144)
        # # save newest in-memory actor before collect_episode()
        # latest_actor_for_visual = Path(args.output_dir) / 'ppo_uvr_ddm_actor_current_for_visual.pth'
        #
        # torch.save(
        #     actor.state_dict(),
        #     latest_actor_for_visual
        # )
        #
        # # force collect_episode() to reload the newest actor
        # args.actor_path = str(latest_actor_for_visual)
        #
        # records, _ = collect_episode(args)
        # pos = array([r['pos'] for r in records])
        # target = array([r['target'] for r in records])
        # heading = array([r['heading'] for r in records])
        # vel = array([r['vel'] for r in records])
        # actions = array([r['action'] for r in records])
        # rewards = array([r['reward'] for r in records])
        # obs_all = array([r['obs'] for r in records])
        #
        # n_points = 31
        # sensor_angles = linspace(-60, 60, n_points)
        # final_success = records[-1]['success']
        # for k in range(len(records)):
        #     # clf()
        #
        #     obs = obs_all[k]
        #     depth = obs[0:n_points]
        #     mask = obs[n_points:2 * n_points]
        #
        #     plot(pos[:k + 1, 0], pos[:k + 1, 1], 'b')
        #     plot(target[:k + 1, 0], target[:k + 1, 1], 'r--')
        #
        #     if actions[k] == 'W':
        #         c = 'k'  # forward
        #     elif actions[k] == 'A':
        #         c = 'b'  # left
        #     elif actions[k] == 'D':
        #         c = 'r'  # right
        #     else:
        #         c = 'g'  # stop / others
        #
        #     plot(pos[k, 0], pos[k, 1], 'o', color=c, markersize=8)
        #     plot(target[k, 0], target[k, 1], 'ro', markersize=8)
        #
        #     # arrow_len = 0.08
        #     # arrow(
        #     #     pos[k, 0],
        #     #     pos[k, 1],
        #     #     arrow_len * cos(heading[k]),
        #     #     arrow_len * sin(heading[k]),
        #     #     head_width=0.025,
        #     #     length_includes_head=True
        #     # )
        #
        #     obstacle = Circle(
        #         (WORLD_SIZE * 0.5, WORLD_SIZE * 0.5),
        #         OBSTACLE_RADIUS,
        #         fill=False
        #     )
        #     gca().add_patch(obstacle)
        #
        #     xlim([0, WORLD_SIZE])
        #     ylim([0, WORLD_SIZE])
        #     xticks([])
        #     yticks([])
        #     axis('equal')
        #     text(
        #         WORLD_SIZE * 0.5,
        #         WORLD_SIZE * 0.5,
        #         'SUCCESS' if final_success else 'FAIL',
        #         ha='center',
        #         va='center',
        #         fontsize=12,
        #         color='g' if final_success else 'r'
        #     )

        if should_update_training_plot:
            tight_layout()
            plot_fig.canvas.draw_idle()
            plot_fig.canvas.flush_events()
            pump_gui_events()
        epoch_times["plot"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        logger.update(epoch + 1, avg_reward, success_rate, crash_rate, timeout_rate)
        policy_score = (success_rate, avg_reward, -crash_rate)
        if best_policy_score is None or policy_score > best_policy_score:
            best_policy_score = policy_score
            torch.save(actor.state_dict(), best_policy_paths["policy"])
            best_policy_snapshot = {
                "epoch": epoch + 1,
                "avg_reward": builtins.float(avg_reward),
                "success_rate": builtins.float(success_rate),
                "crash_rate": builtins.float(crash_rate),
                "timeout_rate": builtins.float(timeout_rate),
                "policy_path": best_policy_paths["policy"],
            }
            best_marker = " best=*"
        else:
            best_marker = ""
        publish_training_progress(
            cfg.output_dir,
            {
                "status": "training",
                "epoch": epoch + 1,
                "total_epochs": cfg.train_epochs,
                "avg_reward": builtins.float(avg_reward),
                "success_rate": builtins.float(success_rate),
                "crash_rate": builtins.float(crash_rate),
                "timeout_rate": builtins.float(timeout_rate),
                "distill_loss": builtins.float(distill_loss),
                "best": builtins.bool(best_marker),
                "history": performance_history,
            },
        )
        # if args.print_every > 0 and ((epoch + 1) % args.print_every == 0 or epoch + 1 == cfg.train_epochs):
            # print(
            #     f"\n[epoch {epoch + 1}/{cfg.train_epochs}] "
            #     f"avg_reward={avg_reward:.4f} "
            #     f"success_rate={success_rate:.1f}% "
            #     f"crash_rate={crash_rate:.1f}% "
            #     f"timeout_rate={timeout_rate:.1f}% "
            #     f"distill_loss={distill_loss:.4f}"
            #     f"{best_marker}",
            #     flush=True,
            # )
        if args.save_best and avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(actor.state_dict(), cfg.output_dir / "ppo_uvr_ddm_teacher_actor_best_reward.pth")
            torch.save(critic.state_dict(), cfg.output_dir / "ppo_uvr_ddm_teacher_critic_best_reward.pth")
            torch.save(student_actor.state_dict(), cfg.output_dir / "ppo_uvr_ddm_student_actor_best_reward.pth")
        if args.save_best and success_rate > best_success:
            best_success = success_rate
            torch.save(actor.state_dict(), cfg.output_dir / "ppo_uvr_ddm_teacher_actor_best_success.pth")
            torch.save(critic.state_dict(), cfg.output_dir / "ppo_uvr_ddm_teacher_critic_best_success.pth")
            torch.save(student_actor.state_dict(), cfg.output_dir / "ppo_uvr_ddm_student_actor_best_success.pth")

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == cfg.train_epochs:
            torch.save(actor.state_dict(), cfg.output_dir / "ppo_uvr_ddm_teacher_actor_latest.pth")
            torch.save(critic.state_dict(), cfg.output_dir / "ppo_uvr_ddm_teacher_critic_latest.pth")
            torch.save(student_actor.state_dict(), cfg.output_dir / "ppo_uvr_ddm_student_actor_latest.pth")
            checkpoint_stamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_epoch = epoch + 1
            torch.save(
                actor.state_dict(),
                cfg.output_dir / f"ppo_uvr_ddm_teacher_actor_epoch_{checkpoint_epoch:04d}_{checkpoint_stamp}.pth",
            )
            torch.save(
                critic.state_dict(),
                cfg.output_dir / f"ppo_uvr_ddm_teacher_critic_epoch_{checkpoint_epoch:04d}_{checkpoint_stamp}.pth",
            )
            torch.save(
                student_actor.state_dict(),
                cfg.output_dir / f"ppo_uvr_ddm_student_actor_epoch_{checkpoint_epoch:04d}_{checkpoint_stamp}.pth",
            )
            print(f"\nSaved timestamped epoch checkpoint: epoch {checkpoint_epoch} ({checkpoint_stamp})", flush=True)
        if cfg.device.startswith("cuda") and args.empty_cache_each_epoch:
            torch.cuda.empty_cache()
        sync_if_cuda(cfg.device)
        epoch_times["bookkeep"] = time.perf_counter() - t0
        epoch_times["total"] = time.perf_counter() - t_epoch
        if args.profile:
            parts = " ".join(f"{k}={v:.3f}s" for k, v in epoch_times.items())
            print(f"\n[profile epoch {epoch + 1}] {parts}")

    teacher_actor_final = cfg.output_dir / "ppo_uvr_ddm_teacher_actor_final.pth"
    teacher_critic_final = cfg.output_dir / "ppo_uvr_ddm_teacher_critic_final.pth"
    student_actor_final = cfg.output_dir / "ppo_uvr_ddm_student_actor_final.pth"
    teacher_actor_final_unique = cfg.output_dir / f"ppo_uvr_ddm_teacher_actor_final_{run_stamp}.pth"
    teacher_critic_final_unique = cfg.output_dir / f"ppo_uvr_ddm_teacher_critic_final_{run_stamp}.pth"
    student_actor_final_unique = cfg.output_dir / f"ppo_uvr_ddm_student_actor_final_{run_stamp}.pth"
    torch.save(actor.state_dict(), teacher_actor_final)
    torch.save(critic.state_dict(), teacher_critic_final)
    torch.save(student_actor.state_dict(), student_actor_final)
    torch.save(actor.state_dict(), teacher_actor_final_unique)
    torch.save(critic.state_dict(), teacher_critic_final_unique)
    torch.save(student_actor.state_dict(), student_actor_final_unique)
    logger.save_copy(cfg.output_dir / f"ppo_training_log_{run_stamp}.csv")
    if training_plots_enabled:
        plot_training_curves(plot_fig, avg_rewards, success_rates, crash_rates, baseline_loaded)
        tight_layout()
    if best_policy_snapshot is not None:
        performance_index = build_performance_index(performance_history, best_policy_snapshot)
        best_paths = save_best_policy_and_performance_index(cfg, best_policy_snapshot, performance_index)
        loaded_policy_state, loaded_performance_index, loaded_paths = load_best_policy_and_performance_index(cfg.output_dir)
        actor.load_state_dict(loaded_policy_state)
        if training_plots_enabled:
            loaded_fig = plot_loaded_performance_index(loaded_performance_index, loaded_paths["performance_plot"])
            with loaded_paths["performance_figure"].open("wb") as f:
                pickle.dump(loaded_fig, f)
        print(
            f"Saved and reloaded best policy: {best_paths['policy']} "
            f"(epoch {performance_index['best_epoch']}, success={performance_index['best_success_rate']:.1f}%)"
        )
    if training_plots_enabled and baseline_loaded:
        overlay_png = cfg.output_dir / f"training_log_overlay_{run_stamp}.png"
        overlay_fig = cfg.output_dir / f"training_log_overlay_{run_stamp}.fig.pkl"
        plot_fig.savefig(overlay_png, dpi=150)
        with overlay_fig.open("wb") as f:
            pickle.dump(plot_fig, f)
        print(f"Saved training overlay plot: {overlay_png}")
        print(f"Saved training overlay figure: {overlay_fig}")
    print(f"Saved timestamped training log: {cfg.output_dir / f'ppo_training_log_{run_stamp}.csv'}")
    print(f"Saved timestamped final teacher actor: {teacher_actor_final_unique}")
    print(f"Saved timestamped final teacher critic: {teacher_critic_final_unique}")
    print(f"Saved timestamped final student actor: {student_actor_final_unique}")
    print(f"Saved PPO actor/critic to {cfg.output_dir}")


DEFAULT_RUN_DIR = PROJECT_DIR / "ppo_with_uvr_ddm_safe_best_saved"
DEFAULT_ACTOR = DEFAULT_RUN_DIR / "ppo_uvr_ddm_actor_best_success.pth"
DEFAULT_VIS_OUTPUT_DIR = DEFAULT_RUN_DIR / "visualization"


def choose_action(actor, obs, deterministic):
    with torch.no_grad():
        logits = actor(obs)
        if deterministic:
            return torch.argmax(logits, dim=-1, keepdim=True)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().unsqueeze(-1)


def collect_episode(args):
    config_args = SimpleNamespace(
        device=args.device,
        agents=1,
        max_step=args.max_step,
        epochs=1,
        ddm_path=args.ddm_path,
        output_dir=args.output_dir,
        gamma=0.99,
        lam=0.95,
        clip=0.2,
        actor_lr=3e-4,
        critic_lr=1e-3,
        update_epochs=1,
        batch_size=256,
        hidden=args.hidden,
        world_size=args.world_size,
        max_range=args.max_range,
        target_radius=args.target_radius,
        obstacle_radius=args.obstacle_radius,
        success_radius=args.success_radius,
        crash_margin=args.crash_margin,
        u_reward_scale=args.u_reward_scale,
        target_distance_levels=args.target_distance_levels,
        target_angle_deg=args.target_angle_deg,
        heading_jitter_deg=args.heading_jitter_deg,
        no_obstacle_privileged=getattr(args, "no_obstacle_privileged", False),
    )

    cfg = Config(config_args)

    env = USVHunterEnvDDM(cfg)
    env.set_distance_level(args.distance_level)

    actor = Actor(cfg.obs_dim, cfg.hidden, cfg.action_dim).to(cfg.device)
    actor.load_state_dict(torch.load(args.actor_path, map_location=cfg.device))
    actor.eval()

    obs = env.reset()

    records = []
    action_names = ['S', 'W', 'A', 'D']  # v16: S/W/A/D

    for step in range(args.max_step):

        with torch.no_grad():
            logits = actor(obs)

            if args.deterministic:
                action = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().unsqueeze(-1)

        [next_obs, reward, over, success, crash] = env.step(action)

        obs_now = next_obs[0].detach().cpu().numpy().copy()

        records.append(
            {
                'step': step,
                'pos': env.pos[0].detach().cpu().numpy().copy(),
                'yaw': float(env.psi[0, 0].detach().cpu()),
                'heading': float((env.psi[0, 0] + env.ddm.heading_offset).detach().cpu()),
                'vel': env.vel[0].detach().cpu().numpy().copy(),
                'target': env.target_pos[0].detach().cpu().numpy().copy(),
                'action_id': int(action[0, 0].detach().cpu()),
                'action': action_names[int(action[0, 0].detach().cpu())],
                'reward': float(reward[0, 0].detach().cpu()),
                'success': bool(success[0, 0].detach().cpu()),
                'crash': bool(crash[0, 0].detach().cpu()),
                'over': bool(over[0, 0].detach().cpu()),
                'obs': obs_now,
            }
        )

        obs = next_obs

        if bool(over[0, 0]):
            break

    return records, env


def draw_camera_fov(ax, xy, heading, max_range, half_angle_deg=60.0, color="tab:cyan", alpha=0.12, linewidth=1.4):
    start_deg = math.degrees(heading) - half_angle_deg
    end_deg = math.degrees(heading) + half_angle_deg
    wedge = Wedge(
        center=(xy[0], xy[1]),
        r=max_range,
        theta1=start_deg,
        theta2=end_deg,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=linewidth,
    )
    ax.add_patch(wedge)
    for sign in (-1.0, 1.0):
        a = heading + sign * math.radians(half_angle_deg)
        ax.plot(
            [xy[0], xy[0] + max_range * math.cos(a)],
            [xy[1], xy[1] + max_range * math.sin(a)],
            color=color,
            linestyle="--",
            linewidth=linewidth,
            alpha=builtins.min(1.0, alpha + 0.25),
        )
    ax.arrow(
        xy[0],
        xy[1],
        0.10 * math.cos(heading),
        0.10 * math.sin(heading),
        color="tab:blue",
        head_width=0.025,
        length_includes_head=True,
        alpha=0.9,
    )


def plot_overview(records, output_path, world_size, obstacle_radius):
    steps = np.array([r["step"] for r in records])
    pos = np.array([r["pos"] for r in records])
    target = np.array([r["target"] for r in records])
    yaw = np.array([r["yaw"] for r in records])
    heading = np.array([r["heading"] for r in records])
    vel = np.array([r["vel"] for r in records])
    actions = np.array([r["action_id"] for r in records])
    rewards = np.array([r["reward"] for r in records])
    sees_target = np.array([r["sees_target"] for r in records], dtype=float)

    fig = figure(figsize=(13, 10))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.4, 1.0, 1.0, 0.8])
    ax_xy = fig.add_subplot(gs[0, :])
    ax_yaw = fig.add_subplot(gs[1, 0])
    ax_reward = fig.add_subplot(gs[1, 1])
    ax_action = fig.add_subplot(gs[1, 2])
    ax_u = fig.add_subplot(gs[2, 0])
    ax_v = fig.add_subplot(gs[2, 1])
    ax_r = fig.add_subplot(gs[2, 2])
    ax_text = fig.add_subplot(gs[3, :])

    ax_xy.plot(pos[:, 0], pos[:, 1], color="tab:blue", linewidth=2, label="Boat")
    ax_xy.plot(target[:, 0], target[:, 1], color="tab:orange", linewidth=2, label="Target")
    ax_xy.scatter(pos[0, 0], pos[0, 1], color="tab:green", s=50, label="Start")
    ax_xy.scatter(pos[-1, 0], pos[-1, 1], color="tab:red", s=50, label="End")
    skip = builtins.max(1, len(records) // 6)
    for i in range(0, len(records), skip):
        draw_camera_fov(ax_xy, pos[i], heading[i], max_range=0.35, alpha=0.04, linewidth=0.8)
    draw_camera_fov(ax_xy, pos[-1], heading[-1], max_range=0.45, alpha=0.16, linewidth=1.5)
    circle = Circle((world_size * 0.5, world_size * 0.5), obstacle_radius, color="black", alpha=0.15,
                    label="Obstacle")
    ax_xy.add_patch(circle)
    ax_xy.set_xlim(0, world_size)
    ax_xy.set_ylim(0, world_size)
    ax_xy.set_aspect("equal")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xy.grid(True, linestyle="--", alpha=0.4)
    ax_xy.legend(fontsize=8)

    ax_yaw.plot(steps, yaw, color="tab:purple", label="mocap yaw")
    ax_yaw.plot(steps, heading, color="tab:purple", linestyle="--", label="heading yaw+offset")
    ax_yaw.set_title("Yaw / heading")
    ax_yaw.grid(True, linestyle="--", alpha=0.4)
    ax_yaw.legend(fontsize=8)
    ax_reward.plot(steps, rewards, color="tab:gray")
    ax_reward.fill_between(steps, rewards.min() - 0.5, rewards.max() + 0.5, where=sees_target > 0, color="tab:cyan",
                           alpha=0.15, label="target in FOV")
    ax_reward.set_title("Reward")
    ax_reward.grid(True, linestyle="--", alpha=0.4)
    ax_reward.legend(fontsize=8)
    ax_action.step(steps, actions, where="post", color="tab:gray")
    ax_action.set_yticks([0, 1, 2, 3])  # v16: 4 actions S/W/A/D
    ax_action.set_yticklabels(["S", "W", "A", "D"])  # v16: S/W/A/D
    ax_action.set_title("Action")
    ax_action.grid(True, linestyle="--", alpha=0.4)

    for ax, idx, name, color in [(ax_u, 0, "u", "tab:blue"), (ax_v, 1, "v", "tab:green"), (ax_r, 2, "r", "tab:red")]:
        ax.plot(steps, vel[:, idx], color=color)
        ax.set_title(name)
        ax.grid(True, linestyle="--", alpha=0.4)

    final = records[-1]
    status = "SUCCESS" if final["success"] else "CRASH" if final["crash"] else "TIMEOUT/ONGOING"
    ax_text.axis("off")
    ax_text.text(
        0.01,
        0.65,
        f"Episode status: {status} | steps: {len(records)} | final action: {final['action']} | "
        f"final reward: {final['reward']:.3f} | target_in_fov: {final['sees_target']}",
        fontsize=12,
    )
    ax_text.text(
        0.01,
        0.25,
        f"Final boat=({pos[-1, 0]:.3f}, {pos[-1, 1]:.3f}), target=({target[-1, 0]:.3f}, {target[-1, 1]:.3f}), "
        f"distance={np.linalg.norm(pos[-1] - target[-1]):.3f}",
        fontsize=12,
    )

    fig.tight_layout()
    show()


def make_animation(records, output_path, world_size, obstacle_radius):
    pos = np.array([r["pos"] for r in records])
    target = np.array([r["target"] for r in records])
    fig, ax = subplots(figsize=(6, 6))

    def draw_frame(i):
        ax.clear()
        ax.plot(pos[: i + 1, 0], pos[: i + 1, 1], color="tab:blue", linewidth=2, label="Boat trail")
        ax.plot(target[: i + 1, 0], target[: i + 1, 1], color="tab:orange", linewidth=1.5, label="Target trail")
        ax.scatter(pos[i, 0], pos[i, 1], color="tab:blue", s=60)
        ax.scatter(target[i, 0], target[i, 1], color="tab:orange", s=60)
        heading = records[i]["heading"]
        draw_camera_fov(ax, pos[i], heading, max_range=0.45, alpha=0.14, linewidth=1.2)
        circle = Circle((world_size * 0.5, world_size * 0.5), obstacle_radius, color="black", alpha=0.15)
        ax.add_patch(circle)
        ax.set_xlim(0, world_size)
        ax.set_ylim(0, world_size)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_title(
            f"step {records[i]['step']} | action {records[i]['action']} | reward {records[i]['reward']:.2f} "
            f"| in FOV {records[i]['sees_target']}"
        )
        ax.legend(loc="upper right", fontsize=8)

    anim = __import__("matplotlib.animation").animation.FuncAnimation(fig, draw_frame, frames=len(records),
                                                                      interval=250)
    show()


def evaluate_actor_success(actor, cfg, episodes, distance_level, deterministic=True, obs_kind="teacher"):
    env = USVHunterEnvDDM(cfg)
    env.set_distance_level(distance_level)
    actor.eval()
    teacher_obs = env.reset()
    obs = teacher_obs if obs_kind == "teacher" else env.last_student_obs
    active = torch.ones((episodes, 1), dtype=torch.bool, device=cfg.device)
    outcomes = {}
    for _ in range(cfg.max_step):
        with torch.no_grad():
            logits = actor(obs)
            if deterministic:
                action = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                action = torch.distributions.Categorical(logits=logits).sample().unsqueeze(-1)
            action = torch.where(active, action, torch.zeros_like(action))
        next_teacher_obs, _, over, success, crash = env.step(action)
        over_np = over.bool().detach().cpu().numpy()
        success_np = success.bool().detach().cpu().numpy()
        crash_np = crash.bool().detach().cpu().numpy()
        active_indices = torch.nonzero(active.flatten(), as_tuple=False).flatten().detach().cpu().numpy().tolist()
        for i in active_indices:
            if over_np[i, 0] and i not in outcomes:
                outcomes[i] = "success" if success_np[i, 0] else "failed" if crash_np[i, 0] else "timeout"
        active = active & ~over.bool()
        obs = next_teacher_obs if obs_kind == "teacher" else env.last_student_obs
        if not active.any():
            break
    for i in range(episodes):
        outcomes.setdefault(i, "timeout")
    success_count = builtins.sum(1 for v in outcomes.values() if v == "success")
    failed_count = builtins.sum(1 for v in outcomes.values() if v == "failed")
    timeout_count = builtins.sum(1 for v in outcomes.values() if v == "timeout")
    return {
        "success": 100.0 * success_count / episodes,
        "failed": 100.0 * failed_count / episodes,
        "timeout": 100.0 * timeout_count / episodes,
    }


def refine_student_from_teacher(args):
    cfg = Config(args)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    teacher_path = Path(args.teacher_policy_path)
    if str(teacher_path) == "auto":
        candidates = [
            cfg.output_dir / "ppo_uvr_ddm_teacher_actor_final.pth",
            cfg.output_dir / "ppo_uvr_ddm_teacher_actor_latest.pth",
            cfg.output_dir / "ppo_uvr_ddm_teacher_actor_best_success.pth",
        ]
        candidates = [p for p in candidates if p.exists()]
        if not candidates:
            raise FileNotFoundError(f"No teacher checkpoint found in {cfg.output_dir}")
        teacher_path = builtins.max(candidates, key=lambda p: p.stat().st_mtime)

    student_path = Path(args.student_policy_path)
    if str(student_path) == "auto":
        student_path = cfg.output_dir / "ppo_uvr_ddm_student_actor_final.pth"

    teacher = Actor(cfg.teacher_obs_dim, cfg.hidden, cfg.action_dim).to(cfg.device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=cfg.device))
    teacher.eval()
    student = Actor(cfg.student_obs_dim, cfg.hidden, cfg.action_dim).to(cfg.device)
    if student_path.exists():
        student.load_state_dict(torch.load(student_path, map_location=cfg.device))
    optimizer_student = optim.Adam(student.parameters(), lr=cfg.distill_lr)
    env = USVHunterEnvDDM(cfg)
    env.set_distance_level(len(cfg.target_distance_levels) - 1)
    print(f"Refining student from teacher: {teacher_path}")

    for epoch in tqdm(range(args.student_refine_epochs)):
        teacher_obs = env.reset()
        student_batches = []
        teacher_batches = []
        for _ in range(cfg.max_step):
            student_obs = env.last_student_obs
            with torch.no_grad():
                action = torch.argmax(teacher(teacher_obs), dim=-1, keepdim=True)
            student_batches.append(student_obs.detach().clone())
            teacher_batches.append(teacher_obs.detach().clone())
            teacher_obs, _, over, _, _ = env.step(action)
            done_idx = torch.nonzero(over.bool().flatten(), as_tuple=False).flatten()
            if done_idx.numel() > 0:
                teacher_obs[done_idx] = env.reset_indices(done_idx)
        b_student_obs = torch.cat(student_batches, dim=0)
        b_teacher_obs = torch.cat(teacher_batches, dim=0)
        loss = distill_student(student, teacher, optimizer_student, b_student_obs, b_teacher_obs, cfg)
        if args.profile:
            print(f"student refine epoch {epoch + 1}: distill_loss={loss:.4f}")

    torch.save(student.state_dict(), cfg.output_dir / "ppo_uvr_ddm_student_actor_final.pth")
    torch.save(student.state_dict(), cfg.output_dir / "ppo_uvr_ddm_student_actor_refined.pth")
    eval_episodes = builtins.max(128, args.test_episodes)
    eval_args = SimpleNamespace(**vars(args))
    eval_args.agents = eval_episodes
    eval_cfg = Config(eval_args)
    teacher_eval = Actor(eval_cfg.teacher_obs_dim, eval_cfg.hidden, eval_cfg.action_dim).to(eval_cfg.device)
    teacher_eval.load_state_dict(torch.load(teacher_path, map_location=eval_cfg.device))
    student_eval = Actor(eval_cfg.student_obs_dim, eval_cfg.hidden, eval_cfg.action_dim).to(eval_cfg.device)
    student_eval.load_state_dict(torch.load(eval_cfg.output_dir / "ppo_uvr_ddm_student_actor_final.pth", map_location=eval_cfg.device))
    teacher_metrics = evaluate_actor_success(teacher_eval, eval_cfg, eval_episodes, args.test_distance_level, True, "teacher")
    student_metrics = evaluate_actor_success(student_eval, eval_cfg, eval_episodes, args.test_distance_level, True, "student")
    ratio = student_metrics["success"] / builtins.max(teacher_metrics["success"], 1e-6)
    pd = __import__("pandas")
    pd.DataFrame(
        [
            {"policy": "teacher", **teacher_metrics},
            {"policy": "student", **student_metrics, "student_teacher_ratio": ratio},
        ]
    ).to_csv(cfg.output_dir / "teacher_student_eval.csv", index=False)
    print(
        f"Refined distillation eval ({eval_episodes} episodes): "
        f"teacher_success={teacher_metrics['success']:.1f}% "
        f"student_success={student_metrics['success']:.1f}% "
        f"student/teacher={ratio * 100.0:.1f}%"
    )


PURE_SENSOR_RAYS = 31
PURE_SENSOR_FOV_DEG = 120.0
PURE_USV_X_RANGE = (0.0, 0.2)
PURE_USV_Y_RANGE = (1.0, 1.2)
PURE_USV_HEADING_DEG_RANGE = (-90.0, 0.0)
PURE_TARGET_X_RANGE = (0.8, 0.9)
PURE_TARGET_Y_RANGE = (0.1, 0.2)
PURE_TARGET_HEADING_DEG_RANGE = (0.0, 45.0)
PURE_TARGET_SPEED_RANGE = (0.0, 0.01)
PURE_TARGET_RADIUS = 0.02
PURE_TRAJECTORY_STEPS = 50
PURE_SQUARE_CENTER = (1.0, 1.0)
PURE_SQUARE_SIDE = 0.2
PURE_CIRCLE_CENTER = (0.6, 0.6)
PURE_CIRCLE_RADIUS = 0.1
PURE_TRIANGLE_CENTER = (0.3, 0.3)
PURE_TRIANGLE_SIDE = 0.2


def pure_square_vertices(center=PURE_SQUARE_CENTER, side=PURE_SQUARE_SIDE):
    cx, cy = center
    half = side * 0.5
    return [
        (cx - half, cy - half),
        (cx + half, cy - half),
        (cx + half, cy + half),
        (cx - half, cy + half),
    ]


def pure_triangle_vertices(center=PURE_TRIANGLE_CENTER, side=PURE_TRIANGLE_SIDE):
    cx, cy = center
    height = math.sqrt(3.0) * side * 0.5
    return [
        (cx, cy + 2.0 * height / 3.0),
        (cx - side * 0.5, cy - height / 3.0),
        (cx + side * 0.5, cy - height / 3.0),
    ]


def pure_obstacles():
    return [
        {"name": "square", "kind": "polygon", "vertices": pure_square_vertices()},
        {"name": "circle", "kind": "circle", "center": PURE_CIRCLE_CENTER, "radius": PURE_CIRCLE_RADIUS},
        {"name": "triangle", "kind": "polygon", "vertices": pure_triangle_vertices()},
    ]


def pure_sample_initial_cases(test_num, seed=SEED):
    rng = np.random.default_rng(seed)
    cases = []
    for idx in range(test_num):
        usv_heading_deg = rng.uniform(*PURE_USV_HEADING_DEG_RANGE)
        target_heading_deg = rng.uniform(*PURE_TARGET_HEADING_DEG_RANGE)
        cases.append(
            {
                "case": idx,
                "usv_pos": (
                    rng.uniform(*PURE_USV_X_RANGE),
                    rng.uniform(*PURE_USV_Y_RANGE),
                ),
                "usv_heading": math.radians(usv_heading_deg),
                "target_pos": (
                    rng.uniform(*PURE_TARGET_X_RANGE),
                    rng.uniform(*PURE_TARGET_Y_RANGE),
                ),
                "target_heading": math.radians(target_heading_deg),
                "target_speed": rng.uniform(*PURE_TARGET_SPEED_RANGE),
            }
        )
    return cases


def pure_ray_wall_distance(origin, direction):
    ox, oy = origin
    dx, dy = direction
    candidates = []
    if abs(dx) > 1e-12:
        candidates.extend([(0.0 - ox) / dx, (WORLD_SIZE - ox) / dx])
    if abs(dy) > 1e-12:
        candidates.extend([(0.0 - oy) / dy, (WORLD_SIZE - oy) / dy])
    valid = [t for t in candidates if t > 0.0]
    return builtins.min(valid) if valid else math.inf


def pure_ray_circle_distance(origin, direction, center, radius):
    ox, oy = origin
    dx, dy = direction
    cx, cy = center
    ocx = ox - cx
    ocy = oy - cy
    b = ocx * dx + ocy * dy
    c = ocx * ocx + ocy * ocy - radius * radius
    disc = b * b - c
    if disc <= 0.0:
        return math.inf
    t = -b - math.sqrt(disc)
    return t if t > 0.0 else math.inf


def pure_cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def pure_ray_segment_distance(origin, direction, start, end):
    sx, sy = start
    ex, ey = end
    segment = (ex - sx, ey - sy)
    denom = pure_cross(direction, segment)
    if abs(denom) < 1e-12:
        return math.inf
    start_delta = (sx - origin[0], sy - origin[1])
    t = pure_cross(start_delta, segment) / denom
    u = pure_cross(start_delta, direction) / denom
    if t > 0.0 and 0.0 <= u <= 1.0:
        return t
    return math.inf


def pure_ray_polygon_distance(origin, direction, vertices):
    distances = []
    for idx, start in enumerate(vertices):
        end = vertices[(idx + 1) % len(vertices)]
        distances.append(pure_ray_segment_distance(origin, direction, start, end))
    return builtins.min(distances) if distances else math.inf


def pure_ray_obstacle_distance(origin, direction, obstacle):
    if obstacle["kind"] == "circle":
        return pure_ray_circle_distance(origin, direction, obstacle["center"], obstacle["radius"])
    return pure_ray_polygon_distance(origin, direction, obstacle["vertices"])


def pure_sensor_snapshot(case):
    origin = case["usv_pos"]
    ray_angles = np.linspace(
        -math.radians(PURE_SENSOR_FOV_DEG * 0.5),
        math.radians(PURE_SENSOR_FOV_DEG * 0.5),
        PURE_SENSOR_RAYS,
    )
    readings = []
    for idx, body_angle in enumerate(ray_angles):
        world_angle = case["usv_heading"] + float(body_angle)
        direction = (math.cos(world_angle), math.sin(world_angle))
        best_laser_dist = pure_ray_wall_distance(origin, direction)
        laser_hit = "boundary"
        for obstacle in pure_obstacles():
            obstacle_dist = pure_ray_obstacle_distance(origin, direction, obstacle)
            if obstacle_dist < best_laser_dist:
                best_laser_dist = obstacle_dist
                laser_hit = obstacle["name"]

        target_dist = pure_ray_circle_distance(origin, direction, case["target_pos"], PURE_TARGET_RADIUS)
        camera_target = target_dist < builtins.min(best_laser_dist, MAX_RANGE)
        laser_depth = builtins.min(best_laser_dist, MAX_RANGE)
        if best_laser_dist > MAX_RANGE:
            laser_hit = "max_range"
        readings.append(
            {
                "idx": idx,
                "body_angle_deg": math.degrees(float(body_angle)),
                "world_angle_deg": math.degrees(world_angle),
                "laser_depth": laser_depth,
                "laser_hit": laser_hit,
                "camera_target": int(camera_target),
                "target_distance": target_dist if camera_target else math.inf,
                "laser_end": (
                    origin[0] + laser_depth * direction[0],
                    origin[1] + laser_depth * direction[1],
                ),
                "camera_end": (
                    origin[0] + target_dist * direction[0],
                    origin[1] + target_dist * direction[1],
                ) if camera_target else None,
            }
        )
    return readings


def pure_print_case(case):
    print(
        f"case={case['case']} "
        f"usv=({case['usv_pos'][0]:.4f}, {case['usv_pos'][1]:.4f}) "
        f"usv_heading={math.degrees(case['usv_heading']):.2f} deg "
        f"target=({case['target_pos'][0]:.4f}, {case['target_pos'][1]:.4f}) "
        f"target_heading={math.degrees(case['target_heading']):.2f} deg "
        f"target_speed={case['target_speed']:.4f} m/s"
    )


def pure_draw_pool(ax):
    ax.plot([0, WORLD_SIZE, WORLD_SIZE, 0, 0], [0, 0, WORLD_SIZE, WORLD_SIZE, 0], "k-", linewidth=2)
    for obstacle in pure_obstacles():
        if obstacle["kind"] == "circle":
            ax.add_patch(
                Circle(
                    obstacle["center"],
                    obstacle["radius"],
                    color="tab:gray",
                    alpha=0.35,
                    label=f"{obstacle['name']} obstacle",
                )
            )
        else:
            ax.add_patch(
                Polygon(
                    obstacle["vertices"],
                    color="tab:gray",
                    alpha=0.50,
                    label=f"{obstacle['name']} obstacle",
                )
            )
    ax.add_patch(Rectangle((PURE_TARGET_X_RANGE[0], PURE_TARGET_Y_RANGE[0]), 0.1, 0.1, fill=False, edgecolor="tab:orange", linestyle="--", linewidth=1.2, label="target birth zone"))
    ax.add_patch(Rectangle((PURE_USV_X_RANGE[0], PURE_USV_Y_RANGE[0]), 0.2, 0.2, fill=False, edgecolor="tab:blue", linestyle="--", linewidth=1.2, label="USV birth zone"))
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(0, WORLD_SIZE)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.25)


def pure_draw_pose(ax, pos, heading, color, label, arrow_len=0.08):
    ax.plot(pos[0], pos[1], "o", color=color, markersize=6, label=label)
    ax.arrow(
        pos[0],
        pos[1],
        arrow_len * math.cos(heading),
        arrow_len * math.sin(heading),
        head_width=0.025,
        color=color,
        length_includes_head=True,
    )


def pure_integrate_ddm_pose(pos, psi_internal, next_uvr, ddm):
    u = next_uvr[:, 0:1]
    v = next_uvr[:, 1:2]
    r = next_uvr[:, 2:3]
    turn = r * ddm.dt
    abs_r = torch.abs(r)
    r_safe = torch.where(abs_r > ddm.r_eps, r, torch.ones_like(r))
    dx_arc = (u / r_safe) * torch.sin(turn) + (v / r_safe) * (torch.cos(turn) - 1.0)
    dy_arc = (u / r_safe) * (1.0 - torch.cos(turn)) + (v / r_safe) * torch.sin(turn)
    dx_line = u * ddm.dt
    dy_line = v * ddm.dt
    use_arc = abs_r > ddm.r_eps
    dx_body = torch.where(use_arc, dx_arc, dx_line)
    dy_body = torch.where(use_arc, dy_arc, dy_line)
    heading = psi_internal + ddm.heading_offset
    pos[:, 0:1] += dx_body * torch.cos(heading) - dy_body * torch.sin(heading)
    pos[:, 1:2] += dx_body * torch.sin(heading) + dy_body * torch.cos(heading)
    psi_internal += r * ddm.dt
    return pos, psi_internal


def pure_simulate_w_trajectory(case, steps=PURE_TRAJECTORY_STEPS):
    pure_device = "cuda" if str(DEVICE).startswith("cuda") and torch.cuda.is_available() else "cpu"
    ddm = FrozenUVRDDM(DDM_PATH, pure_device)
    w_action = torch.tensor([int(ddm.action_to_id["A"])], dtype=torch.long, device=pure_device)

    pos = torch.tensor([case["usv_pos"]], dtype=torch.float32, device=pure_device)
    psi_internal = torch.tensor(
        [[case["usv_heading"] - ddm.heading_offset]],
        dtype=torch.float32,
        device=pure_device,
    )
    zero_uvr_norm = ddm.normalize(torch.zeros((1, 3), dtype=torch.float32, device=pure_device))
    uvr_history_norm = zero_uvr_norm.unsqueeze(1).repeat(1, ddm.history_len, 1)
    action_history = torch.zeros((1, ddm.history_len), dtype=torch.long, device=pure_device)

    usv_positions = [case["usv_pos"]]
    target_positions = [case["target_pos"]]
    target_x, target_y = case["target_pos"]
    for _ in range(steps):
        next_uvr, _, uvr_history_norm, action_history = ddm.predict_next(
            uvr_history_norm,
            action_history,
            w_action,
        )
        pos, psi_internal = pure_integrate_ddm_pose(pos, psi_internal, next_uvr, ddm)
        usv_positions.append(
            (
                float(pos[0, 0].detach().cpu()),
                float(pos[0, 1].detach().cpu()),
            )
        )

        target_x += case["target_speed"] * math.cos(case["target_heading"]) * ddm.dt
        target_y += case["target_speed"] * math.sin(case["target_heading"]) * ddm.dt
        target_positions.append((target_x, target_y))

    return np.array(usv_positions), np.array(target_positions), ddm.dt


def pure_run_test():
    cases = pure_sample_initial_cases(TEST_NUM)
    print(f"PURE_RUN_TEST = 1, TEST_NUM = {TEST_NUM}")
    print(f"Pool: {WORLD_SIZE:.1f} m x {WORLD_SIZE:.1f} m, origin at lower-left corner")
    print("USV birth rectangle: x=[0.0, 0.2], y=[1.0, 1.2], heading=[0, -90] deg")
    print(f"Square obstacle: center={PURE_SQUARE_CENTER}, side={PURE_SQUARE_SIDE}")
    print(f"Circle obstacle: center={PURE_CIRCLE_CENTER}, diameter={2.0 * PURE_CIRCLE_RADIUS}")
    print(f"Triangle obstacle: center={PURE_TRIANGLE_CENTER}, side={PURE_TRIANGLE_SIDE}")
    print("Target birth rectangle: x=[0.8, 0.9], y=[0.1, 0.2], heading=[0, 90] deg, speed=[0, 0.01] m/s")
    for case in cases:
        pure_print_case(case)

    fig = figure("PURE_RUN_TEST initial conditions", figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    pure_draw_pool(ax)

    if TEST_NUM == 1:
        case = cases[0]
        readings = pure_sensor_snapshot(case)
        pure_draw_pose(ax, case["usv_pos"], case["usv_heading"], "tab:blue", "USV")
        pure_draw_pose(ax, case["target_pos"], case["target_heading"], "tab:orange", "target", arrow_len=0.06)
        ax.add_patch(Circle(case["target_pos"], PURE_TARGET_RADIUS, fill=False, color="tab:orange", linewidth=1.0))
        usv_trajectory, target_trajectory, ddm_dt = pure_simulate_w_trajectory(case, PURE_TRAJECTORY_STEPS)
        ax.plot(
            usv_trajectory[:, 0],
            usv_trajectory[:, 1],
            color="tab:blue",
            linestyle="-",
            marker="*",
            markersize=6,
            linewidth=1.2,
            label="USV W trajectory",
        )
        ax.plot(
            target_trajectory[:, 0],
            target_trajectory[:, 1],
            color="tab:orange",
            linestyle="-",
            marker="*",
            markersize=6,
            linewidth=1.2,
            label="target trajectory",
        )
        print(f"Plotted next {PURE_TRAJECTORY_STEPS} steps with USV action='W' and dt={ddm_dt:.4f}s")

        print("idx  body_deg  world_deg  laser_depth  laser_hit   camera_target  target_dist")
        for row in readings:
            print(
                f"{row['idx']:02d}  "
                f"{row['body_angle_deg']:8.2f}  "
                f"{row['world_angle_deg']:9.2f}  "
                f"{row['laser_depth']:11.4f}  "
                f"{row['laser_hit']:<10s}  "
                f"{row['camera_target']:13d}  "
                f"{row['target_distance'] if math.isfinite(row['target_distance']) else float('nan'):.4f}"
            )
            laser_color = "tab:red" if row["laser_hit"] not in {"boundary", "max_range"} else "tab:blue"
            ax.plot(
                [case["usv_pos"][0], row["laser_end"][0]],
                [case["usv_pos"][1], row["laser_end"][1]],
                color=laser_color,
                alpha=0.35,
                linewidth=0.8,
            )
            if row["camera_end"] is not None:
                ax.plot(
                    [case["usv_pos"][0], row["camera_end"][0]],
                    [case["usv_pos"][1], row["camera_end"][1]],
                    color="tab:orange",
                    linewidth=1.8,
                )
        left_end = readings[0]["laser_end"]
        right_end = readings[-1]["laser_end"]
        ax.plot([case["usv_pos"][0], left_end[0]], [case["usv_pos"][1], left_end[1]], color="tab:purple", linestyle="--", linewidth=1.5, label="camera boundary")
        ax.plot([case["usv_pos"][0], right_end[0]], [case["usv_pos"][1], right_end[1]], color="tab:purple", linestyle="--", linewidth=1.5)
        ax.set_title("PURE_RUN_TEST: sensor rays and 20-step W rollout")
    else:
        for case in cases:
            pure_draw_pose(ax, case["usv_pos"], case["usv_heading"], "tab:blue", f"USV {case['case']}", arrow_len=0.06)
            pure_draw_pose(ax, case["target_pos"], case["target_heading"], "tab:orange", f"target {case['case']}", arrow_len=0.05)
        ax.set_title(f"PURE_RUN_TEST: {TEST_NUM} initial condition pairs")

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    # ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8)
    tight_layout()
    show()


# ============================================================
# SELECT WHAT TO RUN
# ============================================================
# Change these values in the constants block near the top of this file.
# Test is intentionally written inline below, not hidden inside a function.
RUN_TRAINING = True
RUN_TESTING = True

TEST_EPISODES = 16
TEST_DISTANCE_LEVEL = 3
TEST_DETERMINISTIC = True
TEST_REALTIME_PLOT = False
TEST_FRAME_PAUSE = 0.05
TEST_POLICY_PATH = "auto"
TEST_SHOW_PLOT = False
TEST_SAVE_PLOTS = False

if PURE_RUN_TEST:
    pure_run_test()
    sys.exit(0)

args = args_from_constants()
if args.mode == "distill":
    refine_student_from_teacher(args)
    # close("all")
    sys.exit(0)

run_training = RUN_TRAINING
run_testing = RUN_TESTING
if args.mode != "config":
    run_training = args.mode in {"train", "both"}
    run_testing = args.mode in {"test", "both"}

test_episodes = args.test_episodes if args.mode != "config" else TEST_EPISODES
test_distance_level = args.test_distance_level if args.mode != "config" else TEST_DISTANCE_LEVEL
test_deterministic = args.test_deterministic if args.mode != "config" else TEST_DETERMINISTIC
test_realtime_plot = args.test_realtime_plot if args.mode != "config" else TEST_REALTIME_PLOT
test_frame_pause = args.test_frame_pause if args.mode != "config" else TEST_FRAME_PAUSE
test_policy_setting = args.test_policy_path if args.mode != "config" else TEST_POLICY_PATH
test_show_plot = (args.show_test_plot and not args.no_test_plot) if args.mode != "config" else TEST_SHOW_PLOT
test_save_plots = args.save_test_plots if args.mode != "config" else TEST_SAVE_PLOTS

if run_training:
    train(args)
    if not run_testing:
        # close("all")
        sys.exit(0)


validation_args = SimpleNamespace(
    device=args.device,
    agents=test_episodes,
    max_step=args.max_step,
    epochs=1,
    ddm_path=args.ddm_path,
    output_dir=args.output_dir,
    gamma=args.gamma,
    lam=args.lam,
    clip=args.clip,
    actor_lr=args.actor_lr,
    critic_lr=args.critic_lr,
    update_epochs=1,
    batch_size=args.batch_size,
    hidden=args.hidden,
    world_size=args.world_size,
    max_range=args.max_range,
    target_radius=args.target_radius,
    obstacle_radius=args.obstacle_radius,
    success_radius=args.success_radius,
    crash_margin=args.crash_margin,
    u_reward_scale=args.u_reward_scale,
    target_distance_levels=args.target_distance_levels,
    target_angle_deg=args.target_angle_deg,
    heading_jitter_deg=args.heading_jitter_deg,
    no_obstacle_privileged=args.no_obstacle_privileged,
)
validation_cfg = Config(validation_args)
test_plot_dir = validation_cfg.output_dir / "latest_policy_test_plots"
if test_save_plots:
    test_plot_dir.mkdir(parents=True, exist_ok=True)
validation_env = USVHunterEnvDDM(validation_cfg)
validation_env.set_distance_level(test_distance_level)

if test_policy_setting and test_policy_setting != "auto":
    validation_policy_path = Path(test_policy_setting)
else:
    validation_candidates = [
        validation_cfg.output_dir / f"{BEST_ARTIFACT_STEM}.pth",
        validation_cfg.output_dir / "ppo_uvr_ddm_teacher_actor_best_success.pth",
        validation_cfg.output_dir / "ppo_uvr_ddm_teacher_actor_best_reward.pth",
        validation_cfg.output_dir / "ppo_uvr_ddm_teacher_actor_final.pth",
        validation_cfg.output_dir / "ppo_uvr_ddm_teacher_actor_latest.pth",
    ]
    validation_candidates = [p for p in validation_candidates if p.exists()]
    if not validation_candidates:
        raise FileNotFoundError(f"No PPO actor checkpoint found in {validation_cfg.output_dir}")
    validation_policy_path = validation_candidates[0]

validation_policy_state = torch.load(validation_policy_path, map_location=validation_cfg.device)
validation_policy_dim = int(validation_policy_state["net.0.weight"].shape[1])
old_teacher_dim = validation_cfg.obs_dim - 2 * validation_cfg.policy_history_len
if validation_policy_dim != validation_cfg.obs_dim:
    if validation_policy_dim == old_teacher_dim and not validation_args.no_obstacle_privileged:
        print(
            "Loaded policy uses the old teacher observation without privileged obstacle state; "
            "using NO_OBSTACLE_PRIVILEGED compatibility for this test."
        )
        validation_args.no_obstacle_privileged = True
        validation_cfg = Config(validation_args)
        validation_env = USVHunterEnvDDM(validation_cfg)
        validation_env.set_distance_level(test_distance_level)
    else:
        raise ValueError(
            f"Policy input dim {validation_policy_dim} does not match current observation dim {validation_cfg.obs_dim}. "
            "Set NO_OBSTACLE_PRIVILEGED = True for old teacher-student checkpoints."
        )

validation_actor = Actor(validation_cfg.obs_dim, validation_cfg.hidden, validation_cfg.action_dim).to(validation_cfg.device)
validation_actor.load_state_dict(validation_policy_state)
validation_actor.eval()
print(f"Teacher test using policy: {validation_policy_path}")

obs = validation_env.reset()
active = torch.ones((test_episodes, 1), dtype=torch.bool, device=validation_cfg.device)
start_pos = validation_env.pos.detach().cpu().numpy().copy()
start_target_pos = validation_env.target_pos.detach().cpu().numpy().copy()
start_obstacle_pos = validation_env.obstacle_pos.detach().cpu().numpy().copy()
rows = []
case_outcomes = {}

for step in range(args.max_step):
    with torch.no_grad():
        logits = validation_actor(obs)
        if test_deterministic:
            action = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            action = torch.distributions.Categorical(logits=logits).sample().unsqueeze(-1)
        action = torch.where(active, action, torch.zeros_like(action))

    next_obs, reward, over, success, crash = validation_env.step(action)
    active_indices = torch.nonzero(active.flatten(), as_tuple=False).flatten().detach().cpu().numpy().tolist()
    pos_np = validation_env.pos.detach().cpu().numpy().copy()
    target_np = validation_env.target_pos.detach().cpu().numpy().copy()
    action_np = action.detach().cpu().numpy().copy()
    reward_np = reward.detach().cpu().numpy().copy()
    over_np = over.bool().detach().cpu().numpy().copy()
    success_np = success.bool().detach().cpu().numpy().copy()
    crash_np = crash.bool().detach().cpu().numpy().copy()

    for i in active_indices:
        x = float(np.clip(pos_np[i, 0], 0.0, validation_cfg.world_size))
        y = float(np.clip(pos_np[i, 1], 0.0, validation_cfg.world_size))
        target_x = float(np.clip(target_np[i, 0], 0.0, validation_cfg.world_size))
        target_y = float(np.clip(target_np[i, 1], 0.0, validation_cfg.world_size))
        rows.append(
            {
                "case": i,
                "step": step,
                "x": x,
                "y": y,
                "target_x": target_x,
                "target_y": target_y,
                "obstacle_x": float(start_obstacle_pos[i, 0]),
                "obstacle_y": float(start_obstacle_pos[i, 1]),
                "action_id": int(action_np[i, 0]),
                "reward": float(reward_np[i, 0]),
            }
        )
        if over_np[i, 0] and i not in case_outcomes:
            if success_np[i, 0]:
                case_outcomes[i] = "success"
            elif crash_np[i, 0]:
                case_outcomes[i] = "failed"
            else:
                case_outcomes[i] = "timeout"

    active = active & ~over.bool()
    obs = next_obs
    if not active.any():
        break

for i in range(test_episodes):
    case_outcomes.setdefault(i, "timeout")

success_count = builtins.sum(1 for v in case_outcomes.values() if v == "success")
failed_count = builtins.sum(1 for v in case_outcomes.values() if v == "failed")
timeout_count = builtins.sum(1 for v in case_outcomes.values() if v == "timeout")
avg_reward = builtins.sum(row["reward"] for row in rows) / builtins.max(1, len(rows))
success_rate = 100.0 * success_count / test_episodes
crash_rate = 100.0 * failed_count / test_episodes
# print(
#     f"Previous method metrics: avg_reward={avg_reward:.4f} "
#     f"success_rate={success_rate:.1f}% "
#     f"crash_rate={crash_rate:.1f}% "
#     f"timeout_rate={100.0 * timeout_count / test_episodes:.1f}%"
# )

student_policy_path = validation_cfg.output_dir / "ppo_uvr_ddm_student_actor_final.pth"
if args.eval_student_distillation and student_policy_path.exists():
    eval_episodes = builtins.max(128, test_episodes)
    eval_args = SimpleNamespace(**vars(validation_args))
    eval_args.agents = eval_episodes
    eval_cfg = Config(eval_args)
    teacher_eval_actor = Actor(eval_cfg.teacher_obs_dim, eval_cfg.hidden, eval_cfg.action_dim).to(eval_cfg.device)
    teacher_eval_actor.load_state_dict(torch.load(validation_policy_path, map_location=eval_cfg.device))
    student_eval_actor = Actor(eval_cfg.student_obs_dim, eval_cfg.hidden, eval_cfg.action_dim).to(eval_cfg.device)
    student_eval_actor.load_state_dict(torch.load(student_policy_path, map_location=eval_cfg.device))
    teacher_metrics = evaluate_actor_success(
        teacher_eval_actor, eval_cfg, eval_episodes, test_distance_level, test_deterministic, "teacher"
    )
    student_metrics = evaluate_actor_success(
        student_eval_actor, eval_cfg, eval_episodes, test_distance_level, test_deterministic, "student"
    )
    ratio = student_metrics["success"] / builtins.max(teacher_metrics["success"], 1e-6)
    print(
        f"Distillation eval ({eval_episodes} episodes): "
        f"teacher_success={teacher_metrics['success']:.1f}% "
        f"student_success={student_metrics['success']:.1f}% "
        f"student/teacher={ratio * 100.0:.1f}%"
    )
    pd = __import__("pandas")
    pd.DataFrame(
        [
            {"policy": "teacher", **teacher_metrics},
            {"policy": "student", **student_metrics, "student_teacher_ratio": ratio},
        ]
    ).to_csv(validation_cfg.output_dir / "teacher_student_eval.csv", index=False)
elif args.eval_student_distillation:
    print(f"No student policy found at {student_policy_path}")

if rows and test_show_plot:
    action_colors = {
        0: "yellow",
        1: "black",
        2: "blue",
        3: "red",
    }
    fig = figure("PPO test", figsize=(8, 8))
    cases = sorted({row["case"] for row in rows})
    max_plotted_step = builtins.max(row["step"] for row in rows)

    frame_steps = range(max_plotted_step + 1) if test_realtime_plot else [max_plotted_step]
    if test_realtime_plot:
        ion()

    for frame_step in frame_steps:
        fig.clf()
        for case in cases:
            ax = subplot(4, 4, case + 1)
            case_rows = [row for row in rows if row["case"] == case and row["step"] <= frame_step]
            if not case_rows:
                continue
            xs = np.array([row["x"] for row in case_rows])
            ys = np.array([row["y"] for row in case_rows])
            target_xs = np.array([row["target_x"] for row in case_rows])
            target_ys = np.array([row["target_y"] for row in case_rows])
            action_ids = np.array([row["action_id"] for row in case_rows])
            obstacle_center = (case_rows[0]["obstacle_x"], case_rows[0]["obstacle_y"])

            ax.plot([0, validation_cfg.world_size], [0, 0], "k--", linewidth=0.8)
            ax.plot([0, validation_cfg.world_size], [validation_cfg.world_size, validation_cfg.world_size], "k--", linewidth=0.8)
            ax.plot([0, 0], [0, validation_cfg.world_size], "k--", linewidth=0.8)
            ax.plot([validation_cfg.world_size, validation_cfg.world_size], [0, validation_cfg.world_size], "k--", linewidth=0.8)
            ax.plot(target_xs, target_ys, "--", color="tab:orange", linewidth=0.8, alpha=0.7)
            for k in range(len(case_rows)):
                ax.plot(xs[k], ys[k], "o", color=action_colors[int(action_ids[k])], markersize=2)
            ax.text(start_pos[case, 0], start_pos[case, 1], "s", color="tab:blue", fontsize=8, fontweight="bold")
            ax.text(xs[-1], ys[-1], "e", color="tab:blue", fontsize=8, fontweight="bold")
            ax.text(start_target_pos[case, 0], start_target_pos[case, 1], "s", color="tab:orange", fontsize=8, fontweight="bold")
            ax.text(target_xs[-1], target_ys[-1], "e", color="tab:orange", fontsize=8, fontweight="bold")
            ax.add_patch(Circle(obstacle_center, validation_cfg.obstacle_radius, fill=False, color="black", linewidth=1.1))
            outcome = case_outcomes.get(case, "timeout")
            if outcome == "failed":
                ax.set_title("Failed", fontsize=9, color="red")
            elif outcome == "timeout":
                ax.set_title("Timeout", fontsize=9, color="red")
            ax.set_xlim(0, validation_cfg.world_size)
            ax.set_ylim(0, validation_cfg.world_size)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
        tight_layout()
        if test_realtime_plot:
            pause(test_frame_pause)
            pump_gui_events()
    if test_save_plots:
        rollout_plot_path = test_plot_dir / "teacher_test_rollouts.png"
        fig.savefig(rollout_plot_path, dpi=170)
        print(f"Saved test rollout plot: {rollout_plot_path}")


    # x, y, theta vs. time for all test cases
    state_fig = figure('x y theta vs time', figsize=(9, 6))

    for case in cases:
        case_rows = [row for row in rows if row['case'] == case]
        if not case_rows:
            continue

        ts = np.array([row['step'] for row in case_rows]) * validation_env.dt
        xs = np.array([row['x'] for row in case_rows])
        ys = np.array([row['y'] for row in case_rows])

        dxs = np.gradient(xs)
        dys = np.gradient(ys)
        theta = np.unwrap(np.arctan2(dys, dxs))

        subplot(311)
        plot(ts, xs)
        ylabel('x [m]')

        subplot(312)
        plot(ts, ys)
        ylabel('y [m]')

        subplot(313)
        plot(ts, theta*180/pi)
        xlabel('Time [s]')
        ylabel(r'theta [$^\circ$]')

    tight_layout()
    if test_save_plots:
        state_plot_path = test_plot_dir / "teacher_test_state_timeseries.png"
        state_fig.savefig(state_plot_path, dpi=170)
        print(f"Saved test state time-series plot: {state_plot_path}")

    print("Showing test figures. Close the plot window to finish the script.")
    show()
