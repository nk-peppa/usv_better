import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ================= 1. 配置项 =================
class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        self.agents_num = int(os.getenv("PPO_USV_AGENTS", "20000"))
        self.max_step = int(os.getenv("PPO_USV_MAX_STEP", "500"))
        self.dt = 0.2
        self.train_epochs = int(os.getenv("PPO_USV_EPOCHS", "25"))

        self.gamma = 0.99
        self.lam = 0.95
        self.clip = 0.2
        self.actor_lr = 3e-4
        self.critic_lr = 1e-3
        self.update_epochs = int(os.getenv("PPO_USV_UPDATE_EPOCHS", "2"))
        self.batch_size = int(os.getenv("PPO_USV_BATCH_SIZE", "2048"))
        self.enable_plot = os.getenv("PPO_USV_PLOT", "0") == "1"

        # 🚀 定义单帧维度和堆叠帧数
        self.single_obs_dim = 65  # 31(深度) + 31(语义) + 1(角速度r) + 2(记忆)
        self.stack_frames = 3
        self.obs_dim = self.single_obs_dim * self.stack_frames  # 总输入维度: 195
        self.action_dim = 4

# ================= 2. PPO 核心组件 =================
class Agent_State_Buffer:
    def __init__(self, obs_dim, agent_num, max_step, device):
        self.device = device
        self.obs_buffer = torch.zeros((max_step, agent_num, obs_dim), device=device)
        self.action_buffer = torch.zeros((max_step, agent_num, 1), device=device, dtype=torch.long)
        self.logp_buffer = torch.zeros((max_step, agent_num, 1), device=device)
        self.reward_buffer = torch.zeros((max_step, agent_num, 1), device=device)
        self.over_buffer = torch.zeros((max_step, agent_num, 1), device=device)
        self.value_buffer = torch.zeros((max_step, agent_num, 1), device=device)
        self.adv_buffer = torch.zeros((max_step, agent_num, 1), device=device)
        self.return_buffer = torch.zeros((max_step, agent_num, 1), device=device)

    def compute_GAE(self, next_value, gamma, lam):
        advantage = torch.zeros_like(self.reward_buffer[0])
        for t in reversed(range(self.reward_buffer.shape[0])):
            nextnonterminal = 1.0 - self.over_buffer[t]
            nextvalues = next_value if t == self.reward_buffer.shape[0] - 1 else self.value_buffer[t + 1]
            delta = self.reward_buffer[t] + gamma * nextvalues * nextnonterminal - self.value_buffer[t]
            advantage = delta + gamma * lam * nextnonterminal * advantage
            self.adv_buffer[t] = advantage
        self.return_buffer = self.adv_buffer + self.value_buffer
        if self.adv_buffer.std() > 1e-6:
            self.adv_buffer = (self.adv_buffer - self.adv_buffer.mean()) / (self.adv_buffer.std() + 1e-8)

class Actor(nn.Module):
    def __init__(self, input_dim=195, hidden=256, n_actions=4): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, n_actions)
        )
    def forward(self, x): return self.net(x)

class Critic(nn.Module):
    def __init__(self, input_dim=195, hidden=256): 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

# ================= 3. 微型船环境 =================
class USVHunterEnv_Semantic:
    def __init__(self, config):
        self.agent_num = config.agents_num
        self.dt = config.dt
        self.device = config.device
        self.max_steps = config.max_step
        self.world_size = 1.2
        
        self.base_A_u, self.base_B_u = 0.21746, 0.00015
        self.base_A_v, self.base_A_r, self.base_B_r = 0.00011, 0.03265, 0.00793
        
        self.A_u = torch.zeros((self.agent_num, 1), device=self.device)
        self.B_u = torch.zeros((self.agent_num, 1), device=self.device)
        self.A_v = torch.zeros((self.agent_num, 1), device=self.device)
        self.A_r = torch.zeros((self.agent_num, 1), device=self.device)
        self.B_r = torch.zeros((self.agent_num, 1), device=self.device)
        
        self.water_cx = torch.zeros((self.agent_num, 1), device=self.device)
        self.water_cy = torch.zeros((self.agent_num, 1), device=self.device)
        self.bias_r = torch.zeros((self.agent_num, 1), device=self.device)

        self.pos, self.psi, self.vel = torch.zeros((self.agent_num, 2), device=self.device), torch.zeros((self.agent_num, 1), device=self.device), torch.zeros((self.agent_num, 3), device=self.device)
        self.target_pos, self.target_vel = torch.zeros((self.agent_num, 2), device=self.device), torch.zeros((self.agent_num, 2), device=self.device)
        self.target_speed, self.target_heading = torch.zeros((self.agent_num, 1), device=self.device), torch.zeros((self.agent_num, 1), device=self.device)
        self.target_w = torch.zeros((self.agent_num, 1), device=self.device)
        self.step_count = torch.zeros((self.agent_num, 1), device=self.device)
        
        self.n_points = 31
        self.max_range = 1.5
        self.angles_body = torch.linspace(-math.radians(60), math.radians(60), self.n_points, device=self.device)
        self.mem_left, self.mem_right = torch.zeros((self.agent_num, 1), device=self.device), torch.zeros((self.agent_num, 1), device=self.device)

        # 初始化用于存储多帧观测的缓冲区
        self.single_obs_dim = config.single_obs_dim
        self.stack_frames = config.stack_frames
        self.obs_stack = torch.zeros((self.agent_num, self.single_obs_dim * self.stack_frames), device=self.device)

    def reset(self, indices=None):
        if indices is None: indices = torch.arange(self.agent_num, device=self.device)
        n = len(indices)
        
        rand_scale = lambda: 0.8 + 0.4 * torch.rand((n, 1), device=self.device)
        self.A_u[indices] = self.base_A_u * rand_scale()
        self.B_u[indices] = self.base_B_u * rand_scale()
        self.A_v[indices] = self.base_A_v * rand_scale()
        self.A_r[indices] = self.base_A_r * rand_scale()
        self.B_r[indices] = self.base_B_r * rand_scale()
        
        current_mag = torch.rand((n, 1), device=self.device) * 0.015
        current_dir = torch.rand((n, 1), device=self.device) * 2 * math.pi
        self.water_cx[indices] = current_mag * torch.cos(current_dir)
        self.water_cy[indices] = current_mag * torch.sin(current_dir)
        
        self.bias_r[indices] = (torch.rand((n, 1), device=self.device) - 0.5) * 0.04

        valid_boat = torch.zeros(n, dtype=torch.bool, device=self.device)
        while not valid_boat.all():
            idx = (~valid_boat).nonzero(as_tuple=True)[0]
            self.pos[indices[idx]] = torch.rand(len(idx), 2, device=self.device) * 0.8 + 0.2
            dist_to_center = torch.norm(self.pos[indices] - torch.tensor([0.6, 0.6], device=self.device), dim=1)
            valid_boat = dist_to_center > 0.30

        valid_target = torch.zeros(n, dtype=torch.bool, device=self.device)
        while not valid_target.all():
            idx = (~valid_target).nonzero(as_tuple=True)[0]
            self.target_pos[indices[idx]] = torch.rand(len(idx), 2, device=self.device) * 0.8 + 0.2
            dist_to_center = torch.norm(self.target_pos[indices] - torch.tensor([0.6, 0.6], device=self.device), dim=1)
            dist_to_boat = torch.norm(self.target_pos[indices] - self.pos[indices], dim=1)
            valid_target = (dist_to_center > 0.25) & (dist_to_boat > 0.40)

        self.psi[indices] = (torch.rand((n, 1), device=self.device) - 0.5) * 2 * math.pi
        self.vel[indices], self.step_count[indices] = 0.0, 0
        
        self.target_speed[indices] = 0.01 + torch.rand((n, 1), device=self.device) * 0.02
        self.target_heading[indices] = (torch.rand((n, 1), device=self.device) - 0.5) * 2 * math.pi
        self.target_w[indices] = torch.randn((n, 1), device=self.device) * 0.6
        self.mem_left[indices], self.mem_right[indices] = 0.0, 0.0
        
        # Reset时用同一帧画面填满整个堆叠缓冲区
        single_obs, _ = self._update_perception(indices)
        self.obs_stack[indices] = single_obs.repeat(1, self.stack_frames)
        
        if len(indices) == self.agent_num:
            return self.obs_stack.clone()
        return self.obs_stack[indices].clone()

    def _batched_raycast_semantic(self, indices):
        pos, psi, t_pos = self.pos[indices], self.psi[indices], self.target_pos[indices]
        ray_angles = psi + self.angles_body.unsqueeze(0)
        dir_x, dir_y = torch.cos(ray_angles) + 1e-8, torch.sin(ray_angles) + 1e-8

        t_x0 = torch.where(dir_x < 0, -pos[:, 0:1] / dir_x, float('inf'))
        t_x1 = torch.where(dir_x > 0, (self.world_size - pos[:, 0:1]) / dir_x, float('inf'))
        t_y0 = torch.where(dir_y < 0, -pos[:, 1:2] / dir_y, float('inf'))
        t_y1 = torch.where(dir_y > 0, (self.world_size - pos[:, 1:2]) / dir_y, float('inf'))
        dist_env = torch.min(torch.min(t_x0, t_x1), torch.min(t_y0, t_y1))

        oc_x, oc_y = pos[:, 0:1] - 0.6, pos[:, 1:2] - 0.6
        b = oc_x * dir_x + oc_y * dir_y
        c = oc_x**2 + oc_y**2 - 0.15**2
        disc = b**2 - c
        t_c = torch.where(disc > 0, -b - torch.sqrt(torch.relu(disc)), float('inf'))
        dist_env = torch.min(dist_env, torch.where(t_c > 0, t_c, float('inf')))

        toc_x, toc_y = pos[:, 0:1] - t_pos[:, 0:1], pos[:, 1:2] - t_pos[:, 1:2]
        tb = toc_x * dir_x + toc_y * dir_y
        tc = toc_x**2 + toc_y**2 - 0.05**2
        tdisc = tb**2 - tc
        t_target = torch.where(tdisc > 0, -tb - torch.sqrt(torch.relu(tdisc)), float('inf'))
        t_target = torch.where(t_target > 0, t_target, float('inf'))

        hit_target = (t_target < dist_env) & (t_target < self.max_range)
        color_array = hit_target.float()
        physical_dist = torch.min(dist_env, t_target)
        sensor_depth = torch.clamp(physical_dist, 0.02, self.max_range)

        # =======================================================
        # 🚀 核心修改：注入 D435i 的物理视场角盲区 (Sim2Real 严格对齐)
        # 根据内参计算: 
        # 深度在 640x480 下真实水平 FOV = 79.4° (即 ±39.7°)
        # RGB 真实水平 FOV = 69.4° (即 ±34.7°)
        # =======================================================
        valid_depth_mask = (self.angles_body >= -math.radians(39.7)) & (self.angles_body <= math.radians(39.7))
        valid_rgb_mask = (self.angles_body >= -math.radians(34.7)) & (self.angles_body <= math.radians(34.7))

        # 1. 屏蔽盲区深度：超出 79.4° 的外侧射线，强制设为最大安全量程 (看不见墙)
        sensor_depth = torch.where(
            valid_depth_mask.unsqueeze(0), 
            sensor_depth, 
            torch.full_like(sensor_depth, self.max_range)
        )
        
        # 2. 屏蔽盲区语义：超出 69.4° 的外侧射线，即使物理上撞到了猎物，也强制清零 (因为彩色镜头看不见)
        color_array = torch.where(
            valid_rgb_mask.unsqueeze(0), 
            color_array, 
            torch.zeros_like(color_array)
        )
        # =======================================================

        return sensor_depth, color_array, physical_dist

    def _update_perception(self, indices=None):
        if indices is None: indices = torch.arange(self.agent_num, device=self.device)
        sensor_depth, color_array, physical_dist = self._batched_raycast_semantic(indices)

        sees_target = (color_array.sum(dim=1, keepdim=True) > 0).float()
        ray_idx = torch.arange(31, device=self.device).float()
        center_idx = (color_array * ray_idx).sum(dim=1, keepdim=True) / (color_array.sum(dim=1, keepdim=True) + 1e-6)

        is_left, is_right = (center_idx < 15.0).float(), (center_idx > 15.0).float()
        self.mem_left[indices] = torch.where(sees_target > 0, is_left, self.mem_left[indices])
        self.mem_right[indices] = torch.where(sees_target > 0, is_right, self.mem_right[indices])

        # 去除了 u，仅截取角速度 r 即 self.vel[indices][:, 2:3]
        single_obs = torch.cat((sensor_depth / self.max_range, color_array, self.vel[indices][:, 2:3], self.mem_left[indices], self.mem_right[indices]), dim=1)
        return single_obs, physical_dist

    def step(self, action_idx):
        u, v, r = self.vel[:, 0:1], self.vel[:, 1:2], self.vel[:, 2:3]
        
        t_x = torch.where(action_idx == 0, 100.0, torch.where(action_idx > 1, 50.0, 0.0))
        t_z = torch.where(action_idx == 2, 50.0, torch.where(action_idx == 3, -50.0, 0.0))

        u_dot = -self.A_u * u + self.B_u * t_x
        v_dot = -self.A_v * v - u * r
        r_dot = -self.A_r * r + self.B_r * t_z + self.bias_r

        self.vel[:, 0:1] += u_dot * self.dt
        self.vel[:, 1:2] += v_dot * self.dt
        self.vel[:, 2:3] += r_dot * self.dt
        
        u, v, r = self.vel[:, 0:1], self.vel[:, 1:2], self.vel[:, 2:3]
        self.psi += r * self.dt
        
        self.pos[:, 0:1] += (u * torch.cos(self.psi) - v * torch.sin(self.psi) + self.water_cx) * self.dt
        self.pos[:, 1:2] += (u * torch.sin(self.psi) + v * torch.cos(self.psi) + self.water_cy) * self.dt

        self.target_heading += self.target_w * self.dt
        self.target_w = torch.where(torch.rand((self.agent_num, 1), device=self.device) < 0.05, torch.randn_like(self.target_w) * 1.0, self.target_w)
        self.target_pos[:, 0:1] += self.target_speed * torch.cos(self.target_heading) * self.dt
        self.target_pos[:, 1:2] += self.target_speed * torch.sin(self.target_heading) * self.dt
        
        mask_x = (self.target_pos[:, 0:1] < 0.15) | (self.target_pos[:, 0:1] > 1.05)
        self.target_heading = torch.where(mask_x, math.pi - self.target_heading, self.target_heading)
        mask_y = (self.target_pos[:, 1:2] < 0.15) | (self.target_pos[:, 1:2] > 1.05)
        self.target_heading = torch.where(mask_y, -self.target_heading, self.target_heading)
        self.target_pos = torch.clamp(self.target_pos, 0.15, 1.05)
        
        self.step_count += 1

        single_obs, physical_dist = self._update_perception()
        
        # 更新多帧堆叠张量 (整体左移一个单帧身位，最新帧放在最右侧)
        self.obs_stack = torch.roll(self.obs_stack, shifts=-self.single_obs_dim, dims=1)
        self.obs_stack[:, -self.single_obs_dim:] = single_obs
        stacked_obs = self.obs_stack.clone()

        min_clearance, _ = torch.min(physical_dist, dim=1, keepdim=True)
        dist_to_target = torch.norm(self.target_pos - self.pos, dim=1, keepdim=True)
        dist_to_center = torch.norm(self.pos - torch.tensor([[0.6, 0.6]], device=self.device), dim=1, keepdim=True)

        crash_wall = (self.pos < 0.05).any(dim=1, keepdim=True) | (self.pos > 1.15).any(dim=1, keepdim=True)
        crash_obs = dist_to_center < 0.17 
        crash = crash_wall | crash_obs
        
        success = dist_to_target < 0.15
        valid_success = success & ~crash

        # ================= 归一化防刷分 Reward (使用真实的 u 监督) =================
        reward = torch.full_like(dist_to_target, -0.2)
        sees_target = (single_obs[:, 31:62].sum(dim=1, keepdim=True) > 0).float()
        
        center_idx = (single_obs[:, 31:62] * torch.arange(31, device=self.device).float()).sum(dim=1, keepdim=True) / (single_obs[:, 31:62].sum(dim=1, keepdim=True) + 1e-6)
        center_err = torch.abs(torch.where(sees_target > 0, center_idx, torch.full_like(center_idx, 15.0)) - 15.0) / 15.0
        
        u_norm = u / 0.068
        r_norm = torch.clamp(r / 5.0, -1.0, 1.0)
        
        torpedo_reward = 0.1 * (1.0 - center_err) + 0.05 * u_norm
        spin_correctness = r_norm * self.mem_left - r_norm * self.mem_right
        spin_reward = torch.where((self.mem_left == 0) & (self.mem_right == 0), torch.abs(r_norm), spin_correctness)
        roomba_reward = 0.05 * spin_reward 
        
        reward += sees_target * torpedo_reward + (1.0 - sees_target) * roomba_reward
        
        is_danger = min_clearance < 0.35
        danger_penalty = -1.0 * (0.35 - min_clearance) / 0.27
        brake_penalty = torch.where(u_norm > 0.4, -1.0 * u_norm, torch.zeros_like(u_norm))
        reward += is_danger * (danger_penalty + brake_penalty)
        
        reward[crash] = -60.0
        reward[valid_success] = 100.0
        
        over = crash | (self.step_count >= self.max_steps) | valid_success

        if valid_success.any():
            succ_idx = torch.nonzero(valid_success.flatten()).flatten()
            self.target_pos[succ_idx] = torch.rand(len(succ_idx), 2, device=self.device) * 0.6 + 0.3
            self.mem_left[succ_idx], self.mem_right[succ_idx] = 0.0, 0.0

        # 返回的是堆叠后的 195 维特征
        return stacked_obs, reward, over.float(), valid_success, crash

# ================= 4. 实时监控绘图类 =================
class TrainingPlotter:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.history = {'reward': [], 'success': [], 'crash': []}
        self.epochs = []
        if not self.enabled:
            self.fig, self.axs = None, None
            return
        plt.ion()
        self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))

    def update(self, epoch, reward, success_rate, crash_rate):
        self.epochs.append(epoch)
        self.history['reward'].append(reward)
        self.history['success'].append(success_rate)
        self.history['crash'].append(crash_rate)
        if not self.enabled:
            return
        
        for ax in self.axs: ax.clear()
        
        self.axs[0].plot(self.epochs, self.history['reward'], color='blue')
        self.axs[0].set_title('Avg Reward'); self.axs[0].grid(True)
        self.axs[1].plot(self.epochs, self.history['success'], color='green')
        self.axs[1].set_title('Real Success Rate (%)'); self.axs[1].grid(True)
        self.axs[2].plot(self.epochs, self.history['crash'], color='red')
        self.axs[2].set_title('Real Crash Rate (%)'); self.axs[2].grid(True)
        
        plt.draw(); plt.pause(0.01)

# ================= 5. 主训练循环 =================
if __name__ == '__main__':
    cfg = Config()
    env = USVHunterEnv_Semantic(cfg)
    
    # 显式传入新的 obs_dim
    actor = Actor(input_dim=cfg.obs_dim).to(cfg.device)
    critic = Critic(input_dim=cfg.obs_dim).to(cfg.device)
    
    optimizer_a = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    optimizer_c = optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    buffer = Agent_State_Buffer(cfg.obs_dim, cfg.agents_num, cfg.max_step, cfg.device)
    plotter = TrainingPlotter(enabled=cfg.enable_plot)

    print(f"🌊 [Sim2Real] 启动！模型加入了D435i真实硬件的物理视野遮罩，输入维度: {cfg.obs_dim} on {cfg.device} ...")
    print(f"Config | agents={cfg.agents_num} | max_step={cfg.max_step} | epochs={cfg.train_epochs} | batch={cfg.batch_size} | update_epochs={cfg.update_epochs} | plot={cfg.enable_plot}")

    for epoch in range(cfg.train_epochs):
        obs = env.reset()
        total_success, total_crashes, total_episodes = 0, 0, 0
        
        for step in range(cfg.max_step):
            with torch.no_grad():
                logits = actor(obs)
                dist_p = torch.distributions.Categorical(logits=logits)
                action = dist_p.sample().unsqueeze(-1)
                val = critic(obs)
            
            next_obs, reward, over, success, crash = env.step(action)
            
            total_success += success.sum().item()
            total_crashes += crash.sum().item()
            
            buffer.obs_buffer[step], buffer.action_buffer[step] = obs, action
            buffer.reward_buffer[step], buffer.over_buffer[step] = reward, over
            buffer.value_buffer[step], buffer.logp_buffer[step] = val, dist_p.log_prob(action.squeeze(-1)).unsqueeze(-1)
            
            obs = next_obs
            done_idx = torch.nonzero(over.flatten()).flatten()
            if len(done_idx) > 0: 
                # 这里 env.reset(done_idx) 已经正确返回了重置后的堆叠特征
                obs[done_idx] = env.reset(done_idx)
                total_episodes += len(done_idx) 

        with torch.no_grad(): next_val = critic(obs)
        buffer.compute_GAE(next_val, cfg.gamma, cfg.lam)
        
        b_obs = buffer.obs_buffer.view(-1, cfg.obs_dim)
        b_act = buffer.action_buffer.view(-1)
        b_logp = buffer.logp_buffer.view(-1)
        b_adv = buffer.adv_buffer.view(-1)
        b_ret = buffer.return_buffer.view(-1)
        dataset_size = b_obs.shape[0]
        
        for _ in range(cfg.update_epochs):
            idxs = torch.randperm(dataset_size, device=cfg.device)
            for start in range(0, dataset_size, cfg.batch_size):
                batch_idx = idxs[start:start + cfg.batch_size]
                
                logits = actor(b_obs[batch_idx])
                dist = torch.distributions.Categorical(logits=logits, validate_args=False)
                
                # 🛠️ FIX: Clamp actions to ensure they are strictly 0, 1, 2, or 3.
                # This prevents the CUDA crash if invalid values snuck into the buffer.
                safe_actions = b_act[batch_idx].to(dtype=torch.long).view(-1)
                safe_actions = torch.clamp(safe_actions, 0, cfg.action_dim - 1)
                
                new_logp = dist.log_prob(safe_actions) 
                
                ratio = torch.exp(new_logp - b_logp[batch_idx])
                surr1 = ratio * b_adv[batch_idx]
                # ... rest of the code remains the same ...
                ratio = torch.exp(new_logp - b_logp[batch_idx])
                surr1 = ratio * b_adv[batch_idx]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * b_adv[batch_idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                optimizer_a.zero_grad()
                actor_loss.backward()
                optimizer_a.step()
                
                val_pred = critic(b_obs[batch_idx]).squeeze(-1)
                critic_loss = (b_ret[batch_idx] - val_pred).pow(2).mean()
                
                optimizer_c.zero_grad()
                critic_loss.backward()
                optimizer_c.step()

        avg_reward = buffer.reward_buffer.mean().item()
        episodes_run = max(total_episodes, 1) 
        success_rate = (total_success / episodes_run) * 100.0
        crash_rate = (total_crashes / episodes_run) * 100.0
        timeout_rate = 100.0 - success_rate - crash_rate

        print(f"Epoch {epoch + 1:03d} | Avg Reward: {avg_reward:+.3f} | 真实胜率: {success_rate:05.1f}% | 撞墙率: {crash_rate:05.1f}% | 苟活率: {timeout_rate:05.1f}%")
        plotter.update(epoch + 1, avg_reward, success_rate, crash_rate)

    torch.save(actor.state_dict(), 'ppo_usv_sim2real_aligned.pth')
    print("✅ 纯端到端架构训练完成！包含物理相机盲区的权重已保存至 ppo_usv_sim2real_aligned.pth。")
    if cfg.enable_plot:
        plt.ioff(); plt.show()
