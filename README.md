# RS1 — V33 DDM + PPO

> 基准: v16 (91.6% @ 4096ep) | 架构: Tanh + /scale + 64 hidden + 13-dim

---

## 文件清单

| 文件 | 说明 |
|------|------|
| `train_plot_remove_v33.py` | v33 DDM 训练脚本 |
| `teacher_student_v33.py` | v33 Teacher-Student PPO 训练脚本 |
| `uvr_next_policy_0718_2040.pt` | v33 DDM 模型 (test_loss=1.91) |
| `loss_train_test_0718_2040.png` | DDM 训练 loss 曲线 |
| `train_test_comparison_0718_2040.png` | DDM 轨迹对比图 |
| `action_tests_50_0718_2040.png` | DDM 单动作测试图 |
| `all_trajectories_0718_2040.png` | 全轨迹概览 |
| `training_curves.png` | v16 TS 训练曲线 (91.6%) |

---

## v33 vs v16 改动

| 改动 | v16 | v33 |
|------|-----|-----|
| 架构 | Tanh + /scale + 64 hidden | **不变** |
| 数据源 | 先重采样→后窗口 (16段) | **先20Hz窗口→后重采样 (45段)** |
| Test 分割 | 按段 (数据泄露) | **按 CSV 文件 (独立)** |
| 最佳模型 | 保存最后一个 | **track test loss 最低** |
| 梯度裁剪 | 无 | **clip_grad_norm_(1.0)** |
| DDM test loss | ~2.0 | **1.91** |
