# Real Observation Workspace

This workspace is for the first real-sensor PPO bridge test.

Files:

- `real_obs_adapter.py`: connects to the USV TCP gateway, parses fused D435i ACK details, converts `r/depth/gyro` into the PPO actor observation, and prints action probabilities.
- `models/ppo_usv_sim2real_aligned.pth`: copied PPO actor weights.
- `reference/usv_test_client.py`: copied TCP client reference.
- `reference/PPO_USV_Hunter_Semantic3_gpu.py`: copied training reference.

Default behavior is safe: every probe frame sends `S`, so the boat should stay stopped while the script prints the policy output.

Example:

```powershell
cd C:\Users\ROG\Desktop\git_clone\usv_runtime1.2\real_obs_workspace
python .\real_obs_adapter.py --host 192.168.31.31 --frames 50
```

Only after the printed action probabilities look reasonable should closed-loop action sending be tested:

```powershell
python .\real_obs_adapter.py --host 192.168.31.31 --send-policy-action --max-power 20 --soft-hz 5
```
