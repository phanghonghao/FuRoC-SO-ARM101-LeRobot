# Next Steps — SO-ARM101 LeRobot Pipeline

> Auto-generated from session 2026-05-10/11. Current pipeline status and actionable next steps.

---

## Current Status

| Phase | Status | Notes |
|:-----:|:------:|:------|
| 0 | Done | Local venv + RTX 6000D + HF Hub |
| 1 | Done | PushT validation (CPU) |
| 2 | Done | SO-101 MuJoCo data collection (10 eps, 800 frames) |
| 3 | **TODO** | ACT v2 training on RTX (need batch_size=128 relaunch) |
| 4 | Done | SmolVLA exploration (not mainstream for SO-101) |
| 5 | Designed | Push task pipeline (see `task_pipeline.md`) |

---

## Priority 1: ACT v2 Training (batch_size=128)

The previous ACT training (batch_size=16, step 2181/10000) was killed due to low GPU utilization. Relaunch with higher batch.

### Launch Command

```bash
# Clean old output + start v2 training
ssh phh@192.168.120.155 "rm -rf /tmp/so101_act_train && source ~/miniconda3/etc/profile.d/conda.sh && conda activate lerobot && nohup python -c \"
import sys
sys.argv = ['lerobot-train',
    '--dataset.repo_id=PhangHongHao/so101_sim',
    '--dataset.root=/tmp/so101_sim_lerobot',
    '--dataset.video_backend=pyav',
    '--policy.type=act',
    '--policy.repo_id=PhangHongHao/so101_act_v2',
    '--policy.device=cuda:0',
    '--batch_size=128',
    '--steps=20000',
    '--save_freq=5000',
    '--output_dir=/tmp/so101_act_train_v2',
    '--num_workers=0',
]
from lerobot.scripts.lerobot_train import main
main()
\" > /tmp/so101_act_train_v2.log 2>&1 & echo PID=\$!"
```

### Expected

- ~0.8 step/s with batch_size=128 (800 frames / 128 = 6.25 batch/epoch)
- Checkpoints at step 5000, 10000, 15000, 20000
- Total ~6-7h

### Monitor

```bash
ssh phh@192.168.120.155 'tail -5 /tmp/so101_act_train_v2.log'
```

---

## Priority 2: Increase Data Volume (50-100 episodes)

Current dataset (`PhangHongHao/so101_sim`) has only 10 episodes (800 frames). This is insufficient for a generalizable policy.

### Options

| Approach | Episodes | Time | Notes |
|----------|----------|------|-------|
| A. Scale random reaching | 100 | ~30 min | Modify `collect_sim_data.py` to loop 100 episodes |
| B. Task-specific (push) | 300 | ~2-3h | Use Phase 5 push task pipeline (`task_pipeline.md`) |
| C. Multiple tasks | 500+ | ~5h+ | Combine reaching + pushing + picking |

### Recommended: Start with Option A (quick), then B (useful)

**Option A — Scale random reaching:**

```python
# Edit collect_sim_data.py: change N_EPISODES = 10 → 100
# Re-run locally, push to HF Hub as PhangHongHao/so101_sim_v2
```

**Option B — Push task (see `docs/task_pipeline.md`):**

Write `collect_task.py` with scripted IK demonstrations for pushing a block to a target zone. Run on RTX (EGL rendering).

---

## Priority 3: Evaluation Script

Write `eval_sim_act.py` to evaluate trained ACT checkpoints in MuJoCo.

### ACT Policy API

```python
from lerobot.policies.act.modeling_act import ACTPolicy
import torch

policy = ACTPolicy.from_pretrained("outputs/so101_act_checkpoints/010000")
policy.eval()
policy.reset()  # Call on env reset

batch = {
    "observation.state": torch.tensor(state),      # (B, 6)
    "observation.images": [torch.tensor(image)],    # list of (B, 3, H, W)
}
action = policy.select_action(batch)  # (B, 6)
```

### Eval Pipeline

1. Load checkpoint
2. Reset MuJoCo env
3. Loop: render image → get state → policy.select_action → apply action → record
4. Save as MP4/GIF
5. Compare across checkpoints (5K, 10K, 15K, 20K)

---

## Priority 4: Git Push

Push current changes to `myfork` remote:

```bash
git add README.md docs/training_logs/ docs/so101_references/videos/*.gif docs/next_steps.md
git commit -m "Add Phase 4 results, eval GIFs, next steps doc"
git push myfork main
```

**Note:** Push to `myfork` (user's fork), NOT `origin` (upstream).

---

## File Locations

| File | Purpose |
|------|---------|
| `collect_sim_data.py` | Random reaching data collection |
| `outputs/so101_act_checkpoints/010000/` | ACT 10K step checkpoint (local) |
| `outputs/so101_sim_lerobot/` | LeRobot native dataset (local) |
| `docs/so101_references/videos/*.gif` | Eval + demo GIFs for README |
| `docs/task_pipeline.md` | Phase 5 push task design |
| `docs/training_logs/` | Session-by-session logs |
