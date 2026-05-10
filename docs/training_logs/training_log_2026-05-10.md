# Training Log — 2026-05-10

---

## [02:02] Session Summary — Phase 5 Execution on RTX (MuJoCo Push Task)

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | Phase 3: ACT training monitoring | Training running on RTX GPU 4 (PID 191236), batch_size=128, steps=1250; at step 492/1250 (39%), ~6.5s/step, ETA ~1.5h |
| 2 | Phase 5.1: SO-101 files transferred to RTX | `scp -r Simulation/SO101/ phh@192.168.120.155:~/so101_sim/` — scene.xml, so101_new_calib.xml, assets/, joints_properties.xml all transferred |
| 3 | Phase 5.1: scene_v2_push.xml created on RTX | Push task scene: red block (2cm cube, freejoint) at (0.25, 0, 0.02) + green target zone (cylinder, 5cm radius) at (0.15, 0.15, 0.001); floor + skybox inherited from original |
| 4 | Phase 5.1: MuJoCo scene verification on RTX | Scene loads OK: nq=13 (6 arm + 7 freejoint), nv=12, nu=6, nbody=9 (world + base + shoulder + upper_arm + lower_arm + wrist + gripper + moving_jaw + red_block) |
| 5 | Phase 5.1: EGL offscreen rendering verified | `MUJOCO_GL=egl` renders 480x640x3 uint8 images successfully; test image saved to `/tmp/so101_push_scene_test.png` |
| 6 | Phase 5.1: Jacobian IK tested for SO-101 | IK converges to target (0.25, 0, 0.05) in 6 iterations using `mj_jac` + pseudo-inverse with 0.5 step; gripper workspace: ~15-30cm forward, ~10cm lateral |
| 7 | Phase 5.1: LeRobot API import path confirmed on RTX | `from lerobot.datasets.lerobot_dataset import LeRobotDataset` works (LeRobot 0.5.1, import path is `lerobot.datasets.lerobot_dataset` not `lerobot.common.datasets`) |
| 8 | Phase 5 plan doc: docs/task_pipeline.md | Created comprehensive Phase 5 pipeline plan: task hierarchy (T1-T4), Push task design, scripted data collection strategy, multi-policy training, evaluation metrics |
| 9 | Bug fix: LeRobot import path on RTX | **Root cause**: LeRobot 0.5.1 uses `lerobot.datasets.lerobot_dataset` not `lerobot.common.datasets.lerobot_dataset` **Fix**: Use correct import path **Files**: `collect_task.py` (to be written) |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | Phase 3: ACT training completion | Still running, step 492/1250 (~39%), ETA ~1.5h; will complete overnight | Check `ssh phh@192.168.120.155 'tail -3 /tmp/so101_act_train.log'`; checkpoint at `/tmp/so101_act_train/` |
| 2 | Phase 5.2: collect_task.py script | Not yet written; need to implement scripted Push task data collection with IK + trajectory interpolation | Write `collect_task.py` locally, scp to RTX, run 300 episodes |
| 3 | Phase 5.2: Data collection (300 episodes) | Blocked behind collect_task.py | Run on RTX GPU 5 or 6 (idle), upload to `PhangHongHao/so101_push` |
| 4 | Phase 5.3: Multi-policy training on Push data | Blocked behind data collection | ACT (cuda:4) + Diffusion (cuda:5) in parallel |
| 5 | Phase 5.4: Eval + success rate | Blocked behind training | Write `eval_rollout.py`, test in MuJoCo, target >70% success rate |

### Key Decisions

- Push task (T2) chosen as first task: no gripper finesse needed, clear visual feedback, block + target zone
- Scripted demonstrations (IK-based) instead of human teleoperation: controllable, scalable, no hardware needed
- Block has freejoint (7 DOF) for physics-simulated pushing; target is visual-only (contype=0, conaffinity=0)
- Jacobian IK with pseudo-inverse sufficient for SO-101 (6 DOF arm, converges in <10 iterations)
- RTX GPU 4 used for ACT training; GPUs 5-7 available for Phase 5 data collection/training
- LeRobot 0.5.1 import path: `lerobot.datasets.lerobot_dataset` (not `lerobot.common.datasets`)

---

## [17:35] Session Summary — ACT v2 Training Design + Eval Script Research

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | Analyzed project structure | Key files: `collect_sim_data.py`, `sim_viewer.py`, `scene.xml` (at `SO-ARM101-LeRobot/Simulation/SO101/scene.xml`) |
| 2 | Researched ACT policy API | `ACTPolicy.from_pretrained(path)` loads checkpoint; `select_action(batch)` returns `(B,6)`; batch = `{"observation.state": (B,6), "observation.images": [(B,3,H,W)]}`; call `policy.reset()` on env reset |
| 3 | Designed v2 training params | batch_size=128 (was 16), steps=20000 (was 10000), save_freq=5000; dataset: `PhangHongHao/so101_sim` (800 frames, 10 ep × 80 steps); estimated ~6-7h total, first checkpoint ~1.7h |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | Clean RTX old outputs + restart training | Session interrupted (user closing computer) | `ssh phh@192.168.120.155 'rm -rf /tmp/so101_act_train'`, then launch with new params (see command below) |
| 2 | Write `eval_sim_act.py` | Research done, code not written | Write eval script using ACT API research notes |
| 3 | Wait for checkpoint + local eval | Training not started | Monitor `/gpu-train --tail`, download checkpoint at step 5000 |

### Key Decisions

- v2 training: batch_size=128 to improve GPU utilization (800 frames / 128 = 6.25 batch/epoch)
- 4 checkpoints at steps 5k, 10k, 15k, 20k for comparing training stages
- Only 800 frames of data → model will overfit, only reproduces training trajectories
- For generalizable policy need 50-100+ episodes with task-specific demonstrations

### Training Launch Command (copy-paste for next session)

```bash
# Clean old output + start training
ssh phh@192.168.120.155 "rm -rf /tmp/so101_act_train && source ~/miniconda3/etc/profile.d/conda.sh && conda activate lerobot && nohup python -m lerobot.scripts.train \
  policy=act \
  dataset_repo_id=PhangHongHao/so101_sim \
  training.batch_size=128 \
  training.offline_steps=20000 \
  training.save_freq=5000 \
  training.save_checkpoint=true \
  output_dir=/tmp/so101_act_train \
  device=cuda:0 \
  > /tmp/so101_act_train_v2.log 2>&1 & echo PID=\$!"
```

### ACT Policy API Notes (for eval_sim_act.py)

```python
from lerobot.policies.act.modeling_act import ACTPolicy

policy = ACTPolicy.from_pretrained("checkpoint_path")
policy.eval()
policy.reset()  # On env reset

batch = {
    "observation.state": torch.tensor(state),      # (B, 6)
    "observation.images": [torch.tensor(image)],    # list of (B, 3, H, W)
}
action = policy.select_action(batch)  # (B, 6)
```

---

## [23:30] Session Summary — Phase 4 VLA Exploration + Pipeline Review

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | Phase 4.1: SmolVLA dependencies installed | `transformers 5.8.0`, `tokenizers 0.22.2`, `num2words 0.5.14` |
| 2 | Bug fix: groot dataclass Python 3.12 | **Root cause**: `GR00TN15Config` fields without defaults before fields with defaults **Fix**: Added `default=None` to 4 fields **Files**: `.venv/.../lerobot/policies/groot/groot_n1.py` |
| 3 | Phase 4.1: SmolVLA base model loaded | `SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')` on CPU; 450M total, 99.9M trainable (VLM backbone frozen) |
| 4 | SmolVLA vs ACT analysis | ACT is SO-101 mainstream (30+ models on HF); SmolVLA has only 4-5 SO-101 models; SmolVLA is for VLA exploration, not recommended for SO-101 |
| 5 | ACT training status check | Step 2181/10000 (22%), 1.27s/step, ~2h45m ETA on RTX cuda:0 |
| 6 | ACT training killed | Killed PID 2089820 per user request; batch_size=16 too low, GPU utilization poor |
| 7 | Eval videos converted to GIF | `eval_10k_100steps.gif` (7.6MB), `eval_test_10steps.gif` (0.7MB) saved to `docs/so101_references/videos/` |
| 8 | README.md updated | Added Results section with 4 GIFs (eval + community demos); updated Phase status |
| 9 | Outputs cleanup | Deleted: `pusht_diffusion_test/` (3G), `so101_sim_dataset/` (114M, old format), `so101_demo_videos/` (119M, has GIF), eval mp4s. Kept: `so101_act_checkpoints/` (198M), `so101_sim_lerobot/` (7.5M). PushT regenerated in background |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | ACT v2 training (batch_size=128) on RTX | Need to clean old output + relaunch | See `docs/next_steps.md` for launch command |
| 2 | PushT regeneration | Running in background (task bfalc8mio) | Auto-completes, ~10 min for download + 10 steps |

### Key Decisions

- SmolVLA (450M) is for exploration only; ACT (52M) is the mainstream choice for SO-101
- PushT outputs kept (user request) — regenerated in background
- Old format `so101_sim_dataset/` (JSON+PNG) deleted; `so101_sim_lerobot/` (LeRobot native) kept
- Git push target: `myfork` remote (not `origin` which is upstream)
