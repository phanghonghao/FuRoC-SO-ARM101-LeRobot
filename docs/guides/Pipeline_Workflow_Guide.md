# Pipeline Workflow Guide — Orchestrator / Sim / Plot-Train

> Three core commands for the Z1 12DOF training lifecycle.
> Each covers different stages: **automated training pipeline**, **simulation video recording**, **learning curve plots**.

---

## Feature Matrix

| Feature | `--resume --orchestrator` | `--sim` | `/plot-train-Z1` |
|---------|--------------------------|---------|-------------------|
| Training execution | **Yes** | No | No |
| Overfitting detection | **Yes** | No | No |
| JIT export | **Yes** | Yes | No |
| Isaac Sim video | **Yes** | Yes | No |
| MuJoCo video | **Yes** | Yes | No |
| Learning curve plots (server) | **Yes** | No | **Yes** |
| Download plots to local | No | No | **Yes** |
| PDF report generation | No | No | **Yes** |
| Save models to local `models/` | **Yes** | No | No |
| Update `bestmodel_phase.json` | **Yes** | No | No |
| Video label overlay | **Yes** | **Yes** | No |

---

## 1. `--resume --orchestrator` (5-Phase Automated Pipeline)

### Overview

Fully automated multi-phase training pipeline. Runs 10 sub-phases across 5 terrain difficulty levels, with embedded monitoring, overfitting detection, auto-rollback, and post-phase artifact generation.

### When to Use

- Start a fresh training pipeline from scratch
- Resume an interrupted pipeline from saved state
- Let the system train autonomously with minimal human intervention

### Sub-commands

| Sub-command | Description |
|-------------|-------------|
| `--automation --start` | Launch fresh pipeline |
| `--automation --resume` | Resume from saved state (reads `orchestrator_state.json`) |
| `--automation --status` | Check pipeline status + current sub-phase progress |
| `--automation --tail` | Show orchestrator log tail |
| `--automation --stop` | Stop pipeline (preserves state for resume) |
| `--automation --dry-run` | Print all 10 sub-phase configs without running |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--plan <yaml>` | `training_plans/z1_5phase_plan.yaml` | Training plan file |
| `--gpus <N>` | 4 | Number of GPUs |
| `--from <SUB_PHASE>` | p1_coarse | Start from specific sub-phase |
| `--fresh` | Off | Ignore saved state |
| `--poll <N>` | 120 | Monitor poll interval (seconds) |

### 5-Phase Curriculum

```
p1 (Flat — Bootstrap)
  p1_coarse → p1_fine → post-phase
    ↓ resume from best checkpoint
p2 (Flat — Velocity Tracking)
  p2_coarse → p2_fine → post-phase
    ↓
p3 (Gentle Terrain)
  p3_coarse → p3_fine → post-phase
    ↓
p4 (Rough Terrain)
  p4_coarse → p4_fine → post-phase
    ↓
p5 (Full Terrain + Polish)
  p5_coarse → p5_fine → post-phase → final model
```

10 sub-phases, ~210K total iterations.

### Per Sub-Phase Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Sub-Phase Lifecycle                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Config Generation                                               │
│     ├── config_generator → velocity_env_cfg.py (terrain, rewards)   │
│     └── ppo_override → temp PPO config (LR, entropy)               │
│                                                                     │
│  2. Training Launch                                                 │
│     └── torchrun on N GPUs → train.py / train_multigpu.py          │
│                                                                     │
│  3. Embedded Monitoring (polls every 120s)                          │
│     ├── TensorBoard data parsing                                    │
│     ├── Overfitting detection (reward decline, action rate, etc.)   │
│     └── Auto-stop on overfitting or max iterations                  │
│                                                                     │
│  4. Best Checkpoint Resolution                                      │
│     └── Find best model from monitor state → model_N.pt             │
│                                                                     │
│  5. Rollback Check                                                  │
│     └── if best_reward < starting_reward × 0.95                     │
│         → discard, retry with LR×0.5 (max 1 retry)                 │
│                                                                     │
│  6. Post-Phase Artifact Pipeline                                    │
│     ├── JIT export → policy.pt                                      │
│     ├── MuJoCo video recording                                      │
│     ├── Isaac Sim video recording                                   │
│     └── Learning curve plot generation (server-side)                │
│                                                                     │
│  7. Advance to Next Sub-Phase                                       │
│     └── Resume from best checkpoint of current sub-phase            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Post-Phase Artifacts (on server)

Generated by `_run_post_phase_pipeline()` in `phase_orchestrator.py`:

| Step | Script | Output | Config Key |
|------|--------|--------|------------|
| 1. JIT export | `scripts/export_jit.py` | `<run_dir>/exported/policy.pt` | `enable_jit_export` |
| 2. MuJoCo video | `sim2sim/mujoco_manual.py` | `videos/phase_pipeline/<sp_id>_mujoco.mp4` | `enable_mujoco_video` |
| 3. Isaac video | `scripts/rsl_rl/play.py` | `videos/phase_pipeline/<sp_id>.mp4` | `enable_isaac_video` |
| 4. Plots | `scripts/plot_learning_curves.py` | `plots/<sp_id>/*.png` | `enable_plots` |

All enabled by default in `z1_5phase_plan.yaml`.

### Local Post-Processing (models + videos + labels + JSON)

**This is done by the gpu-train skill** (not the orchestrator script itself). When `--tail` detects a just-completed sub-phase, it automatically runs the full post-processing pipeline:

```
┌─ Post-Processing ─────────────────────────────┐
│  Phase completed: p3_coarse                     │
│  Downloaded: model_9500.pt + policy.pt          │
│  Videos: isaaclab + mujoco (labeled)            │
│  Updated: docs/bestmodel_phase.json             │
└────────────────────────────────────────────────┘
```

**Step 1: Download models to local**
```bash
LOCAL_DIR="D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/models/p/${SUB_PHASE}"
mkdir -p "${LOCAL_DIR}"

# Best checkpoint (for resuming)
scp phh@192.168.120.155:~/magiclab_rl_lab/logs/rsl_rl/.../${RUN_DIR}/${BEST_MODEL} \
    "${LOCAL_DIR}/${SUB_PHASE}_${BEST_MODEL}"

# JIT policy
scp phh@192.168.120.155:~/magiclab_rl_lab/logs/rsl_rl/.../${RUN_DIR}/exported/policy.pt \
    "${LOCAL_DIR}/${SUB_PHASE}_policy.pt"
```

**Step 2: Download videos from server (if generated)**

Check if the orchestrator generated videos on the server (`videos/phase_pipeline/`), then download:

```bash
VIDEO_DIR="D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/videos/p/${SUB_PHASE}"
mkdir -p "${VIDEO_DIR}"

# Isaac Sim video
scp phh@192.168.120.155:~/magiclab_rl_lab/videos/phase_pipeline/${SUB_PHASE}.mp4 \
    "${VIDEO_DIR}/${SUB_PHASE}_isaaclab.mp4"

# MuJoCo video
scp phh@192.168.120.155:~/magiclab_rl_lab/videos/phase_pipeline/${SUB_PHASE}_mujoco.mp4 \
    "${VIDEO_DIR}/${SUB_PHASE}_mujoco.mp4"

# Training params (mandatory — same as --sim)
scp -r phh@192.168.120.155:~/magiclab_rl_lab/logs/rsl_rl/.../${RUN_DIR}/params/ \
    "${VIDEO_DIR}/params/"
```

**Step 3: Add Labels to videos (mandatory)**

Same as `--sim` — burn model/run/reward/terrain/iteration labels into each video:

```bash
LABEL_SCRIPT="D:/Desktop_Files/GPU-Train/RTX6000/Magicbot_Z1/scripts/label_video.py"

# Label Isaac Sim video
python "$LABEL_SCRIPT" "${VIDEO_DIR}/${SUB_PHASE}_isaaclab.mp4" \
    --model ${BEST_MODEL} --run ${RUN_DIR} \
    --reward <REWARD> --terrain <TERRAIN> \
    --iteration <ITER_NUM> --action-mean <ACTION_MEAN_ABS>
mv "${VIDEO_DIR}/${SUB_PHASE}_isaaclab_labeled.mp4" "${VIDEO_DIR}/${SUB_PHASE}_isaaclab.mp4"

# Label MuJoCo video
python "$LABEL_SCRIPT" "${VIDEO_DIR}/${SUB_PHASE}_mujoco.mp4" \
    --model ${BEST_MODEL} --run ${RUN_DIR} \
    --reward <REWARD> --terrain <TERRAIN> \
    --iteration <ITER_NUM> --action-mean <ACTION_MEAN_ABS>
mv "${VIDEO_DIR}/${SUB_PHASE}_mujoco_labeled.mp4" "${VIDEO_DIR}/${SUB_PHASE}_mujoco.mp4"
```

**Step 4: Update bestmodel_phase.json**

File: `D:\Desktop_Files\GPU-Train\RTX6000\Magicbot_Z1\docs\bestmodel_phase.json`

```json
{
  "id": "p1_coarse",
  "best_model": "model_2900.pt",
  "best_iteration": 2900,
  "local_models": {
    "checkpoint": "models/p/p1_coarse/p1_coarse_model_2900.pt",
    "jit_policy": "models/p/p1_coarse/p1_coarse_policy.pt"
  },
  "video": "videos/p/p1_coarse/"
}
```

### Files on Server

| File | Path |
|------|------|
| Pipeline log | `/tmp/z1_5phase_pipeline.log` |
| State file | `~/magiclab_rl_lab/orchestrator_state.json` |
| Training log | `~/magiclab_rl_lab/logs/train_<sub_phase>.log` |
| Run dirs | `~/magiclab_rl_lab/logs/rsl_rl/magiclab_z1_12dof_velocity/<run_dir>/` |
| Generated cfgs | `~/magiclab_rl_lab/tmp/phase_configs/<sub_phase>/` |
| Server plots | `~/magiclab_rl_lab/plots/<sp_id>/` |
| Server videos | `~/magiclab_rl_lab/videos/phase_pipeline/` |

### Local Files

| File | Path |
|------|------|
| Models | `RTX6000/Magicbot_Z1/models/p/<sub_phase>/` |
| Phase status | `RTX6000/Magicbot_Z1/docs/bestmodel_phase.json` |
| Pipeline state | Downloaded from server as needed |

### Architecture

```
phase_orchestrator.py
  ├── phase_manager.py       — Parse YAML, 3-layer merge (defaults → phase → sub_phase)
  ├── config_generator.py    — Generate velocity_env_cfg.py from params dict
  ├── ppo_override.py        — Generate temp PPO config (LR, entropy override)
  ├── training_launcher.py   — torchrun launch with --agent_cfg
  ├── embedded_monitor.py    — TensorBoard polling, overfitting detection
  └── state_store.py         — JSON state persistence for resume
```

---

## 2. `--sim` (Simulation Video Recording)

### Overview

Record both Isaac Lab and MuJoCo simulation videos from a trained checkpoint. All recording happens on the RTX server. Used for evaluating trained policies visually.

### When to Use

- Record a video for a specific checkpoint
- Compare policies across different training runs
- Evaluate a model after manual training (non-pipeline)

### Important: Does NOT Include

- **No training** — only records video from existing checkpoints
- **No model saving** — doesn't download models to local
- **No bestmodel_phase.json update**
- **No plot generation** — use `/plot-train-Z1` separately

### Modes

#### Auto-resolve (`--best <VERSION>`)

```bash
/gpu-train --sim --best s2_gentle              # From best_models.json
/gpu-train --sim --best p1_coarse              # Pipeline sub-phase
/gpu-train --sim --best s2_gentle --mujoco_only
/gpu-train --sim --best s2_gentle --isaac_only
```

#### Manual (`--checkpoint <PATH>`)

```bash
/gpu-train --sim --checkpoint logs/rsl_rl/.../model_5000.pt
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--checkpoint` | — | Model checkpoint path (required if no --best) |
| `--best <VERSION>` | — | Auto-resolve from best_models.json |
| `--vel_x` | 0.5 | Forward velocity (m/s) |
| `--vel_y` | 0.0 | Lateral velocity (m/s) |
| `--vel_yaw` | 0.0 | Yaw angular velocity (rad/s) |
| `--duration` | 10 | MuJoCo simulation duration (seconds) |
| `--video_length` | 200 | Isaac Sim video steps |
| `--mujoco_only` | — | MuJoCo recording only |
| `--isaac_only` | — | Isaac Sim recording only |
| `--skip_export` | — | Skip JIT export step |

### Pipeline Steps

```
┌─────────────────────────────────────────────────────────────────────┐
│                     --sim Pipeline (5 Steps)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Export JIT Policy (skip with --skip_export)                │
│     ssh → export_jit.py --checkpoint <PATH>                        │
│     Output: <run_dir>/exported/policy.pt                           │
│                                                                     │
│  Step 2: Isaac Sim Recording (headless)                             │
│     ssh → play_z1_video.py --checkpoint --video --headless         │
│     Output: <run_dir>/videos/play/rl-video-step-0.mp4              │
│                                                                     │
│  Step 3: MuJoCo Recording (EGL offscreen)                          │
│     ssh → mujoco_sim2sim.py --record --headless                    │
│     Output: /tmp/<NAME>_mujoco.mp4                                 │
│                                                                     │
│  Step 4: Download Videos                                            │
│     scp → local videos/<VERSION>/ directory                        │
│     - <NAME>_rtx_isaaclab.mp4                                      │
│     - <NAME>_mujoco.mp4                                            │
│                                                                     │
│  Step 5: Add Labels (mandatory)                                     │
│     label_video.py — model name, reward, terrain, iteration        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Quick Options

| Flag | Steps Executed |
|------|---------------|
| (default) | 1→2→3→4→5 |
| `--best <VERSION>` | 1→2→3→4→5 |
| `--mujoco_only` | 1→3→4→5 |
| `--isaac_only` | 1→2→4→5 |
| `--skip_export` | 2→3→4→5 |

### One-click Script

```bash
bash D:/Desktop_Files/GPU-Train/RTX6000/rtx_record_video.sh <RUN_DIR> <CHECKPOINT> [VIDEO_LENGTH] [VEL_X] [DURATION]
```

### Output Location

```
videos/
└── <VERSION_NAME>/
    ├── <NAME>_rtx_isaaclab.mp4    ← Isaac Lab recording
    └── <NAME>_mujoco.mp4          ← MuJoCo recording
```

### Common Issues

| Symptom | Fix |
|---------|-----|
| `size mismatch` in export | Check checkpoint format (rsl-rl 3.x vs 5.x) |
| Robot falls in MuJoCo | Use `mujoco_sim2sim.py` (has sim2sim fixes) |
| Isaac Sim video all black | Renderer warmup — use updated `play_z1_video.py` |
| RTX Isaac Sim `--video` fails | Do NOT use xvfb-run; just `--headless` is enough |
| MuJoCo EGL fails | Check `~/miniconda3/envs/isaaclab/share/glvnd/egl_vendor.d/10_nvidia.json` |

---

## 3. `/plot-train-Z1` (Learning Curve Plots + PDF Report)

### Overview

Generate learning curve plots from TensorBoard data on the RTX server, download to local, and compile a single-page A4 PDF report with data analysis.

### When to Use

- Visualize training progress and trends
- Generate PDF reports for each training run
- Compare reward trends, termination reasons, efficiency across runs
- After a pipeline sub-phase completes (to get local plots + PDF)

### Important: Does NOT Include

- **No training** — reads existing TensorBoard data only
- **No model saving** — doesn't download checkpoints
- **No video recording** — use `--sim` for that
- **No bestmodel_phase.json update** — reads it but doesn't write

### Commands

| Command | Description |
|---------|-------------|
| (no args) | Generate all plots + download + PDF report for all runs |
| `--focus <RUN>` | Plots + PDF for a specific run |
| `--pipeline` | Plots + PDF for all runs in `bestmodel_phase.json` |
| `--all-runs` | Plots + PDF for every run with >5000 data points |
| `--pdf-only <RUN>` | Only regenerate PDF from existing PNGs (skip server) |
| `--sync` | Same as default (always re-reads TensorBoard) |
| `--update-readme` | Update `plots/README.md` only |

### Pipeline Steps

```
┌─────────────────────────────────────────────────────────────────────┐
│                  /plot-train-Z1 Pipeline                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Step 1: Update best_models.json (on server)                        │
│     ssh → train_monitor.py --once --terrain gentle                  │
│     scp → download best_models.json to local                        │
│                                                                     │
│  Step 2: Generate Plots (on server)                                 │
│     ssh → plot_learning_curves.py --focus_run <DIR>                 │
│     Output: 4 PNG files per run                                     │
│       1_reward_trend_<alias>.png                                    │
│       2_reward_decomposition_<alias>.png                            │
│       3_termination_<alias>.png                                     │
│       4_efficiency_<alias>.png                                      │
│                                                                     │
│  Step 3: Download + Organize (local)                                │
│     scp → download PNGs                                             │
│     Organize into plots/<alias>/ directory                          │
│       ├── 1_reward_trend.png                                        │
│       ├── 2_reward_decomposition.png                                │
│       ├── 3_termination.png                                         │
│       └── 4_efficiency.png                                          │
│                                                                     │
│  Step 4: Generate PDF Report (local)                                │
│     gen_report_pdf.py --alias <ALIAS>                               │
│     Output: plots/<alias>/report_<alias>.pdf                        │
│                                                                     │
│  Step 5: Open PDF                                                   │
│     start report PDF for user review                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Run Aliases

**5-phase pipeline runs:**

| Alias | Run Directory |
|-------|--------------|
| `p1_coarse` | `2026-05-06_15-47-12_p1_coarse` |
| `p1_fine` | `2026-05-06_17-40-13_p1_fine` |
| `p2_coarse` | `2026-05-06_18-49-40_p2_coarse` |
| `p2_fine` | `2026-05-06_19-33-51_p2_fine` |
| `p3_coarse` | `2026-05-07_03-56-16_p3_coarse` |

**Legacy single-stage runs:**

| Alias | Run Directory |
|-------|--------------|
| `s1_flat` | `2026-04-30_04-53-17_s1_flat` |
| `s2_gentle` | `2026-05-01_04-50-05_s2_gentle` |
| `s3_rough_l2` | `2026-05-01_07-04-35_s3_rough_l2` |
| `s4_full` | `2026-05-04_16-56-05_s4_full_terrain` |

### Output Files

```
plots/
├── phase_p1/
│   ├── 1_reward_trend.png         ← Reward curve with peak/best annotations
│   ├── 2_reward_decomposition.png ← Reward components + curriculum
│   ├── 3_termination.png          ← Termination reasons + episode length
│   ├── 4_efficiency.png           ← Throughput, time, entropy, LR
│   ├── report_phase_p1.pdf        ← Auto-generated A4 PDF report
│   ├── report_phase_p1.tex        ← LaTeX source (preserved)
│   └── report_phase_p1.md         ← Markdown summary
├── s4_full/
│   └── ...
└── README.md
```

### Key File Locations

| File | Location |
|------|----------|
| Plot script (server) | `~/magiclab_rl_lab/scripts/plot_learning_curves.py` |
| Output (server) | `~/magiclab_rl_lab/plots/` |
| Local plots | `RTX6000/Magicbot_Z1/plots/<alias>/` |
| Report script (local) | `RTX6000/Magicbot_Z1/scripts/gen_report_pdf.py` |
| best_models.json | `RTX6000/Magicbot_Z1/best_models.json` |
| bestmodel_phase.json | `RTX6000/Magicbot_Z1/docs/bestmodel_phase.json` |

### Plot Types

| Plot | Description |
|------|-------------|
| `1_reward_trend` | Reward over iterations with peak marker, best model marker, and progress bar |
| `2_reward_decomposition` | Individual reward components (tracking, orientation, action penalty, etc.) |
| `3_termination` | Termination reasons (timeout, bad orientation, etc.) + episode length |
| `4_efficiency` | Training throughput (steps/s), wall time, entropy, learning rate |

---

## Typical Workflow: Pipeline + Sim + Plot

### Scenario: Running the full 5-phase pipeline

```
Day 1: Launch pipeline
  └── /gpu-train --automation --start --gpus 4

Day 1-3: Monitor progress
  └── /gpu-train --automation --status    # Check pipeline state
  └── /gpu-train --automation --tail      # See recent logs
  └── /gpu-train --monitor --realtime     # Live training metrics

  On each sub-phase completion (auto-detected by --tail):
  ├── Models saved to local models/p/<sub_phase>/
  ├── bestmodel_phase.json updated
  ├── Server-side: JIT + videos + plots generated
  └── (Plots NOT downloaded locally — do manually if needed)

Day 3-4: Pipeline complete (p5_fine done)
  └── All 10 sub-phases finished

  Post-pipeline:
  ├── /gpu-train --sim --best p5_fine     # Record final evaluation video
  ├── /plot-train-Z1 --pipeline           # Download plots + PDF for all sub-phases
  └── Review reports, pick best model for deployment
```

### Scenario: Manual single-run training + evaluation

```
1. Start training
   └── /gpu-train --start --resume

2. Monitor
   └── /gpu-train --check                 # Health check
   └── /gpu-train --monitor --realtime    # Live metrics

3. Record video
   └── /gpu-train --sim --best <VERSION>  # Isaac Sim + MuJoCo video

4. Generate plots
   └── /plot-train-Z1 --focus <VERSION>   # Learning curves + PDF report

5. Review
   └── Check PDF report, video, and model metrics
```

### Scenario: Resume interrupted pipeline

```
1. Resume
   └── /gpu-train --automation --resume   # Reads orchestrator_state.json

2. Check where it left off
   └── /gpu-train --automation --status

3. Continue monitoring normally
   └── /gpu-train --automation --tail
```

---

## Overlap & Differences Summary

| Aspect | Orchestrator | Sim | Plot-Train |
|--------|-------------|-----|------------|
| **JIT export** | Auto (post-phase) | Manual step | No |
| **Isaac video** | Auto (server-side) | Manual → local | No |
| **MuJoCo video** | Auto (server-side) | Manual → local | No |
| **Plots (server)** | Auto (server-side) | No | **Yes (server-side)** |
| **Plots (local)** | No | No | **Yes (downloaded)** |
| **PDF report** | No | No | **Yes** |
| **Video labels** | **Yes** | **Yes** | No |
| **Model download** | **Yes** | No | No |
| **Phase JSON update** | **Yes** | No | No |

### Recommended: Full Evaluation After Sub-Phase

After a sub-phase completes in the orchestrator pipeline:

1. **Orchestrator auto-generates** (on server): JIT + videos + plots
2. **gpu-train skill auto-downloads** (local): model checkpoint + policy + videos (with labels) + updates `bestmodel_phase.json`
3. **Manual**: Run `/plot-train-Z1 --focus <sub_phase>` to download plots + generate PDF report
4. **Manual**: Run `/gpu-train --sim --best <sub_phase>` only if you need a separate re-recording with different parameters (default videos are already downloaded and labeled in step 2)
