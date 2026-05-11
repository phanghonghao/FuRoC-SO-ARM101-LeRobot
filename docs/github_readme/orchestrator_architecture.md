# Orchestrator Arm101 — Architecture

> Automated phase-based training pipeline for SO-101 imitation learning (MuJoCo + LeRobot + Remote GPU).

## Overview

`orchestrator_arm101` is a YAML-driven pipeline orchestrator that automates the full imitation learning workflow:

```
Collection → Training → Evaluation → (Comparison)
```

Each phase is defined in a YAML plan file, with three-layer config merging, crash recovery, and overfitting detection.

Ported from the Z1 12DOF RL training orchestrator, adapted for IL (replacing reward-based tracking with loss-based monitoring).

## Module Architecture

```
orchestrator_arm101/
├── __init__.py                 # Package marker
├── arm101_orchestrator.py      # Main controller + CLI entry point
├── phase_manager.py            # YAML plan parser + 3-layer config merge
├── state_store.py              # Crash-recovery state (atomic JSON)
├── data_collector.py           # MuJoCo scripted demo collection
├── training_launcher.py        # lerobot-train subprocess manager
├── loss_monitor.py             # Log-based loss parser + overfitting detector
└── eval_runner.py              # Checkpoint evaluation in MuJoCo
```

### Module Dependency Graph

```
arm101_orchestrator.py (Main Controller)
  ├── phase_manager.py      → PhaseConfig parsing, DAG ordering
  ├── state_store.py        → OrchestratorState persistence
  ├── data_collector.py     → MuJoCo + LeRobotDataset collection
  ├── training_launcher.py  → subprocess.Popen + sys.argv wrapper
  │     └── loss_monitor.py → regex log parsing + overfitting detection
  └── eval_runner.py        → ACTPolicy/DiffusionPolicy + MuJoCo rollout
```

## Core Components

### 1. Arm101Orchestrator (`arm101_orchestrator.py`)

Main controller with a 2-level event loop:

```python
while True:
    match state.current_phase_status:
        "pending"   → start_phase()      # dispatch by phase_type
        "running"   → monitor_phase()     # poll progress
        "complete"  → advance()           # move to next phase
        "failed"    → handle_failure()    # retry (max 2) or stop
```

**Phase dispatch table:**

| phase_type  | Start                     | Monitor              |
|-------------|---------------------------|----------------------|
| collection  | `DataCollector.run()`     | synchronous          |
| training    | `TrainingLauncher.launch()` | `LossMonitor.poll()` |
| evaluation  | `EvalRunner.evaluate_checkpoint()` | synchronous  |
| comparison  | same as training          | same as training     |

**Features:**
- `--fresh`: clear saved state, start from scratch
- `--start-from <phase_id>`: resume from a specific phase
- `--dry-run`: print plan without executing
- `--device cuda:N`: override GPU device
- Crash recovery: re-attaches to running training PID on restart
- Auto-retry: up to 2 retries on phase failure

### 2. PhaseManager (`phase_manager.py`)

Parses YAML training plans into `PhaseConfig` dataclasses with **three-layer config merging**:

```
defaults (plan-level) → phase override → result
```

Example:
```yaml
defaults:
  training:
    batch_size: 128
    lr: 1e-4

phases:
  - id: train_act
    training:
      batch_size: 256     # overrides default 128
      # lr inherits 1e-4
```

**PhaseConfig fields:**

| Field | Description |
|-------|-------------|
| `id` | Unique phase identifier |
| `name` | Human-readable name |
| `phase_type` | `collection` / `training` / `evaluation` / `comparison` |
| `enabled` | Skip phase if false |
| `depends_on` | List of prerequisite phase IDs |
| `dataset` | Dataset config (repo_id, root, fps, etc.) |
| `policy` | Policy config (type, chunk_size, dim_model, etc.) |
| `training` | Training config (batch_size, lr, max_steps, etc.) |
| `collection` | Collection config (trajectory_type, n_episodes, etc.) |
| `eval` | Eval config (n_episodes, max_steps, success_threshold) |
| `monitor` | Monitor config (poll_interval, overfitting thresholds) |

### 3. StateStore (`state_store.py`)

Crash-recovery state persistence with **atomic JSON writes** (write-to-tmp + rename):

```python
@dataclass
class OrchestratorState:
    plan_name: str
    current_phase_id: str          # "collect", "train_act", etc.
    current_phase_status: str      # pending / running / complete / failed
    training_pid: int              # subprocess PID
    best_checkpoint_path: str      # best checkpoint found
    best_loss: float               # best loss seen
    training_progress: float       # 0.0 - 1.0
    phase_history: list[dict]      # completed phase results
```

State is saved to `orchestrator_state.json` after every poll cycle (60s default).

### 4. DataCollector (`data_collector.py`)

Scripted demonstration collection in MuJoCo:

```
scene.xml → MjModel → scripted trajectory → render → LeRobotDataset
```

**Supported trajectory types:**

| Type | Description |
|------|-------------|
| `ik_reach` | Reach random waypoints (joint-space interpolation) |
| `ik_push` | Push task: reach → push → retract |
| `scripted_pick` | Pick and place (planned) |

**Features:**
- Randomization: joint noise, target range scaling
- Automatic push to HuggingFace Hub
- Progress callback for orchestrator integration

### 5. TrainingLauncher (`training_launcher.py`)

Wraps `lerobot-train` as a subprocess using `sys.argv` manipulation (proven compatible with draccus CLI):

```python
# Generated subprocess command:
python -u -c "
  import sys
  sys.argv = ['lerobot-train', '--dataset.repo_id=...', '--policy.type=act', ...]
  from lerobot.scripts.lerobot_train import main
  main()
"
```

**Environment variable handling:**

| Variable | Purpose | When Applied |
|----------|---------|-------------|
| `PYTHONUNBUFFERED=1` | Real-time log output | Always |
| `LD_PRELOAD` | torchcodec libstdc++ fix (CXXABI_1.3.15) | Auto-detected from `$CONDA_PREFIX/lib/libstdc++.so.6` |
| `CUDA_VISIBLE_DEVICES` | LeRobot v0.5.1 `is_amp_available` bug workaround | When `device` is `cuda:N` |

**Device bug workaround:** LeRobot v0.5.1's `is_amp_available("cuda:N")` crashes with `ValueError` — only accepts `"cuda"`, `"cpu"`, `"xpu"`, `"mps"`. The launcher automatically sets `CUDA_VISIBLE_DEVICES=N` and overrides `--policy.device=cuda` when a specific GPU index is requested.

**Features:**
- `video_backend="torchcodec"` by default (8.6x faster than pyav for video decoding)
- Graceful stop: SIGTERM → wait 30s → SIGKILL
- PID tracking for crash recovery
- Default `num_workers=8` (override via training config)

### 6. LossMonitor (`loss_monitor.py`)

Parses training log files via regex and detects:

| Status | Condition |
|--------|-----------|
| `TRAINING` | Loss decreasing or stable, steps < min_steps |
| `CONVERGED` | Loss plateau: <1% change over 10 polls |
| `OVERFITTING` | Loss increased >20% from best |
| `NO_DATA` | No loss values parsed yet |

Loss is extracted via regex patterns matching lerobot-train log format:
```
loss: 0.1234
train_loss=0.1234
'loss': 0.1234
```

### 7. EvalRunner (`eval_runner.py`)

Loads trained policy checkpoints and runs MuJoCo rollouts:

```
checkpoint → ACTPolicy.from_pretrained() → MuJoCo rollout × N episodes → metrics
```

**Metrics computed:**
- `success_rate`: fraction of episodes with final distance < threshold
- `avg_distance`: average final joint distance
- `avg_steps`: average episode length

**Features:**
- Auto-detects policy type (ACT or Diffusion)
- Saves evaluation videos (mp4)
- `evaluate_multiple()`: compare checkpoints, sort by metric

## Training Plan YAML Format

```yaml
plan_name: my_pipeline
device: cuda:7

defaults:              # Shared across all phases
  dataset: { ... }
  training: { ... }
  monitor: { ... }

phases:
  - id: collect
    type: collection
    collection:
      trajectory_type: ik_push
      n_episodes: 300

  - id: train_act
    type: training
    depends_on: [collect]
    policy:
      type: act
      chunk_size: 100
    training:
      max_steps: 50000

  - id: eval_act
    type: evaluation
    depends_on: [train_act]
    eval:
      n_episodes: 20

  - id: train_diffusion    # Optional comparison
    type: comparison
    enabled: false
    depends_on: [eval_act]
```

## Usage

```bash
# Full pipeline (collection → training → evaluation)
python -m orchestrator_arm101.arm101_orchestrator \
    --plan training_plans/so101_push_plan.yaml \
    --device cuda:7 --fresh

# Dry run (print plan, no execution)
python -m orchestrator_arm101.arm101_orchestrator \
    --plan training_plans/so101_push_plan.yaml --dry-run

# Resume from specific phase (skip collection)
python -m orchestrator_arm101.arm101_orchestrator \
    --plan training_plans/rtx_train_plan.yaml \
    --start-from train_act --device cuda:6 --fresh

# RTX remote training only
python -m orchestrator_arm101.arm101_orchestrator \
    --plan training_plans/rtx_train_plan.yaml \
    --device cuda:6
```
