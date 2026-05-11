# FuRoC-SO-ARM101-LeRobot

> Based on [SO-ARM101-LeRobot](https://github.com/horndeer/SO-ARM101-LeRobot) by TheRobotStudio & Hugging Face.
> Licensed under the [Apache License 2.0](LICENSE).

No-hardware simulation learning pipeline for SO-101 robot arm: MuJoCo simulation + LeRobot + remote GPU training.

[Orchestrator Architecture](docs/github_readme/orchestrator_architecture.md) | [Pipeline Guide](docs/no_hardware_deployment.md) | [Training Logs](docs/training_logs/)

---

## Pipeline

| Phase | Status | Description |
|:-----:|:------:|:------------|
| 0 | Done | Environment setup (local venv + RTX 6000D + HF Hub) |
| 1 | Done | PushT simulation training (CPU, validation) |
| 2 | Done | SO-101 MuJoCo data collection (300 eps, 30K frames) |
| 3 | **Active** | **ACT training on RTX 6000D (84M params, 50K steps, torchcodec + AMP)** |
| 4 | **Active** | **PushT Diffusion on RTX 6000D (263M params, 100K steps, torchcodec)** |

## Results

### ACT Policy Evaluation (SO-101 MuJoCo Simulation)

<p align="center">
  <img src="docs/so101_references/videos/eval_10k_100steps.gif" width="45%" alt="ACT eval 10K checkpoint, 100 steps" />
  <img src="docs/so101_references/videos/eval_test_10steps.gif" width="45%" alt="ACT eval quick test, 10 steps" />
</p>

<p align="center"><em>Left: ACT checkpoint (10K steps) evaluated for 100 steps &nbsp;|&nbsp; Right: Quick 10-step eval</em></p>

### PushT Diffusion Policy

<p align="center">
  <img src="docs/so101_references/videos/pusht_eval.gif" width="45%" alt="PushT diffusion policy evaluation" />
</p>

### Community ACT Reference Demos

<p align="center">
  <img src="docs/so101_references/videos/act_so101_pick_pen.gif" width="45%" alt="ACT SO-101 pick pen" />
  <img src="docs/so101_references/videos/act_so101_pick_rag.gif" width="45%" alt="ACT SO-101 pick rag" />
</p>

<p align="center"><em>Community ACT models for SO-101 (pick pen / pick rag tasks)</em></p>

## Architecture

```
Hardware Layer          Simulation Layer          Training Layer          Documentation
─────────────          ─────────────────          ──────────────          ─────────────
STL/ STEP/             sim_viewer.py              Remote RTX 6000D:      docs/
Optional/              render_test.py             ACT (84M) ★            no_hardware_deployment.md
Simulation/SO101/      collect_sim_data.py ★      Diffusion (263M)       GPU_Train_Command_Reference.md
URDF + MJCF            convert_to_lerobot_dataset  batch=128, workers=8  RTX_Server_Guide.md
                       ─────────────────────       torchcodec + AMP       so101_references/
                       MuJoCo offscreen render     ~1.9 step/s (ACT)      training_logs/
                       → LeRobot native format     ~13 step/s (Diffusion)
```

## Data Flow

```
MuJoCo scene.xml → collect_sim_data.py → LeRobotDataset → HF Hub → RTX 6000D → ACT Policy → Eval
                   (offscreen render)    (images+meta)    (sync)   (lerobot-train)  (MuJoCo)
```

## Tech Stack

| Component | Version | Role |
|-----------|---------|------|
| MuJoCo | 3.8.0 | Physics simulation + offscreen rendering |
| LeRobot | 0.5.1 | Dataset management + training framework |
| PyTorch | 2.x | Model training (CPU local / CUDA remote) |
| ACT | 84M params | SO-101 推物 policy (chunk_size=100) |
| Diffusion | 263M params | PushT 验证 policy |
| torchcodec | latest | 视频解码 (比 pyav 快 8-20x) |
| RTX 6000D | 8x 85GB | Remote GPU training server |

## Policy Comparison

| Policy | Params | Speed (RTX 6000D) | SO-101 HF Models | Status |
|--------|--------|-------------------|-------------------|--------|
| **ACT** | 84M | 1.9 step/s (torchcodec+AMP) | 30+ | Training (50K steps) |
| **Diffusion** | 263M | 13 step/s (torchcodec) | — | Training (100K steps) |

> Speed benchmark & optimization details: [docs/so101_references/README.md](docs/so101_references/README.md)

## Quick Start

```bash
# 1. Setup
python -m venv .venv && .venv/Scripts/activate
pip install lerobot mujoco

# 2. Collect simulation data
python collect_sim_data.py

# 3. Train on remote GPU (with optimal config)
# See docs/so101_references/README.md for full benchmark & bug workarounds
export CUDA_VISIBLE_DEVICES=6
export LD_PRELOAD=~/miniconda3/envs/lerobot/lib/libstdc++.so.6
export HF_ENDPOINT=https://hf-mirror.com
```

Full walkthrough: [docs/no_hardware_deployment.md](docs/no_hardware_deployment.md)

## Automated Pipeline (Orchestrator)

YAML-driven automated training pipeline with crash recovery and overfitting detection.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Collection   │ ──► │   Training   │ ──► │  Evaluation  │ ──► │  Comparison  │
│ DataCollector │     │ TrainLauncher│     │  EvalRunner  │     │ (optional)   │
│ + LossMonitor │     │ + LossMonitor│     │  + metrics   │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       ▲                    ▲                    ▲
       │                    │                    │
  ┌────┴────────────────────┴────────────────────┴────┐
  │            Arm101Orchestrator (main loop)          │
  │  phase_manager ─ state_store ─ crash recovery      │
  └───────────────────────────────────────────────────┘
```

```bash
# Full pipeline (collection → training → evaluation)
python -m orchestrator_arm101.arm101_orchestrator \
    --plan training_plans/so101_push_plan.yaml --device cuda:7 --fresh

# Dry run (preview plan)
python -m orchestrator_arm101.arm101_orchestrator \
    --plan training_plans/so101_push_plan.yaml --dry-run

# Resume from specific phase
python -m orchestrator_arm101.arm101_orchestrator \
    --plan training_plans/rtx_train_plan.yaml --start-from train_act --device cuda:6
```

**Features:** Three-layer config merging, overfitting detection (loss plateau + increase), auto-retry (2x), atomic state persistence, PID-based crash recovery.

Full architecture docs: [docs/github_readme/orchestrator_architecture.md](docs/github_readme/orchestrator_architecture.md)

## Project Structure

```
FuRoC-SO-ARM101-LeRobot/
├── collect_sim_data.py              # Core data collection (MuJoCo → LeRobot)
├── render_test.py                   # Offscreen render verification
├── convert_to_lerobot_dataset.py    # Format conversion
├── sim_viewer.py                    # MuJoCo interactive viewer
├── eval_pusht.py                    # PushT policy evaluation
├── eval_rollout.py                  # ACT rollout evaluation
├── run_collect.py                   # Data collection launcher
├── run_pipeline_rtx.sh              # RTX training pipeline script
├── orchestrator_arm101/             # Automated training orchestrator
├── training_plans/                  # Training plan YAML configs
├── Simulation/SO101/                # MuJoCo scene + URDF/MJCF
├── docs/
│   ├── github_readme/               # GitHub README assets
│   │   └── orchestrator_architecture.md  # Orchestrator full docs
│   ├── no_hardware_deployment.md    # 5-phase pipeline guide
│   ├── architecture_overview.html   # Full architecture document
│   ├── so101_references/            # HF model survey + demos
│   └── training_logs/               # Session-by-session logs
└── README.md
```
