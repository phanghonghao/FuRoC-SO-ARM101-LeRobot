# Training Log — 2026-05-11

---

## [03:16] Session Summary — ARM101 Pipeline Orchestrator Implementation + Deployment

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | Orchestrator module: `state_store.py` | Ported from Z1 state_store.py; `OrchestratorState` dataclass with atomic JSON persistence, `StateStore` class with load/save/clear |
| 2 | Orchestrator module: `phase_manager.py` | YAML plan parser with 3-layer merge (defaults→phase); `PhaseConfig` dataclass; phase types: collection/training/evaluation/comparison; `depends_on` ordering, `enabled` flag |
| 3 | Orchestrator module: `training_launcher.py` | Wraps lerobot-train as subprocess using sys.argv manipulation (proven pattern from PushT); maps plan config to draccus CLI args (`--policy.xxx`, `--dataset.xxx`, top-level `--batch_size`, `--steps`) |
| 4 | Orchestrator module: `loss_monitor.py` | Polls training log file, parses loss/step via regex; detects overfitting (loss > best×1.2) and plateau (<1% change over N polls); returns status: TRAINING/CONVERGED/OVERFITTING/COMPLETE |
| 5 | Orchestrator module: `data_collector.py` | Wraps MuJoCo scripted demo collection; supports ik_reach/ik_push trajectories; outputs LeRobot native dataset format |
| 6 | Orchestrator module: `eval_runner.py` | Loads policy via `ACTPolicy.from_pretrained()`, runs N episodes in MuJoCo, computes success_rate, saves video via imageio |
| 7 | Orchestrator module: `arm101_orchestrator.py` | Main controller with 2-level event loop; phase-type dispatch (collection→DataCollector, training→TrainingLauncher+LossMonitor, evaluation→EvalRunner); CLI: `--plan`, `--fresh`, `--start-from`, `--dry-run`, `--device` |
| 8 | Training plans: `so101_push_plan.yaml` + `rtx_train_plan.yaml` | Full 4-phase plan (collect→train_act→eval_act→train_diffusion disabled); RTX-specific 2-phase plan for cuda:6 |
| 9 | Deployment scripts: `run_collect.py` + `run_pipeline_rtx.sh` | Standalone collection script with EGL; bash pipeline runner that waits for collection then launches orchestrator |
| 10 | Bug fix: Scene XML include error | **Root cause**: `scene.xml` includes `so101_new_calib.xml` which wasn't uploaded **Fix**: Uploaded full `Simulation/SO101/` directory (XMLs + STL assets) **Files**: RTX `~/SO-ARM101-LeRobot/Simulation/SO101/` |
| 11 | Bug fix: Nested scp directory | **Root cause**: `scp -r dir/ server:~/dir/` creates `~/dir/dir/` **Fix**: `cp` from nested to correct location, `rm -rf` nested **Files**: RTX `~/SO-ARM101-LeRobot/Simulation/SO101/` |
| 12 | Bug fix: MuJoCo OpenGL error | **Root cause**: Headless server needs EGL rendering **Fix**: Added `os.environ.setdefault("MUJOCO_GL", "egl")` to `run_collect.py` **Files**: `run_collect.py` |
| 13 | Bug fix: Windows line endings in bash | **Root cause**: Git checkout on Windows adds `\r` **Fix**: `sed -i "s/\r$//"` on remote **Files**: `run_pipeline_rtx.sh` on RTX |
| 14 | Bug fix: lerobot-train unrecognized args (3 iterations) | **Root cause**: lerobot-train uses draccus (not argparse); wrong arg names (`--device`, `--run_name`, `--lr`); missing `--policy.repo_id` **Fix**: Rewrote launcher to use sys.argv manipulation matching working PushT pattern; auto-generate `policy.repo_id` **Files**: `orchestrator_arm101/training_launcher.py` |
| 15 | Bug fix: LossMonitor plateau detection | **Root cause**: Only stored last loss per poll, not all parsed losses **Fix**: Iterate over all `new_losses` and append each to `loss_history` **Files**: `orchestrator_arm101/loss_monitor.py` |
| 16 | Data collection: 300 episodes on RTX | 300 episodes, 30,000 frames collected to `/tmp/so101_push_sim_lerobot/` via `run_collect.py` |
| 17 | ACT training launched on RTX cuda:6 | Training running at step ~23/50,000, ~40h ETA; PushT Diffusion training continues on cuda:7 |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | ACT training completion | Running autonomously on RTX, step ~23/50,000 (~40h ETA) | Monitor via `/gpu-train --tail`; training log at `~/SO-ARM101-LeRobot/outputs/logs/train_act.log` |
| 2 | eval_act phase | Blocked behind training completion | Should auto-start if orchestrator is still running |
| 3 | Push to HF Hub | Dataset collected locally, not yet pushed | `push_to_hub=True` after verification |
| 4 | Pipeline state file monitoring | Need to check `orchestrator_state.json` for progress | `ssh phh@192.168.120.155 'cat ~/SO-ARM101-LeRobot/orchestrator_state.json'` |

### Key Decisions

- Single GPU for IL training (LeRobot lerobot-train doesn't support DDP); device parameter reserved for future
- PushT training on cuda:7 kept running alongside ARM101 on cuda:6
- Pipeline designed for overnight autonomous execution: `run_pipeline_rtx.sh` waits for collection then runs train+eval
- sys.argv manipulation approach for lerobot-train (draccus-based CLI, not standard argparse)
- GPU conflict resolved: collection runs on CPU, training on cuda:6, PushT on cuda:7
