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
