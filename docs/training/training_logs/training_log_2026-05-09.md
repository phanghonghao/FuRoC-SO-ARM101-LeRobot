# Training Log — 2026-05-09

---

## [20:50] Session Summary — SO-ARM101 LeRobot Pipeline (Phase 0-3)

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | Phase 0.1: Local venv verification | Python 3.12.10, torch 2.10.0+cpu, mujoco 3.8.0, lerobot 0.5.1, Dataset API OK |
| 2 | Phase 0.2: gym-pusht install | Installed gym-pusht 0.1.6 + pymunk 6.11.1 (downgraded from 7.x) |
| 3 | Phase 0.3: Remote RTX6000 verification | RTX accessible via VPN `phh@192.168.120.155`, 8x RTX 6000D (85.7GB each); Spark unreachable (timeout) |
| 4 | Phase 0.4: HuggingFace Hub login | Logged in as `PhangHongHao`, org `Thu-FuRoc` |
| 5 | Phase 1: PushT dataset + CPU training | Downloaded lerobot/pusht (206 eps, 25650 frames); trained Diffusion Policy 10 steps on CPU (~3s/step) |
| 6 | Phase 1: gym-pusht env verification | Obs space (5,) state + render (680,680,3) RGB; confirmed working after pymunk fix |
| 7 | Phase 2: render_test.py + collect_sim_data.py | Created both scripts; MuJoCo offscreen render verified (480x640x3 uint8) |
| 8 | Phase 2: Data collection v1 | First run: 800 frames, 10 episodes, saved as custom format to `outputs/so101_sim_dataset/` |
| 9 | Phase 2->3: Rewrote collect_sim_data.py | Switched to LeRobot native API (`LeRobotDataset.create/add_frame/save_episode/finalize/push_to_hub`) |
| 10 | Phase 3: LeRobot-native data collection + HF Hub upload | 10 episodes x 80 frames, AV1 video encoding, uploaded to `PhangHongHao/so101_sim` (7.8MB) |
| 11 | Phase 3: RTX6000 conda env setup | Created `lerobot` conda env with Python 3.12.13 on remote server |
| 12 | Phase 3: Background torch+lerobot install on RTX | Launched nohup pip install torch(CU128) + lerobot + HF login on RTX6000 (PID 1943191) |
| 13 | Bug fix: pymunk 7.x incompatible with gym-pusht | **Root cause**: `add_collision_handler` removed in pymunk 7.x API **Fix**: `pip install "pymunk>=6.6.0,<7.0.0"` **Files**: gym-pusht env |
| 14 | Bug fix: Windows GBK encoding | **Root cause**: `✓` Unicode char can't encode in GBK **Fix**: Replace with `[OK]` **Files**: `render_test.py`, `collect_sim_data.py` |
| 15 | Bug fix: LeRobot `consolidate()` not found | **Root cause**: LeRobot 0.5.1 uses `finalize()` not `consolidate()` **Fix**: Changed to `ds.finalize()` **Files**: `collect_sim_data.py` |
| 16 | Bug fix: lerobot-train CLI format | **Root cause**: Doc used wrong format; actual uses draccus `--field.subfield=val` with `--policy.repo_id` required **Fix**: Use `sys.argv` injection with correct arg format |
| 17 | Bug fix: Windows symlink error in lerobot checkpoint | **Root cause**: `os.symlink` requires admin/developer mode on Windows **Fix**: Non-fatal, checkpoint saved successfully without `last` symlink |
| 18 | Docs: Copied RTX6000 docs | Moved files from `D:\Desktop_Files\GPU-Train\RTX6000\docs\` to `SO-ARM101-LeRobot\docs\` |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | RTX6000 lerobot env install completion | nohup install running in background; status at `/tmp/lerobot_install_status.txt` | Check `cat /tmp/lerobot_install_status.txt` after reconnecting VPN |
| 2 | Remote ACT/Diffusion training on RTX6000 | Waiting for env install to complete | `lerobot-train --dataset.repo_id=PhangHongHao/so101_sim --policy.type=act --steps=100000 --device=cuda:0` |
| 3 | Phase 4: VLA (SmolVLA) exploration | Blocked behind Phase 3 remote training | After training completes, test `SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')` |
| 4 | `/mnt/data3/phh` directory | Permission denied (no sudo); home dir only 24G free | Request admin to create `/mnt/data3/phh` and chown, or work within 24G |
| 5 | Remote Spark server (59.66.25.192) | SSH timeout, VPN not connected | Need to connect to aTrust VPN for Spark access |

### Key Decisions

- Use LeRobot native API (`LeRobotDataset.create/push_to_hub`) instead of custom format → eliminates format conversion step entirely
- Upload to HF Hub during collection → one-step pipeline, dataset available anywhere
- Use conda env `lerobot` (Python 3.12) separate from `isaaclab` (Python 3.10) on RTX6000
- Use `nohup` for remote install to survive SSH disconnect (user closing computer)
- Home directory (24G free) sufficient for lerobot env (~5-6GB needed), `/mnt/data3/phh` not accessible

---

## [00:05] Session Summary — Phase 4 VLA Model Exploration (continued)

### Completed

| # | Item | Details |
|---|------|---------|
| 1 | Phase 4.1: SmolVLA dependency install | Installed `transformers 5.8.0`, `tokenizers 0.22.2`, `num2words 0.5.14` for SmolVLA support |
| 2 | Bug fix: groot dataclass Python 3.12 incompatibility | **Root cause**: `GR00TN15Config` in `groot_n1.py` has `field(init=False)` without defaults before fields with defaults, Python 3.12 rejects **Fix**: Added `default=None` to `backbone_cfg`, `action_head_cfg`, `action_horizon`, `action_dim` **Files**: `.venv/Lib/site-packages/lerobot/policies/groot/groot_n1.py` |
| 3 | Phase 4.1: SmolVLA base model loaded | `SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')` loaded successfully on CPU; 450M total params, 99.9M trainable (action expert + state proj), 350.2M frozen (VLM backbone) |
| 4 | Phase 4.1: SmolVLA config explored | VLM backbone: `SmolVLM2-500M-Video-Instruct`; chunk_size=50, n_action_steps=50, max_state_dim=32, max_action_dim=32, resize_imgs=(512,512), freeze_vision_encoder=True, train_expert_only=True |

### Uncompleted / Blocked

| # | Item | Blocker | Next Step |
|---|------|---------|-----------|
| 1 | Phase 4.2: SmolVLA inference test on SO-101 data | Need to construct a proper input batch from `PhangHongHao/so101_sim` dataset and run forward pass | Create test script with `policy.select_action()` using dataset frames |
| 2 | Phase 4.3: SmolVLA fine-tuning on RTX6000 | Requires GPU (model too large for CPU); remote ACT training may still be running | `lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=PhangHongHao/so101_sim --batch_size=8 --steps=20000` |
| 3 | ACT training on RTX6000 — status check | Session continued after disconnect; need to reconnect VPN + SSH to check training progress | `ssh phh@192.168.120.155 'tail -20 /tmp/so101_act_train.log'` |

### Key Decisions

- SmolVLA base model loads successfully on CPU for exploration; fine-tuning must happen on GPU (RTX6000)
- VLM backbone (350M params) is frozen by default; only action expert (100M params) is trained → efficient fine-tuning
- `max_state_dim=32` and `max_action_dim=32` handle SO-101's 6-dim state/action via automatic zero-padding
