# LeRobot Training Versions

Training experiments for LeRobot policies on RTX 6000D.

- **Server**: RTX 6000D (8x 85.7GB)
- **Framework**: lerobot 0.5.1 + draccus CLI

---

## Best Practice Pipeline

**所有训练必须使用以下配置模板：**

```bash
# 1. 必须设置 CUDA_VISIBLE_DEVICES（--policy.device 无法正确隔离）
# 2. 必须使用 torchcodec（比 pyav 快 58-183x）
# 3. 必须设置 LD_PRELOAD（解决 libstdc++ 兼容问题）
# 4. 必须加 --log_freq=100（让 orchestrator loss monitor 能解析 loss）

CUDA_VISIBLE_DEVICES=<GPU_ID> \
LD_PRELOAD=~/miniconda3/envs/lerobot/lib/libstdc++.so.6 \
HF_ENDPOINT=https://hf-mirror.com \
python -u -c "
import sys
sys.argv = ['lerobot-train',
    '--dataset.repo_id=<DATASET>',
    '--dataset.video_backend=torchcodec',     # 必须
    '--policy.type=<act|diffusion>',
    '--policy.device=cuda',                   # 用 cuda 即可，由 CUDA_VISIBLE_DEVICES 控制
    '--policy.use_amp=True',                  # policy-level draccus override
    '--batch_size=<N>',
    '--steps=<N>',
    '--save_freq=<N>',
    '--log_freq=100',                         # 必须，让 loss monitor 可用
    '--num_workers=8',
]
from lerobot.scripts.lerobot_train import main
main()
"
```

**关键规则：**

| 规则 | 错误做法 | 正确做法 |
|------|---------|---------|
| GPU 隔离 | `--policy.device=cuda:6` | `CUDA_VISIBLE_DEVICES=6 --policy.device=cuda` |
| 视频解码 | `--dataset.video_backend=pyav` | `--dataset.video_backend=torchcodec` |
| AMP 开关 | `--use_amp=true` | `--policy.use_amp=True` |
| Loss 可见 | 不设 log_freq | `--log_freq=100` |

---

## Version Summary

| Version | GPU | backend | batch_size | num_workers | AMP | steps | Speed | ETA | Status |
|---------|-----|---------|-----------|-------------|-----|-------|-------|-----|--------|
| v2 | cuda:6 | pyav | 128 | 4 | No | 50,000 | 4.5s/step | 63h | Replaced |
| v3 | cuda:6 | pyav | 256 | 12 | No | 25,000 | 3.3s/step | 23h | Crashed (GPU contention) |
| v5 | cuda:6 | pyav | 128 | 8 | Yes | 50,000 | 2.25s/step | 31h | Killed (superseded by v6) |
| v6 | cuda:4 | pyav | 256 | 12 | Yes | 25,000 | 3.3s/step | ~23h | Killed (superseded by v7) |
| **v7** | **cuda:6** | **torchcodec** | **128** | **8** | **Yes** | **50,000** | **~0.53s/step** | **~7h** | **Done, eval 0%** |

**Current: v7** | **Best config: v7 (torchcodec, 6.3x faster than v6)** | **Eval: 0% success — 训练数据无物体，需重训 v8**

> **v7 eval 结果**：50K steps 训练完成，best checkpoint @ 40K (loss=0.1020)。但 eval rollout 成功率 0%。
> **根因**：训练数据 `PhangHongHao/so101_push_sim` 采集时场景里没有方块，模型只学到了关节运动模式，不是物体交互。
> **正确方案**：见 [`push_task_correct_approach.md`](../guides/push_task_correct_approach.md)。

---

## Version Details

### V2 — Baseline (Original)

- **Launched by**: Orchestrator (arm101_orchestrator)
- **GPU**: cuda:6 (`--policy.device=cuda:6`, no CUDA_VISIBLE_DEVICES)
- **Log**: `outputs/logs/train_act_v2.log`
- **Config**:
  ```yaml
  batch_size: 128
  num_workers: 4
  steps: 50000
  save_freq: 5000
  use_amp: false
  log_freq: default (no loss output)
  ```
- **Speed**: 4.5s/step
- **ETA**: 63 hours
- **Loss**: Not visible (no --log_freq)
- **Issue**: `--policy.device=cuda:6` did not properly isolate to GPU 6; data loaded on GPU 0; orchestrator loss monitor showed `inf` because no loss was printed
- **Result**: Replaced by v3

### V3 — Optimized (No AMP)

- **Launched by**: Manual SSH
- **GPU**: GPU 6 (`CUDA_VISIBLE_DEVICES=6`)
- **Log**: `outputs/logs/train_act_v3.log`
- **Config**:
  ```yaml
  batch_size: 256
  num_workers: 12
  steps: 25000
  save_freq: 5000
  use_amp: false
  log_freq: 100
  ```
- **Speed**: 3.3s/step (2.7x faster than v2)
- **ETA**: 23 hours
- **Loss@100**: 6.940, Loss@200: 2.046
- **Breakdown**: data_s=2.3s (70%) + updt_s=1.0s (30%)
- **Pattern**: Periodic stall every 12 steps (~25s), caused by video decoding workers exhausting prefetched data
- **Issue**: Crashed with `BrokenPipeError` due to GPU 6 contention with v5
- **Result**: Crashed, replaced by v6

### V5 — AMP Test (External Launch)

- **Launched by**: External session (not this Claude session)
- **GPU**: GPU 6 (`CUDA_VISIBLE_DEVICES=6`)
- **Log**: `outputs/logs/train_act_v5.log`
- **Config**:
  ```yaml
  batch_size: 128
  num_workers: 8
  steps: 50000
  save_freq: 5000
  use_amp: true  # --policy.use_amp=True (policy-level config)
  ```
- **Speed**: 2.25s/step (batch=128, so per-sample cost lower but total steps double)
- **ETA**: 31 hours (50000 steps)
- **Key finding**: `--policy.use_amp=True` works as a policy-level draccus override (not `--use_amp=true` as a top-level arg)
- **Result**: Killed to free GPU 6, superseded by v6

### V6 — Optimized with AMP (pyav)

- **Launched by**: Manual SSH (this session)
- **GPU**: GPU 4 (`CUDA_VISIBLE_DEVICES=4`, dedicated, no contention)
- **PID**: 56307
- **Log**: `outputs/logs/train_act_v6.log`
- **Config**:
  ```yaml
  batch_size: 256
  num_workers: 12
  steps: 25000
  save_freq: 5000
  eval_freq: 5000
  use_amp: true   # --policy.use_amp=True
  log_freq: 100
  video_backend: pyav
  ```
- **Speed**: 3.3s/step
- **ETA**: ~23 hours
- **Loss@100**: 6.940 (same as v3, expected — identical data/model)
- **Breakdown**: data_s=2.16s (65%) + updt_s=1.16s (35%)
- **GPU memory**: 83.4 / 85.7 GB (97.4% utilized)
- **Total data seen**: 25,000 x 256 = 6.4M samples (same as v2's 50,000 x 128)
- **First checkpoint**: step 5,000 (~4.6 hours from start)
- **Result**: Killed, superseded by v7 (torchcodec eliminates data bottleneck)

### V7 — torchcodec Breakthrough (Current)

- **Launched by**: External session
- **GPU**: GPU 6 (`CUDA_VISIBLE_DEVICES=6`)
- **PID**: 2064777
- **Log**: `outputs/logs/train_act_v6.log` (reused log path)
- **Config**:
  ```yaml
  batch_size: 128
  num_workers: 8
  steps: 50000
  save_freq: 5000
  eval_freq: 5000
  use_amp: true   # --policy.use_amp=True
  video_backend: torchcodec   # ← KEY CHANGE
  ```
- **Speed**: ~0.53s/step (6.3x faster than v6)
- **ETA**: ~7 hours
- **Loss@200**: 4.646, Loss@400: 1.665
- **Breakdown**: data_s=0.037s (6.5%) + updt_s=0.527s (93.5%)
- **Total data seen**: 50,000 x 128 = 6.4M samples (same total)
- **Key finding**: `torchcodec` eliminates the video decoding bottleneck entirely. Data loading dropped from 2.16s to 0.037s (58x speedup). GPU compute is now the dominant cost.
- **Note**: Mild periodic stall every ~8 steps (= num_workers), but recovers in ~1s vs pyav's ~25s stall

---

## Key Findings

### 1. `torchcodec` Eliminates Video Decoding Bottleneck (GAME CHANGER)

| Component | V6 (pyav) | V7 (torchcodec) | Speedup |
|-----------|-----------|-----------------|---------|
| Data loading (video decode) | 2.16s (65%) | **0.037s (6.5%)** | **58x** |
| GPU forward + backward | 1.16s (35%) | 0.527s (93.5%) | 2.2x* |
| Total per step | 3.3s | **0.53s** | **6.3x** |

*GPU update time also reduced due to smaller batch (128 vs 256).

Previous bottleneck was entirely CPU-bound pyav video decoding. torchcodec uses GPU-accelerated decoding and completely removes this bottleneck. GPU compute is now the dominant cost.

### 2. AMP Has No Effect on RTX 6000D

| Metric | V3 (no AMP) | V6 (with AMP) |
|--------|-------------|---------------|
| updt_s | 1.105s | 1.155s |
| data_s | 2.312s | 2.157s |
| Total | 3.32s/step | 3.31s/step |

RTX 6000D Ada's FP32 tensor cores are already extremely fast. AMP helps on consumer GPUs where FP16 is significantly faster.

### 3. `CUDA_VISIBLE_DEVICES` is Required

`--policy.device=cuda:6` does NOT isolate to GPU 6. lerobot still allocates data tensors on GPU 0. Must use `CUDA_VISIBLE_DEVICES=6` (or 4) for proper isolation.

### 4. `--policy.use_amp=True` is the Correct Syntax

Top-level `--use_amp=true` is rejected by lerobot-train. The correct way is as a policy-level draccus override: `--policy.use_amp=True`.

### 5. pyav Periodic Stalls vs torchcodec Mild Stalls

| Backend | Stall Frequency | Stall Duration | Cause |
|---------|----------------|----------------|-------|
| pyav | Every 12 steps (= num_workers) | ~25s | CPU decode workers exhaust prefetched batches |
| torchcodec | Every ~8 steps (= num_workers) | ~1s | Brief buffer refill, recovers quickly |

---

## Future Optimization Opportunities

| Approach | Expected Impact | Difficulty | Status |
|----------|----------------|------------|--------|
| ~~Use `torchcodec` instead of `pyav`~~ | ~~1.5-2x data loading speedup~~ | ~~Easy~~ | **Done (v7) — actual: 58x speedup** |
| Pre-extract video frames to images | Marginal (torchcodec already fast) | Medium | Low priority |
| Move dataset to RAM disk (`/dev/shm`) | Marginal | Easy | Low priority |
| Increase batch_size (GPU now dominant) | 1.5-2x throughput | Easy | Worth trying |
| Multi-GPU data-parallel training | Linear scaling | Medium | If single GPU saturates |

Current ETA (~7h) is very good. GPU compute is now the bottleneck (93.5%), so further gains require GPU-level optimization.

---

## PushT Diffusion Policy Training

- **Dataset**: lerobot/pusht (206 episodes, 26K frames)
- **Policy**: Diffusion (263M params)
- **Server**: RTX 6000D (8x 85.7GB)

### Version Summary

| Version | GPU | backend | batch_size | num_workers | steps | Speed | ETA | Status |
|---------|-----|---------|-----------|-------------|-------|-------|-----|--------|
| v1 | cuda:7 | pyav | 64 | 4 | 100,000 | ~0.6 step/s | ~36h | Killed (21%, ~10h wasted) |
| **v2** | **cuda:7** | **torchcodec** | **64** | **4** | **100,000** | **~13 step/s** | **~2h** | **Running** |

**Current: v2** | **torchcodec 21x speedup (36h → 2h)**

### V1 — pyav Baseline

- **Launched by**: External session
- **GPU**: cuda:7 (`--policy.device=cuda:7`, no CUDA_VISIBLE_DEVICES)
- **PID**: 2957652
- **Log**: `/tmp/pusht_train.log`
- **Config**:
  ```yaml
  batch_size: 64
  num_workers: 4
  steps: 100000
  save_freq: 20000
  use_amp: false
  video_backend: pyav
  ```
- **Speed**: ~0.6 step/s (heavy periodic stalls: fast ~1.2s, stall ~4-6s, every 4 steps)
- **ETA**: ~36 hours
- **Loss@21K**: 0.018
- **Breakdown**: data_s=1.098s (94.6%) + updt_s=0.063s (5.4%)
- **Progress when killed**: 21,400 / 100,000 (21%, ~9h45m elapsed)
- **Issue**: pyav video decode is 94.6% of step time, GPU mostly idle
- **Result**: Killed at 21% to restart with torchcodec

### V2 — torchcodec (Current)

- **Launched by**: This session
- **GPU**: cuda:7 (`CUDA_VISIBLE_DEVICES=7`, proper isolation)
- **PID**: 687179
- **Log**: `/tmp/pusht_train_torchcodec.log`
- **Config**:
  ```yaml
  batch_size: 64
  num_workers: 4
  steps: 100000
  save_freq: 20000
  video_backend: torchcodec   # ← KEY CHANGE
  log_freq: 100               # added for loss visibility
  ```
- **Speed**: ~13 step/s (21.7x faster than v1)
- **ETA**: ~2 hours
- **Loss@1K**: 0.054 → 0.053 → 0.052 (normal descent)
- **Breakdown**: data_s=0.006-0.025s (0.5-3%) + updt_s=0.069s (97-99.5%)
- **Key finding**: Diffusion policy is so lightweight (updt_s=0.063s) that pyav's 1.1s decode dominated 94.6%. torchcodec drops it to ~0.01s, making total time essentially equal to GPU compute time.

---

## torchcodec Impact Summary

| Policy | Params | pyav data_s | torchcodec data_s | Speedup (data) | Speedup (total) |
|--------|--------|-------------|-------------------|----------------|-----------------|
| ACT | 84M | 2.16s | 0.037s | 58x | 6.3x |
| Diffusion | 263M | 1.098s | ~0.01s | 110x | 21.7x |

**结论：`torchcodec` 对所有使用视频数据的 LeRobot 训练都是 must-have。数据加载加速 58-110x，总体加速 6-22x。越轻量的 policy（如 Diffusion），加速效果越显著，因为 GPU 计算占比本来就很低。**
