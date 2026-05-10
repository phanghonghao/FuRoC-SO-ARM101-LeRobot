参考链接：https://docs.robotsfan.com/isaaclab/source/features/multi_gpu.html
# RTX6000 多 GPU 分布式训练部署指南

## 概述

使用 PyTorch `torchrun` 在 RTX6000 服务器（8x RTX 6000D）上部署多 GPU 并行训练。支持 2-8 卡，通过 NCCL 后端同步梯度。

**实测性能**: 4 卡 4096 envs/GPU → ~100,000 steps/s，约单卡 (~25,000 steps/s) 的 4 倍。

## 快速启动

```bash
# 一键启动（推荐，通过 gpu-train skill）
/gpu-train --start --resume --gpus 4

# 手动启动 4 卡
ssh phh@192.168.120.155
cd ~/magiclab_rl_lab && conda activate isaaclab
torchrun --nproc_per_node=4 scripts/rsl_rl/train_multigpu.py \
  --task=Magiclab-Z1-12dof-Velocity \
  --run_name=v7_mgpu \
  --headless --distributed \
  --num_envs=4096 --max_iterations=50000 \
  --resume --load_run=<RUN_DIR> --checkpoint=model_<N>.pt
```

## GPU 选择

| 卡数 | `--nproc_per_node` | 使用 GPU | 说明 |
|------|-------------------|----------|------|
| 1 | (不用 torchrun) | GPU 0 | 单卡模式 |
| 2 | 2 | GPU 0-1 | |
| 4 | 4 | GPU 0-3 | 推荐 |
| 7 | 7 | GPU 0-6 | 跳过 GPU 5 (他人占用) |
| 8 | 8 | GPU 0-7 | 需 GPU 5 空闲 |

**注意**: 不使用 `CUDA_VISIBLE_DEVICES`（会导致 Isaac Sim rank 0 环境创建卡死）。GPU 从 0 开始连续分配，无法跳过单张卡。

## 脚本说明

### `train_multigpu.py`（推荐）

自定义多 GPU 训练脚本，在原版 `train.py` 基础上增加：
- 调试屏障 (`[BARRIER]`) 输出，便于排查死锁
- 仅 rank 0 写配置文件和日志，其他 rank 写到 `_rankN` 子目录

位置: `~/magiclab_rl_lab/scripts/rsl_rl/train_multigpu.py`

### `train.py`（原版）

原版也支持 `--distributed` 标志，用法相同：

```bash
torchrun --nproc_per_node=4 scripts/rsl_rl/train.py \
  --task=Magiclab-Z1-12dof-Velocity --distributed ...
```

## 完整启动命令模板

```bash
# 从 checkpoint 恢复，4 卡训练
cd ~/magiclab_rl_lab && \
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab && \
nohup torchrun --nproc_per_node=4 --master_port=29500 \
  scripts/rsl_rl/train_multigpu.py \
  --task=Magiclab-Z1-12dof-Velocity \
  --run_name=<RUN_NAME> \
  --headless --distributed \
  --num_envs=4096 \
  --max_iterations=<MAX_ITER> \
  --resume \
  --load_run=<SOURCE_RUN_DIR> \
  --checkpoint=model_<N>.pt \
  > /tmp/z1_mgpu_<NAME>.log 2>&1 & echo PID=$!
```

**多个训练并行时**，使用不同 `--master_port`（如 29500, 29501, 29502）避免端口冲突。

## 参数说明

| 参数 | 说明 | 默认 |
|------|------|------|
| `--nproc_per_node` | GPU 数量 (2-8) | — |
| `--master_port` | torchrun 通信端口 | 29500 |
| `--distributed` | 启用分布式模式 | — |
| `--num_envs` | **每张卡**的环境数 | 4096 |
| `--max_iterations` | 总迭代次数 | 50000 |
| `--resume` | 从 checkpoint 恢复 | — |
| `--load_run` | 源运行目录名 | — |
| `--checkpoint` | 源模型文件名 | — |

4 卡 4096 envs → 总共 16,384 并行环境。

## 监控

```bash
# 查看训练日志
tail -f /tmp/z1_mgpu_<NAME>.log

# 通过 gpu-train skill
/gpu-train --tail
/gpu-train --status
/gpu-train --mycuda
```

## 已知限制与注意事项

| 事项 | 说明 |
|------|------|
| **禁用 CUDA_VISIBLE_DEVICES** | Omniverse Kit 的 `carb.cudainterop` 插件检测到 CUDA_VISIBLE_DEVICES 后直接标记 GPU 为 "bad state"，导致所有 GPU 被跳过。**不可绕过**，这是 Omniverse 层面硬限制（2026-05-04 实测确认）。 |
| **GPU 连续分配** | torchrun 从 GPU 0 开始，无法跳过中间某张卡。如果 GPU 5 被占，只能用 0-4 (5卡) 或 0-6 (7卡) |
| **EGL 前置条件** | 确保 `~/miniconda3/envs/isaaclab/share/glvnd/egl_vendor.d/10_nvidia.json` 存在 |
| **端口冲突** | 多个 torchrun 实例需不同 `--master_port` |
| **ETA 参考值** | 4 卡 4096 envs: ~3.7s/iter, 50000 iter ≈ 51h (会自动 resume，无需一次跑完) |

## 故障排查

| 症状 | 原因 | 修复 |
|------|------|------|
| 进程卡在 "Creating environment" | `CUDA_VISIBLE_DEVICES` 已设置 | 去掉 CUDA_VISIBLE_DEVICES |
| `Default process group has not been initialized` | 在 runner 外调用 `torch.distributed.barrier()` | 只在 runner 创建后调用 barrier |
| `Synchronizing parameters for rank N...` 永久等待 | 某个 rank 未到达同步点 | 等更久（Isaac Sim 初始化慢），或检查 GPU 是否够用 |
| `init_process_group` 被调用两次 | 自定义脚本提前调用了 init_process_group | 让 runner 自行调用，不要提前 init |
