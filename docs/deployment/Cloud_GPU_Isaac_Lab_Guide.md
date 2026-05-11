# 云GPU运行Isaac Lab仿真部署指南

> 使用场景：RTX 6000D本地服务器已运行Isaac Sim（MagicBotZ1 sim2sim），需租用云GPU独立运行Isaac Lab仿真，加载本地训练好的LeRobot checkpoint，**实时观看仿真画面**。

## 1. 架构概览

```
┌─────────────────────────────────┐     ┌─────────────────────────────────┐
│   RTX 6000D 本地服务器           │     │   Paratera 云GPU (Windows)       │
│                                 │     │                                 │
│  Isaac Sim (MagicBotZ1 sim2sim) │     │  Isaac Sim + Isaac Lab          │
│  LeRobot ACT 训练 (cuda:6)      │     │  加载 LeRobot ACT checkpoint     │
│  MuJoCo 仿真 (SO-ARM101)        │     │  RDP 远程桌面实时看到仿真画面     │
│                                 │     │                                 │
│  用途：训练 + MagicBotZ1验证     │     │  用途：Isaac Lab 高保真仿真展示   │
└─────────────────────────────────┘     └─────────────────────────────────┘
         │                                        ▲
         │      SCP/RDP 拷贝 checkpoint 文件       │
         └────────────────────────────────────────┘
```

**核心依赖关系**：Isaac Lab **不能独立运行**，必须依赖 Isaac Sim 作为基础运行环境。

---

## 2. 推荐方案：Paratera Windows 实例 + RDP

### 2.1 为什么选 Windows + RDP

Isaac Sim 需要图形界面进行 GPU 渲染。Paratera 的 **Windows 实例自带 RDP 远程桌面**（端口3389），连上就是一个完整的 Windows 桌面环境，GPU 渲染画面通过 RDP 实时传回本地，**无需额外配置任何显示服务**。

对比其他方案：

| 方案 | 能否实时看 | 配置难度 | 说明 |
|------|-----------|---------|------|
| **Windows + RDP** (推荐) | 能 | 低 | Paratera原生支持，mstsc直连 |
| Ubuntu + WebSSH | 不能 | — | 只有命令行，看不到画面 |
| Ubuntu + VNC | 能 | 中 | 需自装 xvfb + x11vnc，较卡 |
| Ubuntu + NICE DCV | 能 | 中 | NVIDIA官方方案，流畅但安装复杂 |
| Headless + 录制回传 | 不能 | 低 | 只能事后看视频 |

### 2.2 硬件配置建议

| 项目 | 推荐配置 |
|------|---------|
| GPU | RTX 4090 (24GB) |
| 显存 | >= 24GB |
| 内存 | >= 64GB |
| 存储 | >= 200GB SSD（Isaac Sim安装约100GB+） |
| 系统 | **Windows Server** |
| 带宽 | 按固定带宽 5Mbps（足够RDP使用） |

> Isaac Sim 2023.1+ 硬性要求：NVIDIA RTX GPU + >= 24GB VRAM + CUDA 11.8+

---

## 3. 完整部署步骤

### 3.1 在 Paratera 创建云服务器

1. 登录 [并行智算云](https://ai.paratera.com)
2. 创建云服务器，选择：
   - GPU：**RTX 4090**
   - 系统：**Windows Server**
   - 存储：SSD 云盘 >= 200GB
   - 带宽：按固定带宽 5Mbps（0.335元/h）
3. 开机后，点击【登录信息】获取 IP、用户名、密码

### 3.2 本地连接远程桌面

```
Win + R → 输入 mstsc → 回车
输入云服务器 IP 地址（端口默认3389）
输入用户名、密码 → 连接
```

连上后就是一个完整的 Windows 桌面，和操作本地电脑一样。

### 3.3 云服务器环境搭建

在 RDP 远程桌面内打开 PowerShell：

```powershell
# 1. 确认GPU驱动
nvidia-smi

# 2. 安装 Miniconda（如果没有）
# 从 https://docs.conda.io/en/latest/miniconda.html 下载安装

# 3. 创建环境
conda create -n isaac python=3.10 -y
conda activate isaac

# 4. 安装 Isaac Sim（Isaac Lab的必需依赖）
pip install isaacsim[all]

# 5. 安装 Isaac Lab
pip install isaac-lab

# 6. 安装 LeRobot（版本需与训练时一致）
pip install lerobot==0.5.1

# 7. 验证安装
python -c "import isaacsim; print('Isaac Sim OK')"
python -c "import isaac.lab; print('Isaac Lab OK')"
python -c "import lerobot; print('LeRobot OK')"
```

> 注意：Isaac Sim 安装包较大（~100GB），下载和安装需要较长时间，建议提前准备。

### 3.4 传输 Checkpoint

**方式1：通过 RDP 远程桌面拷贝**
- RDP 连接时勾选"本地资源 → 剪贴板"，可直接复制文件
- 或在 RDP 设置中映射本地磁盘到远程

**方式2：从 RTX 6000D 服务器 SCP 到云服务器**
```bash
# 在RTX 6000D服务器上执行
scp -r outputs/so101_act_checkpoints/ user@cloud-server-ip:C:/checkpoints/
```

> ACT checkpoint 约 300MB，传输速度快。

### 3.5 运行仿真并实时观看

在 RDP 远程桌面中：

```python
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy

# 加载预训练的ACT policy
policy = ACTPolicy.from_pretrained("C:/checkpoints/010000")
policy.to("cuda")
policy.eval()

# 启动Isaac Lab仿真环境
# 仿真画面会在远程桌面窗口中实时渲染显示
# （具体仿真环境配置取决于机器人型号和任务）
```

---

## 4. 成本估算

| 资源 | 配置 | 按量计费（估算） |
|------|------|----------------|
| GPU | RTX 4090 × 1 | ~X 元/h |
| 存储 | SSD 200GB | ~0.06 元/h |
| 带宽 | 固定 5Mbps | 0.335 元/h |
| **合计** | | ~X 元/h |

> 具体GPU价格以 Paratera 平台为准。不使用时选择**关机（节省模式）**可停止GPU计费，仅保留云盘费用。

### 关机模式选择

| 模式 | 计费 | 说明 |
|------|------|------|
| 节省模式 | 仅云盘计费 | 释放GPU和网络，下次开机重配资源+IP |
| 标准模式 | 云盘+网络计费 | 释放GPU，保留网络IP |
| 计费模式 | 全部计费 | 不释放任何资源，下次开机不排队 |

---

## 5. 注意事项

1. **Isaac Lab 依赖 Isaac Sim**：不能单独安装 Isaac Lab，必须先安装 Isaac Sim
2. **Checkpoint 兼容性**：确保云服务器上的 `lerobot` 版本与训练时一致，否则加载失败
3. **SO-ARM101 vs MagicBotZ1**：
   - SO-ARM101 当前使用 MuJoCo 仿真，在 Isaac Lab 中需要重新配置 URDF/MJCF
   - MagicBotZ1 的 sim2sim 流程参见 `docs/Sim2Sim_Guide.md`
4. **成本控制**：仿真展示完毕及时关机（节省模式），避免空闲产生费用
5. **数据安全**：checkpoint 用完后建议清理，关机不会删除数据但删除实例会
6. **RDP 带宽**：5Mbps 足够 RDP 日常使用，如果画面卡顿可临时提升带宽
