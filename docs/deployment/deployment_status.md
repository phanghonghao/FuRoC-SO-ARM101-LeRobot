# SO-ARM101-LeRobot 部署状态记录

> 最后更新：2026-05-09

---

## 本地 vs 远程分工

```
┌──────────────────┐      HuggingFace Hub       ┌──────────────┐
│  本地电脑 (核显)   │    ← 数据集上传/下载 →      │  RTX6000 服务器│
│                  │    ← 模型上传/下载 →          │              │
│  ① 遥操作采集     │                             │  ② 训练策略    │
│  ③ 推理部署       │                             │              │
│  ④ MuJoCo 仿真   │                             │              │
│  可看屏幕 ✓      │                             │  无头模式 ✗   │
└──────────────────┘                             └──────────────┘
```

| 阶段 | GPU 需求 | 跑在哪 |
|------|---------|--------|
| 遥操作采集 | 不需要 | 本地（USB + 摄像头） |
| 训练 ACT | ~1GB VRAM | 远程 RTX6000 |
| 训练 Diffusion | ~5GB VRAM | 远程 RTX6000 |
| 推理部署 | 不需要 | 本地 CPU |
| MuJoCo 仿真 | 不需要 | 本地 CPU |

本地没有独显完全不影响，采集和推理都是 CPU 工作。训练交给远程服务器。

---

## 已完成

### 1. 仓库克隆
- 地址：`https://github.com/horndeer/SO-ARM101-LeRobot`
- 路径：`D:\Desktop_Files\FuRoC\SO-ARM101-LeRobot`
- 状态：已完成

### 2. Python 3.12 独立安装
- 版本：Python 3.12.10
- 路径：`C:\Users\20174\AppData\Local\Programs\Python\Python312\`
- 不影响全局 Python 3.14

### 3. 虚拟环境创建
- 路径：`D:\Desktop_Files\FuRoC\SO-ARM101-LeRobot\.venv\`
- Python 版本：3.12.10
- pip 镜像：清华 TUNA（`https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`）

### 4. 依赖安装
| 包 | 版本 | 说明 |
|---|------|------|
| torch | 2.10.0+cpu | CPU 版本（本地无 GPU，训练在远程） |
| torchvision | 0.25.0 | |
| lerobot | 0.5.1 | 机器人学习框架 |
| mujoco | 3.8.0 | 物理仿真 |

### 5. MuJoCo 本地仿真查看器
- SO-101 模型可在本地正常可视化
- 启动脚本：`python sim_viewer.py`（仓库根目录）
- 注意：需要在 `.venv` 环境下运行

### 6. 项目框架梳理
- 仓库结构：
  - `STL/SO101/` — Leader + Follower 3D 打印文件
  - `Simulation/SO101/` — URDF + MuJoCo MJCF 模型
  - `STEP/` — CAD 源文件
  - `Optional/` — 可选配件（摄像头支架、柔性夹爪等）
  - `media/` — 图片资源

---

## 未完成 / 待推进

### 1. 仿真 Pipeline 脚本
- 计划：用 LeRobot + MuJoCo 搭建仿真遥操作 → 训练 → 推理全流程
- 状态：未开始，环境已就绪

### 2. 远程训练环境
- 计划：在 RTX6000 服务器上安装 CUDA 版 torch + lerobot
- 数据集和模型通过 HuggingFace Hub 同步
- 状态：未开始

### 3. 实体部署指南
- 计划：硬件组装 + LeRobot 连接实体 + 遥操作采集 + 训练 + 自主执行
- 状态：未开始，硬件不齐全

---

## 环境信息

### 本地环境
| 项目 | 状态 |
|------|------|
| 系统 | Windows 11 Home China |
| Python (全局) | 3.14.2 |
| Python (venv) | 3.12.10（`.venv`） |
| torch | 2.10.0+cpu |
| MuJoCo | 3.8.0 |
| LeRobot | 0.5.1 |
| 本地 GPU | 核显（无独显） |

### 远程环境
| 项目 | 状态 |
|------|------|
| GPU | 8 × RTX PRO 6000 |
| 当前任务 | Isaac Lab 训练 |

---

## 快速激活环境

```bash
# 激活虚拟环境
D:\Desktop_Files\FuRoC\SO-ARM101-LeRobot\.venv\Scripts\activate

# 或直接用绝对路径运行
D:\Desktop_Files\FuRoC\SO-ARM101-LeRobot\.venv\Scripts\python sim_viewer.py
```
