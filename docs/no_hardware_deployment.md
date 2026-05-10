# 无实体硬件的仿真学习 Pipeline

> 最后更新：2026-05-09
> 适用环境：本地 Windows（无独显）+ 远程 RTX6000 服务器

---

## 概览

```
┌─────────────────────────────────────────────────────────────────────┐
│  Phase 0          Phase 1          Phase 2          Phase 3         │
│  环境验证          PushT 仿真        SO-101 数据采集   远程 GPU 训练  │
│  ──────────       ──────────       ──────────────    ────────────── │
│  本地 venv ✓       lerobot/pusht    MuJoCo offscreen  RTX6000       │
│  远程 CUDA ✓       数据集下载        RGB 渲染 + 轨迹   ACT/Diffusion │
│  HF Hub ✓         训练 + 评估       → LeRobot 格式     模型同步      │
│                                                                    │
│  Phase 4                                                           │
│  VLA 探索                                                           │
│  ──────────                                                        │
│  SmolVLA 推理                                                       │
│  可选 LoRA 微调                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**核心思路**：本地 CPU 做仿真和数据采集，远程 GPU 做训练，HuggingFace Hub 做数据/模型同步。

---

## Phase 0: 环境验证

### 0.1 本地 venv 验证

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 验证 Python 版本（需要 3.12）
python --version
# Expected: Python 3.12.x

# 验证 torch（CPU 版本，本地不需要 GPU）
python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
# Expected: torch 2.x.x+cpu, CUDA: False

# 验证 MuJoCo
python -c "import mujoco; print(f'mujoco {mujoco.__version__}')"
# Expected: mujoco 3.x.x

# 验证 LeRobot
python -c "import lerobot; print(f'lerobot {lerobot.__version__}')"
# Expected: lerobot 0.5.1

# 验证 LeRobot 数据集 API
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('Dataset API OK')"
# Expected: Dataset API OK
```

### 0.2 安装 gym-pusht

`gym-pusht` 是 LeRobot 内置的 PushT 仿真环境，Phase 1 的训练/评估依赖它。

```bash
pip install gym-pusht
```

验证：

```bash
python -c "import gym_pusht; print('gym-pusht OK')"
# Expected: gym-pusht OK
```

### 0.3 远程 RTX6000 环境安装

在远程服务器上执行：

```bash
# 创建 venv
python3.12 -m venv .venv
source .venv/bin/activate

# 安装 CUDA 版 torch（根据服务器 CUDA 版本选择，假设 CUDA 12.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# Expected: CUDA: True, GPU: NVIDIA RTX PRO 6000 ...

# 安装 lerobot
pip install lerobot==0.5.1
pip install gym-pusht

# 验证
python -c "import lerobot; print(f'lerobot {lerobot.__version__}')"
# Expected: lerobot 0.5.1
```

### 0.4 HuggingFace Hub 登录

本地和远程都需要登录：

```bash
pip install huggingface_hub
huggingface-cli login
# 粘贴你的 HF token（从 https://huggingface.co/settings/tokens 获取）
```

验证：

```bash
python -c "from huggingface_hub import HfApi; api = HfApi(); print(f'Logged in as: {api.whoami()[\"name\"]}')"
# Expected: Logged in as: <your_username>
```

### Phase 0 检查点

```bash
# 一键验证所有依赖
python -c "
import torch, mujoco, lerobot, gym_pusht
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi
print(f'✓ torch {torch.__version__}')
print(f'✓ mujoco {mujoco.__version__}')
print(f'✓ lerobot {lerobot.__version__}')
print(f'✓ gym-pusht')
print(f'✓ LeRobot Dataset API')
print(f'✓ HuggingFace Hub')
"
```

---

## Phase 1: LeRobot 内置仿真（PushT）

PushT 是一个简单的推物任务，LeRobot 自带预采集数据集，适合先跑通训练-评估全流程。

### 1.1 下载 PushT 数据集

```bash
# 从 HuggingFace Hub 下载预采集数据集（约 50 episodes）
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('lerobot/pusht')
print(f'Episodes: {dataset.num_episodes}')
print(f'Frames:   {dataset.num_frames}')
print(f'FPS:      {dataset.fps}')
print(f'Features: {list(dataset.features.keys())}')
"
# Expected:
# Episodes: 50
# Frames:   ~25000
# Features: ['observation.image', 'observation.state', 'action']
```

### 1.2 训练 Diffusion Policy（本地 CPU 先跑通）

```bash
# 本地 CPU 小规模训练（1 epoch，验证流程可用）
lerobot-train \
  policy.type=diffusion \
  env.type=pusht \
  dataset_repo_id=lerobot/pusht \
  training.num_epochs=1 \
  training.batch_size=8 \
  device=cpu \
  output_dir=outputs/pusht_diffusion_test
```

> **注意**：本地 CPU 训练非常慢，这里只跑 1 epoch 验证流程。大规模训练在 Phase 3 移到远程 GPU。

Expected 输出关键信息：
```
[INFO] Loading dataset from lerobot/pusht
[INFO] Training started
[INFO] Epoch 1/1 ...
[INFO] Training complete
```

### 1.3 训练 ACT（可选，同理）

```bash
lerobot-train \
  policy.type=act \
  env.type=pusht \
  dataset_repo_id=lerobot/pusht \
  training.num_epochs=1 \
  training.batch_size=8 \
  device=cpu \
  output_dir=outputs/pusht_act_test
```

### 1.4 评估预训练模型

LeRobot Hub 上有社区训练好的 PushT 模型，可以直接下载评估：

```bash
# 下载并评估预训练模型
python -c "
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 加载预训练模型
policy = DiffusionPolicy.from_pretrained('lerobot/diffusion_pusht')
print(f'✓ Model loaded: {policy.config.name}')
"
```

### 1.5 用 gym-pusht 可视化推理

```bash
python -c "
import gym_pusht
import gymnasium as gym

env = gym.make('gym_pusht/PushT-v0', render_mode='rgb_array')
obs, info = env.reset()
print(f'Obs space: {env.observation_space}')
print(f'Act space:  {env.action_space}')
print(f'Initial obs shape: {obs[\"image\"].shape}')
env.close()
print('✓ PushT env OK')
"
# Expected:
# Obs space: Dict(image: Box(0, 255, (480, 640, 3)), state: Box(-inf, inf, (2,)))
# Act space: Box(-inf, inf, (2,))
# Initial obs shape: (480, 640, 3)
# ✓ PushT env OK
```

### Phase 1 检查点

- [ ] `lerobot/pusht` 数据集下载成功
- [ ] `lerobot-train` 能正常启动训练（CPU 上跑 1 epoch 即可）
- [ ] `gym-pusht` 环境可以 `reset()` 和 `step()`

---

## Phase 2: SO-101 MuJoCo 仿真数据采集

本阶段基于现有的 `sim_viewer.py` 和 `Simulation/SO101/scene.xml`，扩展为脚本化数据采集。

### 2.1 SO-101 MuJoCo 模型概览

```
关节结构（6 DOF）:
  base → shoulder_pan → shoulder_lift → elbow_flex → wrist_flex → wrist_roll → gripper
```

| 关节名 | 类型 | 范围 (rad) | 范围 (deg) | 说明 |
|--------|------|-----------|-----------|------|
| `shoulder_pan` | hinge | [-1.92, 1.92] | [-110, 110] | 底座旋转 |
| `shoulder_lift` | hinge | [-1.75, 1.75] | [-100, 100] | 肩部俯仰 |
| `elbow_flex` | hinge | [-1.69, 1.69] | [-97, 97] | 肘部弯曲 |
| `wrist_flex` | hinge | [-1.66, 1.66] | [-95, 95] | 腕部俯仰 |
| `wrist_roll` | hinge | [-2.74, 2.84] | [-157, 163] | 腕部旋转 |
| `gripper` | hinge | [-0.17, 1.75] | [-10, 100] | 夹爪开合 |

执行器均为 `position` 控制模式，`kp=17.8`，`forcerange=[-3.35, 3.35]`。

### 2.2 MuJoCo Offscreen 渲染验证

```python
# render_test.py — 验证 offscreen 渲染
import mujoco
import numpy as np
import os

SCENE_XML = os.path.join(os.path.dirname(__file__), "Simulation", "SO101", "scene.xml")
model = mujoco.MjModel.from_xml_path(SCENE_XML)
data = mujoco.MjData(model)

# 设置渲染分辨率
model.vis.global_.offwidth = 640
model.vis.global_.offheight = 480

# 创建 offscreen 渲染器
renderer = mujoco.Renderer(model, height=480, width=640)

# 设置初始关节角度
mujoco.mj_resetData(model, data)
data.qpos[:6] = [0, 0, 0, 0, 0, 0]
mujoco.mj_forward(model, data)

# 渲染
renderer.update_scene(data)
image = renderer.render()
print(f"Image shape: {image.shape}")  # (480, 640, 3)
print(f"Image dtype: {image.dtype}")  # uint8
print("✓ Offscreen rendering OK")
```

```bash
python render_test.py
# Expected:
# Image shape: (480, 640, 3)
# Image dtype: uint8
# ✓ Offscreen rendering OK
```

### 2.3 编写脚本轨迹

以下脚本生成 reaching 任务数据，保存为 LeRobot 数据集格式：

```python
# collect_sim_data.py — SO-101 MuJoCo 数据采集脚本
import mujoco
import numpy as np
import os
import json
from pathlib import Path
from PIL import Image

SCENE_XML = os.path.join(os.path.dirname(__file__), "Simulation", "SO101", "scene.xml")

# ---------- 轨迹定义 ----------
def generate_reaching_trajectory(n_steps=100, n_waypoints=3):
    """生成 reaching 任务的关键帧，线性插值为密集轨迹"""
    # 随机目标关节角度
    rng = np.random.default_rng(42)

    # 关节范围 (from scene.xml)
    joint_ranges = np.array([
        [-1.92, 1.92],   # shoulder_pan
        [-1.75, 1.75],   # shoulder_lift
        [-1.69, 1.69],   # elbow_flex
        [-1.66, 1.66],   # wrist_flex
        [-2.74, 2.84],   # wrist_roll
        [-0.17, 1.75],   # gripper (保持闭合)
    ])

    # 初始位置 (home)
    home = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 1.0])

    # 生成 waypoints
    waypoints = [home.copy()]
    for _ in range(n_waypoints):
        target = np.array([rng.uniform(lo, hi) for lo, hi in joint_ranges])
        target[5] = 1.0  # gripper 保持闭合
        waypoints.append(target)
    waypoints.append(home.copy())  # 回到初始位置

    # 线性插值
    steps_per_segment = n_steps // len(waypoints)
    trajectory = []
    for i in range(len(waypoints) - 1):
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            pose = waypoints[i] * (1 - t) + waypoints[i + 1] * t
            trajectory.append(pose)
    trajectory.append(waypoints[-1])

    return np.array(trajectory)


# ---------- 采集循环 ----------
def collect_episode(model, data, renderer, trajectory):
    """执行一次轨迹采集，返回 observation + action 列表"""
    frames = []
    mujoco.mj_resetData(model, data)

    for i in range(len(trajectory) - 1):
        # 设置目标关节角度
        data.ctrl[:6] = trajectory[i]
        # 仿真若干步让位置稳定
        for _ in range(10):
            mujoco.mj_step(model, data)
        # 渲染图像
        renderer.update_scene(data)
        image = renderer.render()  # (480, 640, 3) uint8

        # 当前关节角度作为 observation.state
        state = data.qpos[:6].copy()
        # action = 下一帧的目标角度
        action = trajectory[i + 1].copy()

        frames.append({
            "observation.image": image,
            "observation.state": state,
            "action": action,
        })

    return frames


# ---------- 保存为 LeRobot 数据集格式 ----------
def save_as_lerobot_dataset(all_episodes, output_dir="outputs/so101_sim_dataset"):
    """将采集数据保存为 LeRobot 兼容格式"""
    out = Path(output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)

    meta = {
        "repo_id": "local/so101_sim",
        "fps": 30,
        "features": {
            "observation.image": {"dtype": "video", "shape": [480, 640, 3]},
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [6]},
        },
        "num_episodes": len(all_episodes),
    }

    all_data = []
    for ep_idx, episode in enumerate(all_episodes):
        for frame_idx, frame in enumerate(episode):
            # 保存图像
            img_path = f"images/ep{ep_idx:04d}_frame{frame_idx:06d}.png"
            Image.fromarray(frame["observation.image"]).save(out / img_path)

            all_data.append({
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "observation.image.path": img_path,
                "observation.state": frame["observation.state"].tolist(),
                "action": frame["action"].tolist(),
            })

    # 保存 meta
    with open(out / "meta" / "info.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {len(all_data)} frames from {len(all_episodes)} episodes to {out}")
    return out


# ---------- Main ----------
def main():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    n_episodes = 10
    n_steps_per_episode = 100

    all_episodes = []
    for ep in range(n_episodes):
        trajectory = generate_reaching_trajectory(
            n_steps=n_steps_per_episode,
            n_waypoints=3,
        )
        frames = collect_episode(model, data, renderer, trajectory)
        all_episodes.append(frames)
        print(f"Episode {ep+1}/{n_episodes}: {len(frames)} frames")

    save_as_lerobot_dataset(all_episodes)
    print("✓ Data collection complete")


if __name__ == "__main__":
    main()
```

运行：

```bash
python collect_sim_data.py
# Expected:
# Episode 1/10: 99 frames
# Episode 2/10: 99 frames
# ...
# Episode 10/10: 99 frames
# Saved 990 frames from 10 episodes to outputs/so101_sim_dataset
# ✓ Data collection complete
```

### 2.4 Pick-Place 轨迹扩展

在 `collect_sim_data.py` 的基础上，可以添加 pick-place 轨迹。核心思路：

```python
def generate_pick_place_trajectory(n_steps=200):
    """Pick-place: approach → grasp → lift → place → release → retract"""
    rng = np.random.default_rng()
    joint_ranges = np.array([
        [-1.92, 1.92], [-1.75, 1.75], [-1.69, 1.69],
        [-1.66, 1.66], [-2.74, 2.84], [-0.17, 1.75],
    ])

    home = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 1.0])
    gripper_open = np.array([0, 0, 0, 0, 0, -0.1])
    gripper_closed = np.array([0, 0, 0, 0, 0, 1.5])

    # 随机 pick 位置
    pick = np.array([rng.uniform(lo * 0.3, hi * 0.3) for lo, hi in joint_ranges[:5]] + [0])
    pick[1] = -1.0  # 俯身

    # 随机 place 位置
    place = pick.copy()
    place[0] += rng.uniform(-0.5, 0.5)

    # 组装关键帧
    waypoints = [
        home,
        pick + gripper_open[:6].__add__(np.zeros(6)),  # approach (open gripper)
        pick + np.array([0, 0, 0, 0, 0, 1.5]),          # grasp (close gripper)
        home + np.array([0, 0, 0, 0, 0, 1.5]),          # lift
        place + np.array([0, 0, 0, 0, 0, 1.5]),         # place
        place + np.array([0, 0, 0, 0, 0, -0.1]),        # release
        home,                                               # retract
    ]

    # 线性插值
    steps_per_segment = n_steps // (len(waypoints) - 1)
    trajectory = []
    for i in range(len(waypoints) - 1):
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            pose = np.array(waypoints[i]) * (1 - t) + np.array(waypoints[i + 1]) * t
            trajectory.append(pose)
    trajectory.append(waypoints[-1])

    return np.array(trajectory)
```

> **注意**：pick-place 轨迹需要在 `scene.xml` 中添加可抓取物体（如一个 box geom），这里只给出框架。实际使用时需要修改场景文件。

### Phase 2 检查点

- [ ] `render_test.py` offscreen 渲染输出正确图像
- [ ] `collect_sim_data.py` 生成 LeRobot 格式数据集
- [ ] 数据集包含 `observation.image`、`observation.state`、`action`
- [ ] 至少 10 episodes，每 episode ~100 frames

---

## Phase 3: 远程 GPU 训练

### 3.1 传输数据集到远程

**方式 A：通过 HuggingFace Hub（推荐）**

```bash
# 本地：上传数据集到 HF Hub
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='outputs/so101_sim_dataset',
    repo_id='<your_username>/so101_sim',
    repo_type='dataset',
)
print('✓ Dataset uploaded')
"

# 远程：下载数据集
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('<your_username>/so101_sim')
print(f'Downloaded: {dataset.num_episodes} episodes')
"
```

**方式 B：rsync 直传**

```bash
# 本地 → 远程
rsync -avz outputs/so101_sim_dataset/ user@remote:/path/to/outputs/so101_sim_dataset/
```

### 3.2 远程训练 ACT

```bash
# 在 RTX6000 服务器上
lerobot-train \
  policy.type=act \
  dataset_repo_id=<your_username>/so101_sim \
  training.num_epochs=100 \
  training.batch_size=16 \
  training.lr=1e-4 \
  device=cuda \
  output_dir=outputs/so101_act

# 监控 GPU 使用
watch -n 1 nvidia-smi
# Expected: ~1GB VRAM usage for ACT
```

### 3.3 远程训练 Diffusion Policy

```bash
lerobot-train \
  policy.type=diffusion \
  dataset_repo_id=<your_username>/so101_sim \
  training.num_epochs=200 \
  training.batch_size=64 \
  training.lr=1e-4 \
  device=cuda \
  output_dir=outputs/so101_diffusion

# Expected: ~5GB VRAM usage for Diffusion Policy
```

### 3.4 模型同步回本地

```bash
# 远程：上传模型到 HF Hub
python -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path='outputs/so101_act/checkpoints/best',
    repo_id='<your_username>/so101_act',
)
"

# 本地：下载模型
python -c "
from lerobot.common.policies.act.modeling_act import ACTPolicy
policy = ACTPolicy.from_pretrained('<your_username>/so101_act')
print('✓ Model downloaded')
"
```

### Phase 3 检查点

- [ ] 数据集成功上传到 HuggingFace Hub 或 rsync 到远程
- [ ] 远程训练启动正常，GPU 使用率 > 80%
- [ ] 训练 loss 下降
- [ ] 模型同步回本地并加载成功

---

## Phase 4: VLA 模型探索

LeRobot 0.5.1 内置 SmolVLA，一个轻量级 Vision-Language-Action 模型（~500M 参数）。

### 4.1 推理测试（不需要硬件）

```bash
# 下载预训练 SmolVLA
python -c "
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')
print(f'✓ SmolVLA loaded')
print(f'  Parameters: {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M')
"
```

### 4.2 可选：LoRA 微调（需要远程 GPU）

```bash
# 在 RTX6000 服务器上
lerobot-train \
  policy.type=smolvla \
  dataset_repo_id=<your_username>/so101_sim \
  training.num_epochs=50 \
  training.batch_size=8 \
  policy.use_lora=true \
  policy.lora_rank=16 \
  device=cuda \
  output_dir=outputs/so101_smolvla_lora
```

### Phase 4 检查点

- [ ] SmolVLA 预训练模型加载成功
- [ ] （可选）LoRA 微调完成

---

## Appendix

### A. SO-101 关节参考表

| # | 关节名 | 范围 (rad) | 范围 (deg) | 执行器类型 | kp | 力矩限制 (Nm) |
|---|--------|-----------|-----------|-----------|-----|-------------|
| 1 | shoulder_pan | [-1.92, 1.92] | [-110, 110] | position | 17.8 | 3.35 |
| 2 | shoulder_lift | [-1.75, 1.75] | [-100, 100] | position | 17.8 | 3.35 |
| 3 | elbow_flex | [-1.69, 1.69] | [-97, 97] | position | 17.8 | 3.35 |
| 4 | wrist_flex | [-1.66, 1.66] | [-95, 95] | position | 17.8 | 3.35 |
| 5 | wrist_roll | [-2.74, 2.84] | [-157, 163] | position | 17.8 | 3.35 |
| 6 | gripper | [-0.17, 1.75] | [-10, 100] | position | 17.8 | 3.35 |

### B. LeRobot CLI 命令参考

| 命令 | 用途 |
|------|------|
| `lerobot-train` | 训练策略 |
| `lerobot-eval` | 评估策略 |
| `lerobot-record` | 遥操作录制（需硬件） |
| `lerobot-replay` | 回放录制数据 |
| `lerobot-push-dataset` | 上传数据集到 HF Hub |
| `lerobot-viz-dataset` | 可视化数据集 |

常用训练参数：

```
policy.type=act|diffusion|smolvla    策略类型
dataset_repo_id=<repo>               数据集
training.num_epochs=N                训练轮数
training.batch_size=N                批大小
training.lr=<float>                  学习率
device=cpu|cuda                      设备
output_dir=<path>                    输出目录
wandb.enable=true                    启用 Weights & Biases 日志
```

### C. 常见错误排查

| 错误 | 原因 | 解决 |
|------|------|------|
| `ModuleNotFoundError: No module named 'lerobot'` | 未在 venv 中运行 | 先执行 `.venv\Scripts\activate` |
| `torch.cuda.is_available() == False` | 安装了 CPU 版 torch | 安装 CUDA 版：`pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| `OSError: Unable to open file` (MuJoCo) | 找不到 scene.xml | 使用绝对路径或从项目根目录运行 |
| `HfApi HTTP 401` | 未登录 HF Hub | 运行 `huggingface-cli login` |
| `gym-pusht` import 失败 | 未安装 | `pip install gym-pusht` |
| 训练 loss 不下降 | 学习率或数据问题 | 检查数据集格式、尝试不同 lr（1e-3 ~ 1e-5） |
| MuJoCo 渲染黑屏 | 未调用 `mj_forward` | 在 `renderer.update_scene()` 前调用 `mujoco.mj_forward(model, data)` |
| `offscreen rendering` 失败 | MuJoCo 版本问题 | 确保 `mujoco >= 3.0`，使用 `mujoco.Renderer` API |

---

## 关键文件索引

| 文件 | 作用 | 状态 |
|------|------|------|
| `docs/no_hardware_deployment.md` | 本文档 | 新建 |
| `docs/deployment_status.md` | 环境信息记录 | 已有 |
| `sim_viewer.py` | MuJoCo 交互查看器 | 已有 |
| `Simulation/SO101/scene.xml` | MuJoCo 场景定义 | 已有 |
| `Simulation/SO101/so101_new_calib.xml` | SO-101 模型（6 关节） | 已有 |
| `collect_sim_data.py` | 数据采集脚本 | 待创建（Phase 2） |
| `render_test.py` | 渲染测试脚本 | 待创建（Phase 2） |
