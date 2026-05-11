# SO-101 任务式 Pipeline：从仿真到有意义的策略

> 创建：2026-05-10
> 前置条件：Phase 0-4 pipeline 已跑通（见 `no_hardware_deployment.md`）

---

## 当前状态

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 0-1: 环境验证 + PushT | 已完成 | 本地 venv、远程 RTX6000、HF Hub 均通 |
| Phase 2: 仿真数据采集 | 已完成 | `PhangHongHao/so101_sim`（10 eps, 800 帧，随机到达） |
| Phase 3: 远程 ACT 训练 | 进行中 | RTX6000 GPU4, batch=128, steps=1250 |
| Phase 4: SmolVLA 探索 | 已完成 | 本地加载 smolvla_base (450M params)，确认可导入 |

**问题**：当前数据是"随机到达"（无物体、无任务、无成功判定），训练出的策略没有实际用途。

**目标**：设计具体任务，收集有任务语义的演示数据，训练出可在 MuJoCo eval 环境中测试成功率的策略。

---

## Phase 5：任务式 Pipeline

### 总览

```
Phase 5.1          Phase 5.2          Phase 5.3          Phase 5.4
MuJoCo 任务环境     数据采集（脚本化）   多策略训练          评估 + 迭代
──────────────     ──────────────     ──────────────     ──────────────
scene_v2.xml       collect_task.py    ACT (GPU A)        eval_rollout.py
+ 物体 + 目标       200+ episodes      Diffusion (GPU B)   成功率统计
+ 碰撞 + 约束       → HF Hub           [可选] SmolVLA      loss 曲线对比
                                         视频可视化
```

### 5.1 MuJoCo 任务环境搭建

#### 5.1.1 任务选择

推荐从简单任务开始，逐步增加难度：

| 等级 | 任务 | 物体 | 成功条件 | 预计难度 |
|------|------|------|----------|----------|
| T1 | Reach — 到达目标点 | 无（虚拟目标点） | 末端距目标 < 2cm | 低 |
| T2 | Push — 推方块到目标 | 1 个方块 | 方块中心距目标 < 3cm | 中 |
| T3 | Pick — 抓取抬起 | 1 个方块 | 方块离桌面 > 5cm | 高（需要夹爪） |
| T4 | PickPlace — 抓放 | 2 个方块 | 方块放到目标位置 | 很高 |

**建议从 T2（Push）开始**：不需要夹爪精细控制，视觉上有明确反馈。

> **重要踩坑记录**：SO-101 夹爪最低只能到 Z=6.7cm，**推不到桌面上的物体**。必须加台面。详见 [`push_task_correct_approach.md`](./push_task_correct_approach.md)。

#### 5.1.2 MuJoCo 场景修改

当前 `scene.xml` 只有机械臂 + 地板。需要创建带台面 + 物体的场景：

```xml
<!-- 在 worldbody 中添加 -->
<!-- 台面（SO-101 够不到地面，必须加台面让 cube 在夹爪可达高度） -->
<geom name="push_table" type="box" size="0.18 0.10 0.02"
      pos="0.20 0.0 0.02" rgba="0.4 0.35 0.3 1"
      contype="1" conaffinity="1"/>

<!-- 红色方块（在台面上，中心 Z=0.065m，夹爪最低可达 Z=0.067m） -->
<body name="push_object" pos="0.16 0.0 0.065">
    <freejoint name="push_object_joint"/>
    <geom name="push_object_geom" type="box" size="0.025 0.025 0.025"
          rgba="0.9 0.15 0.15 1" mass="0.05" friction="0.8 0.8 0.8"/>
</body>

<!-- 目标区域（绿色圆圈） -->
<geom name="target_zone" type="cylinder" size="0.05 0.001"
      pos="0.32 0.0 0.041" rgba="0.1 0.8 0.2 0.35"
      contype="0" conaffinity="0"/>
```

**关键改动**（相比原文档）：
- ~~直接放桌面~~ → **加 4cm 台面**（SO-101 最低 Z=6.7cm）
- ~~方块 pos="0.3 0 0.02"~~ → **pos="0.16 0.0 0.065"**（夹爪可达范围）
- 方块用 `freejoint`（可被推动），不是 `geom`（固定不动）

#### 5.1.3 成功判定函数

```python
def check_success(block_pos, target_pos, threshold=0.03):
    """检查方块是否到达目标位置"""
    dist = np.linalg.norm(block_pos[:2] - target_pos[:2])
    return dist < threshold
```

#### 5.1.4 文件结构

```
Simulation/SO101/
  scene.xml          # 原始（只有机械臂）
  scene_v2_push.xml  # T2: Push 任务
  scene_v3_pick.xml  # T3: Pick 任务（后续）
```

### 5.2 脚本化数据采集

#### 5.2.1 采集策略

使用脚本生成"脚本化演示"（非人类遥操作）：

1. **逆运动学求解** — 计算末端到达目标附近的关节角
2. **轨迹插值** — 从 Home → 绕到方块后方 → 下降到方块高度 → 向前推 → 回 Home
3. **随机化** — 方块初始位置在台面范围内随机
4. **关键**：场景必须包含物体，轨迹必须从物体后方接近

> **踩坑警告**：关节空间线性插值在笛卡尔空间是弧线。如果夹爪从 HOME (X=0.22) 直接下降到 cube (X=0.16)，会从右侧撞击 cube 导致反向推动。必须先移到 cube 后方 (X < cube位置) 再下降。详见 [`push_task_correct_approach.md`](./push_task_correct_approach.md) 坑 3。

```python
# collect_task.py 核心逻辑
for episode in range(N_EPISODES):
    # 1. 随机放置方块和目标
    block_pos = random_in_workspace()
    target_pos = random_near(block_pos, max_dist=0.15)

    # 2. 生成轨迹 waypoints
    waypoints = [
        home_pos,
        above_block(block_pos),     # 移到方块上方
        push_start(block_pos),       # 降到推起始位
        push_end(block_pos, target), # 推到目标
        home_pos,                    # 回 Home
    ]

    # 3. 执行轨迹，记录每一帧
    for wp in waypoints:
        trajectory = interpolate(current_qpos, wp, n_steps=20)
        for qpos in trajectory:
            image = renderer.render()
            ds.add_frame({
                "observation.image": image,
                "observation.state": current_qpos[:6],
                "action": qpos[:6],
            })

    # 4. 检查成功
    success = check_success(block_pos, target_pos)
    ds.save_episode(task="push_block", success=success)
```

#### 5.2.2 数据量规划

| 任务 | Episodes | 帧/episode | 总帧数 | 预计大小 |
|------|----------|-----------|--------|----------|
| T1 Reach | 100 | 50 | 5,000 | ~50 MB |
| T2 Push | 300 | 80 | 24,000 | ~250 MB |
| T3 Pick | 500 | 100 | 50,000 | ~500 MB |

**T2 Push 建议配置**：
- 300 episodes（200 训练 + 50 验证 + 50 测试）
- 每 episode ~80 帧
- 上传到 HF Hub: `PhangHongHao/so101_push`

#### 5.2.3 数据增强

在采集时引入随机化以提高泛化：
- 方块初始位置：工作空间内均匀随机
- 目标位置：方块附近 5-15cm 范围
- 桌面颜色：可选（light/dark）
- 相机角度：微小抖动（±2°）

### 5.3 多策略训练

#### 5.3.1 训练配置

在 RTX6000 上使用不同 GPU 并行训练多个策略：

| 策略 | GPU | 参数量 | 推荐配置 |
|------|-----|--------|----------|
| ACT | cuda:4 | 52M | batch=64, steps=50000 |
| Diffusion Policy | cuda:5 | ~50M | batch=64, steps=100000 |
| [可选] SmolVLA | cuda:6 | 450M | batch=8, steps=20000 |

#### 5.3.2 训练命令

```bash
# ACT
CUDA_VISIBLE_DEVICES=4 python -c "
import sys
sys.argv = ['lerobot-train',
    '--dataset.repo_id=PhangHongHao/so101_push',
    '--dataset.video_backend=pyav',
    '--policy.type=act',
    '--policy.repo_id=PhangHongHao/so101_push_act',
    '--batch_size=64', '--steps=50000',
    '--output_dir=/tmp/so101_push_act',
]
from lerobot.scripts.lerobot_train import main; main()
"

# Diffusion Policy
CUDA_VISIBLE_DEVICES=5 python -c "
import sys
sys.argv = ['lerobot-train',
    '--dataset.repo_id=PhangHongHao/so101_push',
    '--dataset.video_backend=pyav',
    '--policy.type=diffusion',
    '--policy.repo_id=PhangHongHao/so101_push_diffusion',
    '--batch_size=64', '--steps=100000',
    '--output_dir=/tmp/so101_push_diffusion',
]
from lerobot.scripts.lerobot_train import main; main()
"
```

### 5.4 评估与迭代

#### 5.4.1 MuJoCo Eval 环境

创建 `eval_rollout.py`：

```python
def evaluate_policy(policy, scene_xml, n_episodes=50):
    """在 MuJoCo 中 rollout 策略，统计成功率"""
    successes = 0
    for ep in range(n_episodes):
        # 1. 重置环境，随机放置方块和目标
        reset_env(block_pos_random=True, target_pos_random=True)

        # 2. 执行策略
        for step in range(MAX_STEPS):
            image = renderer.render()
            state = data.qpos[:6]
            action = policy.select_action({
                "observation.image": image,
                "observation.state": state,
            })
            apply_action(action)
            mujoco.mj_step(model, data)

        # 3. 检查成功
        if check_success(block_pos, target_pos):
            successes += 1

    return successes / n_episodes
```

#### 5.4.2 评估指标

| 指标 | 说明 | 目标 |
|------|------|------|
| 成功率 | rollout 到达目标的比例 | > 70% |
| 平均距离 | 方块最终位置到目标的距离 | < 3cm |
| 轨迹长度 | 总步数 / 最短路径步数 | < 2.0 |
| 崩溃率 | 关节越界或碰撞异常 | < 5% |

#### 5.4.3 迭代策略

```
成功率 < 30%  → 增加数据量（+200 episodes），检查数据质量
成功率 30-70% → 调参（learning_rate, batch_size），增加训练步数
成功率 > 70%  → 增加任务难度（T3 Pick），或转移到真机
```

---

## 实施顺序

```
Week 1: 5.1 搭建 MuJoCo 任务环境（scene_v2_push.xml + 成功判定）
         → 验证脚本化轨迹生成在物理上合理

Week 2: 5.2 数据采集（300 episodes push 任务）
         → 上传 HF Hub，本地验证数据完整性

Week 3: 5.3 远程训练（ACT + Diffusion Policy 并行）
         → 监控 loss 曲线，选择最佳 checkpoint

Week 4: 5.4 评估 + 迭代
         → MuJoCo eval 统计成功率，视频记录，调参重训
```

---

## 文件清单（需要创建）

| 文件 | 用途 |
|------|------|
| `Simulation/SO101/scene_v2_push.xml` | Push 任务的 MuJoCo 场景 |
| `collect_task.py` | 脚本化数据采集（支持 Push 任务） |
| `eval_rollout.py` | 策略评估 + 成功率统计 |
| `docs/training_logs/` | 训练日志目录（已存在） |

---

## 关键决策

- **任务选择**：从 T2 Push 开始（不需要精细夹爪控制，视觉反馈清晰）
- **数据采集**：脚本化生成（非人类遥操作），可控且可大量生成
- **策略选择**：ACT 为首选（SO-101 社区主流），Diffusion Policy 作为对比
- **评估方式**：MuJoCo offscreen rollout + 成功率统计（不需要 GUI）
- **GPU 分配**：RTX6000 GPU 4-7（GPU 0-3 用于 Z1 locomotion 训练）
