# SO-101 Push Task: 踩坑记录与正确方案

> 创建：2026-05-12
> 前置：`task_pipeline.md` Phase 5 设计，`ACT_Training_Versions.md` 训练配置
> 状态：v7 训练完成（50K steps），eval 发现关键问题，本文档总结正确方案

---

## 1. 问题全景

训练了 50K steps 的 ACT 模型（`act_v7`），eval rollout 成功率 **0%**。根因不是模型或超参数，而是 **数据采集管线存在根本性错误**。

```
训练数据采集 → 场景里没有物体 → 模型学的是"空中推"的运动模式 → eval 有物体时无法交互
```

---

## 2. 踩过的坑

### 坑 1: 训练场景里没有物体

**现象**：`data_collector.py` 使用 `scene.xml`（只有机械臂 + 地面），"push" 轨迹只是关节空间运动模式。

**根因**：
```python
# data_collector.py 第 38 行
self.scene_xml = config.get("scene_xml", "Simulation/SO101/scene.xml")
# ↑ 用的是没有 push_object 的场景
```

**后果**：模型从未在图像中见过方块，学到的 "push" 只是关节运动模式，不是物体交互。

**正确做法**：采集数据时必须用带物体的场景，且物体必须在轨迹路径上。

---

### 坑 2: SO-101 夹爪够不到桌面

**现象**：把 cube 放在 Z=0.025m（桌面），但 eval 时夹爪从不接触 cube。

**实测数据**（RTX 上 sweep 全关节空间）：

| 关节配置 | 夹爪 XYZ | 夹爪 Z |
|---------|----------|--------|
| HOME [0, -0.5, 1.0, -0.5] | [0.223, 0, 0.159] | 15.9cm |
| pre_push [0, -0.8, 1.2, -0.5] | [0.195, 0, 0.167] | 16.7cm |
| **最低可达** [-0.35, 1.51, -0.46] | [0.154, 0, **0.067**] | **6.7cm** |

**结论**：SO-101 夹爪最低只能到 Z=6.7cm，**不可能接触桌面（Z=0）上的物体**。

**解决方案**：加一个 4cm 高的台面（platform），让 cube 坐在台面上（cube 中心 Z=0.065m），夹爪就能到达。

```xml
<!-- 正确方案：加台面 -->
<geom name="push_table" type="box" size="0.18 0.10 0.02"
      pos="0.20 0.0 0.02" rgba="0.4 0.35 0.3 1"
      contype="1" conaffinity="1"/>
<!-- cube 在台面上 -->
<body name="push_object" pos="0.16 0.0 0.065">
    <freejoint name="push_object_joint"/>
    <geom name="push_object_geom" type="box" size="0.025 0.025 0.025"
          rgba="0.9 0.15 0.15 1" mass="0.05"
          friction="0.8 0.8 0.8"/>
</body>
```

---

### 坑 3: 夹爪从错误方向接近 cube

**现象**：设计了 "HOME → 下降 → 推 → 回撤" 轨迹，但 cube 被推往基座方向（反向）。

**实测**（`scene_push_table.xml` + 物理模拟）：

```
cube 起始:  X=0.160
cube 终止:  X=0.135  ← 往回推了 2.5cm，方向反了！
```

**原因分析**：轨迹在关节空间插值，夹爪实际笛卡尔路径如下：

```
step   0: gripper (0.291, 0, 0.234)  ← HOME，在 cube 右上方
step  16: gripper (0.189, 0, 0.096)  ← 下降中，经过 cube 上方
step  32: gripper (0.183, 0, 0.096)  ← 从右侧接触 cube，推向左边
step  48: gripper (0.212, 0, 0.103)  ← 继续向前，但 cube 已被推走
step  96: gripper (0.220, 0, 0.156)  ← 回 HOME
```

**根本问题**：夹爪从 **X=0.29**（HOME）下降到 **X=0.16**（cube 位置），过程中从 cube 右侧经过，把 cube 推向左边（基座方向）。

**正确做法**：夹爪必须从 cube **后方**（靠近基座一侧）接近，然后向前推。需要更多 waypoint 设计 "绕行" 路径。

---

### 坑 4: 关节空间插值 ≠ 笛卡尔空间直线

**现象**：以为在关节空间线性插值就能让夹爪走直线。

**事实**：关节空间线性插值产生的是弧线，夹爪路径不可预测。

**正确做法**：
- 简单任务：多加 waypoint 让关节空间插值近似笛卡尔路径
- 精确任务：用 IK（逆运动学）在笛卡尔空间规划，再转换为关节角

---

## 3. SO-101 运动学参数（实测）

在 RTX 上用 MuJoCo 全关节空间 sweep 得出的 SO-101 实测数据：

### 3.1 工作空间

| 区域 | X 范围 (m) | Z 范围 (m) | 对应关节配置 |
|------|-----------|-----------|-------------|
| 高位 | 0.13 ~ 0.25 | 0.15 ~ 0.25 | HOME 附近 |
| 中位 | 0.15 ~ 0.22 | 0.10 ~ 0.15 | 中间配置 |
| **低位** | **0.12 ~ 0.21** | **0.067 ~ 0.10** | sl∈[-1.0,-0.35], ef∈[1.3,1.69] |
| Y 偏移 | ±0.04 | - | shoulder_pan 变化 |

### 3.2 夹爪最低点搜索结果

```
绝对最低点: Z=0.067m (6.7cm)
关节配置:   shoulder_lift=-0.15, elbow_flex=1.60, wrist_flex=-0.86
夹爪位置:   [0.157, 0.000, 0.067]
```

### 3.3 推物轨迹关键 waypoint（低位）

| 名称 | 关节 [sl, ef, wf] | 夹爪 XYZ | 用途 |
|------|-------------------|----------|------|
| HOME | [-0.50, 1.00, -0.50] | [0.221, 0.000, 0.162] | 起始 |
| descend_high | [-0.65, 1.51, -0.16] | [0.159, 0.000, 0.082] | 高位下降 |
| contact_low | [-0.15, 1.60, -0.86] | [0.157, 0.000, 0.067] | 接触 cube |
| push_forward | [-0.15, 1.10, -0.26] | [0.207, 0.000, 0.081] | 向前推 |

### 3.4 关节限位

```python
JOINT_RANGES = {
    'shoulder_pan':  [-1.92, 1.92],
    'shoulder_lift': [-1.75, 1.75],
    'elbow_flex':    [-1.69, 1.69],
    'wrist_flex':    [-1.66, 1.66],
    'wrist_roll':    [-2.74, 2.84],
    'gripper':       [-0.17, 1.75],
}
```

---

## 4. 正确方案

### 4.1 场景设计：带台面的 Push 场景

文件：`Simulation/SO101/scene_push_table.xml`

```xml
<mujoco model="scene_push_table">
    <include file="so101_new_calib.xml" />

    <visual>
        <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
        <rgba haze="0.15 0.25 0.35 1" />
        <global azimuth="160" elevation="-20" />
    </visual>

    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
            width="512" height="3072" />
        <texture type="2d" name="groundplane" builtin="checker" mark="edge"
            rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
            width="300" height="300" />
        <material name="groundplane" texture="groundplane" texuniform="true"
            texrepeat="5 5" reflectance="0.2" />
    </asset>

    <worldbody>
        <light pos="0 0 3.5" dir="0 0 -1" directional="true" />
        <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane"
            material="groundplane" />

        <!-- 台面：让 SO-101 够得着 cube -->
        <geom name="push_table" type="box" size="0.18 0.10 0.02"
              pos="0.20 0.0 0.02" rgba="0.4 0.35 0.3 1"
              contype="1" conaffinity="1"/>

        <!-- 可推物体：红方块 -->
        <body name="push_object" pos="0.16 0.0 0.065">
            <freejoint name="push_object_joint"/>
            <geom name="push_object_geom" type="box" size="0.025 0.025 0.025"
                  rgba="0.9 0.15 0.15 1" mass="0.05"
                  friction="0.8 0.8 0.8"/>
        </body>

        <!-- 目标区域 -->
        <geom name="target_zone" type="cylinder" size="0.05 0.001"
              pos="0.32 0.0 0.041" rgba="0.1 0.8 0.2 0.35"
              contype="0" conaffinity="0"/>
    </worldbody>
</mujoco>
```

**设计要点**：
- 台面高度 Z=0.04m（8cm），cube 中心 Z=0.065m，在夹爪低位可达范围
- 台面足够大（36cm × 20cm）提供推物空间
- cube 用 `freejoint`（6DOF），可被推动
- target_zone 无碰撞属性，仅视觉标记

### 4.2 轨迹设计：正确接近方向

**核心原则**：夹爪从 cube 后方接近，向前推。避免从上方或侧方撞击。

```
HOME (高处，X=0.22)
  ↓ 绕行到 cube 左后方（避免经过 cube 上方）
  ↓ 需要先移到 X < cube位置 的上方
descend_behind (X=0.12, Z=0.16) ← 在 cube 左后方高处
  ↓ 下降到 cube 高度
contact_low (X=0.157, Z=0.067) ← cube 后方，同高度
  ↓ 向前推
push_forward (X=0.207, Z=0.081) ← 推过 cube 位置
  ↓ 回撤
  ↓ HOME
```

**注意**：由于 SO-101 关节空间插值的非线性，需要实测验证每个 waypoint 的笛卡尔路径。推荐在 MuJoCo 中先跑一遍轨迹，检查夹爪不与 cube 发生意外碰撞。

### 4.3 数据采集管线

```python
# 正确的 data_collector 配置
cfg = {
    "scene_xml": "Simulation/SO101/scene_push_table.xml",  # 带物体的场景！
    "trajectory_type": "ik_push",
    "n_episodes": 300,
    "n_steps": 100,
    "randomize": True,
    "randomization": {
        "joint_noise_std": 0.02,
        "target_range_scale": 0.8,
    },
}
```

**关键修改**：
1. `scene_xml` 指向带 cube 的场景
2. 轨迹设计为从后方接近 → 前推
3. 随机化 cube 位置（而非轨迹形状）

### 4.4 Eval 管线

```python
# eval_runner.py push 模式
PUSH_OBJECT_START = np.array([0.16, 0.0, 0.065])  # cube 初始位置
TARGET_POS = np.array([0.32, 0.0, 0.041])          # 目标位置

# 成功判定：XY 距离 < 0.05m
push_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "push_object")
dist = np.linalg.norm(obj_pos[:2] - TARGET_POS[:2])
success = dist < 0.05
```

**运行命令**：
```bash
# RTX 服务器上（EGL headless 渲染）
PYTHONPATH=. MUJOCO_GL=egl python -u scripts/eval/eval_rollout.py \
    --checkpoint outputs/train_act_v8/checkpoints/040000/pretrained_model \
    --push \
    --episodes 50 \
    --save-video \
    --output outputs/eval_results/act_v8_040000_push.json
```

---

## 5. 完整的正确 Pipeline

```
Step 1: 场景准备
  scene_push_table.xml（台面 + cube + target_zone）
  ↑ 关键：cube 在夹爪可达范围内（Z=0.065m）

Step 2: 轨迹验证
  在 RTX 上跑一遍轨迹，确认：
  - 夹爪从后方接近 cube
  - 推动方向正确（向前，+X）
  - cube 不被推下台面

Step 3: 数据采集（300 eps）
  run_collect.py → 使用 scene_push_table.xml
  → 每帧包含：带 cube 的图像 + 关节状态 + 动作
  → 上传 HF Hub

Step 4: 训练 ACT
  同 v7 配置（torchcodec, batch=128, 50K steps）
  → 关键区别：训练图像里有 cube

Step 5: Eval
  eval_rollout.py --push
  → 同场景（scene_push_table.xml）
  → 成功率目标 > 30%
```

---

## 6. 经验总结

### 6.1 采集前必检清单

| # | 检查项 | 验证方法 |
|---|--------|---------|
| 1 | 场景中有可推物体 | `grep push_object scene.xml` |
| 2 | 夹爪能到达物体高度 | sweep 关节空间，打印夹爪 XYZ |
| 3 | 轨迹从物体后方接近 | 模拟轨迹，逐帧打印夹爪位置 |
| 4 | 推动方向正确（+X） | 查看 cube 位移是否向 target |
| 5 | 物体不掉出工作空间 | 检查 cube Z 是否始终 > 台面高度 |
| 6 | 图像中可见物体 | 渲染第一帧，目视确认 |

### 6.2 常见错误对照

| 错误 | 症状 | 排查方法 |
|------|------|---------|
| 场景没物体 | eval 0% 且 cube 不动 | `grep push_object` 场景文件 |
| 夹爪太高 | eval 0% 且 cube 不动 | 打印夹爪 Z 和 cube Z 对比 |
| 推向错误方向 | cube 离 target 越来越远 | 逐帧打印 cube XYZ |
| 关节限位 | 夹爪到不了预期位置 | 检查关节是否触及 JOINT_RANGES |
| 随机化过大 | 轨迹不可控 | 减小 noise_std |

### 6.3 SO-101 特有的坑

1. **不能推地面物体** — 最低 Z=6.7cm，必须用台面
2. **5DOF 无手腕偏移** — 夹爪朝向有限，不能像 6DOF 那样灵活接近
3. **关节空间插值不直线** — 笛卡尔路径是弧线，需要多 waypoint 或 IK
4. **Y 方向偏移小** — shoulder_pan 变化只给 ±4cm Y 偏移

### 6.4 ACT 训练注意事项

| 参数 | 推荐值 | 原因 |
|------|--------|------|
| chunk_size | 50-100 | Push 任务 80-150 步完成 |
| kl_weight | 10 | ACT 原论文推荐 |
| batch_size | 128 | 配合 torchcodec 加速 |
| 训练步数 | 50K | 300 eps × 100 frames 足够 |
| overfitting 检测 | loss > best×1.2 | 早停，用 best checkpoint |

### 6.5 Eval 注意事项

| 项目 | 说明 |
|------|------|
| EGL 渲染 | `MUJOCO_GL=egl` 必须在 import mujoco 之前设置 |
| 成功阈值 | XY 距离 < 5cm（对于 5cm cube 合理） |
| episode 数 | ≥ 50 才有统计意义 |
| 保存视频 | 必须有视频才能判断失败原因 |

---

## 7. 关键文件索引

| 文件 | 用途 |
|------|------|
| `Simulation/SO101/scene_push_table.xml` | 带台面的 Push 采集+eval 场景 |
| `Simulation/SO101/scene_push_eval.xml` | 升高 cube 的 eval 场景（Task A） |
| `orchestrator_arm101/data_collector.py` | 数据采集器（需改 scene_xml） |
| `orchestrator_arm101/eval_runner.py` | Push eval runner |
| `scripts/eval/eval_rollout.py` | Eval rollout 入口 |
| `scripts/data/run_collect.py` | 数据采集入口 |
| `training_plans/so101_push_plan.yaml` | 完整 4-phase 训练计划 |
| `training_plans/rtx_train_plan.yaml` | RTX 训练配置 |

---

## 8. 参考资料

- **ACT 原论文**: Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", RSS 2023
- **LeRobot 框架**: https://github.com/huggingface/lerobot
- **SO-ARM101 硬件**: https://github.com/TheRobotStudio/SO-ARM101
- **gym_pusht**: HuggingFace 推物 benchmark 环境
- **MuJoCo 文档**: https://mujoco.readthedocs.io/

---

## 附录: 关节空间 Sweep 脚本

用于在 RTX 上验证 SO-101 夹爪可达工作空间：

```bash
ssh phh@192.168.120.155
conda activate lerobot
cd ~/SO-ARM101-LeRobot

python3 -c "
import mujoco, numpy as np
model = mujoco.MjModel.from_xml_path('Simulation/SO101/scene.xml')
data = mujoco.MjData(model)

min_z, min_cfg, min_xyz = 999, None, None
for sl in np.arange(-1.75, -0.1, 0.15):
    for ef in np.arange(-1.69, 1.69, 0.3):
        for wf in np.arange(-1.66, 1.66, 0.4):
            cfg = np.array([0.0, sl, ef, wf, 0.0, 1.0])
            mujoco.mj_resetData(model, data)
            data.ctrl[:6] = cfg
            for _ in range(300): mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            for i in range(model.nbody):
                nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                if nm == 'gripper':
                    z = data.xpos[i][2]
                    if z < min_z:
                        min_z = z
                        min_cfg = cfg.copy()
                        min_xyz = data.xpos[i].copy()
                    break

print(f'Min Z: {min_z:.4f}m at joints {min_cfg[1:4]}')
print(f'XYZ: {min_xyz}')
"
```
