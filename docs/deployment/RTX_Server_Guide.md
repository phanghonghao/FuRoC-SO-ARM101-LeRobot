# RTX 6000D 服务器指南

> RTX 6000D 服务器连接、环境配置、训练启动一站式参考。

---

## 1. 服务器硬件

| 项目 | 配置 |
|------|------|
| CPU | Intel Xeon 6767P × 2 |
| 内存 | 96GB DDR5 × 16 |
| GPU | RTX PRO 6000 × 8 (85.7 GB each) |
| 系统 | Ubuntu 22.04.5 |
| GPU 驱动 | 580.95.05 (Open Kernel Module) |

---

## 2. VPN + SSH 连接

### VPN (iNode)

1. 打开 iNode 客户端：`C:\Program Files (x86)\iNode\iNode Client\iNode Client.exe`
2. 网关：`113.57.110.73:4433`，用户名/密码：`phh` / `thu1234`
3. 点击"连接"

浏览器门户（首次安装客户端）：`https://113.57.110.73:4433/login/login.html`

### SSH

```bash
ssh phh@192.168.120.155
```

SSH 公钥免密已配置（`~/.ssh/id_ed25519`）。VPN 连接后即可直接登录。

| 项目 | 值 |
|------|-----|
| 地址 | `192.168.120.155` |
| 主机名 | `pro6000d` |
| 用户名 | `phh` |
| Conda 环境 | `isaaclab` |
| 项目目录 | `~/magiclab_rl_lab` |

### 文件传输

```bash
scp file phh@192.168.120.155:/mnt/data3/
rsync -avz ./local phh@192.168.120.155:~/magiclab_rl_lab/
```

### BMC 管理界面

| 项目 | 值 |
|------|-----|
| 地址 | `https://192.168.120.154/` |
| 用户名 | `Administrator` |
| 密码 | `Admin@9000` |

---

## 3. 环境配置

### Conda 激活

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate isaaclab
```

### EGL 持久化修复

Isaac Sim 启动时需要 NVIDIA EGL 厂商文件。已写入 conda 激活脚本自动补全：

- 脚本：`~/miniconda3/envs/isaaclab/etc/conda/activate.d/setenv.sh`
- 触发：每次 `conda activate isaaclab` 时自动检查

手动补一次（如果启动卡死）：
```bash
cp /usr/share/glvnd/egl_vendor.d/10_nvidia.json ~/miniconda3/envs/isaaclab/share/glvnd/egl_vendor.d/
```

### Vulkan 权限（已修复）

`phh` 用户已加入 `render` 和 `video` 组（2026-05-04 修复）。

**症状**（如果再次出现）：Isaac Sim 启动卡住，日志显示 `No device could be created`，但 `nvidia-smi` 和 PyTorch CUDA 正常。

**修复**：
```bash
sudo usermod -aG render,video phh
# 重新登录 SSH 让组权限生效
```

---

## 4. 训练启动

### 通过 gpu-train skill（推荐）

```bash
/gpu-train --start --resume              # 恢复最新训练
/gpu-train --start --from s4_gentle      # 从指定版本最佳模型开始
/gpu-train --start --resume --gpus 4     # 4卡训练
```

### 手动启动

```bash
ssh phh@192.168.120.155
cd ~/magiclab_rl_lab && conda activate isaaclab

# 单卡
nohup python -u scripts/rsl_rl/train.py \
    --task Magiclab-Z1-12dof-Velocity \
    --run_name <NAME> --headless \
    --max_iterations 50000 --num_envs 16384 --device cuda:0 \
    > /tmp/z1_train_<NAME>.log 2>&1 & echo PID=$!

# 多卡 (torchrun)
nohup torchrun --nproc_per_node=4 --master_port=29500 \
    scripts/rsl_rl/train_multigpu.py \
    --task=Magiclab-Z1-12dof-Velocity \
    --run_name=<NAME> --headless --distributed \
    --num_envs=16384 --max_iterations=50000 \
    --resume --load_run=<RUN_DIR> --checkpoint=model_<N>.pt \
    > /tmp/z1_mgpu_<NAME>.log 2>&1 & echo PID=$!
```

多卡详情见 `Multi_GPU_Deploy.md`。

### 默认参数

| 参数 | 值 |
|------|-----|
| Task | `Magiclab-Z1-12dof-Velocity` |
| Device | `cuda:0` |
| Num envs | 16384 (单卡) |
| Max iterations | 50000 |
| Save interval | 100 |

### 启动日志正常警告（可忽略）

| 日志 | 含义 |
|------|------|
| `ECC is enabled on physical device 0~7` | GPU 正常识别 |
| `Skipping NVIDIA GPU due CUDA being in bad state` | GPU Foundation 跳过，PhysX 通过 CUDA 正常工作 |
| `Could not load libneuray.so` | iray 渲染库缺失，训练不需要 |

---

## 5. 训练监控

```bash
/gpu-train --status          # 进程是否存活
/gpu-train --tail            # 最近 30 行日志
/gpu-train --mycuda          # 自己的 GPU 占用
/gpu-train --gpu             # 全部 GPU 状态
/gpu-train --monitor         # 过拟合检测 + 最佳模型
```

---

## 6. Isaac Lab 可视化说明

### 速度指令箭头

训练和 play 时，机器人上方显示两个箭头：

| 箭头 | 含义 | 来源 |
|------|------|------|
| 绿色 | 目标速度（命令要求） | `command[:, :2]` |
| 蓝色 | 当前速度（实际运动） | `robot.data.root_lin_vel_b[:, :2]` |

两箭头越重合 → 策略跟踪越好。开启方式：环境配置中 `debug_vis=True`。

---

## 7. 注意事项

- 测试数据归档到 `/mnt/data3` 个人文件夹，防止根目录被占满
- 系统管理员不负责数据备份
- GPU 从 0 开始连续分配，不能用 `CUDA_VISIBLE_DEVICES` 跳过
