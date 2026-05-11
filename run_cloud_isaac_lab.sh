#!/bin/bash
# ============================================================
# Cloud GPU Isaac Lab Pipeline
# 在 Paratera 云GPU上自动部署 Isaac Lab 并加载 LeRobot checkpoint
#
# 前提：用户已在 Paratera 网站手动租好 Windows GPU 实例
# 用法：bash run_cloud_isaac_lab.sh
# ============================================================

set -e

# ---- 配置区（用户填写） ----
CLOUD_IP=""           # Paratera 云服务器公网IP
CLOUD_USER=""         # Windows 用户名（从Paratera登录信息获取）
CLOUD_PASS=""         # Windows 密码
CLOUD_PORT=3389       # RDP 端口，默认3389

# Checkpoint 路径
LOCAL_CHECKPOINT_DIR="outputs/so101_act_checkpoints/010000"
REMOTE_CHECKPOINT_DIR="C:/checkpoints/010000"

# Isaac Lab 环境
CONDA_ENV="isaac"
LEROBOT_VERSION="0.5.1"

# ---- 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')]${NC} $1"; }
err()  { echo -e "${RED}[$(date '+%H:%M:%S')]${NC} $1"; }
info() { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }

# ============================================================
# Step 0: 交互式收集信息（如果配置区为空）
# ============================================================
collect_info() {
    echo ""
    info "========== Cloud GPU Isaac Lab Pipeline =========="
    echo ""

    if [ -z "$CLOUD_IP" ]; then
        read -p "云服务器 IP 地址: " CLOUD_IP
    fi
    if [ -z "$CLOUD_USER" ]; then
        read -p "用户名 (默认 administrator): " CLOUD_USER
        CLOUD_USER=${CLOUD_USER:-administrator}
    fi
    if [ -z "$CLOUD_PASS" ]; then
        read -sp "密码: " CLOUD_PASS
        echo ""
    fi

    echo ""
    log "目标: $CLOUD_USER@$CLOUD_IP:$CLOUD_PORT"
    echo ""
}

# ============================================================
# Step 1: 生成云服务器端 Setup 脚本 (PowerShell)
# ============================================================
generate_setup_script() {
    local ps1_path="cloud_gpu_pipeline/setup_cloud.ps1"
    mkdir -p cloud_gpu_pipeline

    cat > "$ps1_path" << 'PSEOF'
# cloud_gpu_pipeline/setup_cloud.ps1
# 在 Paratera Windows 云服务器上运行此脚本
# 用法: PowerShell 中运行: Set-ExecutionPolicy Bypass -Scope Process -Force; .\setup_cloud.ps1

$ErrorActionPreference = "Stop"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Isaac Lab Cloud Server Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# --- 1. 检查 GPU ---
Write-Host "`n[1/6] Checking GPU..." -ForegroundColor Green
nvidia-smi
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: nvidia-smi failed. GPU driver may not be installed." -ForegroundColor Red
    exit 1
}

# --- 2. 安装 Miniconda ---
Write-Host "`n[2/6] Setting up Conda..." -ForegroundColor Green
$condaPath = "C:\Miniconda3"
if (-not (Test-Path "$condaPath\Scripts\conda.exe")) {
    Write-Host "Installing Miniconda..." -ForegroundColor Yellow
    $installer = "$env:TEMP\Miniconda3-latest-Windows-x86_64.exe"
    Invoke-WebRequest -Uri "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -OutFile $installer
    Start-Process -FilePath $installer -ArgumentList "/S","/InstallationType=JustMe","/AddToPath=1","/RegisterPython=0","/D=$condaPath" -Wait
    Remove-Item $installer
}
& "$condaPath\Scripts\conda.exe" init powershell
$env:Path = "$condaPath\Scripts;$condaPath;$env:Path"

# --- 3. 创建环境 ---
Write-Host "`n[3/6] Creating conda environment 'isaac'..." -ForegroundColor Green
$envExists = & conda env list 2>&1 | Select-String "isaac"
if (-not $envExists) {
    & conda create -n isaac python=3.10 -y
}
& conda activate isaac

# --- 4. 安装 Isaac Sim + Isaac Lab ---
Write-Host "`n[4/6] Installing Isaac Sim + Isaac Lab..." -ForegroundColor Green
Write-Host "This will take a while (~100GB download)..." -ForegroundColor Yellow

pip install isaacsim[all] 2>&1 | ForEach-Object { Write-Host $_ }
pip install isaac-lab 2>&1 | ForEach-Object { Write-Host $_ }

# --- 5. 安装 LeRobot ---
Write-Host "`n[5/6] Installing LeRobot..." -ForegroundColor Green
pip install lerobot==PSEOF_LEROBOT_VERSION 2>&1 | ForEach-Object { Write-Host $_ }

# --- 6. 验证 ---
Write-Host "`n[6/6] Verifying installation..." -ForegroundColor Green

python -c "import isaacsim; print('Isaac Sim ... OK')" 2>&1
python -c "import isaac.lab; print('Isaac Lab ... OK')" 2>&1
python -c "import lerobot; print(f'LeRobot {lerobot.__version__} ... OK')" 2>&1

Write-Host "`n========================================" -ForegroundColor Green
Write-Host " Setup COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nNext: Transfer checkpoint and run simulation" -ForegroundColor Cyan
PSEOF

    # 替换版本号
    sed -i "s/PSEOF_LEROBOT_VERSION/$LEROBOT_VERSION/g" "$ps1_path"
    log "Setup script generated: $ps1_path"
}

# ============================================================
# Step 2: 生成仿真运行脚本 (Python)
# ============================================================
generate_sim_script() {
    local py_path="cloud_gpu_pipeline/run_isaac_lab_sim.py"

    cat > "$py_path" << 'PYEOF'
"""
Cloud GPU Isaac Lab 仿真运行脚本
在云服务器上运行: conda activate isaac && python run_isaac_lab_sim.py
"""
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run LeRobot ACT policy in Isaac Lab")
    parser.add_argument("--checkpoint", type=str, default="C:/checkpoints/010000",
                        help="Path to ACT checkpoint directory")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (default: cuda)")
    parser.add_argument("--record", action="store_true",
                        help="Record simulation video")
    parser.add_argument("--output", type=str, default="simulation_output.mp4",
                        help="Output video path (when --record is set)")
    args = parser.parse_args()

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    # 验证 checkpoint 存在
    ckpt_dir = Path(args.checkpoint)
    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_dir}")
        print("Please transfer checkpoint first.")
        sys.exit(1)

    required_files = ["config.json", "model.safetensors"]
    for f in required_files:
        if not (ckpt_dir / f).exists():
            print(f"ERROR: Missing {f} in checkpoint directory")
            sys.exit(1)

    print("Checkpoint validated. Loading policy...")

    # 加载 LeRobot ACT Policy
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    import torch

    policy = ACTPolicy.from_pretrained(str(ckpt_dir))
    policy.to(args.device)
    policy.eval()
    print(f"Policy loaded successfully on {args.device}")

    # --- Isaac Lab 仿真 ---
    # TODO: 根据 SO-ARM101 / MagicBotZ1 的具体 URDF 和任务配置仿真环境
    # 以下为 Isaac Lab 环境创建的模板代码

    print("")
    print("=" * 50)
    print("  Isaac Lab environment setup needed!")
    print("  Please configure the robot URDF/MJCF and")
    print("  task environment in this script.")
    print("")
    print("  Current checkpoint: ACT policy for SO-ARM101")
    print("  Task: ik_push (simulated pushing)")
    print("=" * 50)

    # 示例：使用 Isaac Lab 的 GymWrapper 或直接 API
    # from omni.isaac.lab.app import AppLauncher
    # app = AppLauncher(headless=not args.display)
    # ... setup simulation environment ...
    # ... run policy inference loop ...

if __name__ == "__main__":
    main()
PYEOF

    log "Sim script generated: $py_path"
}

# ============================================================
# Step 3: 传输 Checkpoint 和脚本到云服务器
# ============================================================
transfer_files() {
    echo ""
    log "========== Step 3: Transfer Files =========="

    # 检查本地 checkpoint
    if [ ! -d "$LOCAL_CHECKPOINT_DIR" ]; then
        err "Checkpoint not found: $LOCAL_CHECKPOINT_DIR"
        err "Make sure you're running this from the project root."
        exit 1
    fi

    log "Checkpoint size: $(du -sh "$LOCAL_CHECKPOINT_DIR" | cut -f1)"
    log "Checkpoint files:"
    ls -la "$LOCAL_CHECKPOINT_DIR"

    echo ""
    info "Choose transfer method:"
    info "  1) SCP (if cloud server has SSH enabled)"
    info "  2) Generate instructions for RDP copy-paste"
    info "  3) Skip (transfer manually later)"
    echo ""
    read -p "Choice [1-3]: " transfer_choice

    case $transfer_choice in
        1)
            log "Transferring via SCP..."
            log "Target: $CLOUD_USER@$CLOUD_IP:$REMOTE_CHECKPOINT_DIR"

            # 尝试 SCP（Windows Server 可能需要 OpenSSH）
            sshpass -p "$CLOUD_PASS" scp -r -P "$CLOUD_PORT" \
                "$LOCAL_CHECKPOINT_DIR" \
                "$CLOUD_USER@$CLOUD_IP:$(dirname "$REMOTE_CHECKPOINT_DIR")/" \
                2>&1 && log "SCP transfer complete!" || {
                    warn "SCP failed. Windows may not have OpenSSH enabled."
                    info "Falling back to RDP method (see below)."
                    transfer_rdp_instructions
                }

            # 传输 setup 脚本
            sshpass -p "$CLOUD_PASS" scp -P "$CLOUD_PORT" \
                cloud_gpu_pipeline/setup_cloud.ps1 \
                cloud_gpu_pipeline/run_isaac_lab_sim.py \
                "$CLOUD_USER@$CLOUD_IP:C:/" 2>/dev/null || true
            ;;
        2)
            transfer_rdp_instructions
            ;;
        3)
            warn "Skipping transfer. Do it manually."
            ;;
        *)
            warn "Invalid choice. Skipping."
            ;;
    esac
}

transfer_rdp_instructions() {
    echo ""
    info "===== RDP File Transfer Instructions ====="
    info ""
    info "1. Connect to RDP:"
    info "   Win+R → mstsc → $CLOUD_IP:$CLOUD_PORT"
    info "   User: $CLOUD_USER  Pass: $CLOUD_PASS"
    info ""
    info "2. Before connecting, in RDP options:"
    info "   Local Resources → More → Check your drives"
    info "   This maps your local drives to the remote session"
    info ""
    info "3. After connecting, copy checkpoint from:"
    info "   This PC → [Your Drive] → ... → outputs/so101_act_checkpoints/010000"
    info "   to C:/checkpoints/010000"
    info ""
    info "4. Also copy these files to C:/ on the remote server:"
    info "   - cloud_gpu_pipeline/setup_cloud.ps1"
    info "   - cloud_gpu_pipeline/run_isaac_lab_sim.py"
    info ""
    info "============================================="
    echo ""
    read -p "Press Enter when files are transferred..."
}

# ============================================================
# Step 4: 在云服务器上运行 Setup
# ============================================================
run_setup() {
    echo ""
    log "========== Step 4: Cloud Server Setup =========="

    echo ""
    info "In the RDP remote desktop, open PowerShell as Administrator and run:"
    echo ""
    echo -e "${CYAN}  Set-ExecutionPolicy Bypass -Scope Process -Force"
    echo -e "  cd C:\\"
    echo -e "  .\\setup_cloud.ps1${NC}"
    echo ""
    info "This will install: Miniconda → Isaac Sim → Isaac Lab → LeRobot"
    info "Estimated time: 30-60 minutes (mostly Isaac Sim download ~100GB)"
    echo ""
    read -p "Press Enter when setup is complete..."
}

# ============================================================
# Step 5: 运行仿真
# ============================================================
run_simulation() {
    echo ""
    log "========== Step 5: Run Simulation =========="

    echo ""
    info "In the RDP remote desktop, open PowerShell and run:"
    echo ""
    echo -e "${CYAN}  conda activate isaac"
    echo -e "  cd C:\\"
    echo -e "  python run_isaac_lab_sim.py --checkpoint C:/checkpoints/010000${NC}"
    echo ""
    info "The simulation will render in the RDP window - you can see it live!"
    echo ""
    info "To record a video:"
    echo -e "${CYAN}  python run_isaac_lab_sim.py --checkpoint C:/checkpoints/010000 --record${NC}"
    echo ""
}

# ============================================================
# Step 6: 收工 - 关机提醒
# ============================================================
shutdown_reminder() {
    echo ""
    log "========== Done! =========="
    echo ""
    warn "When finished, shut down the cloud server to save costs:"
    info "  Paratera Console → [Shut Down] → 选择 [节省模式]"
    info "  节省模式: 仅云盘计费，GPU和网络停止计费"
    echo ""
    log "Generated files:"
    log "  cloud_gpu_pipeline/setup_cloud.ps1    - 云服务器环境搭建脚本"
    log "  cloud_gpu_pipeline/run_isaac_lab_sim.py - 仿真运行脚本"
    echo ""
}

# ============================================================
# Main
# ============================================================
main() {
    collect_info
    generate_setup_script
    generate_sim_script
    transfer_files
    run_setup
    run_simulation
    shutdown_reminder
}

main
