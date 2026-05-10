#!/bin/bash
# ARM101 Pipeline Runner for RTX Server
# Waits for data collection to finish, then runs train_act + eval_act via orchestrator.
# Usage: nohup bash run_pipeline_rtx.sh > /tmp/arm101_pipeline.log 2>&1 &

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
cd ~/SO-ARM101-LeRobot

DATASET_DIR="/tmp/so101_push_sim_lerobot"
TARGET_EPISODES=300
DEVICE="cuda:6"
LOG="/tmp/arm101_pipeline.log"

echo "[$(date)] ARM101 Pipeline Runner started"
echo "[$(date)] Device: $DEVICE"
echo "[$(date)] Waiting for data collection to complete..."

# Phase 1: Wait for collection to finish
while true; do
    if [ -f "$DATASET_DIR/meta/info.json" ]; then
        EPISODES=$(python3 -c "import json; print(json.load(open('$DATASET_DIR/meta/info.json'))['total_episodes'])")
        echo "[$(date)] Collection progress: $EPISODES/$TARGET_EPISODES episodes"
        if [ "$EPISODES" -ge "$TARGET_EPISODES" ]; then
            echo "[$(date)] Collection COMPLETE! $EPISODES episodes collected."
            break
        fi
    fi

    # Check if collection process is still running
    if ! ps aux | grep "run_collect.py" | grep -v grep > /dev/null 2>&1; then
        echo "[$(date)] Collection process stopped. Checking final state..."
        if [ -f "$DATASET_DIR/meta/info.json" ]; then
            EPISODES=$(python3 -c "import json; print(json.load(open('$DATASET_DIR/meta/info.json'))['total_episodes'])")
            if [ "$EPISODES" -ge "$TARGET_EPISODES" ]; then
                echo "[$(date)] Collection finished with $EPISODES episodes."
                break
            else
                echo "[$(date)] WARNING: Collection stopped at $EPISODES/$TARGET_EPISODES episodes!"
                echo "[$(date)] Proceeding with available data..."
                break
            fi
        else
            echo "[$(date)] ERROR: No dataset found. Collection may have failed."
            exit 1
        fi
    fi

    sleep 60
done

echo "[$(date)] ==========================================="
echo "[$(date)] Phase 2: ACT Training on $DEVICE"
echo "[$(date)] ==========================================="

# Phase 2: Run the orchestrator starting from train_act
export MUJOCO_GL=egl
export HF_ENDPOINT=https://hf-mirror.com

python -u -c "
import sys, os
sys.path.insert(0, '$HOME/SO-ARM101-LeRobot')
os.environ['MUJOCO_GL'] = 'egl'

from orchestrator_arm101.arm101_orchestrator import Arm101Orchestrator

# Run train_act and eval_act phases
orchestrator = Arm101Orchestrator(
    plan_path='training_plans/rtx_train_plan.yaml',
    project_root='$HOME/SO-ARM101-LeRobot',
    start_from='train_act',
    fresh=True,
    device='$DEVICE',
    state_path='orchestrator_state.json',
)
orchestrator.run()
" 2>&1 | tee -a "$LOG"

echo "[$(date)] ==========================================="
echo "[$(date)] ARM101 Pipeline COMPLETE!"
echo "[$(date)] ==========================================="

# Show final results
if [ -f "orchestrator_state.json" ]; then
    echo "[$(date)] Final state:"
    cat orchestrator_state.json
fi
