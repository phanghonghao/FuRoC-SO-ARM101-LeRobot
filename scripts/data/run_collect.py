"""Run data collection phase only on RTX server."""
import sys
import os

# EGL for headless MuJoCo rendering on GPU server
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.expanduser("~/SO-ARM101-LeRobot"))

from orchestrator_arm101.data_collector import DataCollector
from pathlib import Path

# Collection config (from so101_push_plan.yaml collect phase)
cfg = {
    "scene_xml": str(Path(os.path.expanduser("~/SO-ARM101-LeRobot/Simulation/SO101/scene.xml"))),
    "trajectory_type": "ik_push",
    "n_episodes": 300,
    "n_steps": 100,
    "n_waypoints": 3,
    "randomize": True,
    "randomization": {
        "joint_noise_std": 0.05,
        "target_range_scale": 0.8,
    },
    "repo_id": "PhangHongHao/so101_push_sim",
    "root": "/tmp/so101_push_sim_lerobot",
    "push_to_hub": False,  # will push after verifying
    "fps": 30,
    "image_height": 480,
    "image_width": 640,
}

print(f"[collect] Starting: {cfg['n_episodes']} episodes, type={cfg['trajectory_type']}")
print(f"[collect] Output: {cfg['root']}")

collector = DataCollector(cfg)

def on_progress(p):
    if int(p * 100) % 10 == 0:
        print(f"[collect] Progress: {p:.0%}")

result = collector.run(progress_callback=on_progress)
print(f"\n[collect] DONE!")
print(f"  Episodes: {result['n_episodes']}")
print(f"  Frames:   {result['n_frames']}")
print(f"  Output:   {result['output_dir']}")
