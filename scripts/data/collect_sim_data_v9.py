# collect_sim_data_v9.py — SO-101 MuJoCo v9 push 数据采集 (扩展推距)
# v8 push 只推 5cm; v9 扩展到 ~14cm (gripper x=0.29)
# 输出 LeRobot 原生数据集格式, push_to_hub 上传到 HF Hub
#
# Usage:
#   LD_PRELOAD=~/miniconda3/envs/lerobot/lib/libstdc++.so.6 \
#   HF_ENDPOINT=https://hf-mirror.com \
#   python -u scripts/data/collect_sim_data_v9.py
#
# Or with custom episode count:
#   python -u scripts/data/collect_sim_data_v9.py --episodes 300

import argparse
import os

# EGL for headless MuJoCo rendering on GPU server (must be before mujoco import)
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Paths — works both locally and on RTX server
SCENE_SEARCH_PATHS = [
    Path.home() / "SO-ARM101-LeRobot" / "Simulation" / "SO101" / "scene_push_table.xml",
    Path("SO-ARM101-LeRobot/Simulation/SO101/scene_push_table.xml"),
    Path("Simulation/SO101/scene_push_table.xml"),
    Path(__file__).parent.parent.parent / "SO-ARM101-LeRobot" / "Simulation" / "SO101" / "scene_push_table.xml",
]

REPO_ID = "PhangHongHao/so101_push_v9_table_act"
OUTPUT_DIR = "/tmp/so101_push_v9_lerobot"
FPS = 30
N_EPISODES = 300
N_STEPS = 150  # v9: longer trajectory (v8 was 100)
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

HOME = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 1.0])


def find_scene() -> Path:
    for p in SCENE_SEARCH_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(f"Scene not found, searched: {[str(p) for p in SCENE_SEARCH_PATHS]}")


def generate_push_trajectory(n_steps=N_STEPS, seed=None):
    """Generate v9 push trajectory with extended push distance.

    Waypoints (FK-verified gripper body positions):
      HOME:    (0.222, 0, 0.169) — arm up
      descend: (0.162, 0, 0.085) — above cube
      contact: (0.152, 0, 0.065) — cube level
      push_end:(0.290, 0, 0.067) — max reach at cube height (v9: 14cm push)
    """
    rng = np.random.default_rng(seed)

    home = HOME.copy()
    descend = np.array([0.0, -0.65, 1.51, -0.16, 0.0, 1.0])
    contact = np.array([0.0, -0.15, 1.60, -0.86, 0.0, 1.0])

    # v9: extended push — gripper reaches x=0.29 at cube height
    push_end = np.array([0.0, 0.50, 0.32, -0.11, 0.0, 1.0])
    # Small random perturbation for diversity
    push_end[:4] += rng.uniform(-0.08, 0.08, 4)

    waypoints = [home, descend, contact, push_end, contact, descend, home]

    steps_per_segment = n_steps // (len(waypoints) - 1)
    trajectory = []
    for i in range(len(waypoints) - 1):
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            pose = waypoints[i] * (1 - t) + waypoints[i + 1] * t
            trajectory.append(pose)
    trajectory.append(waypoints[-1])
    return np.array(trajectory)


def main():
    parser = argparse.ArgumentParser(description="Collect v9 push demonstrations")
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--no-push-hub", action="store_true", help="Skip push_to_hub")
    args = parser.parse_args()

    scene_path = find_scene()
    print(f"[v9] Scene: {scene_path}")
    print(f"[v9] Episodes: {args.episodes}, Steps/ep: {N_STEPS}")
    print(f"[v9] Output: {args.output}")

    # MuJoCo setup
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

    # LeRobot dataset
    features = {
        "observation.image": {
            "shape": (3, IMAGE_HEIGHT, IMAGE_WIDTH),
            "dtype": "video",
            "names": ["channel", "height", "width"],
        },
        "observation.state": {
            "shape": (6,),
            "dtype": "float32",
            "names": None,
        },
        "action": {
            "shape": (6,),
            "dtype": "float32",
            "names": None,
        },
    }

    task = "Push cube to target zone with SO-101 arm in MuJoCo simulation (v9 extended push)"

    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        root=args.output,
        use_videos=True,
        image_writer_threads=1,
    )

    for ep in range(args.episodes):
        trajectory = generate_push_trajectory(seed=ep)
        mujoco.mj_resetData(model, data)

        for i in range(len(trajectory) - 1):
            data.ctrl[:6] = trajectory[i]
            for _ in range(10):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            renderer.update_scene(data)
            image = renderer.render()

            state = data.qpos[:6].copy().astype(np.float32)
            action = trajectory[i + 1].copy().astype(np.float32)

            frame = {
                "observation.image": image,
                "observation.state": state,
                "action": action,
                "task": task,
            }
            ds.add_frame(frame)

        ds.save_episode()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[v9] Episode {ep+1}/{args.episodes}: {len(trajectory)-1} frames")

    ds.finalize()
    print(f"[v9] Dataset saved: {ds.num_episodes} episodes, {ds.num_frames} frames")

    # Push to Hub
    if not args.no_push_hub:
        print(f"[v9] Pushing to HuggingFace Hub: {REPO_ID} ...")
        ds.push_to_hub(private=False, tag_version=False)
        print(f"[v9] Push complete")
    else:
        print(f"[v9] Skipping push_to_hub (--no-push-hub)")


if __name__ == "__main__":
    main()
