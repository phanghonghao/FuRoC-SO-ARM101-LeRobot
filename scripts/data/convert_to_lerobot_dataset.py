# convert_to_lerobot_dataset.py
# Converts the SO-101 MuJoCo simulation data collected in outputs/so101_sim_dataset/
# to a proper LeRobot dataset format using LeRobot's native API (v0.5.1).
#
# The original collection script (collect_sim_data.py) only saved images and a
# metadata file -- the per-frame state/action arrays were never persisted.  Because
# the simulation is fully deterministic (fixed RNG seed, fixed XML model), we
# re-simulate here to recover the exact same trajectories and state/action data,
# then feed every frame to LeRobotDataset.add_frame() / save_episode().
#
# Usage:
#   python convert_to_lerobot_dataset.py [--push] [--output-dir OUTPUT_DIR]
#
# Options:
#   --push          Upload the finished dataset to HuggingFace Hub after creation.
#   --output-dir    Local directory for the LeRobot dataset (default: datasets/PhangHongHao/so101_sim)

import argparse
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ---------------------------------------------------------------------------
# Constants (must match collect_sim_data.py exactly)
# ---------------------------------------------------------------------------
SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Simulation", "SO101", "scene.xml")
N_EPISODES = 10
N_STEPS_PER_EPISODE = 100
N_WAYPOINTS = 3
FPS = 30

TASK_DESCRIPTION = "Reach to random target joint configurations and return home"

# LeRobot feature specification
FEATURES = {
    "observation.image": {"dtype": "video", "shape": [3, 480, 640]},
    "observation.state": {"dtype": "float32", "shape": [6]},
    "action": {"dtype": "float32", "shape": [6]},
}

# Repo ID on HuggingFace Hub
HF_REPO_ID = "PhangHongHao/so101_sim"


# ---------------------------------------------------------------------------
# Trajectory generation (copied from collect_sim_data.py for determinism)
# ---------------------------------------------------------------------------
def generate_reaching_trajectory(n_steps=100, n_waypoints=3):
    """Generate a reaching trajectory with linear interpolation between waypoints.

    Uses the same RNG seed (42) as the original collection script so the
    trajectories are bit-identical.
    """
    rng = np.random.default_rng(42)

    joint_ranges = np.array([
        [-1.92, 1.92],  # shoulder_pan
        [-1.75, 1.75],  # shoulder_lift
        [-1.69, 1.69],  # elbow_flex
        [-1.66, 1.66],  # wrist_flex
        [-2.74, 2.84],  # wrist_roll
        [-0.17, 1.75],  # gripper (keep closed)
    ])

    home = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 1.0])

    waypoints = [home.copy()]
    for _ in range(n_waypoints):
        target = np.array([rng.uniform(lo, hi) for lo, hi in joint_ranges])
        target[5] = 1.0  # gripper stays closed
        waypoints.append(target)
    waypoints.append(home.copy())

    steps_per_segment = n_steps // len(waypoints)
    trajectory = []
    for i in range(len(waypoints) - 1):
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            pose = waypoints[i] * (1 - t) + waypoints[i + 1] * t
            trajectory.append(pose)
    trajectory.append(waypoints[-1])

    return np.array(trajectory)


# ---------------------------------------------------------------------------
# Simulation (mirrors collect_sim_data.py exactly)
# ---------------------------------------------------------------------------
def collect_episode(model, data, renderer, trajectory):
    """Run one episode of the trajectory in MuJoCo and return per-frame data."""
    frames = []
    mujoco.mj_resetData(model, data)

    for i in range(len(trajectory) - 1):
        data.ctrl[:6] = trajectory[i]
        for _ in range(10):
            mujoco.mj_step(model, data)
        renderer.update_scene(data)
        image = renderer.render()  # (480, 640, 3) uint8

        state = data.qpos[:6].copy()
        action = trajectory[i + 1].copy()

        frames.append({
            "observation.image": image,
            "observation.state": state.astype(np.float32),
            "action": action.astype(np.float32),
        })

    return frames


# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert SO-101 MuJoCo sim data to LeRobot dataset format"
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push the dataset to HuggingFace Hub after creation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Local directory for the LeRobot dataset. "
        "Defaults to datasets/PhangHongHao/so101_sim",
    )
    args = parser.parse_args()

    # Determine output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("datasets") / HF_REPO_ID
    output_dir = output_dir.resolve()

    # ------------------------------------------------------------------
    # Step 1: Re-simulate to recover the full per-frame data
    # ------------------------------------------------------------------
    print("[1/4] Re-simulating to recover state/action data ...")
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=480, width=640)

    all_episodes = []
    for ep_idx in range(N_EPISODES):
        trajectory = generate_reaching_trajectory(
            n_steps=N_STEPS_PER_EPISODE,
            n_waypoints=N_WAYPOINTS,
        )
        episode_frames = collect_episode(model, data, renderer, trajectory)
        all_episodes.append(episode_frames)
        print(f"  Episode {ep_idx + 1}/{N_EPISODES}: {len(episode_frames)} frames")

    total_frames = sum(len(ep) for ep in all_episodes)
    print(f"  Total frames recovered: {total_frames}")

    # ------------------------------------------------------------------
    # Step 2: Create a new LeRobot dataset with LeRobotDataset.create()
    # ------------------------------------------------------------------
    print(f"[2/4] Creating LeRobot dataset at: {output_dir}")

    # Remove output directory if it already exists (LeRobotDataset.create
    # requires the directory to not exist).
    if output_dir.exists():
        import shutil
        print(f"  Removing existing directory: {output_dir}")
        shutil.rmtree(output_dir)

    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=FPS,
        features=FEATURES,
        root=str(output_dir),
        robot_type="so101",
        use_videos=True,
        # Use h264 codec for broader compatibility on Windows
        vcodec="h264",
    )

    # ------------------------------------------------------------------
    # Step 3: Add frames and save episodes
    # ------------------------------------------------------------------
    print("[3/4] Writing frames to LeRobot dataset ...")
    for ep_idx, episode_frames in enumerate(all_episodes):
        for frame_idx, frame in enumerate(episode_frames):
            # Build the frame dict that add_frame() expects.
            # The "task" key is required -- it provides the natural-language
            # task description for this frame.
            frame_dict = {
                "observation.image": frame["observation.image"],
                "observation.state": frame["observation.state"],
                "action": frame["action"],
                "task": TASK_DESCRIPTION,
            }
            dataset.add_frame(frame_dict)

        # Save the episode after all its frames have been added.
        dataset.save_episode()
        print(f"  Saved episode {ep_idx + 1}/{N_EPISODES}")

    # Finalize: flush parquet writers, encode videos, compute stats.
    print("  Finalizing dataset (encoding videos, writing metadata) ...")
    dataset.finalize()
    print("  Dataset finalized.")

    # Print a summary
    print(f"\n  Dataset summary:")
    print(f"    Repo ID   : {dataset.repo_id}")
    print(f"    Root      : {dataset.root}")
    print(f"    Episodes  : {dataset.num_episodes}")
    print(f"    Frames    : {dataset.num_frames}")
    print(f"    FPS       : {dataset.fps}")
    print(f"    Features  : {list(dataset.features.keys())}")

    # ------------------------------------------------------------------
    # Step 4: (Optional) Push to HuggingFace Hub
    # ------------------------------------------------------------------
    if args.push:
        print(f"\n[4/4] Pushing dataset to HuggingFace Hub as '{HF_REPO_ID}' ...")
        dataset.push_to_hub(
            tags=["lerobot", "so101", "mujoco", "simulation"],
            private=False,
        )
        print(f"  Successfully pushed to: https://huggingface.co/datasets/{HF_REPO_ID}")
    else:
        print("\n[4/4] Skipping push to Hub (use --push to upload).")
        print(f"  Dataset is available locally at: {output_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
