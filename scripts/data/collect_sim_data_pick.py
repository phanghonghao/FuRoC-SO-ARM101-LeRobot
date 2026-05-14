# collect_sim_data_pick.py — SO-101 MuJoCo T3 pick 数据采集
# 基于 workspace check 验证的 waypoints:
#   home → above → descend → grasp → lift → carry → release → retreat
#
# Usage:
#   LD_PRELOAD=~/miniconda3/envs/lerobot/lib/libstdc++.so.6 \
#   HF_ENDPOINT=https://hf-mirror.com \
#   python -u scripts/data/collect_sim_data_pick.py
#
#   python -u scripts/data/collect_sim_data_pick.py --episodes 300

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SCENE_SEARCH_PATHS = [
    Path.home() / "SO-ARM101-LeRobot" / "Simulation" / "SO101" / "scene_v3_pick.xml",
    Path("SO-ARM101-LeRobot/Simulation/SO101/scene_v3_pick.xml"),
    Path("Simulation/SO101/scene_v3_pick.xml"),
    Path(__file__).parent.parent.parent / "SO-ARM101-LeRobot" / "Simulation" / "SO101" / "scene_v3_pick.xml",
]

REPO_ID = "PhangHongHao/so101_pick_v3"
OUTPUT_DIR = "/tmp/so101_pick_v3_lerobot"
FPS = 30
N_EPISODES = 300
N_STEPS = 160  # 8 waypoints × 20 steps each
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

GRIPPER_OPEN = -0.17
GRIPPER_CLOSED = 1.0

# Waypoints from workspace validation (check_workspace_pick.py)
# Gripper body FK verified:
#   home:    (0.222, 0, 0.169)
#   above:   (0.160, 0, 0.110)
#   descend: (0.160, 0, 0.065)
#   lift:    (0.174, 0, 0.118)
#   carry:   (0.250, 0, 0.244)
PICK_WAYPOINTS = {
    "home":    np.array([0.0, -0.5, 1.0, -0.5, 0.0, GRIPPER_OPEN]),
    "above":   np.array([0.0, -0.4873, 1.6548, -1.2215, 0.0, GRIPPER_OPEN]),
    "descend": np.array([0.0, 0.2975, 1.5141, -1.5000, 0.0, GRIPPER_OPEN]),
    "grasp":   np.array([0.0, 0.2975, 1.5141, -1.5000, 0.0, GRIPPER_CLOSED]),
    "lift":    np.array([0.0, -0.2359, 1.5141, -1.5000, 0.0, GRIPPER_CLOSED]),
    "carry":   np.array([0.0, -0.3692, 0.3000, 0.1053, 0.0, GRIPPER_CLOSED]),
    "release": np.array([0.0, -0.3692, 0.3000, 0.1053, 0.0, GRIPPER_OPEN]),
    "retreat": np.array([0.0, -0.5, 1.0, -0.5, 0.0, GRIPPER_OPEN]),
}


def find_scene() -> Path:
    for p in SCENE_SEARCH_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(f"Scene not found, searched: {[str(p) for p in SCENE_SEARCH_PATHS]}")


def generate_pick_trajectory(n_steps=N_STEPS, seed=None):
    """Generate pick trajectory with randomization."""
    rng = np.random.default_rng(seed)
    wp = PICK_WAYPOINTS

    # Randomize approach (±2cm in Y)
    approach_y = rng.uniform(-0.02, 0.02)
    # Randomize target placement (±3cm in XY)
    target_dx = rng.uniform(-0.03, 0.03)
    target_dy = rng.uniform(-0.03, 0.03)

    above = wp["above"].copy()
    above[1] = approach_y

    descend = wp["descend"].copy()
    descend[1] = approach_y

    grasp = wp["grasp"].copy()
    grasp[1] = approach_y

    lift = wp["lift"].copy()
    lift[1] = approach_y

    carry = wp["carry"].copy()
    carry[0] += target_dx
    carry[1] = target_dy

    release = carry.copy()
    release[5] = GRIPPER_OPEN

    waypoints = [wp["home"], above, descend, grasp, lift, carry, release, wp["retreat"]]

    # Interpolate
    steps_per_segment = n_steps // (len(waypoints) - 1)
    trajectory = []
    for i in range(len(waypoints) - 1):
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            pose = waypoints[i] * (1 - t) + waypoints[i + 1] * t
            trajectory.append(pose)
    trajectory.append(waypoints[-1])
    return np.array(trajectory)


def main():
    parser = argparse.ArgumentParser(description="Collect T3 pick demonstrations")
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--no-push-hub", action="store_true")
    args = parser.parse_args()

    scene_path = find_scene()
    print(f"[pick] Scene: {scene_path}")
    print(f"[pick] Episodes: {args.episodes}, Steps/ep: {N_STEPS}")
    print(f"[pick] Output: {args.output}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

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

    task = "pick up the red block"

    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        root=args.output,
        use_videos=True,
        image_writer_threads=1,
    )

    # Track pick object
    pick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pick_object")
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
    target_pos = model.geom_pos[target_geom_id].copy() if target_geom_id >= 0 else None

    n_success = 0

    for ep in range(args.episodes):
        trajectory = generate_pick_trajectory(seed=ep)
        mujoco.mj_resetData(model, data)

        initial_block_z = data.xpos[pick_body_id][2] if pick_body_id >= 0 else 0

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

        # Check if block was successfully placed near target
        if pick_body_id >= 0 and target_pos is not None:
            block_pos = data.xpos[pick_body_id].copy()
            dist = float(np.linalg.norm(block_pos[:2] - target_pos[:2]))
            lifted = block_pos[2] > initial_block_z + 0.03
            if dist < 0.05 or lifted:
                n_success += 1

        ds.save_episode()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[pick] Episode {ep+1}/{args.episodes}: {len(trajectory)-1} frames")

    ds.finalize()
    print(f"\n[pick] Dataset saved: {ds.num_episodes} episodes, {ds.num_frames} frames")
    print(f"[pick] Estimated success: {n_success}/{args.episodes}")

    if not args.no_push_hub:
        print(f"[pick] Pushing to HuggingFace Hub: {REPO_ID} ...")
        ds.push_to_hub(private=False, tag_version=False)
        print(f"[pick] Push complete")
    else:
        print(f"[pick] Skipping push_to_hub (--no-push-hub)")


if __name__ == "__main__":
    main()
