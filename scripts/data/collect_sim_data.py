# collect_sim_data.py — SO-101 MuJoCo 数据采集脚本
# 直接输出 LeRobot 原生数据集格式，可 push_to_hub 上传到 HF Hub
import mujoco
import numpy as np
import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

SCENE_XML = os.path.join(os.path.dirname(__file__), "Simulation", "SO101", "scene.xml")
REPO_ID = "PhangHongHao/so101_sim"
OUTPUT_DIR = "outputs/so101_sim_lerobot"
FPS = 30
N_EPISODES = 10
N_STEPS = 100
N_WAYPOINTS = 3
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

JOINT_RANGES = np.array([
    [-1.92, 1.92],   # shoulder_pan
    [-1.75, 1.75],   # shoulder_lift
    [-1.69, 1.69],   # elbow_flex
    [-1.66, 1.66],   # wrist_flex
    [-2.74, 2.84],   # wrist_roll
    [-0.17, 1.75],   # gripper
])
HOME = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 1.0])


def generate_reaching_trajectory(n_steps=N_STEPS, n_waypoints=N_WAYPOINTS, seed=None):
    rng = np.random.default_rng(seed)
    waypoints = [HOME.copy()]
    for _ in range(n_waypoints):
        target = np.array([rng.uniform(lo, hi) for lo, hi in JOINT_RANGES])
        target[5] = 1.0  # gripper closed
        waypoints.append(target)
    waypoints.append(HOME.copy())

    steps_per_segment = n_steps // len(waypoints)
    trajectory = []
    for i in range(len(waypoints) - 1):
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            pose = waypoints[i] * (1 - t) + waypoints[i + 1] * t
            trajectory.append(pose)
    trajectory.append(waypoints[-1])
    return np.array(trajectory)


def main():
    # --- MuJoCo setup ---
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

    # --- LeRobot dataset ---
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

    task = "Reach random target positions with SO-101 arm in MuJoCo simulation"

    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        root=OUTPUT_DIR,
        use_videos=True,
        image_writer_threads=1,
    )

    for ep in range(N_EPISODES):
        trajectory = generate_reaching_trajectory(seed=ep)
        mujoco.mj_resetData(model, data)

        for i in range(len(trajectory) - 1):
            data.ctrl[:6] = trajectory[i]
            for _ in range(10):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            renderer.update_scene(data)
            image = renderer.render()  # (H, W, 3) uint8

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
        print(f"Episode {ep+1}/{N_EPISODES}: {len(trajectory)-1} frames saved")

    ds.finalize()
    print(f"[OK] Dataset saved to {OUTPUT_DIR}")
    print(f"  Episodes: {ds.num_episodes}")
    print(f"  Frames:   {ds.num_frames}")

    # --- Push to Hub ---
    print("Pushing to HuggingFace Hub...")
    ds.push_to_hub(private=False, tag_version=False)
    print(f"[OK] Dataset pushed to {REPO_ID}")


if __name__ == "__main__":
    main()
