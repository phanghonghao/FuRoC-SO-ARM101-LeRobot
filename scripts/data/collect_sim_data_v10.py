# collect_sim_data_v10.py — SO-101 MuJoCo v10 push 数据采集 (精准停止)
# v9 问题: 推到目标区后继续推 (过推), cube 最终停在 x=0.41 而非 target x=0.32
# v10 改进: 保持 v9 推轨迹不变, 但在运行时检测 cube 位置,
#           当 cube 进入目标区 (dist < 4cm) 后冻结手臂, 示范 "推到就停"
#
# 核心思路: 不改轨迹, 改数据采集循环 — 加 early-stop 规则
#   - 前 N 步: 正常执行 v9 轨迹 (push)
#   - 检测到 cube 进入目标区后: 冻结手臂位置 (hold)
#   - 如果 trajectory 结束也没进入: 正常收回 (和 v9 一样)
#
# Usage:
#   LD_PRELOAD=~/miniconda3/envs/lerobot/lib/libstdc++.so.6 \
#   HF_ENDPOINT=https://hf-mirror.com \
#   python -u scripts/data/collect_sim_data_v10.py
#
#   python -u scripts/data/collect_sim_data_v10.py --episodes 300

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

REPO_ID = "PhangHongHao/so101_push_v10_table_act"
OUTPUT_DIR = "/tmp/so101_push_v10_lerobot"
FPS = 30
N_EPISODES = 300
N_STEPS = 150  # same as v9
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

# Early-stop: when cube enters this radius from target center, freeze arm
STOP_THRESHOLD = 0.04  # 4cm (target zone radius is 5cm, stop a bit early for safety)
# Don't check distance in the first MIN_PUSH_STEPS (let arm approach first)
MIN_PUSH_STEPS = 30

HOME = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 1.0])


def find_scene() -> Path:
    for p in SCENE_SEARCH_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(f"Scene not found, searched: {[str(p) for p in SCENE_SEARCH_PATHS]}")


def generate_push_trajectory(n_steps=N_STEPS, seed=None):
    """Generate v9-style push trajectory (unchanged from v9).

    The early-stop logic is in the data collection loop, not here.
    This function produces the same trajectory as v9.
    """
    rng = np.random.default_rng(seed)

    home = HOME.copy()
    descend = np.array([0.0, -0.65, 1.51, -0.16, 0.0, 1.0])
    contact = np.array([0.0, -0.15, 1.60, -0.86, 0.0, 1.0])

    # Same as v9: extended push — gripper reaches x=0.29 at cube height
    push_end = np.array([0.0, 0.50, 0.32, -0.11, 0.0, 1.0])
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
    parser = argparse.ArgumentParser(description="Collect v10 push demonstrations (early-stop)")
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--no-push-hub", action="store_true", help="Skip push_to_hub")
    parser.add_argument("--stop-threshold", type=float, default=STOP_THRESHOLD,
                        help="Freeze arm when cube distance < threshold (m)")
    args = parser.parse_args()

    scene_path = find_scene()
    print(f"[v10] Scene: {scene_path}")
    print(f"[v10] Episodes: {args.episodes}, Steps/ep: {N_STEPS}")
    print(f"[v10] Output: {args.output}")
    print(f"[v10] Strategy: v9 trajectory + early-stop at {args.stop_threshold}m from target")
    print(f"[v10] Min push steps before checking: {MIN_PUSH_STEPS}")

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

    task = "Push cube to target zone and STOP with SO-101 arm in MuJoCo simulation (v10 early-stop)"

    ds = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=features,
        root=args.output,
        use_videos=True,
        image_writer_threads=1,
    )

    # Detect push object body for distance checking
    push_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "push_object")
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
    target_pos = model.geom_pos[target_geom_id].copy() if target_geom_id >= 0 else None

    n_stopped = 0  # episodes where arm was frozen

    for ep in range(args.episodes):
        trajectory = generate_push_trajectory(seed=ep)
        mujoco.mj_resetData(model, data)

        min_dist = float("inf")
        stopped = False
        hold_position = None
        stop_step = -1

        for i in range(len(trajectory) - 1):
            # Early-stop logic: freeze arm when cube enters target zone
            if not stopped and i >= MIN_PUSH_STEPS and push_body_id >= 0 and target_pos is not None:
                obj_pos = data.xpos[push_body_id].copy()
                dist = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
                if dist < args.stop_threshold:
                    stopped = True
                    hold_position = data.qpos[:6].copy().astype(np.float32)
                    stop_step = i
                    n_stopped += 1

            # Apply control: either follow trajectory or hold position
            if stopped and hold_position is not None:
                data.ctrl[:6] = hold_position
            else:
                data.ctrl[:6] = trajectory[i]

            for _ in range(10):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            renderer.update_scene(data)
            image = renderer.render()

            state = data.qpos[:6].copy().astype(np.float32)
            # Action: next step's control (hold position if stopped)
            if stopped and hold_position is not None:
                action = hold_position.copy()
            else:
                action = trajectory[i + 1].copy().astype(np.float32)

            # Track distance for logging
            if push_body_id >= 0 and target_pos is not None:
                obj_pos = data.xpos[push_body_id].copy()
                dist = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
                min_dist = min(min_dist, dist)

            frame = {
                "observation.image": image,
                "observation.state": state,
                "action": action,
                "task": task,
            }
            ds.add_frame(frame)

        ds.save_episode()
        status = f"STOPPED@step{stop_step}" if stopped else "no-stop"
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"[v10] Episode {ep+1}/{args.episodes}: {len(trajectory)-1} frames, "
                  f"min_dist={min_dist:.4f}m, {status}")

    ds.finalize()
    print(f"\n[v10] Dataset saved: {ds.num_episodes} episodes, {ds.num_frames} frames")
    print(f"[v10] Early-stopped: {n_stopped}/{args.episodes} episodes")

    # Push to Hub
    if not args.no_push_hub:
        print(f"[v10] Pushing to HuggingFace Hub: {REPO_ID} ...")
        ds.push_to_hub(private=False, tag_version=False)
        print(f"[v10] Push complete")
    else:
        print(f"[v10] Skipping push_to_hub (--no-push-hub)")


if __name__ == "__main__":
    main()
