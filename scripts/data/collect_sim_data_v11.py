# collect_sim_data_v11.py — SO-101 MuJoCo v11 push 数据采集 (随机位置 + IK 自适应轨迹)
# v9r2 问题: 0% eval 成功率, 方块位置/目标位置/轨迹全部固定, policy 记忆而非泛化
# v10: 加了 early-stop, 但位置仍然固定
# v11 改进: 随机化方块和目标位置, 用 Jacobian IK 自适应生成轨迹
#
# 核心改动 vs v10:
#   1. 方块位置: 固定 (0.16, 0, 0.065) → 随机 X: 0.10-0.22, Y: ±0.04
#   2. 目标位置: 固定 (0.32, 0, 0.041) → 随机 X: 0.26-0.38, Y: ±0.04
#   3. 轨迹生成: 固定 joint waypoints → IK 自适应 (基于位置)
#   4. Episode 数: 300 → 500 (覆盖位置空间)
#
# Usage:
#   LD_PRELOAD=~/miniconda3/envs/lerobot/lib/libstdc++.so.6 \
#   HF_ENDPOINT=https://hf-mirror.com \
#   python -u scripts/data/collect_sim_data_v11.py
#
#   python -u scripts/data/collect_sim_data_v11.py --episodes 500

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

REPO_ID = "PhangHongHao/so101_push_v11_random_pos_act"
OUTPUT_DIR = "/tmp/so101_push_v11_lerobot"
FPS = 30
N_EPISODES = 500
N_STEPS = 150
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

# Early-stop: when cube enters this radius from target center, freeze arm
STOP_THRESHOLD = 0.04  # 4cm
MIN_PUSH_STEPS = 30

# Random position ranges (tuned for SO-101 arm reachability)
CUBE_X_RANGE = (0.10, 0.22)
CUBE_Y_RANGE = (-0.04, 0.04)
TARGET_X_RANGE = (0.24, 0.34)  # max 0.34 to keep push_end within arm reach
TARGET_Y_RANGE = (-0.04, 0.04)
MIN_PUSH_DIST = 0.08  # cube to target minimum distance

# IK solver — use gripper body (not site) since gripperframe has 9.8cm offset
GRIPPER_BODY_NAME = "gripper"  # MuJoCo body name for end-effector

HOME = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 1.0])


def find_scene() -> Path:
    for p in SCENE_SEARCH_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(f"Scene not found, searched: {[str(p) for p in SCENE_SEARCH_PATHS]}")


def solve_ik(model, data, target_xyz, init_qpos, max_iter=100, tol=0.002):
    """Jacobian pseudoinverse IK for 6-DOF arm using gripper body."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, GRIPPER_BODY_NAME)
    data.qpos[:6] = init_qpos.copy()
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        current = data.xpos[body_id].copy()
        error = target_xyz - current
        if np.linalg.norm(error) < tol:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        # Jacobian pseudoinverse with damping
        J = jacp[:, :6]
        delta_q = J.T @ np.linalg.solve(J @ J.T + 0.01 * np.eye(3), error)
        data.qpos[:6] = np.clip(data.qpos[:6] + delta_q,
                                 model.jnt_range[:6, 0], model.jnt_range[:6, 1])
    mujoco.mj_forward(model, data)
    final_error = np.linalg.norm(target_xyz - data.xpos[body_id])
    return data.qpos[:6].copy(), final_error


def randomize_positions(rng, model):
    """Randomize cube and target positions, return them."""
    while True:
        cube_x = rng.uniform(*CUBE_X_RANGE)
        cube_y = rng.uniform(*CUBE_Y_RANGE)
        target_x = rng.uniform(*TARGET_X_RANGE)
        target_y = rng.uniform(*TARGET_Y_RANGE)
        # Ensure target is in front of cube and distance is sufficient
        if target_x - cube_x >= MIN_PUSH_DIST:
            break

    # Set target position (geom_pos — static, persists across mj_resetData)
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
    target_z = 0.041  # Fixed Z (table height + thin marker)
    model.geom_pos[target_geom_id] = [target_x, target_y, target_z]

    cube_z = 0.065  # Fixed Z (table height + half cube height)
    return np.array([cube_x, cube_y, cube_z]), np.array([target_x, target_y, target_z])


def reset_cube_position(model, data, cube_pos):
    """Set cube body position in data.qpos after mj_resetData.

    model.body_pos changes don't survive mj_resetData for bodies with free joints.
    Must set data.qpos directly (same approach as local_play.py).
    """
    push_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "push_object")
    if push_body_id < 0:
        return
    qpos_adr = model.jnt_qposadr[model.body_jntadr[push_body_id]]
    data.qpos[qpos_adr:qpos_adr + 3] = cube_pos
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]  # identity quaternion
    qvel_adr = model.jnt_dofadr[model.body_jntadr[push_body_id]]
    data.qvel[qvel_adr:qvel_adr + 6] = 0


def generate_push_trajectory_ik(model, data, cube_pos, target_pos, n_steps=N_STEPS, seed=None):
    """Generate push trajectory adapted to random cube/target positions using IK."""
    rng = np.random.default_rng(seed)

    home = HOME.copy()

    # IK solve contact waypoint: gripper to 2cm behind cube
    contact_xyz = cube_pos + np.array([-0.02, 0, 0])  # approach from -X direction
    contact_qpos, err1 = solve_ik(model, data, contact_xyz, home)

    # IK solve push_end waypoint: gripper pushes past cube toward target
    push_dir = (target_pos - cube_pos)
    push_dir[2] = 0  # keep in horizontal plane
    push_dir = push_dir / np.linalg.norm(push_dir)
    overshoot = 0.01  # push 1cm past target (arm reach limited at x>0.35)
    push_end_xyz = target_pos + push_dir * overshoot
    push_end_xyz[2] = cube_pos[2] - 0.005  # slightly below cube center
    push_end_qpos, err2 = solve_ik(model, data, push_end_xyz, contact_qpos)

    # Add noise (consistent with v10)
    push_end_qpos[:4] += rng.uniform(-0.05, 0.05, 4)

    # IK solve descend waypoint: above and behind cube (smooth approach)
    descend_xyz = cube_pos + np.array([-0.02, 0, 0.04])  # 4cm above contact
    descend_qpos, _ = solve_ik(model, data, descend_xyz, home)

    # IK solve retract waypoint: same as descend (for smooth return)
    retract_qpos, _ = solve_ik(model, data, descend_xyz, push_end_qpos)

    waypoints = [home, descend_qpos, contact_qpos, push_end_qpos, contact_qpos, retract_qpos, home]

    steps_per_segment = n_steps // (len(waypoints) - 1)
    trajectory = []
    for i in range(len(waypoints) - 1):
        for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
            pose = waypoints[i] * (1 - t) + waypoints[i + 1] * t
            trajectory.append(pose)
    trajectory.append(waypoints[-1])

    return np.array(trajectory), err1, err2


def main():
    parser = argparse.ArgumentParser(description="Collect v11 push demonstrations (random positions + IK)")
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--no-push-hub", action="store_true", help="Skip push_to_hub")
    parser.add_argument("--stop-threshold", type=float, default=STOP_THRESHOLD,
                        help="Freeze arm when cube distance < threshold (m)")
    args = parser.parse_args()

    scene_path = find_scene()
    print(f"[v11] Scene: {scene_path}")
    print(f"[v11] Episodes: {args.episodes}, Steps/ep: {N_STEPS}")
    print(f"[v11] Output: {args.output}")
    print(f"[v11] Strategy: random positions + IK adaptive trajectory + early-stop at {args.stop_threshold}m")
    print(f"[v11] Cube range: X={CUBE_X_RANGE}, Y={CUBE_Y_RANGE}")
    print(f"[v11] Target range: X={TARGET_X_RANGE}, Y={TARGET_Y_RANGE}")
    print(f"[v11] Min push distance: {MIN_PUSH_DIST}m")

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

    task = "Push cube to target zone and STOP with SO-101 arm in MuJoCo simulation (v11 random positions + IK)"

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

    rng = np.random.default_rng(42)
    n_stopped = 0
    ik_errors = []

    for ep in range(args.episodes):
        # Randomize cube and target positions for this episode
        cube_pos, target_pos = randomize_positions(rng, model)

        # Reset simulation with new positions
        mujoco.mj_resetData(model, data)
        reset_cube_position(model, data, cube_pos)

        # Generate IK-adaptive trajectory
        trajectory, err1, err2 = generate_push_trajectory_ik(
            model, data, cube_pos, target_pos, n_steps=N_STEPS, seed=ep
        )
        ik_errors.append((err1, err2))

        # Reset again for actual data collection
        mujoco.mj_resetData(model, data)
        reset_cube_position(model, data, cube_pos)

        min_dist = float("inf")
        stopped = False
        hold_position = None
        stop_step = -1

        for i in range(len(trajectory) - 1):
            # Early-stop logic: freeze arm when cube enters target zone
            if not stopped and i >= MIN_PUSH_STEPS and push_body_id >= 0 and target_geom_id >= 0:
                obj_pos = data.xpos[push_body_id].copy()
                current_target = model.geom_pos[target_geom_id].copy()
                dist = float(np.linalg.norm(obj_pos[:2] - current_target[:2]))
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
            if push_body_id >= 0 and target_geom_id >= 0:
                obj_pos = data.xpos[push_body_id].copy()
                current_target = model.geom_pos[target_geom_id].copy()
                dist = float(np.linalg.norm(obj_pos[:2] - current_target[:2]))
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
            print(f"[v11] Episode {ep+1}/{args.episodes}: {len(trajectory)-1} frames, "
                  f"cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f}), "
                  f"target=({target_pos[0]:.3f},{target_pos[1]:.3f}), "
                  f"IK_err=({err1:.4f},{err2:.4f}), "
                  f"min_dist={min_dist:.4f}m, {status}")

    ds.finalize()
    avg_err1 = np.mean([e[0] for e in ik_errors])
    avg_err2 = np.mean([e[1] for e in ik_errors])
    print(f"\n[v11] Dataset saved: {ds.num_episodes} episodes, {ds.num_frames} frames")
    print(f"[v11] Early-stopped: {n_stopped}/{args.episodes} episodes")
    print(f"[v11] IK errors: contact={avg_err1:.4f}m, push_end={avg_err2:.4f}m (avg)")

    # Push to Hub
    if not args.no_push_hub:
        print(f"[v11] Pushing to HuggingFace Hub: {REPO_ID} ...")
        ds.push_to_hub(private=False, tag_version=False)
        print(f"[v11] Push complete")
    else:
        print(f"[v11] Skipping push_to_hub (--no-push-hub)")


if __name__ == "__main__":
    main()
