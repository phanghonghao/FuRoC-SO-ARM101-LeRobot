"""FK Sweep — find the maximum reachable X position for push trajectory.

Scans (shoulder_lift, elbow_flex, wrist_flex) joint space using MuJoCo FK,
looking for gripper positions at Z ≈ 0.065-0.10 (cube/table height) with
maximum X reach.

Usage:
    python scripts/calibration/fk_sweep_push.py
"""

import mujoco
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCENE_PATH = PROJECT_ROOT / "SO-ARM101-LeRobot" / "Simulation" / "SO101" / "scene_push_table.xml"

HOME = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 1.0])

# Gripper end-effector site name in SO-101
GRIPPER_SITE = "gripper"


def forward_kinematics(model, data, qpos_6):
    """Compute gripper position for given 6-DOF joint angles."""
    data.qpos[:6] = qpos_6
    data.ctrl[:6] = qpos_6
    mujoco.mj_forward(model, data)

    # Try site first (more precise), fall back to last body
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, GRIPPER_SITE)
    if site_id >= 0:
        return data.site_xpos[site_id].copy()

    # Fallback: use the last link body
    # SO-101 links are named link1..link6, gripper is typically last
    for name in ["link6", "gripper", "ee"]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return data.xpos[body_id].copy()

    # Last resort: return the last body position
    return data.xpos[-1].copy()


def main():
    print(f"[FK Sweep] Scene: {SCENE_PATH}")
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    # Check available sites/bodies
    print("\nAvailable sites:")
    for i in range(model.nsite):
        print(f"  {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)}")

    print("\nAvailable bodies (last 10):")
    for i in range(max(0, model.nbody - 10), model.nbody):
        print(f"  [{i}] {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)}")

    # Print HOME position FK
    home_pos = forward_kinematics(model, data, HOME)
    print(f"\nHOME FK: gripper at ({home_pos[0]:.4f}, {home_pos[1]:.4f}, {home_pos[2]:.4f})")

    # --- Sweep joint space ---
    # shoulder_lift range: [-1.75, 1.75]
    # elbow_flex range:    [-1.69, 1.69]
    # wrist_flex range:    [-1.66, 1.66]
    # We fix shoulder_pan=0 (forward), wrist_roll=0, gripper=1.0

    RESOLUTION = 80  # points per joint
    shoulder_lift_range = np.linspace(-1.5, 0.5, RESOLUTION)
    elbow_flex_range = np.linspace(0.3, 1.69, RESOLUTION)
    wrist_flex_range = np.linspace(-1.5, 0.5, RESOLUTION)

    Z_MIN = 0.055   # just above table top (Z=0.04)
    Z_MAX = 0.10    # well above cube

    best_x = 0.0
    best_qpos = None
    best_pos = None

    results = []  # (x, z, qpos) for all valid configs

    print(f"\nSweeping {RESOLUTION}^3 = {RESOLUTION**3} configs...")
    print(f"Z filter: [{Z_MIN}, {Z_MAX}]")

    n_valid = 0
    for sl in shoulder_lift_range:
        for ef in elbow_flex_range:
            for wf in wrist_flex_range:
                qpos = np.array([0.0, sl, ef, wf, 0.0, 1.0])
                pos = forward_kinematics(model, data, qpos)

                if Z_MIN <= pos[2] <= Z_MAX and pos[0] > 0.1:
                    n_valid += 1
                    results.append((pos[0], pos[2], qpos.copy(), pos.copy()))

                    if pos[0] > best_x:
                        best_x = pos[0]
                        best_qpos = qpos.copy()
                        best_pos = pos.copy()

    print(f"\nValid configs: {n_valid}")
    results.sort(key=lambda r: r[0], reverse=True)

    if not results:
        print("ERROR: No valid configurations found!")
        return

    print(f"\n{'='*70}")
    print(f"MAX REACH: gripper X = {best_pos[0]:.4f}m at Z = {best_pos[2]:.4f}m")
    print(f"  qpos = [{', '.join(f'{v:.3f}' for v in best_qpos)}]")
    print(f"{'='*70}")

    # Target analysis
    target_x = 0.32
    cube_x = 0.16
    push_needed = target_x - cube_x
    print(f"\nTarget analysis:")
    print(f"  Cube start:   x = {cube_x:.3f}m")
    print(f"  Target zone:  x = {target_x:.3f}m")
    print(f"  Push needed:  {push_needed:.3f}m = {push_needed*100:.1f}cm")
    print(f"  Max gripper X: {best_pos[0]:.3f}m")

    if best_pos[0] >= target_x:
        print(f"  -> REACHABLE! gripper can reach {best_pos[0] - target_x:.3f}m past target")
        print(f"  -> Plan A: single long push")
    elif best_pos[0] >= 0.27:
        print(f"  -> PARTIAL REACH: gripper reaches x={best_pos[0]:.3f} (need x>=0.27 for success)")
        print(f"  -> Plan A feasible: push to x={best_pos[0]:.3f} should be within target radius")
    else:
        print(f"  -> NOT REACHABLE: max x={best_pos[0]:.3f} < target x={target_x:.3f}")
        print(f"  -> Need Plan B: multi-stage push or shorter target")

    # Top 10 configs sorted by X (for waypoint selection)
    print(f"\nTop 20 configs by X reach (Z in [{Z_MIN}, {Z_MAX}]):")
    print(f"  {'X':>8s}  {'Y':>8s}  {'Z':>8s}  {'shoulder_lift':>14s}  {'elbow_flex':>12s}  {'wrist_flex':>12s}")
    for x, z, qpos, pos in results[:20]:
        print(f"  {pos[0]:8.4f}  {pos[1]:8.4f}  {pos[2]:8.4f}  {qpos[1]:14.4f}  {qpos[2]:12.4f}  {qpos[3]:12.4f}")

    # Also find configs near specific X targets with lowest Z (for pushing)
    print(f"\nConfigs near key X positions (lowest Z):")
    for target in [0.20, 0.25, 0.30, 0.32]:
        near = [r for r in results if abs(r[0] - target) < 0.01]
        if near:
            near.sort(key=lambda r: r[1])  # sort by Z (lowest first)
            r = near[0]
            print(f"  x≈{target:.2f}: X={r[3][0]:.4f} Z={r[3][2]:.4f}  "
                  f"qpos=[0, {r[2][1]:.3f}, {r[2][2]:.3f}, {r[2][3]:.3f}, 0, 1.0]")

    # Find "descend" waypoint: directly above cube with low Z
    print(f"\nConfigs above cube (x≈0.16, lowest Z):")
    above = [r for r in results if abs(r[0] - 0.16) < 0.02]
    if above:
        above.sort(key=lambda r: r[1])  # lowest Z first
        for r in above[:5]:
            print(f"  X={r[3][0]:.4f} Z={r[3][2]:.4f}  "
                  f"qpos=[0, {r[2][1]:.3f}, {r[2][2]:.3f}, {r[2][3]:.3f}, 0, 1.0]")

    # Verify current v8 waypoints
    print(f"\n--- V8 waypoint verification ---")
    v8_waypoints = {
        "descend": np.array([0.0, -0.65, 1.51, -0.16, 0.0, 1.0]),
        "contact": np.array([0.0, -0.15, 1.60, -0.86, 0.0, 1.0]),
        "push_target": np.array([0.0, -0.15, 1.10, -0.26, 0.0, 1.0]),
    }
    for name, qpos in v8_waypoints.items():
        pos = forward_kinematics(model, data, qpos)
        print(f"  {name:15s}: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")


if __name__ == "__main__":
    main()
