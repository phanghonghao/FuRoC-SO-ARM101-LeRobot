"""Local eval for ACT v11 — random position push task.

v11 dataset: so101_push_v11_random_pos_act (500 eps, 75K frames)
  - Cube:   X 0.10-0.22, Y ±0.04
  - Target: X 0.26-0.38, Y ±0.04

Usage:
    python scripts/eval/local_eval_v11.py --best
    python scripts/eval/local_eval_v11.py --best --episodes 10 --csv outputs/eval_results/v11.csv
    python scripts/eval/local_eval_v11.py --checkpoint outputs/checkpoints/act_v11_025000/pretrained_model
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCENE_DIR = PROJECT_ROOT / "SO-ARM101-LeRobot" / "Simulation" / "SO101"

SIM_STEPS_PER_FRAME = 10
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

# v11 random position ranges (from training plan rtx_train_plan_v11.yaml)
CUBE_X_RANGE = (0.10, 0.22)
CUBE_Y_RANGE = (-0.04, 0.04)
CUBE_Z = 0.065
TARGET_X_RANGE = (0.26, 0.38)
TARGET_Y_RANGE = (-0.04, 0.04)
TARGET_Z = 0.041

HOME_QPOS = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 1.0])


def find_scene(scene_name: str) -> Path:
    candidates = [Path(scene_name), PROJECT_ROOT / scene_name, SCENE_DIR / scene_name]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Scene not found: {scene_name}")


def find_v11_checkpoint(checkpoint_arg: str | None, best: bool) -> Path:
    ckpt_base = PROJECT_ROOT / "outputs" / "checkpoints"
    if not best and checkpoint_arg:
        p = Path(checkpoint_arg)
        if not p.exists():
            p = ckpt_base / checkpoint_arg
        if not p.exists():
            p = ckpt_base / checkpoint_arg / "pretrained_model"
        if p.exists():
            return p
        raise FileNotFoundError(f"Not found: {checkpoint_arg}")

    # --best: find highest act_v11_* checkpoint
    best_path = None
    best_step = -1
    for d in ckpt_base.iterdir():
        pretrained = d / "pretrained_model"
        if not pretrained.exists():
            continue
        m = re.match(r"act_v11(?:_\w+)?_(\d+)", d.name)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_path = pretrained
    if best_path is None:
        raise FileNotFoundError(f"No act_v11_* checkpoints in {ckpt_base}")
    return best_path


def load_policy(checkpoint_path: Path, device: str = "cpu"):
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(str(checkpoint_path))
    policy.to(device)
    policy.eval()
    return policy


def randomize_positions(model, data, push_body_id, target_geom_id):
    """Randomize cube and target positions, return (cube_pos, target_pos)."""
    cube_x = np.random.uniform(*CUBE_X_RANGE)
    cube_y = np.random.uniform(*CUBE_Y_RANGE)
    target_x = np.random.uniform(*TARGET_X_RANGE)
    target_y = np.random.uniform(*TARGET_Y_RANGE)

    # Update model positions (persistent across mj_forward)
    model.body_pos[push_body_id] = [cube_x, cube_y, CUBE_Z]
    model.geom_pos[target_geom_id] = [target_x, target_y, TARGET_Z]

    # Update freejoint qpos
    qpos_adr = model.jnt_qposadr[model.body_jntadr[push_body_id]]
    data.qpos[qpos_adr:qpos_adr + 3] = [cube_x, cube_y, CUBE_Z]
    data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
    qvel_adr = model.jnt_dofadr[model.body_jntadr[push_body_id]]
    data.qvel[qvel_adr:qvel_adr + 6] = 0

    return np.array([cube_x, cube_y]), np.array([target_x, target_y])


def main():
    parser = argparse.ArgumentParser(description="Local eval for ACT v11 (random position push)")
    parser.add_argument("--checkpoint", default=None, help="Path to v11 checkpoint")
    parser.add_argument("--best", action="store_true", help="Use best v11 checkpoint")
    parser.add_argument("--scene", default="scene_push_table.xml", help="Base scene XML")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--device", default="cpu", help="Inference device")
    parser.add_argument("--csv", default=None, help="CSV output path")
    parser.add_argument("--record", default=None, help="MP4 output path")
    parser.add_argument("--success-threshold", type=float, default=0.05, help="Dist < this = success (m)")
    args = parser.parse_args()

    if not args.checkpoint and not args.best:
        args.best = True  # default to best v11 checkpoint

    ckpt_path = find_v11_checkpoint(args.checkpoint, args.best)
    scene_path = find_scene(args.scene)

    print(f"[v11_eval] Checkpoint: {ckpt_path}")
    print(f"[v11_eval] Scene: {scene_path}")
    print(f"[v11_eval] Mode: random positions")
    print(f"[v11_eval]   cube   X:[{CUBE_X_RANGE[0]}, {CUBE_X_RANGE[1]}] Y:[{CUBE_Y_RANGE[0]}, {CUBE_Y_RANGE[1]}]")
    print(f"[v11_eval]   target X:[{TARGET_X_RANGE[0]}, {TARGET_X_RANGE[1]}] Y:[{TARGET_Y_RANGE[0]}, {TARGET_Y_RANGE[1]}]")

    policy = load_policy(ckpt_path, device=args.device)
    print(f"[v11_eval] Policy loaded: {type(policy).__name__}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

    push_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "push_object")
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
    assert push_body_id >= 0, "push_object body not found"
    assert target_geom_id >= 0, "target_zone geom not found"

    # CSV
    csv_file = None
    csv_writer = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["episode", "step",
                             "qpos_0", "qpos_1", "qpos_2", "qpos_3", "qpos_4", "qpos_5",
                             "action_0", "action_1", "action_2", "action_3", "action_4", "action_5",
                             "obj_x", "obj_y", "obj_z", "target_x", "target_y", "dist_to_target"])
        print(f"[v11_eval] CSV: {csv_path}")

    # Video
    video_frames = []
    if args.record:
        print(f"[v11_eval] Recording: {args.record}")

    # Viewer
    quit_requested = False

    def key_callback(keycode):
        nonlocal quit_requested
        if keycode in (ord("Q"), ord("q")):
            quit_requested = True

    try:
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
    except Exception:
        viewer = None

    results = []

    try:
        for ep in range(args.episodes):
            if quit_requested or (viewer is not None and not viewer.is_running()):
                break

            mujoco.mj_resetData(model, data)
            data.qpos[:6] = HOME_QPOS
            cube_pos, target_pos = randomize_positions(model, data, push_body_id, target_geom_id)
            mujoco.mj_forward(model, data)
            policy.reset()

            min_dist = float("inf")
            final_dist = float("inf")

            print(f"Ep {ep+1}/{args.episodes}: cube=({cube_pos[0]:.3f},{cube_pos[1]:.3f}) "
                  f"target=({target_pos[0]:.3f},{target_pos[1]:.3f}) dist={np.linalg.norm(cube_pos - target_pos):.3f}m")

            for step in range(args.max_steps):
                if quit_requested or (viewer is not None and not viewer.is_running()):
                    break

                renderer.update_scene(data)
                image = renderer.render()

                if args.record and step % 3 == 0:
                    video_frames.append(image.copy())

                state = data.qpos[:6].copy().astype(np.float32)
                image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
                batch = {
                    "observation.state": torch.tensor(state).unsqueeze(0).to(args.device),
                    "observation.image": image_tensor.to(args.device),
                }

                with torch.no_grad():
                    action = policy.select_action(batch)
                action_np = action.cpu().numpy().squeeze(0)

                data.ctrl[:6] = action_np
                for _ in range(SIM_STEPS_PER_FRAME):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)

                obj_pos = data.xpos[push_body_id].copy()
                dist = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
                min_dist = min(min_dist, dist)
                final_dist = dist

                if csv_writer is not None:
                    csv_writer.writerow(
                        [ep, step] + state.tolist() + action_np.tolist()
                        + [obj_pos[0], obj_pos[1], obj_pos[2], target_pos[0], target_pos[1], dist]
                    )

                if step % 100 == 0:
                    print(f"  step {step:3d}: dist={dist:.4f}m")

                if viewer is not None:
                    try:
                        viewer.sync()
                    except Exception:
                        break
                time.sleep(0.01)

            success = final_dist < args.success_threshold
            results.append({"min_dist": min_dist, "final_dist": final_dist, "success": success})
            print(f"  -> dist final={final_dist:.4f} min={min_dist:.4f} "
                  f"{'SUCCESS' if success else 'FAIL'}")

    finally:
        if viewer:
            try:
                viewer.close()
            except Exception:
                pass
        if csv_file:
            csv_file.close()
            print(f"[v11_eval] CSV saved: {args.csv}")
        if args.record and video_frames:
            import imageio
            Path(args.record).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(args.record, video_frames, fps=10)
            print(f"[v11_eval] Video saved: {args.record} ({len(video_frames)} frames)")

    # Summary
    n = len(results)
    n_success = sum(r["success"] for r in results)
    avg_final = np.mean([r["final_dist"] for r in results])
    avg_min = np.mean([r["min_dist"] for r in results])
    print(f"\n=== v11 Eval Summary ===")
    print(f"  Checkpoint: {ckpt_path.parent.name}")
    print(f"  Episodes:   {n}")
    print(f"  Success:    {n_success}/{n} ({100*n_success/max(n,1):.0f}%)  (threshold < {args.success_threshold}m)")
    print(f"  Avg dist:   {avg_final:.4f}m (final) / {avg_min:.4f}m (min)")
    min_dists = [f"{r['min_dist']:.4f}" for r in results]
    print(f"  Per-episode min dist: {min_dists}")


if __name__ == "__main__":
    main()
