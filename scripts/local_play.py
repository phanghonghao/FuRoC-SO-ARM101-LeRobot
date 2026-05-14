"""Local interactive MuJoCo viewer for SO-ARM101 with trained ACT policy.

Launches a MuJoCo window with a trained policy running inference on CPU.
Supports push task (cube + target zone) or default scene.

Usage:
    python scripts/local_play.py --push --best
    python scripts/local_play.py --push --best --csv logs/play.csv
    python scripts/local_play.py --push --best --record outputs/eval_videos/play.mp4
    python scripts/local_play.py --push --checkpoint outputs/checkpoints/act_v8_040000/pretrained_model

Keys:
    R     - Reset episode
    Q/Esc - Quit
"""

import argparse
import csv
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer  # must import explicitly on Windows
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
SCENE_DIR = PROJECT_ROOT / "SO-ARM101-LeRobot" / "Simulation" / "SO101"

SIM_STEPS_PER_FRAME = 10
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640


def find_scene(scene_name: str) -> Path:
    """Resolve scene XML path, searching multiple locations."""
    candidates = [
        Path(scene_name),
        PROJECT_ROOT / scene_name,
        SCENE_DIR / scene_name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Scene not found: {scene_name} (searched {candidates})")


def _parse_ckpt_name(name: str):
    """Parse checkpoint dir name like 'act_v8_040000' or 'act_v9_r2_020000' -> (version, step)."""
    m = re.match(r"act_v(\d+)(?:_r\d+)?_(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def find_checkpoint(checkpoint_arg: str, best: bool = False) -> Path:
    """Resolve checkpoint directory."""
    if not best:
        p = Path(checkpoint_arg)
        if p.exists():
            return p
        p = PROJECT_ROOT / checkpoint_arg
        if p.exists():
            return p
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_arg}")

    # --best: find highest (version, step) checkpoint
    ckpt_base = PROJECT_ROOT / "outputs" / "checkpoints"
    if not ckpt_base.exists():
        raise FileNotFoundError(f"No checkpoints directory: {ckpt_base}")

    best_path = None
    best_key = (-1, -1)

    for d in ckpt_base.iterdir():
        pretrained = d / "pretrained_model"
        if not pretrained.exists():
            continue
        if d.name.startswith("act_"):
            version, step = _parse_ckpt_name(d.name)
            key = (version, step)
            if key > best_key:
                best_key = key
                best_path = pretrained
        else:
            try:
                step = int(d.name)
                key = (0, step)
                if key > best_key:
                    best_key = key
                    best_path = pretrained
            except ValueError:
                continue

    if best_path is None:
        raise FileNotFoundError(f"No checkpoints with pretrained_model/ found in {ckpt_base}")
    return best_path


def load_policy(checkpoint_path: Path, device: str = "cpu"):
    """Load ACT policy from checkpoint."""
    try:
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(str(checkpoint_path))
        policy.to(device)
        policy.eval()
        return policy
    except Exception:
        pass
    try:
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        policy = DiffusionPolicy.from_pretrained(str(checkpoint_path))
        policy.to(device)
        policy.eval()
        return policy
    except Exception:
        pass
    raise RuntimeError(f"Could not load policy from {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Local MuJoCo interactive viewer with ACT policy")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint directory")
    parser.add_argument("--best", action="store_true", help="Use best available checkpoint")
    parser.add_argument("--push", action="store_true", help="Push task mode (cube + target)")
    parser.add_argument("--scene", default=None, help="MuJoCo scene XML path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--device", default="cpu", help="Device for inference")
    parser.add_argument("--csv", default=None, help="CSV output path for per-step data")
    parser.add_argument("--record", default=None, help="MP4 output path for video recording")
    parser.add_argument("--stop-threshold", type=float, default=0.0,
                        help="Stop pushing when dist < threshold (m). 0=disabled (default)")
    args = parser.parse_args()

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = find_checkpoint(args.checkpoint)
    elif args.best:
        ckpt_path = find_checkpoint(None, best=True)
    else:
        parser.error("Must specify --checkpoint <path> or --best")

    print(f"[local_play] Checkpoint: {ckpt_path}")

    # Resolve scene: use scene_push_table.xml by default for --push (matches training data)
    if args.scene:
        scene_path = find_scene(args.scene)
    elif args.push:
        scene_path = find_scene("scene_push_table.xml")
    else:
        scene_path = find_scene("scene.xml")

    print(f"[local_play] Scene: {scene_path}")

    # Load policy
    print(f"[local_play] Loading policy on {args.device}...")
    policy = load_policy(ckpt_path, device=args.device)
    print(f"[local_play] Policy loaded: {type(policy).__name__}")

    # Setup MuJoCo
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

    # Detect push object
    push_body_id = None
    target_pos = None
    push_start_pos = None
    if args.push:
        push_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "push_object")
        if push_body_id >= 0:
            push_start_pos = model.body_pos[push_body_id].copy()
            print(f"[local_play] Push object start: {push_start_pos}")
            target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
            if target_geom_id >= 0:
                target_pos = model.geom_pos[target_geom_id].copy()
                print(f"[local_play] Target zone: {target_pos}")
        else:
            print("[local_play] Warning: push_object body not found in scene")

    # CSV setup
    csv_file = None
    csv_writer = None
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["episode", "step", "qpos_0", "qpos_1", "qpos_2", "qpos_3", "qpos_4", "qpos_5",
                             "action_0", "action_1", "action_2", "action_3", "action_4", "action_5",
                             "obj_x", "obj_y", "obj_z", "dist_to_target"])
        print(f"[local_play] CSV: {csv_path}")

    # Video recording setup
    video_frames = []
    if args.record:
        print(f"[local_play] Recording to: {args.record}")

    # Key callback
    reset_requested = False
    quit_requested = False

    def key_callback(keycode):
        nonlocal reset_requested, quit_requested
        if keycode == ord("R") or keycode == ord("r"):
            reset_requested = True
        elif keycode == ord("Q") or keycode == ord("q"):
            quit_requested = True

    # Launch viewer
    print(f"\n[local_play] Launching viewer...")
    print(f"[local_play] Episodes: {args.episodes}, Max steps: {args.max_steps}")
    print(f"[local_play] Keys: R=reset episode, Q=quit, Esc=close window\n")

    try:
        viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)
        print("[local_play] Viewer launched successfully")
    except Exception as e:
        print(f"[WARNING] Could not launch viewer: {e}")
        print("[INFO] Running headless mode")
        viewer = None

    # Summary stats
    all_results = []

    try:
        for ep in range(args.episodes):
            # Reset — set arm to HOME (training data always starts from HOME, not zeros)
            mujoco.mj_resetData(model, data)
            HOME = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 1.0])
            data.qpos[:6] = HOME
            if push_body_id is not None and push_body_id >= 0 and push_start_pos is not None:
                qpos_adr = model.jnt_qposadr[model.body_jntadr[push_body_id]]
                data.qpos[qpos_adr:qpos_adr + 3] = push_start_pos
                data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
                qvel_adr = model.jnt_dofadr[model.body_jntadr[push_body_id]]
                data.qvel[qvel_adr:qvel_adr + 6] = 0

            mujoco.mj_forward(model, data)
            policy.reset()

            final_distance = float("inf")
            min_distance = float("inf")
            step_count = 0
            stopped = False  # freeze arm after reaching target

            print(f"=== Episode {ep + 1}/{args.episodes} ===")

            for step in range(args.max_steps):
                if quit_requested:
                    break
                if viewer is not None and not viewer.is_running():
                    break
                if reset_requested:
                    reset_requested = False
                    print("  [R pressed] Resetting episode...")
                    break

                # Render observation
                renderer.update_scene(data)
                image = renderer.render()

                # Record frame (every 3rd step to keep video size reasonable)
                if args.record and step % 3 == 0:
                    video_frames.append(image.copy())

                # Observation
                state = data.qpos[:6].copy().astype(np.float32)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(args.device)
                image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(args.device)
                image_tensor = image_tensor / 255.0

                batch = {
                    "observation.state": state_tensor,
                    "observation.image": image_tensor,
                }

                # Inference
                if stopped:
                    # Already in zone — freeze arm at current position
                    action_np = data.qpos[:6].copy().astype(np.float32)
                else:
                    with torch.no_grad():
                        action = policy.select_action(batch)
                    action_np = action.cpu().numpy().squeeze(0)

                # Step simulation
                data.ctrl[:6] = action_np
                for _ in range(SIM_STEPS_PER_FRAME):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                step_count += 1

                # Compute distance + CSV logging
                obj_pos = None
                dist = None
                if push_body_id is not None and push_body_id >= 0:
                    obj_pos = data.xpos[push_body_id].copy()
                    if target_pos is not None:
                        dist = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
                        min_distance = min(min_distance, dist)
                        final_distance = dist
                        # Stop-threshold: freeze arm once cube enters zone
                        if not stopped and args.stop_threshold > 0 and dist < args.stop_threshold:
                            stopped = True
                            print(f"  Step {step:3d}: STOPPED (dist={dist:.4f}m < {args.stop_threshold}m)")

                # CSV row
                if csv_writer is not None:
                    row = [ep, step] + state.tolist() + action_np.tolist()
                    if obj_pos is not None:
                        row += [obj_pos[0], obj_pos[1], obj_pos[2]]
                    else:
                        row += [0, 0, 0]
                    row.append(dist if dist is not None else 0)
                    csv_writer.writerow(row)

                if dist is not None and step % 50 == 0:
                    print(f"  Step {step:3d}: dist={dist:.4f}m  [{'SUCCESS' if dist < 0.05 else 'pushing'}]")

                # Sync viewer
                if viewer is not None:
                    try:
                        viewer.sync()
                    except Exception:
                        break
                time.sleep(0.01)

            # Episode result
            result = {"steps": step_count}
            if push_body_id is not None and target_pos is not None:
                success = final_distance < 0.05
                result.update({"success": success, "final_dist": final_distance,
                               "min_dist": min_distance, "steps": step_count})
                print(f"  Result: success={success}, final_dist={final_distance:.4f}m, "
                      f"min_dist={min_distance:.4f}m, steps={step_count}")
            else:
                print(f"  Result: steps={step_count}")
            all_results.append(result)

            if quit_requested or (viewer is not None and not viewer.is_running()):
                break
            print()

    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

        # Flush CSV
        if csv_file is not None:
            csv_file.close()
            print(f"[local_play] CSV saved: {args.csv} ({sum(r['steps'] for r in all_results)} rows)")

        # Save video
        if args.record and video_frames:
            import imageio
            record_path = Path(args.record)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(record_path), video_frames, fps=10)
            print(f"[local_play] Video saved: {args.record} ({len(video_frames)} frames)")

        # Summary
        if all_results:
            n_success = sum(1 for r in all_results if r.get("success", False))
            avg_dist = np.mean([r.get("final_dist", 0) for r in all_results])
            avg_min = np.mean([r.get("min_dist", 0) for r in all_results])
            print(f"\n=== Summary ===")
            print(f"  Episodes:  {len(all_results)}")
            print(f"  Success:   {n_success}/{len(all_results)} ({100*n_success/len(all_results):.0f}%)")
            print(f"  Avg dist:  {avg_dist:.4f}m")
            print(f"  Avg min:   {avg_min:.4f}m")

    print("[local_play] Done.")


if __name__ == "__main__":
    main()
