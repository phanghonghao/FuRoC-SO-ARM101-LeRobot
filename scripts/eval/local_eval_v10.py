"""Local eval for ACT v7/v8/v9 — fixed position push task.

All pre-v11 versions use fixed cube and target positions:
  - Cube:   (0.16, 0.0)
  - Target: (0.32, 0.0)

Usage:
    python scripts/eval/local_eval_v10.py --best
    python scripts/eval/local_eval_v10.py --best --episodes 10 --csv outputs/eval_results/v9.csv
    python scripts/eval/local_eval_v10.py --checkpoint outputs/checkpoints/act_v9_r2_050000/pretrained_model
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

HOME_QPOS = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 1.0])


def find_scene(scene_name: str) -> Path:
    candidates = [Path(scene_name), PROJECT_ROOT / scene_name, SCENE_DIR / scene_name]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Scene not found: {scene_name}")


def _parse_ckpt_name(name: str):
    m = re.match(r"act_v(\d+)(?:_\w+)?_(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.match(r"act_r(\d+)_(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def find_checkpoint(checkpoint_arg: str | None, best: bool) -> Path:
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

    # --best: find highest version+step checkpoint (excluding v11+)
    best_path = None
    best_key = (-1, -1)
    for d in ckpt_base.iterdir():
        pretrained = d / "pretrained_model"
        if not pretrained.exists():
            continue
        if not d.name.startswith("act_"):
            continue
        version, step = _parse_ckpt_name(d.name)
        if version >= 11:  # skip v11+ (has own eval script)
            continue
        key = (version, step)
        if key > best_key:
            best_key = key
            best_path = pretrained
    if best_path is None:
        raise FileNotFoundError(f"No pre-v11 checkpoints in {ckpt_base}")
    return best_path


def load_policy(checkpoint_path: Path, device: str = "cpu"):
    from lerobot.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(str(checkpoint_path))
    policy.to(device)
    policy.eval()
    return policy


def main():
    parser = argparse.ArgumentParser(description="Local eval for ACT v7/v8/v9 (fixed position push)")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint")
    parser.add_argument("--best", action="store_true", help="Use best pre-v11 checkpoint")
    parser.add_argument("--scene", default="scene_push_table.xml", help="Base scene XML")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--device", default="cpu", help="Inference device")
    parser.add_argument("--csv", default=None, help="CSV output path")
    parser.add_argument("--record", default=None, help="MP4 output path")
    parser.add_argument("--success-threshold", type=float, default=0.05, help="Dist < this = success (m)")
    args = parser.parse_args()

    if not args.checkpoint and not args.best:
        args.best = True

    ckpt_path = find_checkpoint(args.checkpoint, args.best)
    scene_path = find_scene(args.scene)

    print(f"[v10_eval] Checkpoint: {ckpt_path}")
    print(f"[v10_eval] Scene: {scene_path}")
    print(f"[v10_eval] Mode: fixed positions (cube=0.16,0.0  target=0.32,0.0)")

    policy = load_policy(ckpt_path, device=args.device)
    print(f"[v10_eval] Policy loaded: {type(policy).__name__}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

    push_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "push_object")
    target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
    assert push_body_id >= 0, "push_object body not found"
    assert target_geom_id >= 0, "target_zone geom not found"

    push_start_pos = model.body_pos[push_body_id].copy()
    target_pos = model.geom_pos[target_geom_id].copy()
    print(f"[v10_eval] Cube: {push_start_pos[:2]}  Target: {target_pos[:2]}")

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
                             "obj_x", "obj_y", "obj_z", "dist_to_target"])
        print(f"[v10_eval] CSV: {csv_path}")

    video_frames = []
    if args.record:
        print(f"[v10_eval] Recording: {args.record}")

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

            # Reset — fixed positions from XML
            mujoco.mj_resetData(model, data)
            data.qpos[:6] = HOME_QPOS
            qpos_adr = model.jnt_qposadr[model.body_jntadr[push_body_id]]
            data.qpos[qpos_adr:qpos_adr + 3] = push_start_pos
            data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
            qvel_adr = model.jnt_dofadr[model.body_jntadr[push_body_id]]
            data.qvel[qvel_adr:qvel_adr + 6] = 0
            mujoco.mj_forward(model, data)
            policy.reset()

            min_dist = float("inf")
            final_dist = float("inf")

            print(f"Ep {ep+1}/{args.episodes}: cube=({push_start_pos[0]:.3f},{push_start_pos[1]:.3f}) "
                  f"target=({target_pos[0]:.3f},{target_pos[1]:.3f})")

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
                        + [obj_pos[0], obj_pos[1], obj_pos[2], dist]
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
            print(f"[v10_eval] CSV saved: {args.csv}")
        if args.record and video_frames:
            import imageio
            Path(args.record).parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(args.record, video_frames, fps=10)
            print(f"[v10_eval] Video saved: {args.record} ({len(video_frames)} frames)")

    # Summary
    n = len(results)
    n_success = sum(r["success"] for r in results)
    avg_final = np.mean([r["final_dist"] for r in results])
    avg_min = np.mean([r["min_dist"] for r in results])
    min_dists = [f"{r['min_dist']:.4f}" for r in results]
    print(f"\n=== v10 Eval Summary ===")
    print(f"  Checkpoint: {ckpt_path.parent.name}")
    print(f"  Episodes:   {n}")
    print(f"  Success:    {n_success}/{n} ({100*n_success/max(n,1):.0f}%)  (threshold < {args.success_threshold}m)")
    print(f"  Avg dist:   {avg_final:.4f}m (final) / {avg_min:.4f}m (min)")
    print(f"  Per-episode min dist: {min_dists}")


if __name__ == "__main__":
    main()
