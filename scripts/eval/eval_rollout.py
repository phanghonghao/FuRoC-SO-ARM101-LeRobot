"""Standalone evaluation script for SO-ARM101 trained policies.

Usage:
    # Legacy joint-distance eval
    python eval_rollout.py --checkpoint outputs/.../pretrained_model --episodes 10

    # Push task eval (cube + target zone)
    python eval_rollout.py --checkpoint outputs/.../pretrained_model --push --episodes 10

Can be run locally (MuJoCo CPU) or on RTX (headless with EGL auto-detected).
"""

import argparse
import json
import os
import sys
from pathlib import Path

# EGL auto-setup: must be set before mujoco/torch import
# On Linux without DISPLAY, use EGL for headless GPU rendering
if sys.platform == "linux" and "DISPLAY" not in os.environ and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from orchestrator_arm101.eval_runner import EvalRunner


def main():
    parser = argparse.ArgumentParser(description="Evaluate SO-ARM101 trained policy")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--scene", default=None, help="MuJoCo scene XML (auto-selected by --push)")
    parser.add_argument("--push", action="store_true", help="Push task eval (cube + target zone)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--device", default=None, help="Device (default: auto-detect)")
    parser.add_argument("--output", default=None, help="Output JSON path for results")
    parser.add_argument("--save-video", action="store_true", help="Save eval video")
    parser.add_argument("--video-dir", default="outputs/eval_videos")
    args = parser.parse_args()

    # Auto-select scene
    if args.scene is None:
        if args.push:
            args.scene = "Simulation/SO101/scene_push_eval.xml"
        else:
            args.scene = "Simulation/SO101/scene.xml"

    # Auto-detect device
    if args.device is None:
        import torch
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"[eval_rollout] Checkpoint: {args.checkpoint}")
    print(f"[eval_rollout] Device: {args.device}")
    print(f"[eval_rollout] Episodes: {args.episodes}")
    print(f"[eval_rollout] Mode: {'push (object → target)' if args.push else 'joint-distance'}")
    print(f"[eval_rollout] Scene: {args.scene}")

    eval_config = {
        "n_episodes": args.episodes,
        "max_steps": args.max_steps,
        "success_threshold": 0.05,
        "save_video": args.save_video,
        "video_dir": args.video_dir,
        "push_mode": args.push,
    }

    runner = EvalRunner(eval_config, device=args.device)
    results = runner.evaluate_checkpoint(args.checkpoint, args.scene)

    mode_label = "push dist → target" if args.push else "joint distance"
    print(f"\n[Results]")
    print(f"  Eval mode:    {results.get('eval_mode', mode_label)}")
    print(f"  Success rate: {results['success_rate']:.1%}")
    print(f"  Avg distance: {results['avg_distance']:.4f}")
    if args.push and "min_distance_to_target" in results:
        print(f"  Min distance: {results['min_distance_to_target']:.4f}")
    print(f"  Avg steps:    {results['avg_steps']:.1f}")
    if results.get("video_paths"):
        print(f"  Videos:       {results['video_paths']}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Results saved: {args.output}")


if __name__ == "__main__":
    main()
