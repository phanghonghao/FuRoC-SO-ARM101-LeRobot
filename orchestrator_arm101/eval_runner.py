"""Evaluation runner — load checkpoint, run MuJoCo rollouts, compute metrics.

Loads a trained policy via ACTPolicy/DiffusionPolicy/SmolVLAPolicy.from_pretrained(),
runs N episodes in MuJoCo, and computes success rate + saves video.

Supports eval modes:
- default:   joint-space distance from home (legacy)
- push:      track pushable object distance to target zone
- pick:      T3 — check if block is lifted >5cm from surface
- pickplace: T4 — check if block is placed within target zone
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Default push task config (cube at gripper height Z=0.12m)
PUSH_OBJECT_START = np.array([0.19, 0.0, 0.12])
TARGET_POS = np.array([0.35, 0.0, 0.096])

# Pick task defaults
PICK_OBJECT_START = np.array([0.16, 0.0, 0.065])
PICK_TARGET_POS = np.array([0.25, 0.0, 0.04])
TABLE_TOP_Z = 0.04

# SO-101 HOME position (must match training data start)
HOME_QPOS = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 1.0])


class EvalRunner:
    """Run evaluation episodes with a trained policy in MuJoCo."""

    def __init__(self, config: dict, device: str = "cuda:0"):
        self.cfg = config
        self.device = device
        self.n_episodes = config.get("n_episodes", 20)
        self.max_steps = config.get("max_steps", 300)
        self.success_threshold = config.get("success_threshold", 0.05)
        self.save_video = config.get("save_video", True)
        self.video_dir = config.get("video_dir", "outputs/eval_videos")
        self.select_best_by = config.get("select_best_by", "success_rate")
        self.image_height = config.get("image_height", 480)
        self.image_width = config.get("image_width", 640)

        # Push eval config
        self.push_mode = config.get("push_mode", False)
        self.push_object_start = np.array(config.get("push_object_start", list(PUSH_OBJECT_START)))
        self.target_pos = np.array(config.get("target_pos", list(TARGET_POS)))

        # Pick / PickPlace eval config
        self.pick_mode = config.get("pick_mode", False)
        self.pickplace_mode = config.get("pickplace_mode", False)
        self.pick_object_start = np.array(config.get("pick_object_start", list(PICK_OBJECT_START)))
        self.pick_target_pos = np.array(config.get("pick_target_pos", list(PICK_TARGET_POS)))

    def evaluate_checkpoint(self, checkpoint_path: str, scene_xml: str) -> dict:
        """Evaluate a single checkpoint. Returns metrics dict."""
        logger.info("Evaluating checkpoint: %s", checkpoint_path)

        # Load policy
        policy = self._load_policy(checkpoint_path)
        if policy is None:
            return {"success_rate": 0.0, "avg_distance": float("inf"), "error": "Failed to load policy"}

        # Setup MuJoCo
        scene_path = Path(scene_xml)
        if not scene_path.exists():
            scene_path = Path(__file__).parent.parent / scene_xml
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=self.image_height, width=self.image_width)

        # Detect push object body id and auto-read positions from scene
        push_body_id = None
        pick_body_id = None

        if self.push_mode:
            push_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "push_object")
            if push_body_id < 0:
                logger.warning("push_mode=True but 'push_object' body not found in scene, falling back to joint metric")
                push_body_id = None
            else:
                body_pos = model.body_pos[push_body_id].copy()
                if np.allclose(self.push_object_start, PUSH_OBJECT_START, atol=0.01):
                    self.push_object_start = body_pos
                    logger.info("Auto-detected push_object start from scene: %s", body_pos)
                target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
                if target_geom_id >= 0 and np.allclose(self.target_pos, TARGET_POS, atol=0.01):
                    self.target_pos = model.geom_pos[target_geom_id].copy()
                    logger.info("Auto-detected target_zone from scene: %s", self.target_pos)

        if self.pick_mode or self.pickplace_mode:
            # Try pick_object first (T3 scene), then block_red (T4 scene)
            for body_name in ["pick_object", "block_red"]:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if bid >= 0:
                    pick_body_id = bid
                    body_pos = model.body_pos[bid].copy()
                    self.pick_object_start = body_pos
                    logger.info("Auto-detected %s start from scene: %s", body_name, body_pos)
                    break
            # Auto-read target
            target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
            if target_geom_id < 0:
                target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_1")
            if target_geom_id >= 0:
                self.pick_target_pos = model.geom_pos[target_geom_id].copy()
                logger.info("Auto-detected pick target from scene: %s", self.pick_target_pos)

        # Run episodes
        results = []
        all_video_frames = []

        for ep in range(self.n_episodes):
            ep_result, frames = self._run_episode(
                policy, model, data, renderer, push_body_id, pick_body_id, seed=ep)
            results.append(ep_result)
            if self.save_video and frames:
                all_video_frames.append((ep, frames))
            logger.info("  Episode %d: success=%s, distance=%.4f",
                         ep, ep_result["success"], ep_result["final_distance"])

        # Aggregate metrics
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        avg_distance = np.mean([r["final_distance"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])

        # Save videos
        video_paths = []
        if self.save_video and all_video_frames:
            video_paths = self._save_videos(all_video_frames, checkpoint_path)

        metrics = {
            "checkpoint": checkpoint_path,
            "success_rate": success_rate,
            "avg_distance": float(avg_distance),
            "avg_steps": float(avg_steps),
            "n_episodes": self.n_episodes,
            "eval_mode": ("push" if self.push_mode else "pick" if self.pick_mode
                          else "pickplace" if self.pickplace_mode else "joint_distance"),
            "video_paths": video_paths,
        }
        if self.push_mode:
            metrics["target_pos"] = self.target_pos.tolist()
            metrics["push_object_start"] = self.push_object_start.tolist()
            metrics["min_distance_to_target"] = float(min(r.get("min_distance", r["final_distance"]) for r in results))
        if self.pick_mode or self.pickplace_mode:
            lift_heights = [r.get("max_lift_height", 0) for r in results]
            metrics["avg_lift_height"] = float(np.mean(lift_heights))
            metrics["pick_object_start"] = self.pick_object_start.tolist()
            metrics["pick_target_pos"] = self.pick_target_pos.tolist()
        if self.pickplace_mode:
            place_dists = [r.get("final_distance", float("inf")) for r in results]
            metrics["avg_place_distance"] = float(np.mean(place_dists))

        logger.info("Checkpoint %s: success_rate=%.2f, avg_distance=%.4f",
                     Path(checkpoint_path).name, success_rate, avg_distance)
        return metrics

    def evaluate_multiple(self, checkpoint_paths: list[str], scene_xml: str) -> list[dict]:
        """Evaluate multiple checkpoints and return results sorted by metric."""
        results = []
        for ckpt in checkpoint_paths:
            result = self.evaluate_checkpoint(ckpt, scene_xml)
            results.append(result)

        # Sort by configured metric
        reverse = self.select_best_by == "success_rate"
        results.sort(key=lambda r: r.get(self.select_best_by, 0), reverse=reverse)
        return results

    def _load_policy(self, checkpoint_path: str):
        """Load a policy from checkpoint directory."""
        for policy_cls_name, module_path in [
            ("SmolVLAPolicy", "lerobot.policies.smolvla.modeling_smolvla"),
            ("ACTPolicy", "lerobot.policies.act.modeling_act"),
            ("DiffusionPolicy", "lerobot.policies.diffusion.modeling_diffusion"),
        ]:
            try:
                import importlib
                module = importlib.import_module(module_path)
                policy_cls = getattr(module, policy_cls_name)
                policy = policy_cls.from_pretrained(checkpoint_path)
                policy.to(self.device)
                policy.eval()
                logger.info("Loaded %s from %s", policy_cls_name, checkpoint_path)
                return policy
            except Exception:
                continue

        logger.error("Could not load policy from %s", checkpoint_path)
        return None

    def _run_episode(self, policy, model, data, renderer,
                     push_body_id=None, pick_body_id=None, seed: int = 0) -> tuple[dict, list]:
        """Run a single evaluation episode."""
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(model, data)

        # Set arm to HOME position (matches training data start)
        data.qpos[:6] = HOME_QPOS

        # Reset object positions
        for body_id, start_pos in [(push_body_id, self.push_object_start),
                                    (pick_body_id, self.pick_object_start)]:
            if body_id is not None and body_id >= 0:
                qpos_adr = model.jnt_qposadr[model.body_jntadr[body_id]]
                data.qpos[qpos_adr:qpos_adr + 3] = start_pos
                data.qpos[qpos_adr + 3:qpos_adr + 7] = [1, 0, 0, 0]
                qvel_adr = model.jnt_dofadr[model.body_jntadr[body_id]]
                data.qvel[qvel_adr:qvel_adr + 6] = 0

        mujoco.mj_forward(model, data)
        policy.reset()

        frames = []
        min_distance = float("inf")
        final_distance = float("inf")
        max_lift_height = 0.0
        steps = 0

        # Track initial block Z for pick success
        initial_block_z = TABLE_TOP_Z
        if pick_body_id is not None and pick_body_id >= 0:
            initial_block_z = data.xpos[pick_body_id][2]

        for step in range(self.max_steps):
            # Render
            renderer.update_scene(data)
            image = renderer.render()

            if self.save_video and step % 3 == 0:
                frames.append(image.copy())

            # Prepare observation
            state = data.qpos[:6].copy().astype(np.float32)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)
            image_tensor = image_tensor / 255.0

            batch = {
                "observation.state": state_tensor,
                "observation.image": image_tensor,
            }

            with torch.no_grad():
                action = policy.select_action(batch)

            action_np = action.cpu().numpy().squeeze(0)

            # Step simulation
            data.ctrl[:6] = action_np
            for _ in range(10):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            steps += 1

            # Compute metrics based on mode
            if push_body_id is not None:
                # Push mode: distance from object center to target
                obj_pos = data.xpos[push_body_id].copy()
                dist = float(np.linalg.norm(obj_pos[:2] - self.target_pos[:2]))
                min_distance = min(min_distance, dist)
                final_distance = dist

            elif pick_body_id is not None:
                # Pick/PickPlace mode
                block_pos = data.xpos[pick_body_id].copy()
                lift = block_pos[2] - initial_block_z
                max_lift_height = max(max_lift_height, lift)

                if self.pickplace_mode:
                    # T4: distance to target
                    dist = float(np.linalg.norm(block_pos[:2] - self.pick_target_pos[:2]))
                    min_distance = min(min_distance, dist)
                    final_distance = dist
                else:
                    # T3: distance = inverse of lift height (closer to 0 = more lifted)
                    final_distance = max(0, 0.05 - lift)  # 0 if lifted >5cm

            else:
                # Legacy: distance from joint home position
                current = data.qpos[:6].copy()
                final_distance = float(np.linalg.norm(current))

        # Determine success
        if push_body_id is not None:
            success = final_distance < self.success_threshold
        elif pick_body_id is not None:
            if self.pickplace_mode:
                # T4: block placed within target AND lifted at some point
                success = final_distance < self.success_threshold and max_lift_height > 0.02
            else:
                # T3: block lifted >5cm from initial position
                success = max_lift_height > 0.05
        else:
            success = final_distance < self.success_threshold

        result = {"success": success, "final_distance": final_distance, "steps": steps}
        if push_body_id is not None:
            result["min_distance"] = min_distance
        if pick_body_id is not None:
            result["max_lift_height"] = max_lift_height
        return result, frames

    def _save_videos(self, episode_frames: list, checkpoint_path: str) -> list[str]:
        """Save evaluation videos."""
        import imageio

        video_dir = Path(self.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

        ckpt_name = Path(checkpoint_path).name
        saved = []

        # Save combined video
        all_frames = []
        for ep, frames in episode_frames:
            all_frames.extend(frames)
            # Black separator frame
            if frames:
                all_frames.append(np.zeros_like(frames[0]))

        if all_frames:
            all_frames = all_frames[:-1]  # Remove last separator
            video_path = video_dir / f"eval_{ckpt_name}.mp4"
            try:
                imageio.mimsave(str(video_path), all_frames, fps=10)
                saved.append(str(video_path))
                logger.info("Saved eval video: %s", video_path)
            except Exception as e:
                logger.warning("Failed to save video: %s", e)

        return saved
