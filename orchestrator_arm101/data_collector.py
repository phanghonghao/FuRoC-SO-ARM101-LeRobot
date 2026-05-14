"""Scripted demo collection wrapper for MuJoCo SO-101.

Wraps the logic from collect_sim_data.py with YAML-driven configuration.
Supports trajectory types: ik_reach, ik_push, scripted_pick.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

# SO-101 joint limits
JOINT_RANGES = np.array([
    [-1.92, 1.92],   # shoulder_pan
    [-1.75, 1.75],   # shoulder_lift
    [-1.69, 1.69],   # elbow_flex
    [-1.66, 1.66],   # wrist_flex
    [-2.74, 2.84],   # wrist_roll
    [-0.17, 1.75],   # gripper
])
HOME = np.array([0.0, -0.5, 1.0, -0.5, 0.0, 1.0])

# Gripper constants for pick
GRIPPER_OPEN = -0.17
GRIPPER_CLOSED = 1.0

# Pick waypoints from workspace validation (check_workspace_pick.py)
# FK verified (gripper body positions):
#   home:    (0.222, 0, 0.169)
#   above:   (0.160, 0, 0.110) — above block, gripper open
#   descend: (0.160, 0, 0.065) — at block center
#   lift:    (0.174, 0, 0.118) — lifted 5.3cm above table
#   carry:   (0.250, 0, 0.244) — above target
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


class DataCollector:
    """Collect scripted demonstrations in MuJoCo and save as LeRobot dataset."""

    def __init__(self, config: dict):
        self.cfg = config
        self.scene_xml = config.get("scene_xml", "Simulation/SO101/scene.xml")
        self.n_episodes = config.get("n_episodes", 10)
        self.n_steps = config.get("n_steps", 150)
        self.n_waypoints = config.get("n_waypoints", 3)
        self.trajectory_type = config.get("trajectory_type", "ik_reach")
        self.randomize = config.get("randomize", False)
        self.randomization = config.get("randomization", {})
        self.push_to_hub = config.get("push_to_hub", False)
        self.hub_repo_id = config.get("hub_repo_id", "")
        self.image_height = config.get("image_height", 480)
        self.image_width = config.get("image_width", 640)
        self.fps = config.get("fps", 30)

        # Dataset config
        self.repo_id = config.get("repo_id", "PhangHongHao/so101_sim")
        self.output_dir = config.get("output_dir", config.get("root", "outputs/so101_sim_lerobot"))

    def run(self, progress_callback=None) -> dict:
        """Run data collection. Returns summary dict."""
        logger.info("Starting data collection: %d episodes, type=%s",
                     self.n_episodes, self.trajectory_type)

        # MuJoCo setup
        scene_path = self._resolve_scene_path()
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=self.image_height, width=self.image_width)

        # LeRobot dataset
        features = {
            "observation.image": {
                "shape": (3, self.image_height, self.image_width),
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
        task = self._get_task_description()

        ds = LeRobotDataset.create(
            repo_id=self.repo_id,
            fps=self.fps,
            features=features,
            root=self.output_dir,
            use_videos=True,
            image_writer_threads=1,
        )

        total_frames = 0

        # Early-stop setup: freeze arm when cube enters target zone (push only)
        is_push = self.trajectory_type in ("ik_push",)
        is_pick = self.trajectory_type in ("scripted_pick", "scripted_pickplace")
        push_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "push_object")
        target_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
        target_pos = model.geom_pos[target_geom_id].copy() if target_geom_id >= 0 else None
        stop_threshold = self.cfg.get("stop_threshold", 0.04)  # 4cm default
        min_push_steps = self.cfg.get("min_push_steps", 30)

        # Pick success detection: check if block is near target zone
        pick_body_name = "pick_object" if is_pick else None
        pick_body_id = (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pick_body_name)
                        if pick_body_name else -1)
        pick_target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_zone")
        pick_target_pos = model.geom_pos[pick_target_id].copy() if pick_target_id >= 0 else None

        for ep in range(self.n_episodes):
            trajectory = self._generate_trajectory(seed=ep)
            mujoco.mj_resetData(model, data)

            stopped = False
            hold_position = None

            for i in range(len(trajectory) - 1):
                # Early-stop for push: freeze arm when cube in target zone
                if (not stopped and is_push and i >= min_push_steps
                        and push_body_id >= 0 and target_pos is not None):
                    obj_pos = data.xpos[push_body_id].copy()
                    dist = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
                    if dist < stop_threshold:
                        stopped = True
                        hold_position = data.qpos[:6].copy().astype(np.float32)

                # Apply control: follow trajectory or hold position
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
                # Action: next control target (hold if stopped)
                if stopped and hold_position is not None:
                    action = hold_position.copy()
                else:
                    action = trajectory[i + 1].copy().astype(np.float32)

                frame = {
                    "observation.image": image,
                    "observation.state": state,
                    "action": action,
                    "task": task,
                }
                ds.add_frame(frame)

            ds.save_episode()
            total_frames += len(trajectory) - 1
            logger.info("Episode %d/%d: %d frames saved",
                         ep + 1, self.n_episodes, len(trajectory) - 1)

            if progress_callback:
                progress_callback((ep + 1) / self.n_episodes)

        ds.finalize()
        logger.info("Dataset saved: %d episodes, %d frames", ds.num_episodes, ds.num_frames)

        # Push to Hub
        if self.push_to_hub and self.hub_repo_id:
            logger.info("Pushing to HuggingFace Hub: %s", self.hub_repo_id)
            ds.push_to_hub(private=False, tag_version=False)
            logger.info("Push complete")

        return {
            "output_dir": self.output_dir,
            "n_episodes": ds.num_episodes,
            "n_frames": ds.num_frames,
        }

    def _resolve_scene_path(self) -> Path:
        """Find the MuJoCo scene XML file."""
        # Try relative to project root
        candidates = [
            Path(self.scene_xml),
            Path(__file__).parent.parent / self.scene_xml,
            Path.cwd() / self.scene_xml,
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"Scene XML not found: {self.scene_xml}")

    def _get_task_description(self) -> str:
        descriptions = {
            "ik_reach": "Reach random target positions with SO-101 arm in MuJoCo simulation",
            "ik_push": "Push object to target positions with SO-101 arm in MuJoCo simulation",
            "scripted_pick": "pick up the red block",
            "scripted_pickplace": "pick up the red block and place it on the green target",
        }
        # Allow custom task description from config
        custom = self.cfg.get("task_description", None)
        if custom:
            return custom
        return descriptions.get(self.trajectory_type, descriptions["ik_reach"])

    def _generate_trajectory(self, seed: int) -> np.ndarray:
        """Generate a trajectory based on configured type."""
        rng = np.random.default_rng(seed)
        noise_std = self.randomization.get("joint_noise_std", 0.0) if self.randomize else 0.0
        range_scale = self.randomization.get("target_range_scale", 1.0) if self.randomize else 1.0

        if self.trajectory_type == "ik_push":
            return self._generate_push_trajectory(rng, noise_std, range_scale)
        elif self.trajectory_type == "scripted_pick":
            return self._generate_pick_trajectory(rng, noise_std, range_scale)
        elif self.trajectory_type == "scripted_pickplace":
            return self._generate_pickplace_trajectory(rng, noise_std, range_scale)
        else:
            return self._generate_reach_trajectory(rng, noise_std, range_scale)

    def _generate_reach_trajectory(self, rng, noise_std: float, range_scale: float) -> np.ndarray:
        """Generate reaching trajectory with random waypoints."""
        waypoints = [HOME.copy()]
        for _ in range(self.n_waypoints):
            ranges = JOINT_RANGES * range_scale
            target = np.array([rng.uniform(lo, hi) for lo, hi in ranges])
            target[5] = 1.0  # gripper closed
            if noise_std > 0:
                target += rng.normal(0, noise_std, 6)
            waypoints.append(target)
        waypoints.append(HOME.copy())

        return self._interpolate_waypoints(waypoints)

    def _generate_push_trajectory(self, rng, noise_std: float, range_scale: float) -> np.ndarray:
        """Generate push trajectory: descend → contact → long push → retract.

        v10: same trajectory as v9, but the run() loop adds early-stop:
             when cube enters target zone (dist < 4cm), the arm freezes in place.
             This teaches the policy to "push and stop".

        FK sweep results (gripper body positions):
          - descend:  (0.162, 0, 0.085) — above cube
          - contact:  (0.152, 0, 0.065) — cube level, behind cube
          - push_end: (0.290, 0, 0.067) — max reachable X at cube height
        Jaw tip extends ~3cm forward from gripper body, so effective
        push contact reaches further than gripper body position.
        """
        home = HOME.copy()

        # 下降到 cube 后方高处 (gripper ~(0.16, 0, 0.085))
        descend = np.array([0.0, -0.65, 1.51, -0.16, 0.0, 1.0])

        # 接触 cube (gripper ~(0.152, 0, 0.065) — 夹爪最低点)
        contact = np.array([0.0, -0.15, 1.60, -0.86, 0.0, 1.0])

        # v9/v10: 推到最远可达位置 (gripper ~(0.290, 0, 0.067) — 向前推 ~14cm)
        push_end = np.array([0.0, 0.50, 0.32, -0.11, 0.0, 1.0])
        # 加入随机偏移增加多样性
        push_end[:4] += rng.uniform(-0.08, 0.08, 4) * range_scale
        if noise_std > 0:
            push_end += rng.normal(0, noise_std, 6)
        push_end[5] = 1.0  # gripper closed

        waypoints = [home, descend, contact, push_end, contact, descend, home]
        return self._interpolate_waypoints(waypoints)

    def _interpolate_waypoints(self, waypoints: list) -> np.ndarray:
        """Linear interpolation between waypoints."""
        steps_per_segment = self.n_steps // (len(waypoints) - 1)
        trajectory = []
        for i in range(len(waypoints) - 1):
            for t in np.linspace(0, 1, steps_per_segment, endpoint=False):
                pose = waypoints[i] * (1 - t) + waypoints[i + 1] * t
                trajectory.append(pose)
        trajectory.append(waypoints[-1])
        return np.array(trajectory)

    def _generate_pick_trajectory(self, rng, noise_std: float, range_scale: float) -> np.ndarray:
        """Generate pick trajectory: home -> above -> descend -> grasp -> lift -> carry -> release -> retreat.

        Uses waypoints verified by workspace check (gripper body FK).
        Randomization: slight offset in block approach and target placement.
        """
        wp = PICK_WAYPOINTS

        # Randomize block approach position (slight XY offset ±2cm)
        approach_offset = rng.uniform(-0.02, 0.02, 2) * range_scale
        # Randomize target placement position (offset ±3cm)
        target_offset = rng.uniform(-0.03, 0.03, 2) * range_scale

        # Build waypoints with randomization
        above = wp["above"].copy()
        above[:2] = [0.0, approach_offset[1]]  # slight Y offset

        descend = wp["descend"].copy()
        descend[:2] = [0.0, approach_offset[1]]

        grasp = wp["grasp"].copy()
        grasp[:2] = [0.0, approach_offset[1]]

        lift = wp["lift"].copy()
        lift[:2] = [0.0, approach_offset[1]]

        carry = wp["carry"].copy()
        carry[0] = wp["carry"][0] + target_offset[0]
        carry[1] = target_offset[1]

        release = carry.copy()
        release[5] = GRIPPER_OPEN

        waypoints = [
            wp["home"], above, descend, grasp, lift, carry, release, wp["retreat"]
        ]

        if noise_std > 0:
            for i in range(1, len(waypoints) - 1):
                waypoints[i] = waypoints[i] + rng.normal(0, noise_std, 6) * 0.3
                # Keep gripper value exact
                waypoints[i][5] = PICK_WAYPOINTS[list(PICK_WAYPOINTS.keys())[i]][5]

        return self._interpolate_waypoints(waypoints)

    def _generate_pickplace_trajectory(self, rng, noise_std: float, range_scale: float) -> np.ndarray:
        """Generate pick-place trajectory for T4 (two blocks).

        Same as pick but with wider randomization for two-block scenario.
        The trajectory picks up one block and places it at the target.
        """
        # For T4, use same pick trajectory with more randomization
        return self._generate_pick_trajectory(rng, noise_std * 1.2, range_scale * 1.1)
