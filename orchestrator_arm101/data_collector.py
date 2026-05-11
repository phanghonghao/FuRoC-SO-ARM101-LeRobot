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


class DataCollector:
    """Collect scripted demonstrations in MuJoCo and save as LeRobot dataset."""

    def __init__(self, config: dict):
        self.cfg = config
        self.scene_xml = config.get("scene_xml", "Simulation/SO101/scene.xml")
        self.n_episodes = config.get("n_episodes", 10)
        self.n_steps = config.get("n_steps", 100)
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
        self.output_dir = config.get("root", "outputs/so101_sim_lerobot")

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
        for ep in range(self.n_episodes):
            trajectory = self._generate_trajectory(seed=ep)
            mujoco.mj_resetData(model, data)

            for i in range(len(trajectory) - 1):
                data.ctrl[:6] = trajectory[i]
                for _ in range(10):
                    mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)

                renderer.update_scene(data)
                image = renderer.render()

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
            "scripted_pick": "Pick and place objects with SO-101 arm in MuJoCo simulation",
        }
        return descriptions.get(self.trajectory_type, descriptions["ik_reach"])

    def _generate_trajectory(self, seed: int) -> np.ndarray:
        """Generate a trajectory based on configured type."""
        rng = np.random.default_rng(seed)
        noise_std = self.randomization.get("joint_noise_std", 0.0) if self.randomize else 0.0
        range_scale = self.randomization.get("target_range_scale", 1.0) if self.randomize else 1.0

        if self.trajectory_type == "ik_push":
            return self._generate_push_trajectory(rng, noise_std, range_scale)
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
        """Generate push trajectory: descend → contact → push → retract.

        SO-101 kinematic constraints (实测):
          - 夹爪最低 Z=0.067m，无法到达桌面
          - 需要 4cm 台面让 cube 在 Z=0.065m
          - trajectory waypoints from joint-space sweep on RTX
        """
        # Waypoints (实测: [shoulder_lift, elbow_flex, wrist_flex])
        home = HOME.copy()

        # 下降到 cube 后方高处 (gripper ~(0.16, 0, 0.08))
        descend = np.array([0.0, -0.65, 1.51, -0.16, 0.0, 1.0])

        # 接触 cube (gripper ~(0.157, 0, 0.067) — 夹爪最低点)
        contact = np.array([0.0, -0.15, 1.60, -0.86, 0.0, 1.0])

        # 推过 cube (gripper ~(0.207, 0, 0.081) — 向前推 5cm)
        push_target = np.array([0.0, -0.15, 1.10, -0.26, 0.0, 1.0])
        # 加入随机偏移增加多样性
        push_target[:4] += rng.uniform(-0.1, 0.1, 4) * range_scale
        if noise_std > 0:
            push_target += rng.normal(0, noise_std, 6)
        push_target[5] = 1.0  # gripper closed

        waypoints = [home, descend, contact, push_target, contact, descend, home]
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
