"""Training launcher wrapping lerobot-train as a subprocess.

Uses the sys.argv manipulation approach (same as manual PushT training)
which is proven to work with lerobot's draccus-based CLI.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TrainingLauncher:
    """Launch and manage lerobot-train subprocess."""

    def __init__(self, log_dir: str = "outputs/logs", cwd: Optional[str] = None):
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._cwd = cwd or str(Path.cwd())
        self._proc: Optional[subprocess.Popen] = None

    def launch(
        self,
        policy_type: str,
        dataset_repo_id: str,
        dataset_root: Optional[str] = None,
        device: str = "cuda:0",
        policy_cfg: Optional[dict] = None,
        training_cfg: Optional[dict] = None,
        run_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        video_backend: str = "pyav",
    ) -> subprocess.Popen:
        """Launch training as a subprocess using sys.argv manipulation."""
        policy_cfg = policy_cfg or {}
        training_cfg = training_cfg or {}

        # Build sys.argv list (same format as working PushT training)
        argv = self._build_argv(
            policy_type=policy_type,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
            device=device,
            policy_cfg=policy_cfg,
            training_cfg=training_cfg,
            run_name=run_name,
            output_dir=output_dir,
            video_backend=video_backend,
        )

        # Build the inline Python script
        argv_str = ",\n    ".join(f"'{a}'" for a in argv)
        inline_script = (
            f"import sys\n"
            f"sys.argv = [{argv_str}]\n"
            f"from lerobot.scripts.lerobot_train import main\n"
            f"main()"
        )

        cmd = [sys.executable, "-u", "-c", inline_script]

        # Log file
        run_name_safe = (run_name or "train").replace("/", "_")
        log_file = self._log_dir / f"{run_name_safe}.log"
        log_fh = open(log_file, "w", encoding="utf-8")

        logger.info("Launching training with argv: %s", argv)
        logger.info("Log file: %s", log_file)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        self._proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=self._cwd,
            env=env,
        )

        logger.info("Training PID: %d", self._proc.pid)
        return self._proc

    def _build_argv(
        self,
        policy_type: str,
        dataset_repo_id: str,
        dataset_root: Optional[str],
        device: str,
        policy_cfg: dict,
        training_cfg: dict,
        run_name: Optional[str],
        output_dir: Optional[str],
        video_backend: str,
    ) -> list[str]:
        """Build sys.argv list for lerobot-train.

        lerobot-train uses draccus for config. Accepted top-level args:
          --batch_size, --steps, --output_dir, --num_workers, --seed, --resume, --eval_freq, --log_freq

        Everything else uses draccus override syntax:
          --dataset.xxx=yyy, --policy.xxx=yyy, --optimizer.xxx=yyy, --scheduler.xxx=yyy
        """
        argv = ["lerobot-train"]

        # Dataset config (draccus overrides)
        argv.append(f"--dataset.repo_id={dataset_repo_id}")
        if dataset_root:
            argv.append(f"--dataset.root={dataset_root}")
        argv.append(f"--dataset.video_backend={video_backend}")

        # Policy config (draccus overrides)
        policy_name = self._resolve_policy_name(policy_type)
        argv.append(f"--policy.type={policy_name}")
        argv.append(f"--policy.device={device}")
        # policy.repo_id is required by lerobot-train
        repo_id = policy_cfg.get("repo_id", f"{dataset_repo_id}_{policy_name}")
        argv.append(f"--policy.repo_id={repo_id}")
        for key, val in policy_cfg.items():
            if key in ("type", "repo_id"):
                continue
            argv.append(f"--policy.{key}={val}")

        # Top-level training args (direct CLI parameters)
        if output_dir:
            argv.append(f"--output_dir={output_dir}")
        if run_name:
            argv.append(f"--job_name={run_name}")

        # Map training config to lerobot-train top-level args
        top_level_keys = {
            "batch_size": "batch_size",
            "max_steps": "steps",
            "eval_every": "eval_freq",
            "num_workers": "num_workers",
            "seed": "seed",
        }
        for plan_key, cli_key in top_level_keys.items():
            if plan_key in training_cfg:
                argv.append(f"--{cli_key}={training_cfg[plan_key]}")

        # Defaults
        if "batch_size" not in training_cfg:
            argv.append("--batch_size=128")
        if "max_steps" not in training_cfg:
            argv.append("--steps=50000")

        argv.append("--num_workers=4")

        return argv

    def _resolve_policy_name(self, policy_type: str) -> str:
        """Map plan policy type to LeRobot policy class name."""
        mapping = {
            "act": "act",
            "diffusion": "diffusion",
        }
        return mapping.get(policy_type.lower(), policy_type)

    @staticmethod
    def is_running(pid: Optional[int]) -> bool:
        """Check if a process is still alive."""
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    @staticmethod
    def graceful_stop(pid: int, timeout: int = 30) -> None:
        """Gracefully stop a training process."""
        if not TrainingLauncher.is_running(pid):
            return

        logger.info("Stopping PID %d...", pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        start = time.time()
        while time.time() - start < timeout:
            if not TrainingLauncher.is_running(pid):
                logger.info("PID %d stopped gracefully", pid)
                return
            time.sleep(1)

        logger.warning("Force killing PID %d", pid)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
