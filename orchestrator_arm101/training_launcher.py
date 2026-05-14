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
        video_backend: str = "torchcodec",
        resume_from: Optional[str] = None,
    ) -> subprocess.Popen:
        """Launch training as a subprocess using sys.argv manipulation.

        Args:
            resume_from: Path to a checkpoint's ``pretrained_model/`` directory.
                When set, ``--policy.pretrained_path=<path>`` is added to load
                model weights into a fresh optimizer.  This is the *weight-only*
                resume approach that works even when optimizer state is corrupted.
                The output_dir is suffixed with ``_r<N>`` to avoid FileExistsError.
        """
        policy_cfg = policy_cfg or {}
        training_cfg = training_cfg or {}

        # Handle resume: suffix output_dir to avoid collisions
        if resume_from and output_dir:
            output_dir = self._make_resume_output_dir(output_dir)

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
            resume_from=resume_from,
        )

        # Build the inline Python script
        argv_str = ",\n    ".join(f"'{a}'" for a in argv)
        inline_script = (
            f"import sys\n"
            f"sys.argv = [{argv_str}]\n"
            f"from lerobot.scripts.lerobot_train import main\n"
            f"try:\n"
            f"    main()\n"
            f"except Exception as e:\n"
            f"    import traceback\n"
            f"    tb = traceback.format_exc()\n"
            f"    if 'push_model_to_hub' in tb or 'push_to_hub' in tb:\n"
            f"        print(f'\\nWARNING: push_to_hub failed (non-fatal): {{e}}', file=sys.stderr)\n"
            f"        print('Training checkpoints saved locally. Hub upload skipped.', file=sys.stderr)\n"
            f"        sys.exit(0)\n"
            f"    raise\n"
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

        # torchcodec fix: LD_PRELOAD conda libstdc++ for CXXABI_1.3.15
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            libstdcxx = os.path.join(conda_prefix, "lib", "libstdc++.so.6")
            if os.path.exists(libstdcxx):
                env["LD_PRELOAD"] = libstdcxx
                logger.info("LD_PRELOAD set to %s (torchcodec fix)", libstdcxx)

        # Use CUDA_VISIBLE_DEVICES with --policy.device=cuda to avoid
        # LeRobot v0.5.1 is_amp_available("cuda:N") bug
        if device and device.startswith("cuda:"):
            gpu_id = device.split(":")[1]
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            # Override device in argv to plain "cuda"
            for i, a in enumerate(argv):
                if a.startswith("--policy.device="):
                    argv[i] = "--policy.device=cuda"
            # Rebuild inline script with updated argv
            argv_str = ",\n    ".join(f"'{a}'" for a in argv)
            inline_script = (
                f"import sys\n"
                f"sys.argv = [{argv_str}]\n"
                f"from lerobot.scripts.lerobot_train import main\n"
                f"try:\n"
                f"    main()\n"
                f"except Exception as e:\n"
                f"    import traceback\n"
                f"    tb = traceback.format_exc()\n"
                f"    if 'push_model_to_hub' in tb or 'push_to_hub' in tb:\n"
                f"        print(f'\\nWARNING: push_to_hub failed (non-fatal): {{e}}', file=sys.stderr)\n"
                f"        print('Training checkpoints saved locally. Hub upload skipped.', file=sys.stderr)\n"
                f"        sys.exit(0)\n"
                f"    raise\n"
            )
            cmd = [sys.executable, "-u", "-c", inline_script]
            logger.info("CUDA_VISIBLE_DEVICES=%s, --policy.device=cuda", gpu_id)

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
        resume_from: Optional[str] = None,
    ) -> list[str]:
        """Build sys.argv list for lerobot-train.

        lerobot-train uses draccus for config. Accepted top-level args:
          --batch_size, --steps, --output_dir, --num_workers, --seed, --resume, --eval_freq, --log_freq

        Everything else uses draccus override syntax:
          --dataset.xxx=yyy, --policy.xxx=yyy, --optimizer.xxx=yyy, --scheduler.xxx=yyy
        """
        argv = ["lerobot-train"]

        # Weight-only resume: load pretrained weights into fresh optimizer
        if resume_from:
            argv.append(f"--policy.pretrained_path={resume_from}")

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
            "save_every": "save_freq",
            "num_workers": "num_workers",
            "log_freq": "log_freq",
            "seed": "seed",
        }
        for plan_key, cli_key in top_level_keys.items():
            if plan_key in training_cfg:
                argv.append(f"--{cli_key}={training_cfg[plan_key]}")

        # Defaults (only if not already set above)
        if "batch_size" not in training_cfg:
            argv.append("--batch_size=128")
        if "max_steps" not in training_cfg:
            argv.append("--steps=50000")
        if "num_workers" not in training_cfg:
            argv.append("--num_workers=8")

        return argv

    def _resolve_policy_name(self, policy_type: str) -> str:
        """Map plan policy type to LeRobot policy class name."""
        mapping = {
            "act": "act",
            "diffusion": "diffusion",
        }
        return mapping.get(policy_type.lower(), policy_type)

    @staticmethod
    def find_latest_checkpoint(output_dir: str) -> Optional[str]:
        """Find the latest checkpoint's pretrained_model in an output directory.

        Scans ``<output_dir>/checkpoints/`` for the highest-numbered checkpoint
        directory and returns the ``pretrained_model`` subpath if it exists.

        Returns:
            Path to ``pretrained_model/`` directory, or ``None`` if not found.
        """
        output_path = Path(output_dir)
        checkpoints_dir = output_path / "checkpoints"
        if not checkpoints_dir.exists():
            return None

        # Find checkpoint directories (e.g. 005000, 010000, ...)
        ckpt_dirs = sorted(checkpoints_dir.iterdir(), key=lambda p: p.name)
        for ckpt_dir in reversed(ckpt_dirs):
            pretrained = ckpt_dir / "pretrained_model"
            if pretrained.exists() and pretrained.is_dir():
                return str(pretrained)

        return None

    @staticmethod
    def _make_resume_output_dir(output_dir: str) -> str:
        """Generate a non-colliding output_dir for resumed training.

        Appends ``_r2``, ``_r3``, etc. to avoid ``FileExistsError`` when
        resuming into the same base directory.
        """
        base = output_dir.rstrip("/").rstrip("\\")
        n = 2
        while Path(f"{base}_r{n}").exists():
            n += 1
        return f"{base}_r{n}"

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
