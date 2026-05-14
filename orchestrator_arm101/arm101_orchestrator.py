"""SO-ARM101 Imitation Learning Pipeline Orchestrator.

Ported from Z1's PhaseOrchestrator. Main 2-level event loop with
phase-type dispatch:
  - collection  -> DataCollector.run()
  - training    -> TrainingLauncher + LossMonitor
  - evaluation  -> EvalRunner.evaluate_checkpoint()
  - comparison  -> same as training (different policy type)

Usage:
    python -m orchestrator_arm101.arm101_orchestrator \\
        --plan training_plans/so101_push_plan.yaml \\
        --device cuda:7 --fresh

    # Dry run (print plan, no execution)
    python -m orchestrator_arm101.arm101_orchestrator \\
        --plan training_plans/so101_push_plan.yaml --dry-run

    # Resume from specific phase
    python -m orchestrator_arm101.arm101_orchestrator \\
        --plan training_plans/so101_push_plan.yaml --start-from train_act
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from orchestrator_arm101.data_collector import DataCollector
from orchestrator_arm101.eval_runner import EvalRunner
from orchestrator_arm101.loss_monitor import LossMonitor
from orchestrator_arm101.phase_manager import PhaseConfig, PhaseManager
from orchestrator_arm101.state_store import OrchestratorState, StateStore
from orchestrator_arm101.training_launcher import TrainingLauncher

logger = logging.getLogger("arm101_orchestrator")

# Poll interval for non-training phases (seconds)
_DEFAULT_POLL = 10


class Arm101Orchestrator:
    """Main controller for the SO-ARM101 imitation learning pipeline."""

    def __init__(
        self,
        plan_path: str | Path,
        project_root: Optional[str | Path] = None,
        fresh: bool = False,
        start_from: Optional[str] = None,
        dry_run: bool = False,
        device: Optional[str] = None,
        state_path: str = "orchestrator_state.json",
        resume: bool = False,
    ):
        self._project_root = Path(project_root or Path.cwd())
        self._dry_run = dry_run
        self._resume = resume

        # Components
        self._phase_mgr = PhaseManager(plan_path)
        self._state_store = StateStore(self._project_root / state_path)
        self._launcher = TrainingLauncher(
            log_dir=str(self._project_root / "outputs" / "logs"),
            cwd=str(self._project_root),
        )

        # Device override
        self._device = device or self._phase_mgr.device

        # Runtime state
        self._state: Optional[OrchestratorState] = None
        self._monitor: Optional[LossMonitor] = None
        self._proc = None

        # Resolve start phase
        if fresh:
            self._state_store.clear()
            self._start_id = start_from or self._phase_mgr.get_start_phase_id()
        elif start_from:
            self._start_id = start_from
        else:
            self._start_id = self._phase_mgr.get_start_phase_id()

    # ── Public entry ────────────────────────────────────────────── #

    def run(self) -> None:
        """Main entry point."""
        self._setup_logging()

        if self._dry_run:
            self._print_dry_run()
            return

        # Crash recovery
        self._state = self._state_store.load()
        if self._state and self._state.current_phase_status == "running":
            pid = self._state.training_pid
            if pid and TrainingLauncher.is_running(pid):
                logger.info("Recovering: phase '%s' still running (PID %d)",
                            self._state.current_phase_id, pid)
                # Re-attach monitor for training phases
                if self._state.current_phase_type == "training":
                    self._resume_monitor()
            else:
                logger.info("State found but PID %s is dead -- marking as failed", pid)
                self._state.current_phase_status = "failed"

        if self._state is None or self._state.current_phase_status in ("complete", "failed"):
            self._init_new_run()

        # Main event loop
        logger.info("ARM101 Orchestrator running (device=%s)", self._device)
        try:
            while True:
                status = self._state.current_phase_status
                if status == "pending":
                    self._start_phase()
                elif status == "running":
                    self._monitor_phase()
                elif status == "complete":
                    self._advance()
                elif status == "failed":
                    self._handle_failure()
                else:
                    logger.error("Unknown status: %s", status)
                    break

                self._state_store.save(self._state)
                time.sleep(self._get_poll_interval())
        except KeyboardInterrupt:
            logger.info("Interrupted -- saving state and exiting")
            self._state_store.save(self._state)

    # ── Phase lifecycle ─────────────────────────────────────────── #

    def _init_new_run(self) -> None:
        """Initialize a fresh orchestration run."""
        start_phase = self._phase_mgr.get_phase(self._start_id)
        if start_phase is None:
            logger.error("Start phase '%s' not found in plan", self._start_id)
            sys.exit(1)

        self._state = OrchestratorState(
            plan_name=self._phase_mgr.plan_name,
            current_phase_id=self._start_id,
            current_phase_type=start_phase.phase_type,
            current_phase_status="pending",
            started_at=datetime.now().isoformat(),
        )
        self._state_store.save(self._state)
        logger.info("New orchestration run: plan='%s', start='%s' (type=%s)",
                     self._state.plan_name, self._start_id, start_phase.phase_type)

    def _start_phase(self) -> None:
        """Dispatch phase start based on phase type."""
        phase_id = self._state.current_phase_id
        phase = self._phase_mgr.get_phase(phase_id)
        if phase is None:
            logger.error("Phase '%s' not found in plan", phase_id)
            self._state.current_phase_status = "failed"
            return

        logger.info("=== Starting phase: %s (%s) type=%s ===", phase_id, phase.name, phase.phase_type)

        if phase.phase_type == "collection":
            self._start_collection(phase)
        elif phase.phase_type in ("training", "comparison"):
            self._start_training(phase)
        elif phase.phase_type == "evaluation":
            self._start_evaluation(phase)
        else:
            logger.error("Unknown phase type: %s", phase.phase_type)
            self._state.current_phase_status = "failed"

    def _monitor_phase(self) -> None:
        """Monitor running phase based on type."""
        phase_type = self._state.current_phase_type
        if phase_type in ("training", "comparison"):
            self._monitor_training()
        # collection and evaluation are synchronous -- they don't stay in "running" long
        else:
            # Check if process is still alive
            pid = self._state.training_pid
            if pid and not TrainingLauncher.is_running(pid):
                self._state.current_phase_status = "complete"

    def _start_collection(self, phase: PhaseConfig) -> None:
        """Run data collection (synchronous)."""
        cfg = phase.collection
        cfg.update(phase.dataset)  # merge dataset config

        collector = DataCollector(cfg)

        def on_progress(p: float):
            self._state.collection_progress = p
            self._state_store.save(self._state)

        try:
            result = collector.run(progress_callback=on_progress)
            self._state.phase_history.append({
                "phase_id": phase.id,
                "phase_type": "collection",
                "result": result,
                "completed_at": datetime.now().isoformat(),
            })
            self._state.current_phase_status = "complete"
            logger.info("Collection complete: %s", result)
        except Exception as e:
            logger.error("Collection failed: %s", e)
            self._state.current_phase_status = "failed"

    def _start_training(self, phase: PhaseConfig) -> None:
        """Launch training subprocess."""
        policy_type = phase.policy.get("type", "act")
        dataset_repo_id = phase.dataset.get("repo_id", "PhangHongHao/so101_sim")
        dataset_root = phase.dataset.get("root")
        video_backend = phase.dataset.get("video_backend", "pyav")

        run_name = phase.id
        output_dir = str(self._project_root / "outputs" / phase.id)

        # Resolve resume checkpoint if --resume flag is set
        resume_from = None
        if self._resume:
            resume_from = TrainingLauncher.find_latest_checkpoint(output_dir)
            if resume_from:
                output_dir = TrainingLauncher._make_resume_output_dir(output_dir)
                logger.info("RESUME: found checkpoint '%s' -> weight-only resume into '%s'",
                            resume_from, output_dir)
            else:
                logger.warning("RESUME: no checkpoint found in '%s', starting fresh", output_dir)
            self._resume = False  # Only resume the first training phase

        self._proc = self._launcher.launch(
            policy_type=policy_type,
            dataset_repo_id=dataset_repo_id,
            dataset_root=dataset_root,
            device=self._device,
            policy_cfg=phase.policy,
            training_cfg=phase.training,
            run_name=run_name,
            output_dir=output_dir,
            video_backend=video_backend,
            resume_from=resume_from,
        )

        # Setup loss monitor
        self._monitor = LossMonitor(phase.monitor)
        log_file = str(self._project_root / "outputs" / "logs" / f"{run_name}.log")
        self._monitor.start(log_file)

        # Update state
        self._state.training_pid = self._proc.pid
        self._state.training_run_dir = output_dir
        self._state.current_phase_status = "running"
        self._state.retry_count = 0
        self._state.training_progress = 0.0

        logger.info("Training '%s' launched: PID=%d, device=%s", phase.id, self._proc.pid, self._device)

    def _start_evaluation(self, phase: PhaseConfig) -> None:
        """Run evaluation (synchronous)."""
        # Find best checkpoint from previous training phase
        checkpoint = self._resolve_checkpoint(phase)
        if checkpoint is None:
            logger.error("No checkpoint found for evaluation")
            self._state.current_phase_status = "failed"
            return

        scene_xml = phase.collection.get("scene_xml", "Simulation/SO101/scene.xml")
        runner = EvalRunner(phase.eval, device=self._device)

        try:
            result = runner.evaluate_checkpoint(checkpoint, scene_xml)
            self._state.eval_results[phase.id] = result
            self._state.phase_history.append({
                "phase_id": phase.id,
                "phase_type": "evaluation",
                "checkpoint": checkpoint,
                "result": result,
                "completed_at": datetime.now().isoformat(),
            })
            self._state.current_phase_status = "complete"
            logger.info("Evaluation complete: success_rate=%.2f", result.get("success_rate", 0))
        except Exception as e:
            logger.error("Evaluation failed: %s", e)
            self._state.current_phase_status = "failed"

    def _monitor_training(self) -> None:
        """Poll training progress and check for overfitting."""
        pid = self._state.training_pid

        # Check if process exited
        if not TrainingLauncher.is_running(pid):
            returncode = self._proc.returncode if self._proc else -1
            if returncode == 0:
                logger.info("Training process exited normally")
            else:
                logger.error("Training process exited (returncode=%s)", returncode)

            # Final monitor poll
            if self._monitor:
                summary = self._monitor.poll()
                if summary.get("best_loss") is not None:
                    self._state.best_loss = summary["best_loss"]

            # Training complete (max steps reached or error)
            self._handle_training_complete()
            return

        # Poll monitor
        if self._monitor:
            summary = self._monitor.poll()
            status = summary.get("status", "NO_DATA")

            # Update state
            self._state.training_current_step = summary.get("current_step", 0)
            self._state.training_current_loss = summary.get("current_loss")
            max_steps = self._get_max_steps()
            if max_steps > 0:
                self._state.training_progress = self._state.training_current_step / max_steps

            if status != "NO_DATA":
                logger.info(
                    "[%s] step=%d, loss=%s, best=%s, trend=%s",
                    self._state.current_phase_id,
                    summary.get("current_step", 0),
                    f"{summary['current_loss']:.4f}" if summary.get("current_loss") else "?",
                    f"{summary['best_loss']:.4f}" if summary.get("best_loss") else "?",
                    summary.get("loss_trend", "?"),
                )

            if status == "OVERFITTING":
                logger.warning("Overfitting detected -- stopping training")
                TrainingLauncher.graceful_stop(pid)
                self._handle_training_complete()

    def _handle_training_complete(self) -> None:
        """Handle completed/overfitted training: save best checkpoint, record history."""
        phase_id = self._state.current_phase_id
        phase = self._phase_mgr.get_phase(phase_id)

        # Find best checkpoint
        best_ckpt = self._find_best_checkpoint(phase_id)

        # Record in history
        self._state.phase_history.append({
            "phase_id": phase_id,
            "phase_type": self._state.current_phase_type,
            "best_checkpoint_path": best_ckpt,
            "best_loss": self._state.best_loss,
            "training_run_dir": self._state.training_run_dir,
            "completed_at": datetime.now().isoformat(),
        })
        self._state.best_checkpoint_path = best_ckpt
        self._state.current_phase_status = "complete"

        logger.info("Phase '%s' complete -- best: %s (loss: %s)",
                     phase_id, best_ckpt, self._state.best_loss)

    def _advance(self) -> None:
        """Move to next phase or finish."""
        current_id = self._state.current_phase_id
        next_phase = self._phase_mgr.get_next_phase(current_id)

        if next_phase is None:
            logger.info("=== ALL PHASES COMPLETE! ===")
            logger.info("Plan '%s' finished at %s",
                         self._state.plan_name, datetime.now().isoformat())

            # Print summary
            self._print_summary()

            self._state_store.save(self._state)
            sys.exit(0)

        logger.info("Advancing: '%s' -> '%s'", current_id, next_phase.id)
        self._state.current_phase_id = next_phase.id
        self._state.current_phase_type = next_phase.phase_type
        self._state.current_phase_status = "pending"
        self._state.training_pid = None
        self._state.training_run_dir = None
        self._state.training_progress = 0.0
        self._state.retry_count = 0

    def _handle_failure(self) -> None:
        """Retry or give up."""
        phase_id = self._state.current_phase_id
        max_retries = 2

        if self._state.retry_count < max_retries:
            old_pid = self._state.training_pid
            if old_pid and TrainingLauncher.is_running(old_pid):
                logger.info("Killing previous PID %d before retry", old_pid)
                TrainingLauncher.graceful_stop(old_pid)
            self._state.retry_count += 1
            logger.warning("Phase '%s' failed -- retry %d/%d",
                            phase_id, self._state.retry_count, max_retries)
            self._state.current_phase_status = "pending"
            self._state.training_pid = None
            self._state.training_run_dir = None
        else:
            logger.error("Phase '%s' failed after %d retries -- stopping", phase_id, max_retries)
            self._state_store.save(self._state)
            sys.exit(1)

    # ── Helpers ──────────────────────────────────────────────────── #

    def _resume_monitor(self) -> None:
        """Re-create monitor for a recovered training phase."""
        phase_id = self._state.current_phase_id
        phase = self._phase_mgr.get_phase(phase_id)
        if phase is None:
            return

        self._monitor = LossMonitor(phase.monitor)
        log_file = str(self._project_root / "outputs" / "logs" / f"{phase_id}.log")
        self._monitor.start(log_file)
        logger.info("Monitor resumed for '%s'", phase_id)

    def _resolve_checkpoint(self, phase: PhaseConfig) -> Optional[str]:
        """Find the best checkpoint from a completed training phase."""
        # Check dependencies
        for dep_id in phase.depends_on:
            for hist in reversed(self._state.phase_history):
                if hist.get("phase_id") == dep_id and hist.get("best_checkpoint_path"):
                    return hist["best_checkpoint_path"]

        # Fallback: use state's best checkpoint
        if self._state.best_checkpoint_path:
            return self._state.best_checkpoint_path

        return None

    def _find_best_checkpoint(self, phase_id: str) -> Optional[str]:
        """Find the best checkpoint in the training output directory."""
        run_dir = self._state.training_run_dir
        if not run_dir:
            return None

        run_path = Path(run_dir)
        if not run_path.exists():
            return None

        # Look for pretrained_model directory (LeRobot convention)
        pretrained = run_path / "pretrained_model"
        if pretrained.exists():
            return str(pretrained)

        # Look for checkpoint directories
        checkpoints = sorted(run_path.glob("checkpoint_*"))
        if checkpoints:
            return str(checkpoints[-1])

        return None

    def _get_max_steps(self) -> int:
        """Get max training steps from current phase config."""
        phase = self._phase_mgr.get_phase(self._state.current_phase_id)
        if phase:
            return phase.training.get("max_steps", 0)
        return 0

    def _get_poll_interval(self) -> int:
        """Get poll interval based on current phase type."""
        if self._state.current_phase_type in ("training", "comparison"):
            phase = self._phase_mgr.get_phase(self._state.current_phase_id)
            if phase:
                return phase.monitor.get("poll_interval", 60)
            return 60
        return _DEFAULT_POLL

    def _print_summary(self) -> None:
        """Print final pipeline summary."""
        logger.info("=" * 60)
        logger.info("PIPELINE SUMMARY: %s", self._state.plan_name)
        for hist in self._state.phase_history:
            phase_id = hist.get("phase_id", "?")
            phase_type = hist.get("phase_type", "?")
            if phase_type == "collection":
                result = hist.get("result", {})
                logger.info("  [%s] Collection: %d episodes, %d frames",
                             phase_id, result.get("n_episodes", 0), result.get("n_frames", 0))
            elif phase_type in ("training", "comparison"):
                logger.info("  [%s] Training: loss=%s, checkpoint=%s",
                             phase_id, hist.get("best_loss"), hist.get("best_checkpoint_path"))
            elif phase_type == "evaluation":
                result = hist.get("result", {})
                logger.info("  [%s] Eval: success_rate=%.2f, avg_distance=%.4f",
                             phase_id, result.get("success_rate", 0), result.get("avg_distance", 0))
        logger.info("=" * 60)

    def _print_dry_run(self) -> None:
        """Print what would be executed without running."""
        phases = self._phase_mgr.enabled_phases
        print(f"\n{'=' * 60}")
        print(f"DRY RUN: {self._phase_mgr.plan_name}")
        print(f"Device: {self._device}")
        if self._resume:
            print(f"Resume:  enabled (weight-only from latest checkpoint)")
        print(f"{'=' * 60}")

        for i, phase in enumerate(phases):
            print(f"\nPhase {i+1}: {phase.id} ({phase.name})")
            print(f"  Type:      {phase.phase_type}")
            print(f"  Enabled:   {phase.enabled}")
            print(f"  Depends:   {phase.depends_on or 'none'}")
            print(f"  Device:    {phase.device}")

            if phase.phase_type == "collection":
                print(f"  Episodes:  {phase.collection.get('n_episodes', '?')}")
                print(f"  Trajectory: {phase.collection.get('trajectory_type', '?')}")
                print(f"  Push Hub:  {phase.collection.get('push_to_hub', False)}")
            elif phase.phase_type in ("training", "comparison"):
                print(f"  Policy:    {phase.policy.get('type', '?')}")
                print(f"  Max steps: {phase.training.get('max_steps', '?')}")
                print(f"  Batch size: {phase.training.get('batch_size', '?')}")
                print(f"  LR:        {phase.training.get('lr', '?')}")
                if self._resume:
                    output_dir = str(self._project_root / "outputs" / phase.id)
                    ckpt = TrainingLauncher.find_latest_checkpoint(output_dir)
                    if ckpt:
                        print(f"  Resume:    {ckpt}")
                    else:
                        print(f"  Resume:    no checkpoint found (will start fresh)")
            elif phase.phase_type == "evaluation":
                print(f"  Episodes:  {phase.eval.get('n_episodes', '?')}")
                print(f"  Metric:    {phase.eval.get('select_best_by', '?')}")

        print(f"\n{'=' * 60}")
        print(f"Total phases: {len(phases)} (enabled)")
        print(f"{'=' * 60}\n")

    # ── Logging ─────────────────────────────────────────────────── #

    def _setup_logging(self) -> None:
        """Configure logging to both console and file."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Also log to file
        log_dir = self._project_root / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "orchestrator.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(fh)


# ── CLI ──────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(description="SO-ARM101 Imitation Learning Pipeline Orchestrator")
    parser.add_argument("--plan", required=True, help="Path to training plan YAML")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (clear saved state)")
    parser.add_argument("--start-from", default=None, help="Start from a specific phase ID")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    parser.add_argument("--device", default=None, help="Override device (e.g. cuda:7)")
    parser.add_argument("--project-root", default=None, help="Project root directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint (weight-only, ignores optimizer state)")
    args = parser.parse_args()

    orchestrator = Arm101Orchestrator(
        plan_path=args.plan,
        project_root=args.project_root,
        fresh=args.fresh,
        start_from=args.start_from,
        dry_run=args.dry_run,
        device=args.device,
        resume=args.resume,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
