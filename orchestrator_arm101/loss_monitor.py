"""Training log monitor — parse loss from log file and detect overfitting.

Unlike Z1's TensorBoard-based monitor, this polls the training log file
directly and parses loss values using regex.

Status values:
  TRAINING   — loss is still decreasing or stable
  CONVERGED  — loss plateau detected (<threshold% change over N polls)
  OVERFITTING — loss increased significantly from best (>threshold%)
  COMPLETE   — training process finished
  NO_DATA    — no loss values found yet
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Regex patterns to match loss in lerobot-train log output
LOSS_PATTERNS = [
    re.compile(r"\bloss:([\d.]+)", re.IGNORECASE),        # "loss:0.115" (no space)
    re.compile(r"loss:\s+([\d.]+)", re.IGNORECASE),        # "loss: 0.115" (with space)
    re.compile(r"train_loss[=:\s]+([\d.]+)", re.IGNORECASE),
    re.compile(r"'loss':\s*([\d.]+)", re.IGNORECASE),
]

STEP_PATTERNS = [
    re.compile(r"(\d+)/\d+\s", re.IGNORECASE),            # "1234/50000 " from tqdm
    re.compile(r"step[=:\s]+(\d+)K\b", re.IGNORECASE),    # "step:6K" → 6000
    re.compile(r"step[=:\s]+(\d+)M\b", re.IGNORECASE),    # "step:1M" → 1000000
    re.compile(r"step[=:\s]+(\d+)(?!\d)", re.IGNORECASE),  # "step:100"
]


@dataclass
class MonitorState:
    """Internal state for the loss monitor."""
    log_file: str = ""
    best_loss: float = float("inf")
    best_step: int = 0
    current_loss: float = float("inf")
    current_step: int = 0
    loss_history: list[float] = field(default_factory=list)
    last_file_pos: int = 0
    last_poll_count: int = 0


class LossMonitor:
    """Monitor training progress by parsing log file for loss values."""

    def __init__(self, config: dict):
        self.poll_interval = config.get("poll_interval", 60)
        self.plateau_window = config.get("plateau_window", 10)
        self.plateau_threshold = config.get("plateau_threshold", 0.01)
        self.loss_increase_threshold = config.get("loss_increase_threshold", 0.20)
        self.min_steps_before_detect = config.get("min_steps", 500)
        self._state: Optional[MonitorState] = None

    def start(self, log_file: str) -> None:
        """Initialize monitoring for a log file."""
        self._state = MonitorState(log_file=log_file)
        logger.info("Loss monitor started: %s (poll every %ds)", log_file, self.poll_interval)

    def poll(self) -> dict:
        """Read new log lines, parse loss, check for overfitting.

        Returns summary dict with keys:
          status, current_step, current_loss, best_loss, best_step, loss_trend
        """
        if self._state is None:
            return {"status": "NO_DATA"}

        # Read new lines from log file
        log_path = Path(self._state.log_file)
        if not log_path.exists():
            return {"status": "NO_DATA"}

        new_losses = []
        new_steps = []
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._state.last_file_pos)
                raw = fh.read()
                self._state.last_file_pos = fh.tell()
            # tqdm uses \r to overwrite progress bars; split on both \r and \n
            for line in raw.replace("\r", "\n").split("\n"):
                loss = self._parse_loss(line)
                if loss is not None:
                    new_losses.append(loss)
                step = self._parse_step(line)
                if step is not None:
                    new_steps.append(step)
        except Exception as e:
            logger.warning("Error reading log file: %s", e)
            return {"status": "NO_DATA"}

        # Update state with ALL parsed losses (needed for plateau detection)
        if new_losses:
            for loss_val in new_losses:
                self._state.loss_history.append(loss_val)
                if loss_val < self._state.best_loss:
                    self._state.best_loss = loss_val
                    self._state.best_step = self._state.current_step
            self._state.current_loss = new_losses[-1]

        if new_steps:
            self._state.current_step = max(new_steps)

        self._state.last_poll_count += 1

        # Check status
        status = self._check_status()

        return {
            "status": status,
            "current_step": self._state.current_step,
            "current_loss": self._state.current_loss,
            "best_loss": self._state.best_loss if self._state.best_loss != float("inf") else None,
            "best_step": self._state.best_step,
            "loss_trend": self._compute_trend(),
        }

    def _check_status(self) -> str:
        """Determine training status based on loss history."""
        s = self._state

        if s.current_step < self.min_steps_before_detect:
            return "TRAINING"

        if s.best_loss == float("inf"):
            return "NO_DATA"

        # Check for loss increase (overfitting)
        if s.current_loss > s.best_loss * (1 + self.loss_increase_threshold):
            logger.warning("OVERFITTING: current_loss=%.4f > best_loss=%.4f * %.2f",
                           s.current_loss, s.best_loss, 1 + self.loss_increase_threshold)
            return "OVERFITTING"

        # Check for plateau
        recent = s.loss_history[-self.plateau_window:]
        if len(recent) >= self.plateau_window:
            max_loss = max(recent)
            min_loss = min(recent)
            if max_loss > 0:
                relative_change = (max_loss - min_loss) / max_loss
                if relative_change < self.plateau_threshold:
                    logger.info("CONVERGED: loss plateau (%.4f change over %d polls)",
                                relative_change, self.plateau_window)
                    return "CONVERGED"

        return "TRAINING"

    def _compute_trend(self) -> str:
        """Compute recent loss trend: decreasing / stable / increasing."""
        recent = self._state.loss_history[-5:]
        if len(recent) < 2:
            return "unknown"

        diffs = [recent[i] - recent[i-1] for i in range(1, len(recent))]
        avg_diff = sum(diffs) / len(diffs)

        if recent[-1] == 0:
            return "unknown"
        relative = abs(avg_diff) / recent[-1]

        if relative < 0.005:
            return "stable"
        elif avg_diff < 0:
            return "decreasing"
        else:
            return "increasing"

    @property
    def state(self) -> Optional[MonitorState]:
        return self._state

    @staticmethod
    def _parse_loss(line: str) -> Optional[float]:
        for pattern in LOSS_PATTERNS:
            m = pattern.search(line)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    continue
        return None

    @staticmethod
    def _parse_step(line: str) -> Optional[int]:
        for i, pattern in enumerate(STEP_PATTERNS):
            m = pattern.search(line)
            if m:
                try:
                    val = int(m.group(1))
                    # Patterns 1 and 2 have K/M suffix
                    if i == 1:  # K suffix
                        val *= 1000
                    elif i == 2:  # M suffix
                        val *= 1_000_000
                    return val
                except ValueError:
                    continue
        return None
