"""Crash-recovery state store using atomic JSON writes.

Ported from Z1's state_store.py. Adapted for imitation learning pipeline:
- Replaces RL-specific fields (best_reward) with IL equivalents (best_loss)
- Adds collection/eval progress tracking
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class OrchestratorState:
    """Serializable snapshot of the orchestrator's progress."""

    plan_name: str = ""
    current_phase_id: str = ""           # e.g. "collect", "train_act"
    current_phase_type: str = ""         # collection / training / evaluation / comparison
    current_phase_status: str = "pending"  # pending / running / complete / failed

    # Training tracking
    training_pid: Optional[int] = None
    training_run_dir: Optional[str] = None
    best_checkpoint_path: Optional[str] = None
    best_loss: Optional[float] = None

    # Progress tracking
    collection_progress: float = 0.0     # 0.0 - 1.0
    training_progress: float = 0.0       # 0.0 - 1.0 (current_step / max_steps)
    training_current_step: int = 0
    training_current_loss: Optional[float] = None

    # Eval results
    eval_results: dict = field(default_factory=dict)  # {phase_id: {success_rate, avg_distance, video_path}}

    # History
    phase_history: list[dict] = field(default_factory=list)
    retry_count: int = 0

    # Timestamps
    started_at: str = ""
    updated_at: str = ""

    def touch(self) -> None:
        """Refresh ``updated_at`` timestamp."""
        self.updated_at = datetime.now().isoformat()


class StateStore:
    """Atomic-read / atomic-write JSON persistence for :class:`OrchestratorState`."""

    def __init__(self, path: str | Path = "orchestrator_state.json"):
        self._path = Path(path)

    def load(self) -> Optional[OrchestratorState]:
        """Load state from disk.  Returns *None* if file does not exist or is corrupt."""
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            valid_keys = {f.name for f in OrchestratorState.__dataclass_fields__.values()}
            filtered = {k: v for k, v in data.items() if k in valid_keys}
            return OrchestratorState(**filtered)
        except Exception:
            return None

    def save(self, state: OrchestratorState) -> None:
        """Atomically write *state* to disk (write-to-tmp + rename)."""
        state.touch()
        payload = asdict(state)
        blob = json.dumps(payload, indent=2, ensure_ascii=False)

        dir_path = self._path.parent
        dir_path.mkdir(parents=True, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(blob)
            os.replace(tmp_path, str(self._path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def clear(self) -> None:
        """Remove the state file (used with ``--fresh``)."""
        try:
            self._path.unlink()
        except FileNotFoundError:
            pass
