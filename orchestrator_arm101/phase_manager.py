"""Phase-based training plan parser with three-layer config merging.

Ported from Z1's phase_manager.py. Adapted for imitation learning:
- Phase types: collection / training / evaluation / comparison
- Config sections: dataset / policy / training / collection / eval
- depends_on for DAG ordering
- enabled flag for optional phases
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PhaseConfig:
    """Fully-resolved configuration for a single pipeline phase."""

    id: str
    name: str
    phase_type: str                  # collection / training / evaluation / comparison
    enabled: bool = True
    depends_on: list[str] = field(default_factory=list)

    # Config sections (3-layer merged)
    dataset: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    collection: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)
    monitor: dict = field(default_factory=dict)
    device: str = "cuda:0"


class PhaseManager:
    """Parse an IL pipeline YAML plan and manage config merging."""

    def __init__(self, plan_path: str | Path):
        self._plan_path = Path(plan_path)
        self._plan_name: str = ""
        self._defaults: dict = {}
        self._phases: list[PhaseConfig] = []
        self._device: str = "cuda:0"
        self._parse()

    def _parse(self) -> None:
        if not self._plan_path.exists():
            raise FileNotFoundError(f"Training plan not found: {self._plan_path}")

        with open(self._plan_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        if not data or "phases" not in data:
            raise ValueError(f"Invalid training plan: missing 'phases' key in {self._plan_path}")

        self._plan_name = data.get("plan_name", self._plan_path.stem)
        self._defaults = data.get("defaults", {})
        self._device = data.get("device", "cuda:0")

        for raw_phase in data["phases"]:
            phase = self._parse_phase(raw_phase)
            self._phases.append(phase)

        # Filter out disabled phases
        enabled = [p for p in self._phases if p.enabled]
        logger.info(
            "Loaded plan '%s': %d phases (enabled: %s)",
            self._plan_name,
            len(enabled),
            [p.id for p in enabled],
        )

    def _parse_phase(self, raw: dict) -> PhaseConfig:
        phase_id = raw["id"]
        phase_type = raw.get("type", "training")

        # Three-layer merge: defaults → phase
        merged_dataset = _deep_merge(self._defaults.get("dataset", {}), raw.get("dataset", {}))
        merged_policy = _deep_merge(self._defaults.get("policy", {}), raw.get("policy", {}))
        merged_training = _deep_merge(self._defaults.get("training", {}), raw.get("training", {}))
        merged_collection = _deep_merge(self._defaults.get("collection", {}), raw.get("collection", {}))
        merged_eval = _deep_merge(self._defaults.get("eval", {}), raw.get("eval", {}))
        merged_monitor = _deep_merge(self._defaults.get("monitor", {}), raw.get("monitor", {}))

        return PhaseConfig(
            id=phase_id,
            name=raw.get("name", phase_id),
            phase_type=phase_type,
            enabled=raw.get("enabled", True),
            depends_on=raw.get("depends_on", []),
            dataset=merged_dataset,
            policy=merged_policy,
            training=merged_training,
            collection=merged_collection,
            eval=merged_eval,
            monitor=merged_monitor,
            device=raw.get("device", self._device),
        )

    # ── Properties ──────────────────────────────────────────────── #

    @property
    def plan_name(self) -> str:
        return self._plan_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def phases(self) -> list[PhaseConfig]:
        return list(self._phases)

    @property
    def enabled_phases(self) -> list[PhaseConfig]:
        return [p for p in self._phases if p.enabled]

    @property
    def defaults(self) -> dict:
        return copy.deepcopy(self._defaults)

    # ── Queries ─────────────────────────────────────────────────── #

    def get_phase(self, phase_id: str) -> Optional[PhaseConfig]:
        for p in self._phases:
            if p.id == phase_id:
                return p
        return None

    def get_phase_index(self, phase_id: str) -> int:
        enabled = self.enabled_phases
        for i, p in enumerate(enabled):
            if p.id == phase_id:
                return i
        raise ValueError(f"Unknown phase id: {phase_id}")

    def get_next_phase(self, current_id: str) -> Optional[PhaseConfig]:
        enabled = self.enabled_phases
        for i, p in enumerate(enabled):
            if p.id == current_id:
                if i + 1 < len(enabled):
                    return enabled[i + 1]
                return None
        return None

    def get_start_phase_id(self, start_from: Optional[str] = None) -> str:
        if start_from:
            if self.get_phase(start_from) is None:
                raise ValueError(f"--start-from phase '{start_from}' not in plan")
            return start_from
        enabled = self.enabled_phases
        if not enabled:
            raise ValueError("Training plan has no enabled phases")
        return enabled[0].id


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result
