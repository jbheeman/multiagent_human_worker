"""Backend interface for MCTS rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class RolloutRequest:
    bench: str
    domain: str
    task_id: str
    persona_id: str
    arm: str
    model: str
    sim_model: str
    sim_seed: int
    block: int
    persona_yaml_path: Path | None  # None => fixed_prompt / no persona
    persona_yaml_text: str | None
    eval_dir: Path
    dry_run: bool = False
    no_satisfaction: bool = True  # STATE-Bench: skip turn-level critic (default for unofficial)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RolloutResult:
    terminal_state: str
    task_success: bool
    transfer: bool
    n_turns: int
    full_transcript: list[Any]
    artifact_path: Path | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class BenchBackend(Protocol):
    name: str

    def run(self, req: RolloutRequest) -> RolloutResult:
        ...
