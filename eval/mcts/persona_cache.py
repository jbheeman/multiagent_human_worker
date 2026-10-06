"""Materialize a single persona YAML from an arm JSONL by persona_id."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .ids import index_arm_jsonl, sanitize_stem


def materialize_persona_yaml(
    *,
    arm_index: dict[str, dict[str, Any]],
    persona_id: str,
    cache_dir: Path,
) -> tuple[Path, str]:
    """Write persona_yaml for persona_id to cache_dir; return (path, yaml_text)."""
    row = arm_index.get(persona_id)
    if row is None:
        raise KeyError(f"persona_id {persona_id!r} not in arm index")
    yaml_text = row.get("persona_yaml")
    if not yaml_text:
        raise ValueError(f"persona_id {persona_id!r} missing persona_yaml")
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{sanitize_stem(persona_id)}.yaml"
    if not path.exists() or path.read_text(encoding="utf-8") != yaml_text:
        path.write_text(str(yaml_text), encoding="utf-8")
    return path, str(yaml_text)


def load_arm_or_none(arm_name: str, path: Path | None) -> dict[str, dict[str, Any]] | None:
    if path is None:
        return None
    if not path.is_file():
        raise SystemExit(f"Arm {arm_name!r} JSONL not found: {path}")
    return index_arm_jsonl(path)
