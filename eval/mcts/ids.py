"""Persona ID resolution for Ablation/Reddit and external (nemotron-style) rows."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def sanitize_stem(name: str) -> str:
    cleaned = name.replace("/", "__").replace("\\", "__").strip()
    return cleaned or "persona"


def persona_id_from_row(row: dict[str, Any], index: int | None = None) -> str:
    """Resolve canonical persona_id from a JSONL row.

    Priority:
      1. user_id (Ablation / Reddit arms)
      2. persona.id (nemotron-style / yaml id)
      3. person.uuid
      4. index fallback (only if index provided)
    """
    user_id = row.get("user_id")
    if user_id is not None and str(user_id).strip():
        return str(user_id).strip()

    persona = row.get("persona")
    if isinstance(persona, dict):
        pid = persona.get("id")
        if pid is not None and str(pid).strip():
            return str(pid).strip()

    person = row.get("person")
    if isinstance(person, dict):
        uuid = person.get("uuid")
        if uuid is not None and str(uuid).strip():
            return str(uuid).strip()

    yaml_text = row.get("persona_yaml") or ""
    if isinstance(yaml_text, str) and yaml_text:
        match = re.search(r"^\s*id:\s*['\"]?(.+?)['\"]?\s*$", yaml_text, re.MULTILINE)
        if match:
            return match.group(1).strip()

    if index is not None:
        return f"persona_{index:04d}"
    raise ValueError("Row has no resolvable persona_id (user_id / persona.id / person.uuid)")


def load_jsonl_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def index_arm_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    """Map persona_id -> row. Duplicate IDs hard-fail."""
    index: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(load_jsonl_rows(path)):
        pid = persona_id_from_row(row, index=i)
        if pid in index:
            raise ValueError(f"Duplicate persona_id {pid!r} in {path}")
        index[pid] = row
    return index


def assert_arm_covers_ids(arm_name: str, arm_index: dict[str, Any], required_ids: set[str]) -> None:
    missing = sorted(required_ids - set(arm_index))
    if missing:
        preview = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise SystemExit(
            f"Arm {arm_name!r} missing {len(missing)} persona_id(s) required by assignment: "
            f"{preview}{more}"
        )
