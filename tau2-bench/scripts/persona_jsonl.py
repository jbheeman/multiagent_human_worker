"""Load persona YAML paths from a JSONL file (persona_yaml column)."""

from __future__ import annotations

import json
import re
from pathlib import Path


def sanitize_stem(name: str) -> str:
    """Filesystem-safe persona label."""
    cleaned = name.replace("/", "__").replace("\\", "__").strip()
    return cleaned or "persona"


def persona_stem_from_row(row: dict, index: int) -> str:
    """Derive a stable filename stem for a JSONL persona row."""
    user_id = row.get("user_id")
    if user_id:
        return sanitize_stem(str(user_id))

    yaml_text = row.get("persona_yaml") or ""
    match = re.search(r"^name:\s*['\"]?(.+?)['\"]?\s*$", yaml_text, re.MULTILINE)
    if match:
        return sanitize_stem(match.group(1).strip())

    persona = row.get("persona")
    if isinstance(persona, dict):
        if persona.get("name"):
            return sanitize_stem(str(persona["name"]))
        if persona.get("id"):
            return sanitize_stem(str(persona["id"]))

    return f"persona_{index:04d}"


def load_jsonl_rows(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def materialize_personas_from_jsonl(
    jsonl_path: Path,
    cache_dir: Path | None = None,
    limit: int | None = None,
) -> list[Path]:
    """
    Write persona_yaml from each JSONL row to <cache_dir>/<stem>.yaml.

    Returns persona file paths in JSONL order (not sorted).
    """
    jsonl_path = jsonl_path.resolve()
    if cache_dir is None:
        cache_dir = jsonl_path.parent / f".personas_yaml_cache_{jsonl_path.stem}"
    else:
        cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl_rows(jsonl_path, limit=limit)
    if not rows:
        raise ValueError(f"No rows in {jsonl_path}")

    seen_stems: dict[str, int] = {}
    paths: list[Path] = []
    for index, row in enumerate(rows):
        yaml_text = row.get("persona_yaml")
        if not yaml_text:
            raise ValueError(f"Row {index} missing persona_yaml in {jsonl_path}")

        stem = persona_stem_from_row(row, index)
        if stem in seen_stems:
            seen_stems[stem] += 1
            stem = f"{stem}_{seen_stems[stem]}"
        else:
            seen_stems[stem] = 0

        out_path = cache_dir / f"{stem}.yaml"
        out_path.write_text(yaml_text, encoding="utf-8")
        paths.append(out_path)

    return paths
