"""Schwartz sidecar parsing and dominant-value / higher-order strata."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .ids import persona_id_from_row

SCHWARTZ_VALUES = (
    "POWER",
    "ACHIEVEMENT",
    "HEDONISM",
    "STIMULATION",
    "SELF_DIRECTION",
    "UNIVERSALISM",
    "BENEVOLENCE",
    "TRADITION",
    "CONFORMITY",
    "SECURITY",
)

HIGHER_ORDER: dict[str, tuple[str, ...]] = {
    "OPENNESS_TO_CHANGE": ("SELF_DIRECTION", "STIMULATION", "HEDONISM"),
    "SELF_ENHANCEMENT": ("POWER", "ACHIEVEMENT"),
    "CONSERVATION": ("SECURITY", "CONFORMITY", "TRADITION"),
    "SELF_TRANSCENDENCE": ("UNIVERSALISM", "BENEVOLENCE"),
}

DOMINANCE_FALLBACK_THRESHOLD = 0.40


def parse_schwartz_vector(raw: Any) -> dict[str, float]:
    """Normalize schwartz_json (dict or JSON string) to {VALUE: float}."""
    if raw is None:
        raise ValueError("schwartz_json is missing")
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise ValueError(f"schwartz_json must be a dict, got {type(raw)}")

    # Accept target_vector nesting or flat 10-dim map.
    if "values" in raw and isinstance(raw["values"], dict):
        raw = raw["values"]
    if "target_vector" in raw and isinstance(raw["target_vector"], dict):
        raw = raw["target_vector"]

    out: dict[str, float] = {}
    for key, value in raw.items():
        name = str(key).upper().replace(" ", "_").replace("-", "_")
        if isinstance(value, (int, float)):
            out[name] = float(value)
    if not out:
        raise ValueError(f"No numeric Schwartz dimensions in {raw!r}")
    return out


def argmax_tie_break(scores: dict[str, float]) -> str:
    """Argmax with deterministic lexicographic tie-break on key."""
    if not scores:
        raise ValueError("empty scores for argmax")
    # Sort by (-score, key) so highest score wins; ties → lexicographically first key.
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def dominant_value(vector: dict[str, float]) -> str:
    """Dominant Schwartz value (10-dim), lex tie-break."""
    filtered = {k: vector[k] for k in SCHWARTZ_VALUES if k in vector}
    if not filtered:
        # Fall back to whatever keys exist.
        filtered = vector
    return argmax_tie_break(filtered)


def higher_order_stratum(vector: dict[str, float]) -> str:
    """Map 10-dim vector to one of 4 higher-order groups via summed scores."""
    group_scores: dict[str, float] = {}
    for group, members in HIGHER_ORDER.items():
        group_scores[group] = sum(vector.get(m, 0.0) for m in members)
    return argmax_tie_break(group_scores)


def choose_strata_mode(dominant_counts: Counter[str], n: int) -> tuple[int, str]:
    """Return (S, strata_mode). Fallback to S=4 if any value >40% of pool."""
    if n <= 0:
        raise ValueError("empty persona pool")
    max_share = max(dominant_counts.values()) / n if dominant_counts else 0.0
    if max_share > DOMINANCE_FALLBACK_THRESHOLD:
        return 4, "higher_order_S4"
    return 10, "schwartz_value_S10"


def load_schwartz_by_persona_id(path: Path) -> dict[str, dict[str, float]]:
    """Load persona_id -> Schwartz vector from seed/sidecar JSONL."""
    out: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pid = persona_id_from_row(row, index=i)
            if "schwartz_json" not in row:
                raise ValueError(f"Row {i} ({pid}) missing schwartz_json in {path}")
            out[pid] = parse_schwartz_vector(row["schwartz_json"])
    if not out:
        raise ValueError(f"No rows in {path}")
    return out


def assign_strata(
    vectors: dict[str, dict[str, float]],
) -> tuple[dict[str, str], int, str, Counter[str]]:
    """Assign each persona_id a stratum label.

    Returns (persona_id -> stratum, S, strata_mode, dominant_value_counts).
    """
    dominant_by_id = {pid: dominant_value(vec) for pid, vec in vectors.items()}
    counts = Counter(dominant_by_id.values())
    s, mode = choose_strata_mode(counts, len(vectors))
    if s == 10:
        strata = dict(dominant_by_id)
    else:
        strata = {pid: higher_order_stratum(vec) for pid, vec in vectors.items()}
    return strata, s, mode, counts


def group_by_stratum(strata: dict[str, str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for pid, label in strata.items():
        groups[label].append(pid)
    return dict(groups)
