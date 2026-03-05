"""Persona difficulty categorization.

Maps layer dimension codes to interaction difficulty (easy / moderate / challenging)
from the perspective of the customer service agent. Derives all valence information
from persona.valence (the single source of truth).

Usage:
    from persona.categories import categorize_persona, compute_difficulty_score

    codes = {"emotional_entry_state": "E6", "bandwidth": "B7", ...}
    cat = categorize_persona(codes)           # → "challenging"
    score = compute_difficulty_score(codes)   # → 0.83
"""

from typing import Literal

from persona.valence import (
    CODE_VALENCE,
    SCORED_DIMENSIONS,
    VALENCE_WEIGHT,
    Valence,
)

PersonaCategory = Literal["easy", "moderate", "challenging"]


def compute_difficulty_score(codes: dict[str, str]) -> float:
    """Compute a difficulty score in [0.0, 1.0] from a {dimension: code} dict.

    Each scored dimension contributes a weight from the 5-level valence scale:
        POSITIVE           → -2
        SLIGHTLY_POSITIVE  → -1
        NEUTRAL            →  0
        SLIGHTLY_NEGATIVE  → +1
        NEGATIVE           → +2

    Final score is linearly mapped from [-2N, +2N] → [0.0, 1.0] where N is the
    number of scored dimensions present in the codes dict.

    Returns 0.5 if no scored dimensions are present.
    """
    raw = 0
    scored = 0

    for dim in SCORED_DIMENSIONS:
        code = codes.get(dim)
        if code is None:
            continue
        valence = CODE_VALENCE.get(code)
        if valence is None:
            continue
        scored += 1
        raw += VALENCE_WEIGHT[valence]

    if scored == 0:
        return 0.5

    # Map [-2*scored, +2*scored] → [0.0, 1.0]
    return (raw + 2 * scored) / (4 * scored)


def categorize_persona(codes: dict[str, str]) -> PersonaCategory:
    """Map a {dimension: code} dict to 'easy', 'moderate', or 'challenging'.

    Thresholds (on the [0, 1] difficulty score):
        < 0.33  → easy
        < 0.67  → moderate
        ≥ 0.67  → challenging
    """
    score = compute_difficulty_score(codes)
    if score < 0.33:
        return "easy"
    if score < 0.67:
        return "moderate"
    return "challenging"


def difficulty_breakdown(codes: dict[str, str]) -> dict:
    """Return a breakdown of which dimensions are easy / neutral / challenging.

    Returns:
        {
            "score": float,
            "category": str,
            "positive": ["dim=code", ...],
            "slightly_positive": ["dim=code", ...],
            "neutral": ["dim=code", ...],
            "slightly_negative": ["dim=code", ...],
            "negative": ["dim=code", ...],
            "missing": ["dim", ...],
        }
    """
    buckets: dict[str, list[str]] = {v.value: [] for v in Valence}
    missing: list[str] = []

    for dim in sorted(SCORED_DIMENSIONS):
        code = codes.get(dim)
        if code is None:
            missing.append(dim)
            continue
        valence = CODE_VALENCE.get(code, Valence.NEUTRAL)
        buckets[valence.value].append(f"{dim}={code}")

    return {
        "score": compute_difficulty_score(codes),
        "category": categorize_persona(codes),
        **buckets,
        "missing": missing,
    }
