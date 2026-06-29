"""Lightweight checks for the Axis A PVQ selection path.

Run from the reddit directory:
    python test_axis_a_pvq.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))

import build_dd_seed
import select_pvq_medoids
from pvq import SCHWARTZ_KEYS, score_pvq_assignment


def test_pvq_scoring():
    item_scores = {str(i): 1 + ((i - 1) % 6) for i in range(1, 41)}
    canonical, value_means, target_vector = score_pvq_assignment(item_scores)

    assert set(canonical) == {str(i) for i in range(1, 41)}
    assert set(value_means) == set(SCHWARTZ_KEYS)
    assert set(target_vector) == set(SCHWARTZ_KEYS)
    assert all(1 <= v <= 6 for v in canonical.values())
    assert all(1.0 <= v <= 6.0 for v in value_means.values())
    assert all(0.0 <= v <= 1.0 for v in target_vector.values())

    try:
        score_pvq_assignment({"1": 7})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid PVQ responses must fail loudly")


def _sample_enriched_row(uid: str, offset: int = 0) -> dict:
    item_scores = {str(i): 1 + ((i + offset) % 6) for i in range(1, 41)}
    _, value_means, target_vector = score_pvq_assignment(item_scores)
    return {
        "user_id": uid,
        "subreddits": ["a", "b", "c"],
        "clean_corpus": [
            {"label": "USER", "text": f"{uid} training text", "sub": "a", "post_idx": 0},
            {"label": "INTERLOCUTOR", "text": "quoted context", "sub": "a", "post_idx": 0},
        ],
        "quote_signals": [],
        "heldout_post": f"{uid} heldout text",
        "pvq_item_scores": item_scores,
        "pvq_value_means": value_means,
        "target_vector": target_vector,
    }


def test_seed_schema_and_no_heldout_leakage():
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "selected.jsonl")
        out = os.path.join(tmp, "seed.jsonl")
        with open(inp, "w", encoding="utf-8") as f:
            f.write(json.dumps(_sample_enriched_row("u0")) + "\n")

        build_dd_seed.build_seed(inp, out)
        seed = json.loads(open(out, encoding="utf-8").readline())

        expected = {
            "user_id",
            "source",
            "subreddits",
            "user_corpus",
            "schwartz_json",
            "pvq_item_scores",
            "pvq_value_means",
            "quote_signals",
            "heldout_post",
        }
        assert set(seed) == expected
        assert seed["heldout_post"] == "u0 heldout text"
        assert "u0 heldout text" not in seed["user_corpus"]
        assert json.loads(seed["schwartz_json"])
        assert json.loads(seed["pvq_item_scores"])
        assert json.loads(seed["pvq_value_means"])


def test_selector_outputs_required_sidecars():
    rows = [_sample_enriched_row(f"u{i}", offset=i) for i in range(12)]
    selected = select_pvq_medoids.select_medoids(rows, k=10, seed=1)
    assert len(selected) == 10
    for row in selected:
        assert {"clean_corpus", "heldout_post", "pvq_item_scores", "pvq_value_means", "target_vector"} <= set(row)
        assert set(row["target_vector"]) == set(SCHWARTZ_KEYS)
        assert all(0.0 <= float(v) <= 1.0 for v in row["target_vector"].values())


def test_generation_prompt_boundary():
    source = open(os.path.join(HERE, "persona_pipeline_datadesigner.py"), encoding="utf-8").read()
    assert "LATENT VALUE PROFILE" in source
    assert "{{ schwartz_json }}" in source

    no_psych_start = source.index("REDDIT_NO_PSYCH_PARAGRAPH_PROMPT")
    no_psych_end = source.index('"""', source.index('"""', no_psych_start) + 3) + 3
    no_psych_prompt = source[no_psych_start:no_psych_end]
    assert "schwartz_json" not in no_psych_prompt
    assert "Schwartz" not in no_psych_prompt
    assert "LATENT VALUE PROFILE" not in no_psych_prompt


def main():
    test_pvq_scoring()
    test_seed_schema_and_no_heldout_leakage()
    test_selector_outputs_required_sidecars()
    test_generation_prompt_boundary()
    print("ALL AXIS A PVQ TESTS PASSED")


if __name__ == "__main__":
    main()
