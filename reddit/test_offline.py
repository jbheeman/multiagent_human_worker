"""Offline structural validation for the persona-generation rework.

Runs WITHOUT NAUT_API_KEY and without the heavy research deps (gepa, smolagents,
tau_bench, sentence-transformers). Requires only: stdlib + pydantic + pyyaml +
data-designer-config (for the DD config-build test, which is skipped if absent).

Run:  python test_offline.py
"""

import json
import os
import subprocess
import sys
import tempfile


def test_alignment_math():
    from alignment import schwartz_alignment, SCHWARTZ_KEYS
    tgt = {k: v for k, v in zip(SCHWARTZ_KEYS, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])}
    pvq = {k: 1 + 5 * v for k, v in tgt.items()}            # exact linear map -> perfect corr
    pvq_anti = {k: 1 + 5 * (1 - v) for k, v in tgt.items()}  # reversed -> anti-corr
    assert abs(schwartz_alignment(tgt, pvq) - 1.0) < 1e-9
    assert abs(schwartz_alignment(tgt, pvq_anti) - 0.0) < 1e-9
    assert schwartz_alignment({}, {}) == 0.5                 # degenerate
    assert schwartz_alignment({k: 0.5 for k in SCHWARTZ_KEYS}, pvq) == 0.5  # flat target
    assert abs(schwartz_alignment(tgt, pvq, metric="spearman") - 1.0) < 1e-9
    print("PASS test_alignment_math")


def test_enrichment_leakage_and_schema():
    import enrich_users as e  # MOCK_LLM must be set in env before import
    src = "thousand_reddit_enriched.jsonl"
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, "in.jsonl")
        out = os.path.join(d, "out.jsonl")
        with open(src) as f, open(inp, "w") as g:
            for i, line in zip(range(5), f):
                g.write(line)
        e.INPUT_FILE, e.OUTPUT_FILE = inp, out
        e.enrich()
        rows = [json.loads(l) for l in open(out)]
        assert rows, "no rows enriched"
        for r in rows:
            assert set(r) == {"user_id", "subreddits", "history", "heldout", "target_vector", "demographics"}
            corpus = [p for it in r["history"] for p in it["posts"]]
            assert r["heldout"]["post"] not in corpus, "held-out post leaked into corpus"
            assert set(r["target_vector"]) == set(e.SCHWARTZ_KEYS)
        # determinism: same user_id -> same held-out slot
        r0 = rows[0]
        hist = [{"subreddit": it["subreddit"], "posts": list(it["posts"])} for it in r0["history"]]
        hist.append({"subreddit": r0["heldout"]["subreddit"], "posts": [r0["heldout"]["post"]]})
        _, ho2 = e.hold_out_one_post(hist, r0["user_id"])
        assert ho2["post"] == r0["heldout"]["post"]
    print("PASS test_enrichment_leakage_and_schema")


def test_seed_builder():
    import build_dd_seed as b
    with tempfile.TemporaryDirectory() as d:
        inp = os.path.join(d, "v2.jsonl")
        out = os.path.join(d, "seed.jsonl")
        rec = {
            "user_id": "u", "subreddits": ["a", "b"],
            "history": [{"subreddit": "a", "posts": ["p1", "p2"]}, {"subreddit": "b", "posts": ["p3"]}],
            "heldout": {"subreddit": "a", "post": "HELDOUT"},
            "target_vector": {k: 0.5 for k in
                              ["POWER", "ACHIEVEMENT", "HEDONISM", "STIMULATION", "SELF_DIRECTION",
                               "UNIVERSALISM", "BENEVOLENCE", "TRADITION", "CONFORMITY", "SECURITY"]},
            "demographics": {"age": "30"},
        }
        with open(inp, "w") as f:
            f.write(json.dumps(rec) + "\n")
        b.build_seed(inp, out)
        row = json.loads(open(out).readline())
        assert set(row) == {"user_id", "source", "subreddits", "user_corpus",
                            "schwartz_json", "demographics", "heldout_post"}
        assert row["heldout_post"] == "HELDOUT"
        assert "HELDOUT" not in row["user_corpus"]
        assert json.loads(row["schwartz_json"])
    print("PASS test_seed_builder")


def test_schema_pin_yaml():
    try:
        import persona_pipeline_datadesigner as P
    except ImportError as ex:
        print(f"SKIP test_schema_pin_yaml (missing dep: {ex})")
        return
    schwartz_seed = json.dumps({"POWER": 0.1, "ACHIEVEMENT": 0.4, "HEDONISM": 0.7, "STIMULATION": 0.3,
                                "SELF_DIRECTION": 0.6, "UNIVERSALISM": 0.5, "BENEVOLENCE": 0.8,
                                "TRADITION": 0.2, "CONFORMITY": 0.4, "SECURITY": 0.9})
    persona = {
        "id": "u1", "demographics": "30, nb",
        "cognitive_profile": {"schwartz": {k: 0.0 for k in
                              ["self_direction", "stimulation", "hedonism", "achievement", "power",
                               "security", "conformity", "tradition", "benevolence", "universalism"]}},
        "communication_style": {"formality": "Low", "sentence_structure": "short",
                                "vocabulary_and_lexicon": "casual", "punctuation_and_formatting": "lowercase",
                                "example_utterances": ["hey", "help pls"]},
        "interaction_policy": {"gratification_delay_tolerance": "Low", "authority_challenge": "Active",
                               "policy_friction_tolerance": "Low", "verification_patience": "Low",
                               "escalation_trigger": "ignored"},
        "state_transition_rules": ["IF delayed THEN complain", "IF denied THEN escalate", "IF error THEN demand fix"],
        "termination_success": "resolved", "termination_abandonment": "gives up"}
    assert P.PersonaProfile(**persona)
    pinned = P.pin_schwartz_verbatim(persona, schwartz_seed)
    assert pinned["cognitive_profile"]["schwartz"]["security"] == 0.9
    assert pinned["cognitive_profile"]["schwartz"]["power"] == 0.1
    assert P.PersonaProfile(**pinned)
    assert P.persona_to_yaml(pinned) == P.persona_to_yaml(pinned)  # deterministic
    print("PASS test_schema_pin_yaml")


def test_dd_config_builds():
    try:
        import warnings
        warnings.filterwarnings("ignore")
        import persona_pipeline_datadesigner as P
        import data_designer.config  # noqa: F401
    except ImportError as ex:
        print(f"SKIP test_dd_config_builds (missing dep: {ex})")
        return
    with tempfile.TemporaryDirectory() as d:
        seed = os.path.join(d, "seed.jsonl")
        with open(seed, "w") as f:
            f.write(json.dumps({"user_id": "u", "source": "s", "subreddits": "a",
                                "user_corpus": "c", "schwartz_json": "{}", "demographics": "{}",
                                "heldout_post": "h"}) + "\n")
        cb = P.build_config(P.Arm.GEPA_FULL, seed, n_personas=1, seed_start=0)
        cb.build()
        assert [c.name for c in cb.get_column_configs()] == ["persona_paragraph", "persona", "value_alignment", "grounding"]
        cb2 = P.build_config(P.Arm.NEMOTRON_BASELINE, None, n_personas=1)
        cb2.build()
        assert [c.name for c in cb2.get_column_configs()] == ["person", "persona"]
    print("PASS test_dd_config_builds")


def main():
    # enrich_users reads MOCK_LLM at import time, so it must be set before any
    # test imports it. Re-exec with MOCK_LLM=1 if it isn't already set.
    if not os.getenv("MOCK_LLM"):
        env = dict(os.environ, MOCK_LLM="1")
        sys.exit(subprocess.run([sys.executable, __file__], env=env,
                                cwd=os.path.dirname(os.path.abspath(__file__))).returncode)
    test_alignment_math()
    test_enrichment_leakage_and_schema()
    test_seed_builder()
    test_schema_pin_yaml()
    test_dd_config_builds()
    print("\nALL OFFLINE TESTS PASSED")


if __name__ == "__main__":
    main()
