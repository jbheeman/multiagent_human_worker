"""Build the NeMo Data Designer GEPA seed from the v2 enriched per-user file.

Pure transformation (no API): turns each enriched user record into one seed row
that the Data Designer pipeline (persona_pipeline_datadesigner.py) consumes. The
Nemotron baseline arm draws from its OWN native Nemotron-Personas population and
does not use this seed.

Seed columns produced (all strings, so they template cleanly into prompts):
  user_id, source, subreddits, user_corpus, schwartz_json, quote_signals,
  heldout_post, pvq_item_scores, pvq_value_means

Note: demographics is intentionally omitted from the GEPA/Reddit arm. The Reddit arm
grounds behavior on the cleaned corpus + value vector, not fabricated demographics.
"""

import json
import os


def render_corpus(clean_corpus):
    """Render labeled segment list into a string for DD prompts.

    Preserves attribution labels so the generator knows to build the persona from
    USER turns and treat INTERLOCUTOR/QUOTED turns as context only.
    """
    lines = []
    current_sub = None
    for seg in clean_corpus:
        if seg.get("sub") != current_sub:
            current_sub = seg.get("sub")
            if current_sub:
                lines.append(f"[r/{current_sub}]")
        lines.append(f"[{seg['label']}] {seg['text']}")
    return "\n".join(lines)


def build_seed(input_file, output_file, source="reddit_gepa"):
    n = 0
    with open(input_file, "r", encoding="utf-8") as fin, \
            open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            row = {
                "user_id": r["user_id"],
                "source": source,
                "subreddits": ", ".join(s for s in r.get("subreddits", []) if s),
                # clean_corpus is the attribution-labeled training corpus (heldout removed).
                "user_corpus": render_corpus(r["clean_corpus"]),
                # Instrument-derived Schwartz sidecar for Reddit+Schwartz arms.
                # The persona YAML never includes these numbers.
                "schwartz_json": json.dumps(r["target_vector"]),
                "pvq_item_scores": json.dumps(r.get("pvq_item_scores", {})),
                "pvq_value_means": json.dumps(r.get("pvq_value_means", {})),
                # Quoted/reproduced external content + stance (empty for all-USER users).
                "quote_signals": "; ".join(
                    r.get("quote_signals") or r.get("mock_targets", [])
                ) or "none",
                "heldout_post": r.get("heldout_post", ""),
            }
            fout.write(json.dumps(row) + "\n")
            n += 1
    print(f"Wrote {n} seed rows to {output_file}")
    return n


if __name__ == "__main__":
    inp = os.getenv("SEED_INPUT", "train_reddit_v2.jsonl")
    out = os.getenv("SEED_OUTPUT", "seed_gepa.jsonl")
    build_seed(inp, out)
