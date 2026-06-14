"""Build the NeMo Data Designer GEPA seed from the v2 enriched per-user file.

Pure transformation (no API): turns each enriched user record into one seed row
that the Data Designer pipeline (persona_pipeline_datadesigner.py) consumes. The
Nemotron baseline arm draws from its OWN native Nemotron-Personas population and
does not use this seed.

Seed columns produced (all strings, so they template cleanly into prompts):
  user_id, source, subreddits, user_corpus, schwartz_json, demographics, heldout_post
"""

import json
import os


def flatten_history(history):
    """Join a user's cross-subreddit history into a single corpus string."""
    chunks = []
    for item in history:
        sub = item.get("subreddit")
        for post in item.get("posts", []):
            chunks.append(f"[r/{sub}] {post}")
    return "\n---\n".join(chunks)


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
                "subreddits": ", ".join(r.get("subreddits", [])),
                "user_corpus": flatten_history(r.get("history", [])),
                # Verbatim Schwartz vector — the structured-compile column copies
                # this into cognitive_profile.schwartz and a post-pass pins it.
                "schwartz_json": json.dumps(r["target_vector"]),
                "demographics": json.dumps(r.get("demographics", {})),
                "heldout_post": r.get("heldout", {}).get("post", ""),
            }
            fout.write(json.dumps(row) + "\n")
            n += 1
    print(f"Wrote {n} seed rows to {output_file}")
    return n


if __name__ == "__main__":
    inp = os.getenv("SEED_INPUT", "train_reddit_v2.jsonl")
    out = os.getenv("SEED_OUTPUT", "seed_gepa.jsonl")
    build_seed(inp, out)
