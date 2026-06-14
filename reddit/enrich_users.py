import json
import os
import re
import time
import random
from functools import wraps

# ---------------------------------------------------------------------------
# Per-user enrichment pass (v2) for the AAAI rework.
#
# Canonical input is the PER-USER history file (one record per user, with a
# cross-subreddit `history` and a single per-user `target_vector`). For every
# user we, IN THIS ORDER:
#   1. hold out exactly one post (leakage-safe) as a behavioral target,
#   2. RE-INFER the Schwartz vector on the held-out-removed corpus (so the
#      target vector never sees the held-out post),
#   3. infer coarse demographics.
# We intentionally do NOT extract OCEAN here: the source-conditioned construct
# is Schwartz + held-out behavior (see plan). OCEAN only lives in the Nemotron
# baseline via its own native population grounding.
#
# Set MOCK_LLM=1 to run the whole pass offline with deterministic stub values
# (no NAUT_API_KEY / network required) for structural validation.
# ---------------------------------------------------------------------------

INPUT_FILE = os.getenv("ENRICH_INPUT", "thousand_reddit_enriched.jsonl")
OUTPUT_FILE = os.getenv("ENRICH_OUTPUT", "thousand_reddit_enriched_v2.jsonl")

# Per-user train/val/test split sizes (carved from the enriched users).
TRAIN_SIZE = int(os.getenv("TRAIN_SIZE", "700"))
VAL_SIZE = int(os.getenv("VAL_SIZE", "150"))
TEST_SIZE = int(os.getenv("TEST_SIZE", "150"))

MOCK_LLM = bool(os.getenv("MOCK_LLM"))

SCHWARTZ_KEYS = [
    "POWER", "ACHIEVEMENT", "HEDONISM", "STIMULATION", "SELF_DIRECTION",
    "UNIVERSALISM", "BENEVOLENCE", "TRADITION", "CONFORMITY", "SECURITY",
]


def retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0):
    """Decorator to retry a function with exponential backoff on timeout or connection errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    is_timeout = "timeout" in error_str or "timed out" in error_str
                    if is_timeout or "connection" in error_str:
                        if attempt == max_retries - 1:
                            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Giving up.")
                            return None
                        print(f"Attempt {attempt + 1}/{max_retries} failed: {e}. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        print(f"Non-retryable error: {e}")
                        return None
            return None
        return wrapper
    return decorator


# The client is only constructed/used when not mocking (lazy import so the
# offline MOCK_LLM path needs no httpx/openai installed).
client = None
if not MOCK_LLM:
    import httpx
    from openai import OpenAI
    http_client = httpx.Client(verify=False)
    client = OpenAI(
        api_key=os.getenv("NAUT_API_KEY"),
        base_url="https://ellm.nrp-nautilus.io/v1",
        http_client=http_client,
    )


def extract_json(raw_text):
    """Robust JSON extractor."""
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except Exception:
        pass
    return None


def hold_out_one_post(history, user_id):
    """Deterministically remove exactly one post from `history` and return it.

    Returns (new_history, heldout) where heldout = {"subreddit", "post"} or None
    if the user has no usable posts. The removal is keyed on user_id so it is
    reproducible across runs without touching the global RNG.
    """
    rng = random.Random(user_id)
    # Flatten to (history_index, post_index) addressable slots.
    slots = []
    for hi, item in enumerate(history):
        for pi, _post in enumerate(item.get("posts", [])):
            slots.append((hi, pi))
    if not slots:
        return history, None

    hi, pi = rng.choice(slots)
    # Deep-ish copy so we don't mutate the caller's structure.
    new_history = [
        {"subreddit": item.get("subreddit"), "posts": list(item.get("posts", []))}
        for item in history
    ]
    heldout_post = new_history[hi]["posts"].pop(pi)
    heldout = {"subreddit": new_history[hi]["subreddit"], "post": heldout_post}
    # Drop now-empty subreddits so the corpus stays clean.
    new_history = [it for it in new_history if it["posts"]]
    return new_history, heldout


def _history_str(history):
    out = ""
    for item in history:
        out += f"\n[SUBREDDIT: r/{item.get('subreddit')}]\n" + "\n".join(item.get("posts", [])[:3])
    return out


@retry_with_backoff()
def get_schwartz_vector_chameleon(user_id, history):
    """Infer the 10-dim Schwartz value vector ([0,1]) from a user's history."""
    if MOCK_LLM:
        return {k: round(0.3 + 0.05 * (i % 7), 2) for i, k in enumerate(SCHWARTZ_KEYS)}

    history_str = _history_str(history)
    schwartz_defs = """
    1. POWER: Social status, prestige, control/dominance over people/resources.
    2. ACHIEVEMENT: Personal success through demonstrating competence.
    3. HEDONISM: Pleasure and sensuous gratification for oneself.
    4. STIMULATION: Excitement, novelty, and challenge in life.
    5. SELF-DIRECTION: Independent thought and action, choosing, creating.
    6. UNIVERSALISM: Understanding, appreciation, tolerance, and protection for the welfare of all people and for nature.
    7. BENEVOLENCE: Preservation and enhancement of the welfare of people with whom one is in frequent personal contact.
    8. TRADITION: Respect, commitment, and acceptance of the customs and ideas that traditional culture or religion provide.
    9. CONFORMITY: Restraint of actions, inclinations, and impulses likely to upset or harm others and violate social expectations.
    10. SECURITY: Safety, harmony, and stability of society, of relationships, and of self.
    """
    prompt = f"""
    You are a Quantitative Psychologist. Analyze the provided Reddit post history to infer the user's stable personal values according to the Schwartz Theory of Basic Human Values.
    Identify the STABLE underlying Schwartz Values that persist across these different social contexts.

    DEFINITIONS:
    {schwartz_defs}

    POST HISTORY FOR USER {user_id}:
    {history_str[:6000]}

    TASK:
    Based on the linguistic cues, tone, and stated beliefs in these posts, estimate the relative importance of each value to this individual on a scale of 0.0 (Not at all important/rejected) to 1.0 (Most important/central to identity).

    Return ONLY a raw JSON object:
    {{
      "POWER": 0.0, "ACHIEVEMENT": 0.0, "HEDONISM": 0.0, "STIMULATION": 0.0, "SELF_DIRECTION": 0.0,
      "UNIVERSALISM": 0.0, "BENEVOLENCE": 0.0, "TRADITION": 0.0, "CONFORMITY": 0.0, "SECURITY": 0.0
    }}
    """
    resp = client.chat.completions.create(model="qwen3", messages=[{"role": "user", "content": prompt}])
    vector = extract_json(resp.choices[0].message.content)
    # Keep only canonical keys to preserve the verbatim-copy contract downstream.
    if vector:
        return {k: float(vector.get(k, 0.0)) for k in SCHWARTZ_KEYS}
    return None


@retry_with_backoff()
def get_demographics(user_id, history):
    """Infer coarse demographics from a user's history (logic mirrors process_tldr)."""
    if MOCK_LLM:
        return {"age": "30", "gender": "non-binary", "occupation": "unknown", "location": "unknown"}

    all_text = " ".join(p for item in history for p in item.get("posts", []))
    prompt = f"""
    Analyze this user's post history and extract their demographics.

    USER HISTORY:
    {all_text[:4000]}

    CRITICAL INSTRUCTION:
    - For 'age', estimate the MOST LIKELY single integer. Do not give ranges.
    - For 'gender', pick 'male', 'female', or 'non-binary' based on the strongest cues.

    Required JSON Structure:
    {{
        "age": "integer",
        "gender": "string",
        "occupation": "string",
        "location": "string"
    }}
    """
    resp = client.chat.completions.create(model="gpt-oss", messages=[{"role": "user", "content": prompt}])
    return extract_json(resp.choices[0].message.content)


def enrich():
    print(f"Starting v2 enrichment: {INPUT_FILE} -> {OUTPUT_FILE} (MOCK_LLM={MOCK_LLM})")
    existing = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    existing.add(json.loads(line)["user_id"])
                except Exception:
                    pass
        print(f"Resuming: {len(existing)} users already enriched.")

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
            open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            user_id = data.get("user_id")
            if not user_id or user_id in existing:
                continue

            history = data.get("history", [])
            # 1. Hold out one post BEFORE any inference (leakage-safe).
            history, heldout = hold_out_one_post(history, user_id)
            if heldout is None:
                print(f"  -> {user_id}: no posts to hold out, skipping.")
                continue

            # 2. Re-infer Schwartz on the held-out-removed corpus.
            vector = get_schwartz_vector_chameleon(user_id, history)
            if not vector:
                print(f"  -> {user_id}: Schwartz inference failed, skipping.")
                continue

            # 3. Infer demographics.
            demographics = get_demographics(user_id, history) or {
                "age": "unknown", "gender": "unknown",
                "occupation": "unknown", "location": "unknown",
            }

            record = {
                "user_id": user_id,
                "subreddits": [it.get("subreddit") for it in history],
                "history": history,
                "heldout": heldout,
                "target_vector": vector,
                "demographics": demographics,
            }
            fout.write(json.dumps(record) + "\n")
            fout.flush()
            if i % 25 == 0:
                print(f"  [{i}] enriched {user_id}")

    print("Enrichment done.")


def split_into_train_val_test():
    """Carve per-user train/val/test from the enriched v2 file (reproducible)."""
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        users = [json.loads(line) for line in f if line.strip()]
    random.seed(42)
    random.shuffle(users)

    total_needed = TRAIN_SIZE + VAL_SIZE + TEST_SIZE
    if len(users) < total_needed:
        print(f"Warning: only {len(users)} enriched users; splitting 70/15/15.")
        train_end = int(len(users) * 0.7)
        val_end = int(len(users) * 0.85)
    else:
        train_end = TRAIN_SIZE
        val_end = TRAIN_SIZE + VAL_SIZE

    splits = {
        "train_reddit_v2.jsonl": users[:train_end],
        "val_reddit_v2.jsonl": users[train_end:val_end],
        "test_reddit_v2.jsonl": users[val_end:val_end + TEST_SIZE],
    }
    for fname, rows in splits.items():
        with open(fname, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(rows)} users to {fname}")


if __name__ == "__main__":
    enrich()
    split_into_train_val_test()
