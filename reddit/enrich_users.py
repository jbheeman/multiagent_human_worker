import json
import os
import re
import time
import random
from collections import defaultdict
from functools import wraps
from dotenv import load_dotenv
from pvq import (
    pvq_survey_text,
    score_pvq_assignment,
)
# ---------------------------------------------------------------------------
# Per-user enrichment pass (v2) for the AAAI rework.
#
# For every user, IN THIS ORDER:
#   1. Deterministic pre-parse: segment posts into lines; pre-label ">"-lines
#      as INTERLOCUTOR; strip URLs. Nothing deleted, nothing re-emitted.
#   2. LLM attribution pass: classify non-INTERLOCUTOR segments by index
#      (USER vs QUOTED). LLM returns label indices only — original text is
#      preserved exactly, immune to tone laundering. Chunked for long corpora;
#      fails closed on coverage gaps.
#   3. Hold out one predominantly-USER post (post granularity, not segment).
#   4. Quote-signal extraction on QUOTED/INTERLOCUTOR turns of training corpus only
#      (post-holdout — never from held-out post; never inferred from bare USER text).
#   5. PVQ-40 administration on USER turns + quote_signals (low-weight hint),
#      scored with the Schwartz key.
#   (Demographics: dropped for Reddit arm — fabrication risk, no behavioral benefit.)
#
# Set MOCK_LLM=1 to run offline. Mock exercises the same reconstruction path.
# ---------------------------------------------------------------------------
load_dotenv()
MOCK_LLM = bool(os.getenv("MOCK_LLM"))
NAUT_API_KEY = os.getenv("NAUT_API_KEY")
if not MOCK_LLM and not NAUT_API_KEY:
    raise ValueError("NAUT_API_KEY is not set")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.getenv("ENRICH_INPUT", os.path.join(BASE_DIR, "thousand_users_raw.jsonl"))
OUTPUT_FILE = os.getenv("ENRICH_OUTPUT", os.path.join(BASE_DIR, "thousand_users_pvq_enriched.jsonl"))

TRAIN_SIZE = int(os.getenv("TRAIN_SIZE", "50"))
VAL_SIZE = int(os.getenv("VAL_SIZE", "50"))
TEST_SIZE = int(os.getenv("TEST_SIZE", "200"))
MAX_USERS = int(os.getenv("MAX_USERS", "300"))
ATTRIBUTION_CHUNK_SIZE = int(os.getenv("ATTRIBUTION_CHUNK_SIZE", "50"))

_URL_RE = re.compile(r'https?://\S+|www\.\S+')


def retry_with_backoff(max_retries=3, initial_delay=2.0, max_delay=60.0, backoff_factor=2.0):
    """Decorator for exponential backoff on timeout/connection errors."""
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


_VALID_ATTRIBUTION_LABELS = frozenset({"USER", "QUOTED"})


def _coerce_attribution_label(raw):
    """Normalize a label string; map common aliases to USER/QUOTED."""
    if raw is None:
        return None
    lbl = str(raw).strip().upper()
    aliases = {
        "FIRST_PERSON": "USER",
        "OWN": "USER",
        "QUOTE": "QUOTED",
        "QUOTATION": "QUOTED",
        "INTERLOCUTOR": "QUOTED",  # should be pre-labeled, but tolerate
    }
    lbl = aliases.get(lbl, lbl)
    return lbl if lbl in _VALID_ATTRIBUTION_LABELS else None


def _normalize_classify_result(result):
    """Parse LLM classification JSON into {idx: label}. Returns None if unparseable."""
    if not isinstance(result, dict):
        return None

    label_map = {}
    labels = result.get("labels", result.get("classifications", result.get("segments")))

    if isinstance(labels, dict):
        for key, value in labels.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if isinstance(value, str):
                lbl = _coerce_attribution_label(value)
            elif isinstance(value, dict):
                lbl = _coerce_attribution_label(
                    value.get("label") or value.get("classification") or value.get("type")
                )
            else:
                continue
            if lbl:
                label_map[idx] = lbl
    elif isinstance(labels, list):
        for entry in labels:
            if isinstance(entry, str):
                continue
            if not isinstance(entry, dict):
                continue
            idx = entry.get("idx", entry.get("index", entry.get("id")))
            lbl = _coerce_attribution_label(
                entry.get("label") or entry.get("classification") or entry.get("type")
            )
            if idx is None or not lbl:
                continue
            try:
                label_map[int(idx)] = lbl
            except (TypeError, ValueError):
                continue

    return label_map


# ---------------------------------------------------------------------------
# Step 1: Deterministic pre-parse — no LLM
# ---------------------------------------------------------------------------

def preparse_history(history):
    """Segment all posts into lines; pre-label ">"-lines as INTERLOCUTOR; strip URLs.

    Returns a flat list of segment dicts:
      {"idx": int, "text": str, "label": "INTERLOCUTOR"|None, "sub": str, "post_idx": int}

    Nothing is deleted. INTERLOCUTOR labels are determined by Reddit ">"-syntax alone.
    Unlabeled segments (label=None) go to the LLM attribution pass.
    """
    segments = []
    idx = 0
    for post_idx, item in enumerate(history):
        sub = item.get("subreddit", "?")
        for post in item.get("posts", []):
            for line in post.splitlines():
                line = _URL_RE.sub("", line).rstrip()
                if not line.strip():
                    continue
                if re.match(r"^\s*>", line):
                    label = "INTERLOCUTOR"
                else:
                    label = None  # to be classified by LLM
                segments.append({
                    "idx": idx, "text": line.strip(),
                    "label": label, "sub": sub, "post_idx": post_idx,
                })
                idx += 1
    return segments


# ---------------------------------------------------------------------------
# Step 2: LLM attribution pass — labels by index only, text untouched
# ---------------------------------------------------------------------------

@retry_with_backoff()
def _classify_chunk(unlabeled_segs, subreddits_hint):
    """Ask the LLM to classify a chunk of segments by index.

    Returns parsed JSON dict or None on failure.
    The LLM emits labels (USER/QUOTED) keyed by idx — it never re-emits text,
    so original content is preserved exactly.
    """
    segs_text = "\n".join(f'{s["idx"]}: "{s["text"]}"' for s in unlabeled_segs)
    prompt = (
        f"Classify each Reddit segment as USER (user's own voice/views/reactions/framing) "
        f"or QUOTED (words authored by someone else the user reproduces — "
        f"pasted blocks, screenshotted text, quoted arguments). "
        f"The user's framing around a quote is USER, even if hostile.\n"
        f"Subreddit context (hint, not a rule): {subreddits_hint}\n\n"
        f"Segments:\n{segs_text}\n\n"
        f"Return one entry per segment index. Every index above MUST appear exactly once.\n"
        f"Return JSON only:\n"
        f'{{\n  "labels": [{{"idx": <n>, "label": "USER"|"QUOTED"}}]\n}}'
    )
    resp = client.chat.completions.create(
        model="gpt-oss", messages=[{"role": "user", "content": prompt}]
    )
    return extract_json(resp.choices[0].message.content)


def attribution_pass(user_id, segments, subreddits_hint):
    """Label all non-INTERLOCUTOR segments as USER or QUOTED.

    Returns (clean_corpus, status) where status is "ok" or "failed".
    Fails closed: if any chunk fails or any index is uncovered, returns "failed"
    rather than falling back to treating contaminated text as USER.
    """
    unlabeled = [s for s in segments if s["label"] is None]

    # Pre-map deterministic INTERLOCUTOR labels.
    label_map = {s["idx"]: s["label"] for s in segments if s["label"] is not None}

    if MOCK_LLM:
        # Mock exercises the same reconstruction path as the real path.
        mock_llm_labels = [
            {"idx": s["idx"], "label": "USER" if len(s["text"]) > 25 else "QUOTED"}
            for s in unlabeled
        ]
        label_map.update({d["idx"]: d["label"] for d in mock_llm_labels})
        clean_corpus = [
            {"label": label_map[s["idx"]], "text": s["text"], "sub": s["sub"], "post_idx": s["post_idx"]}
            for s in segments
        ]
        return clean_corpus, "ok"

    if not unlabeled:
        clean_corpus = [
            {"label": label_map[s["idx"]], "text": s["text"], "sub": s["sub"], "post_idx": s["post_idx"]}
            for s in segments
        ]
        return clean_corpus, "ok"

    chunks = [unlabeled[i:i + ATTRIBUTION_CHUNK_SIZE] for i in range(0, len(unlabeled), ATTRIBUTION_CHUNK_SIZE)]

    for chunk in chunks:
        result = _classify_chunk(chunk, subreddits_hint)
        if result is None:
            return None, "failed"
        parsed = _normalize_classify_result(result)
        if parsed is None:
            print(f"  -> {user_id}: unparseable attribution response — failing closed.")
            return None, "failed"
        label_map.update(parsed)

    # Retry once for indices the model skipped (common off-by-one on long chunks).
    missing_segs = [s for s in unlabeled if s["idx"] not in label_map]
    if missing_segs:
        result = _classify_chunk(missing_segs, subreddits_hint)
        if result is not None:
            parsed = _normalize_classify_result(result)
            if parsed:
                label_map.update(parsed)

    # Fail closed: reject if any unlabeled index is not covered.
    missing = [s["idx"] for s in unlabeled if s["idx"] not in label_map]
    if missing:
        print(f"  -> {user_id}: attribution missing {len(missing)} indices — failing closed.")
        return None, "failed"

    clean_corpus = [
        {"label": label_map[s["idx"]], "text": s["text"], "sub": s["sub"], "post_idx": s["post_idx"]}
        for s in segments
    ]
    return clean_corpus, "ok"


# ---------------------------------------------------------------------------
# Step 3: Hold out one predominantly-USER post (post granularity)
# ---------------------------------------------------------------------------

def holdout_from_user_posts(user_id, clean_corpus):
    """Deterministically hold out one predominantly-USER post.

    A post is eligible if the majority of its segments are USER-labeled,
    ensuring we never hold out a QUOTED block as the behavioral target.

    Returns (training_corpus, heldout_text) or (clean_corpus, None) if no
    eligible posts exist. Keyed on user_id for reproducibility.
    """
    posts = defaultdict(list)
    for seg in clean_corpus:
        posts[seg["post_idx"]].append(seg)

    eligible = [
        pi for pi, segs in posts.items()
        if sum(1 for s in segs if s["label"] == "USER") > len(segs) / 2
    ]
    if not eligible:
        return clean_corpus, None

    rng = random.Random(user_id)
    held_idx = rng.choice(eligible)
    heldout_text = "\n".join(s["text"] for s in posts[held_idx] if s["label"] == "USER")
    training_corpus = [s for s in clean_corpus if s["post_idx"] != held_idx]
    return training_corpus, heldout_text


def render_corpus(clean_corpus):
    """Render labeled segments as a string for seed files and prompts."""
    lines = []
    current_sub = None
    for seg in clean_corpus:
        if seg.get("sub") != current_sub:
            current_sub = seg["sub"]
            lines.append(f"[r/{current_sub}]")
        lines.append(f"[{seg['label']}] {seg['text']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 4: Quote-signal extraction (post-holdout, QUOTED/INTERLOCUTOR only)
# ---------------------------------------------------------------------------

_QUOTE_LABELS = frozenset({"QUOTED", "INTERLOCUTOR"})


def _quote_relevant_segments(training_corpus):
    """QUOTED/INTERLOCUTOR segments plus adjacent USER framing from training corpus."""
    relevant = []
    for i, seg in enumerate(training_corpus):
        if seg["label"] in _QUOTE_LABELS:
            relevant.append(seg)
        elif seg["label"] == "USER":
            prev_q = i > 0 and training_corpus[i - 1]["label"] in _QUOTE_LABELS
            next_q = (
                i + 1 < len(training_corpus)
                and training_corpus[i + 1]["label"] in _QUOTE_LABELS
            )
            if prev_q or next_q:
                relevant.append(seg)
    return relevant


@retry_with_backoff()
def extract_quote_signals(user_id, training_corpus, subreddits_hint):
    """Summarize external content the user quoted and their stance toward it.

    Only runs on post-holdout training_corpus. Only considers QUOTED/INTERLOCUTOR
    segments (not bare USER text). Returns [] when the user has no quoted content.
    """
    context_segs = _quote_relevant_segments(training_corpus)
    if not context_segs:
        return []

    if MOCK_LLM:
        return ["quoted external content with user stance (stub)"]

    excerpt = "\n".join(f'[{s["label"]}] {s["text"]}' for s in context_segs)
    prompt = (
        f"Below are labeled Reddit segments where a user quoted or reproduced external "
        f"content (QUOTED / INTERLOCUTOR) plus their own framing (USER).\n"
        f"Summarize each distinct piece of external content referenced and the user's "
        f"stance toward it. Stance may be: mocks, defends, rebuts, agrees, cites neutrally.\n\n"
        f"CRITICAL — get the direction right:\n"
        f"- 'it's silly to get mad at X' = user DEFENDS X, criticizes X's critics\n"
        f"- 'good for them for posting X' = user DEFENDS X\n"
        f"- Do NOT flatten 'criticizes people who hate X' into 'mocks X'\n"
        f"- State who is the target of criticism when relevant\n\n"
        f"Only summarize actual quoted/reproduced external content. "
        f"Do NOT infer quote signals from USER-only opinions.\n"
        f"Subreddit context (hint): {subreddits_hint}\n\n"
        f"Segments:\n{excerpt[:6000]}\n\n"
        f"Return JSON only:\n"
        f'{{"quote_signals": ["<topic>: <stance, e.g. defends X against Y\'s criticism>"]}}'
    )
    resp = client.chat.completions.create(
        model="gpt-oss", messages=[{"role": "user", "content": prompt}]
    )
    result = extract_json(resp.choices[0].message.content)
    if result:
        return result.get("quote_signals", [])
    return []


# ---------------------------------------------------------------------------
# Step 5: PVQ-40 administration on USER turns + quote_signals
# ---------------------------------------------------------------------------

@retry_with_backoff()
def administer_pvq_to_user_text(user_id, user_text, quote_signals=None):
    """Administer PVQ-40 conditioned on USER-turn text and score Schwartz values.

    quote_signals: optional low-weight hint from quoted/reproduced content + stance.
    Primary signal is always USER-authored text.
    """
    if MOCK_LLM:
        item_scores = {str(i): 1 + ((i + len(user_id)) % 6) for i in range(1, 41)}
        pvq_item_scores, pvq_value_means, target_vector = score_pvq_assignment(item_scores)
        return {
            "pvq_item_scores": pvq_item_scores,
            "pvq_value_means": pvq_value_means,
            "target_vector": target_vector,
        }

    quote_hint = ""
    if quote_signals:
        quote_hint = (
            "\nSecondary context (low weight): quoted/reproduced external content and "
            "the user's stance toward it. USER turns above are the primary signal.\n"
            + "\n".join(f"- {t}" for t in quote_signals)
        )

    prompt = (
        "You are participating in a psychology study as the Reddit user described below.\n"
        "Use ONLY the USER-authored turns as the user's own voice. QUOTED and "
        "INTERLOCUTOR content is not authored by the user.\n\n"
        f"USER TURNS FOR {user_id}:\n{user_text[:6000]}"
        f"{quote_hint}\n\n"
        "Below are descriptions of people. For each item, answer: how much like "
        "this user is this person?\n\n"
        "Rating scale:\n"
        "1 = Not like this user at all\n"
        "2 = Not like this user\n"
        "3 = A little like this user\n"
        "4 = Somewhat like this user\n"
        "5 = Like this user\n"
        "6 = Very much like this user\n\n"
        f"PVQ-40 ITEMS:\n{pvq_survey_text()}\n\n"
        "Return only a valid JSON object mapping item number strings to integer "
        "scores for all 40 items, for example: "
        '{"1": 4, "2": 1, ... "40": 5}'
    )
    resp = client.chat.completions.create(
        model="gpt-oss", messages=[{"role": "user", "content": prompt}]
    )
    item_scores = extract_json(resp.choices[0].message.content)
    if not item_scores:
        return None
    try:
        pvq_item_scores, pvq_value_means, target_vector = score_pvq_assignment(item_scores)
    except ValueError as exc:
        print(f"  -> {user_id}: invalid PVQ response ({exc})")
        return None
    return {
        "pvq_item_scores": pvq_item_scores,
        "pvq_value_means": pvq_value_means,
        "target_vector": target_vector,
    }


# ---------------------------------------------------------------------------
# Main enrichment loop
# ---------------------------------------------------------------------------

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

    enriched_count = len(existing)
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
            open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for line in fin:
            if enriched_count >= MAX_USERS:
                print(f"Reached MAX_USERS={MAX_USERS}, stopping.")
                break
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            user_id = data.get("user_id")
            if not user_id or user_id in existing:
                continue

            history = data.get("history", [])
            subreddits = [it.get("subreddit") for it in history]
            subreddits_hint = ", ".join(f"r/{s}" for s in subreddits if s)

            try:
                # 1. Deterministic pre-parse.
                segments = preparse_history(history)
                if not segments:
                    print(f"  -> {user_id}: no segments after pre-parse, skipping.")
                    continue

                # 2. Attribution pass (fails closed) — labels only, no quote signals yet.
                clean_corpus, status = attribution_pass(user_id, segments, subreddits_hint)
                if status == "failed":
                    print(f"  -> {user_id}: attribution failed, skipping.")
                    continue

                # 3. Hold out one predominantly-USER post.
                training_corpus, heldout_post = holdout_from_user_posts(user_id, clean_corpus)
                if heldout_post is None:
                    print(f"  -> {user_id}: no eligible USER posts to hold out, skipping.")
                    continue

                # 4. Quote signals from training corpus only (QUOTED/INTERLOCUTOR + framing).
                quote_signals = extract_quote_signals(user_id, training_corpus, subreddits_hint)

                # 5. PVQ on USER turns of training corpus + optional quote_signals hint.
                user_text = "\n".join(s["text"] for s in training_corpus if s["label"] == "USER")
                pvq_assignment = administer_pvq_to_user_text(user_id, user_text, quote_signals)
                if not pvq_assignment:
                    print(f"  -> {user_id}: PVQ administration failed, skipping.")
                    continue

                record = {
                    "user_id": user_id,
                    "subreddits": subreddits,
                    "clean_corpus": training_corpus,  # labeled segments, heldout post removed
                    "quote_signals": quote_signals,
                    "heldout_post": heldout_post,      # USER-turn text of the held-out post
                    "pvq_item_scores": pvq_assignment["pvq_item_scores"],
                    "pvq_value_means": pvq_assignment["pvq_value_means"],
                    "target_vector": pvq_assignment["target_vector"],
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()
                enriched_count += 1
                if enriched_count % 25 == 0:
                    print(f"  [{enriched_count}] enriched {user_id}")
            except Exception as e:
                print(f"  -> {user_id}: unexpected error ({type(e).__name__}: {e}), skipping.")
                continue

    print("Enrichment done.")


def split_into_train_val_test():
    """Carve per-user train/val/test from the PVQ-enriched file (reproducible)."""
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
    if os.getenv("SPLIT_ENRICHED"):
        split_into_train_val_test()
