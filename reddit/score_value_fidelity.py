"""Spec A -- value-fidelity scoring across the four persona arms.

For every persona (200 source users x 4 arms = 800), administer PVQ-40 to the
persona's ``persona_paragraph`` and recover a Schwartz vector ``rc[p]``. Score
each recovered vector against the *conditioning* anchor ``tc[source_user(p)]``
(the ``schwartz_json`` field, identical across arms -- the paired anchor):

    v_matched  = cosine(center(rc[p]), center(tc[source_user(p)]))
    v_shuffled = mean over K sampled q != p of cosine(center(rc[p]), center(tc[q]))
    leaked     = leak detector (STUBBED -- see detect_leak)

The only cost is the 800 PVQ administrations (gpt-4o-mini via OpenRouter);
everything after is offline. Recovery is cached to JSONL and resumable.

Usage:
    python score_value_fidelity.py recover   # phase 1: 800 PVQ calls (resumable)
    python score_value_fidelity.py score     # phase 2: offline cosine + aggregate
    python score_value_fidelity.py all        # both (default)

Set MOCK_LLM=1 to exercise the full pipeline offline (deterministic fake PVQ).
"""

import json
import os
import re
import sys
import time
import random
import threading
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from dotenv import load_dotenv

from pvq import SCHWARTZ_KEYS, pvq_survey_text, score_pvq_assignment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(os.path.dirname(BASE_DIR), ".env"))  # repo-root .env
load_dotenv()  # also honor reddit/.env if present

MOCK_LLM = bool(os.getenv("MOCK_LLM"))
MODEL = os.getenv("VF_MODEL", "openai/gpt-4o-mini")
MAX_WORKERS = int(os.getenv("VF_MAX_WORKERS", "8"))
K_SHUFFLE = int(os.getenv("VF_K_SHUFFLE", "10"))
N_BOOTSTRAP = int(os.getenv("VF_N_BOOTSTRAP", "2000"))
SEED = int(os.getenv("VF_SEED", "0"))

# arm -> persona file. "fixed_prompt" is the UNOPTIMIZED arm (per user).
ARM_FILES = {
    "fixed_prompt": "AblationPersonas/unopt_k200.jsonl",
    "value_only": "AblationPersonas/value_only_personas_opt.jsonl",
    "behavior_only": "AblationPersonas/behavior_only_optimized_personas.jsonl",
    "full_gepa": "AblationPersonas/full_arm_optimized_personas.jsonl",
}

RECOVERED_FILE = os.path.join(BASE_DIR, "value_fidelity_recovered.jsonl")
ROWS_FILE = os.path.join(BASE_DIR, "value_fidelity_rows.jsonl")
AGG_FILE = os.path.join(BASE_DIR, "value_fidelity_aggregate.json")


# --------------------------------------------------------------------------- #
# LLM plumbing (OpenRouter / gpt-4o-mini)
# --------------------------------------------------------------------------- #
_client = None
if not MOCK_LLM:
    from openai import OpenAI

    key = os.getenv("OPEN_ROUTER_API_KEY")
    if not key:
        raise ValueError("OPEN_ROUTER_API_KEY is not set (repo-root .env)")
    _client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")


def retry_with_backoff(max_retries=4, initial_delay=2.0, max_delay=60.0, factor=2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001 -- transient API errors
                    if attempt == max_retries - 1:
                        print(f"  giving up after {max_retries} attempts: {exc}")
                        return None
                    print(f"  attempt {attempt + 1} failed ({exc}); retry in {delay:.0f}s")
                    time.sleep(delay)
                    delay = min(delay * factor, max_delay)
            return None

        return wrapper

    return decorator


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(raw_text):
    if not raw_text:
        return None
    match = _JSON_OBJ_RE.search(raw_text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


@retry_with_backoff()
def administer_pvq_to_profile_text(user_id, profile_text):
    """Administer PVQ-40 conditioned on the persona paragraph; return the
    recovered [0,1] Schwartz vector (dict in SCHWARTZ_KEYS space) or None."""
    if MOCK_LLM:
        item_scores = {str(i): 1 + ((i + len(user_id)) % 6) for i in range(1, 41)}
    else:
        prompt = (
            "You are participating in a psychology study as the person described "
            "in the persona profile below.\n"
            "Infer how this person would prioritize values from the profile text only.\n\n"
            f"PERSONA PROFILE FOR {user_id}:\n{profile_text[:6000]}\n\n"
            "Below are descriptions of people. For each item, answer: how much like "
            "this person is this description?\n\n"
            "Rating scale:\n"
            "1 = Not like this person at all\n"
            "2 = Not like this person\n"
            "3 = A little like this person\n"
            "4 = Somewhat like this person\n"
            "5 = Like this person\n"
            "6 = Very much like this person\n\n"
            f"PVQ-40 ITEMS:\n{pvq_survey_text()}\n\n"
            "Return only a valid JSON object mapping item number strings to integer "
            'scores for all 40 items, for example: {"1": 4, "2": 1, ... "40": 5}'
        )
        resp = _client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}]
        )
        item_scores = extract_json(resp.choices[0].message.content)
        if not item_scores:
            return None
    try:
        _items, _means, target_vector = score_pvq_assignment(item_scores)
    except ValueError as exc:
        print(f"  -> {user_id}: invalid PVQ response ({exc})")
        return None
    return target_vector


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _parse_anchor(row):
    raw = row["schwartz_json"]
    anchor = json.loads(raw) if isinstance(raw, str) else raw
    return {k: float(anchor[k]) for k in SCHWARTZ_KEYS}


def load_personas():
    """Return (personas, anchors).

    personas: list of {persona_id, arm, source_user_id, profile_text}
    anchors:  source_user_id -> [0,1] Schwartz dict (the paired conditioning vector)
    """
    personas, anchors = [], {}
    for arm, rel in ARM_FILES.items():
        path = os.path.join(BASE_DIR, rel)
        for line in open(path):
            row = json.loads(line)
            uid = row["user_id"]
            anchor = _parse_anchor(row)
            if uid in anchors and anchors[uid] != anchor:
                raise ValueError(f"anchor mismatch across arms for {uid}")
            anchors[uid] = anchor
            personas.append(
                {
                    "persona_id": uid,
                    "arm": arm,
                    "source_user_id": uid,
                    "profile_text": row["persona_paragraph"],
                }
            )
    return personas, anchors


# --------------------------------------------------------------------------- #
# Phase 1 -- recovery (the 800 PVQ calls)
# --------------------------------------------------------------------------- #
def _load_recovered():
    done = {}
    if os.path.exists(RECOVERED_FILE):
        for line in open(RECOVERED_FILE):
            r = json.loads(line)
            done[(r["persona_id"], r["arm"])] = r
    return done


def recover():
    personas, _ = load_personas()
    done = _load_recovered()
    todo = [p for p in personas if (p["persona_id"], p["arm"]) not in done]
    print(f"recover: {len(personas)} personas, {len(done)} cached, {len(todo)} to do "
          f"(model={MODEL}, mock={MOCK_LLM})")

    lock = threading.Lock()
    out = open(RECOVERED_FILE, "a")
    counter = {"n": 0}

    def work(p):
        rc = administer_pvq_to_profile_text(p["persona_id"], p["profile_text"])
        rec = {
            "persona_id": p["persona_id"],
            "arm": p["arm"],
            "source_user_id": p["source_user_id"],
            "recovered_vector": rc,  # None if the call failed
        }
        with lock:
            out.write(json.dumps(rec) + "\n")
            out.flush()
            counter["n"] += 1
            if counter["n"] % 25 == 0:
                print(f"  {counter['n']}/{len(todo)}")
        return rc is not None

    if MOCK_LLM or MAX_WORKERS <= 1:
        oks = [work(p) for p in todo]
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            oks = list(ex.map(work, todo))
    out.close()
    print(f"recover done: {sum(oks)}/{len(todo)} new succeeded "
          f"({len(todo) - sum(oks)} failed -- rerun to retry)")


# --------------------------------------------------------------------------- #
# Phase 2 -- offline scoring
# --------------------------------------------------------------------------- #
def _vec(d):
    return np.array([d[k] for k in SCHWARTZ_KEYS], dtype=float)


def _centered_cosine(a, b):
    a = a - a.mean()
    b = b - b.mean()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def detect_leak(persona, recovered_vec, anchors):
    """STUB. Returns False for every persona.

    Hook for a contamination detector (e.g. verbatim source-corpus overlap or
    explicit Schwartz-value naming in the persona paragraph). Deliberately not
    defined yet -- wire in the real detector here and the rows/aggregate pick it
    up with no other change.
    """
    return False


def score():
    personas, anchors = load_personas()
    recovered = _load_recovered()
    user_ids = sorted(anchors)
    anchor_vecs = {u: _vec(anchors[u]) for u in user_ids}

    rng = random.Random(SEED)
    rows = []
    skipped = 0
    for p in personas:
        rec = recovered.get((p["persona_id"], p["arm"]))
        if not rec or rec.get("recovered_vector") is None:
            skipped += 1
            continue
        rc = _vec(rec["recovered_vector"])
        u = p["source_user_id"]
        v_matched = _centered_cosine(rc, anchor_vecs[u])

        others = [q for q in user_ids if q != u]
        sampled = rng.sample(others, min(K_SHUFFLE, len(others)))
        v_shuffled = float(np.mean([_centered_cosine(rc, anchor_vecs[q]) for q in sampled]))

        rows.append(
            {
                "persona_id": p["persona_id"],
                "arm": p["arm"],
                "source_user_id": u,
                "v_matched": v_matched,
                "v_shuffled": v_shuffled,
                "leaked": bool(detect_leak(p, rc, anchors)),
            }
        )

    with open(ROWS_FILE, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    agg = _aggregate(rows, user_ids)
    with open(AGG_FILE, "w") as f:
        json.dump(agg, f, indent=2)

    _print_summary(agg, len(rows), skipped)


def _aggregate(rows, user_ids):
    """Per-arm mean +/- sd via paired cluster (user) bootstrap."""
    by_arm_user = {}  # arm -> user -> {matched, shuffled, leaked}
    for r in rows:
        by_arm_user.setdefault(r["arm"], {})[r["source_user_id"]] = r
    np_rng = np.random.default_rng(SEED)

    agg = {}
    for arm, per_user in by_arm_user.items():
        users = [u for u in user_ids if u in per_user]
        matched = np.array([per_user[u]["v_matched"] for u in users])
        shuffled = np.array([per_user[u]["v_shuffled"] for u in users])
        leaked = np.array([per_user[u]["leaked"] for u in users], dtype=float)
        delta = matched - shuffled

        n = len(users)
        bm, bs, bd = [], [], []
        for _ in range(N_BOOTSTRAP):
            idx = np_rng.integers(0, n, n)
            bm.append(matched[idx].mean())
            bs.append(shuffled[idx].mean())
            bd.append(delta[idx].mean())
        agg[arm] = {
            "n_personas": int(n),
            "v_matched_mean": float(matched.mean()),
            "v_matched_sd": float(np.std(bm, ddof=1)),
            "v_shuffled_mean": float(shuffled.mean()),
            "v_shuffled_sd": float(np.std(bs, ddof=1)),
            "delta_mean": float(delta.mean()),
            "delta_sd": float(np.std(bd, ddof=1)),
            "delta_ci95": [float(np.percentile(bd, 2.5)), float(np.percentile(bd, 97.5))],
            "leakage_rate": float(leaked.mean()),
        }
    return agg


def _print_summary(agg, n_rows, skipped):
    print(f"\nscored {n_rows} personas ({skipped} skipped: missing/failed recovery)\n")
    hdr = f"{'arm':<14}{'n':>4}{'matched':>18}{'shuffled':>18}{'delta':>20}{'leak':>7}"
    print(hdr)
    print("-" * len(hdr))
    for arm in ARM_FILES:
        a = agg.get(arm)
        if not a:
            continue
        print(f"{arm:<14}{a['n_personas']:>4}"
              f"{a['v_matched_mean']:>10.3f}±{a['v_matched_sd']:<7.3f}"
              f"{a['v_shuffled_mean']:>10.3f}±{a['v_shuffled_sd']:<7.3f}"
              f"{a['delta_mean']:>10.3f}±{a['delta_sd']:<9.3f}"
              f"{a['leakage_rate']:>7.2f}")
    print(f"\nrows -> {ROWS_FILE}\naggregate -> {AGG_FILE}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("recover", "all"):
        recover()
    if cmd in ("score", "all"):
        score()
    if cmd not in ("recover", "score", "all"):
        sys.exit(f"unknown command {cmd!r}; use recover|score|all")
