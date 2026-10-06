"""
Per-user enrichment pass for the Amazon Reviews 2023 arm (Axis A source ablation).

Mirrors enrich_users.py stage-for-stage so the two arms differ ONLY in source
corpus. Shared code (retry, JSON extraction, LLM client, PVQ scoring key) is
imported rather than reimplemented -- that shared path is itself the parity
argument when a reviewer asks whether the arms were treated identically.

Stage map (Reddit -> Amazon):
  1. Pre-parse         ">"-line segmentation      -> HTML strip + review segmentation
  2. Attribution       LLM USER/QUOTED classify   -> DETERMINISTIC (see note below)
  3. Holdout           one predominantly-USER post-> the dataset's own `heldout` review
  4. Quote signals     quoted content + stance    -> item selection + stance (scrubbed)
  5. PVQ-40            on USER turns              -> on USER review prose (non-scaffold)

NOTE ON STAGE 2 -- the one genuine asymmetry. Reddit needs an LLM attribution
pass because a post can contain text the user did not write. A review body is
wholly user-authored by construction, so attribution is deterministic and cannot
fail. This makes the Amazon arm strictly *less* contaminated, not differently
contaminated, and it means no user is ever dropped for attribution failure.
Report this asymmetry in the paper rather than hiding it; it is the honest
version and it cuts in the arm's favor.

Two leakage guards specific to this arm:
  - Product/brand metadata is NEVER placed in clean_corpus. Only rating and a
    coarse category land there, as [CONTEXT] segments. Proper nouns reach the
    generator through nothing except the (scrubbed) quote_signals.
  - `proper_noun_blocklist` is emitted per user so the generator output can be
    asserted clean before personas enter a tau2-retail environment whose catalog
    does not contain these items.

Env:
  ENRICH_INPUT   (default amazon_users_raw.jsonl)
  ENRICH_OUTPUT  (default amazon_users_pvq_enriched.jsonl)
  MAX_USERS      (default 600 -- enrich the full pool, stratify to 200 AFTER)
  MIN_USER_WORDS (default 150) quality floor, not a stratification
  MIN_STANCE_SENTS (default 4) interactional-density floor
  MOCK_LLM=1     offline dry run
"""

import html
import json
import os
import re

from enrich_users import (  # shared path == parity
    MOCK_LLM,
    client,
    extract_json,
    retry_with_backoff,
)
from pvq import pvq_survey_text, score_pvq_assignment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.getenv("ENRICH_INPUT", os.path.join(BASE_DIR, "amazon_users_raw.jsonl"))
OUTPUT_FILE = os.getenv("ENRICH_OUTPUT", os.path.join(BASE_DIR, "amazon_users_pvq_enriched.jsonl"))

MAX_USERS = int(os.getenv("MAX_USERS", "600"))
MIN_USER_WORDS = int(os.getenv("MIN_USER_WORDS", "150"))
MIN_STANCE_SENTS = int(os.getenv("MIN_STANCE_SENTS", "4"))

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_TAG_RE = re.compile(r"<[^>]+>")
_BR_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)

# Scaffold = the review-template headers this population habitually writes
# ("HOW THIS COMES", "THE BOTTOM LINE", "MY RATING"). Kept in the corpus because
# they are user-authored, but excluded from PVQ text and from depth stats, and
# flagged so the generator can be told not to replicate document structure.
_SCAFFOLD_RE = re.compile(r"^[A-Z0-9][A-Z0-9 \-/&'?().]{2,44}$")

# Sentences carrying stance/reaction/expectation -- the interactional signal.
# Descriptive product sentences carry none and should not count toward depth.
_STANCE_RE = re.compile(
    r"\b("
    r"i wish|i wanted|i expected|i hoped|i thought|i had to|i can'?t|i couldn'?t|"
    r"i don'?t|i didn'?t|i won'?t|i'?d rather|i prefer|i hate|i love|i need|"
    r"annoy\w*|frustrat\w*|disappoint\w*|ridiculous|useless|waste|"
    r"should have|shouldn'?t|would have|could have|"
    r"took a star|taking one off|star off|not worth|worth it|"
    r"better than|worse than|unlike|instead of|"
    r"too (?:small|big|expensive|cheap|slow|short|long|thick|thin)|"
    r"only reason|the problem|the catch|my only|other than that"
    r")\b",
    re.IGNORECASE,
)

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Stage 1: deterministic pre-parse
# ---------------------------------------------------------------------------

def _clean_text(text):
    """HTML -> plain text. <br/> becomes a line break; other tags are dropped."""
    if not text:
        return ""
    text = _BR_RE.sub("\n", text)
    text = _TAG_RE.sub("", text)
    text = html.unescape(text)
    text = _URL_RE.sub("", text)
    return text


def _is_scaffold(line):
    return bool(_SCAFFOLD_RE.match(line.strip())) and not line.strip().endswith((".", "!", "?"))


def preparse_reviews(history):
    """Segment review history into labeled lines.

    Returns flat segment dicts matching the Reddit schema so downstream code
    (render, PVQ, data designer) is unchanged:
      {"idx", "text", "label": "USER"|"CONTEXT", "item_idx", "is_scaffold"}

    Product titles and brands are deliberately NOT emitted. Only rating and
    coarse category appear, as CONTEXT.
    """
    segments = []
    idx = 0
    for item_idx, item in enumerate(history):
        product = item.get("product") or {}
        category = product.get("main_category") or item.get("category") or "?"
        rating = item.get("rating")
        segments.append({
            "idx": idx,
            "text": f"rating {rating}/5 | category: {category}",
            "label": "CONTEXT",
            "item_idx": item_idx,
            "is_scaffold": False,
        })
        idx += 1

        # review_full only. review_excerpt is a truncated duplicate; including
        # both would double-count the same sentences in PVQ and depth stats.
        body = _clean_text(item.get("review_full") or "")
        title = _clean_text(item.get("review_title") or "")
        for line in ([title] if title else []) + body.splitlines():
            line = line.strip()
            if not line:
                continue
            segments.append({
                "idx": idx,
                "text": line,
                "label": "USER",
                "item_idx": item_idx,
                "is_scaffold": _is_scaffold(line),
            })
            idx += 1
    return segments


def render_corpus_amazon(clean_corpus):
    """Render segments for the persona prompt's {{ user_corpus }} slot.

    The legend is emitted INSIDE the corpus block (line 1) rather than as a
    per-arm prompt variable, so the persona prompt string and its placeholder
    dictionary stay byte-identical across Reddit and Amazon arms.
    """
    lines = [
        "[SEGMENT LEGEND: [USER] = written by this user; "
        "[CONTEXT] = item metadata, not user-authored]"
    ]
    current_item = None
    for seg in clean_corpus:
        if seg.get("item_idx") != current_item:
            current_item = seg["item_idx"]
            lines.append(f"[ITEM {current_item}]")
        lines.append(f"[{seg['label']}] {seg['text']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Depth / interactional-density stats (quality floor + cross-arm parity report)
# ---------------------------------------------------------------------------

def compute_depth_stats(segments):
    """Word counts and stance-sentence counts over non-scaffold USER text.

    Word count is a whitespace proxy, not a tokenizer count -- fine for a
    relative cross-arm comparison as long as both arms use the same proxy.
    """
    user_lines = [s["text"] for s in segments if s["label"] == "USER" and not s["is_scaffold"]]
    blob = " ".join(user_lines)
    sents = [s for s in _SENT_SPLIT_RE.split(blob) if s.strip()]
    stance_sents = [s for s in sents if _STANCE_RE.search(s)]
    scaffold_n = sum(1 for s in segments if s["label"] == "USER" and s["is_scaffold"])
    return {
        "user_words": len(blob.split()),
        "user_sentences": len(sents),
        "stance_sentences": len(stance_sents),
        "stance_density": round(len(stance_sents) / len(sents), 4) if sents else 0.0,
        "scaffold_lines": scaffold_n,
        "n_items": len({s["item_idx"] for s in segments}),
    }


def passes_quality_floor(stats):
    return (
        stats["user_words"] >= MIN_USER_WORDS
        and stats["stance_sentences"] >= MIN_STANCE_SENTS
    )


# ---------------------------------------------------------------------------
# Leakage guard: proper nouns that must not survive into generated personas
# ---------------------------------------------------------------------------

_STOP = frozenset("""
the a an and or of for with without in on at to from by is are was were be this that
new pack set size color free all natural best pro plus max mini oz ml fl count pcs
""".split())


def _is_high_specificity(tok):
    """Brand-ish token: contains a digit, or internal capitals (NutriGlow, XXI).

    Plain title nouns ("Hair", "Salt", "Iron") are NOT high-specificity. Blocking
    them would ban the vocabulary this user actually needs -- a persona built from
    someone with fine hair must be allowed to say "hair".
    """
    if len(tok) < 3:
        return False
    if any(c.isdigit() for c in tok):
        return True
    return any(c.isupper() for c in tok[1:])


def build_proper_noun_blocklist(history, heldout):
    """Return (audit_terms, scrub_terms).

    audit_terms  -- every title/brand token + ASIN. For post-generation assertion
                    and manual review; deliberately over-inclusive.
    scrub_terms  -- ASINs, explicit brand values, and brand-ish tokens only.
                    Used to auto-discard quote_signals, so it must not contain
                    generic nouns or it will silently delete valid signals.
    """
    audit, scrub = set(), set()
    for item in list(history) + ([heldout] if heldout else []):
        product = item.get("product") or {}
        brand = product.get("brand")
        if brand:
            scrub.add(brand.strip())
            audit.add(brand.strip())
        title = product.get("title") or ""
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9'\-]{2,}", title):
            if tok.lower() in _STOP:
                continue
            audit.add(tok)
            if _is_high_specificity(tok):
                scrub.add(tok)
        asin = item.get("parent_asin")
        if asin:
            audit.add(asin)
            scrub.add(asin)
    return sorted(audit), sorted(scrub)


# ---------------------------------------------------------------------------
# Stage 4: quote signals -- item selection + stance, brand-scrubbed
# ---------------------------------------------------------------------------

@retry_with_backoff()
def extract_quote_signals_amazon(user_id, history):
    """Amazon analogue of Reddit quote signals.

    Reddit: external content the user reproduced + their stance toward it.
    Amazon: items the user chose to engage with + their stance toward them.
    Selection IS stance, so this is the structurally matched signal.

    Output is scrubbed of brands/models by instruction AND by post-filter, since
    these strings are injected into the generation prompt.
    """
    if not history:
        return []
    if MOCK_LLM:
        return ["a personal-care item: praises performance, faults the packaging (stub)"]

    lines = []
    for i, item in enumerate(history):
        product = item.get("product") or {}
        excerpt = _clean_text(item.get("review_full") or "")[:600]
        lines.append(
            f"[ITEM {i}] category: {product.get('main_category') or '?'} | "
            f"rating: {item.get('rating')}/5\nuser wrote: {excerpt}"
        )
    corpus = "\n\n".join(lines)[:6000]

    prompt = (
        "Below are items a user chose to buy and review, with their rating and "
        "their own words about each.\n"
        "Summarize each distinct thing the user engaged with and their stance "
        "toward it. Stance may be: endorses, faults, tolerates, rejects, "
        "compares against alternatives.\n\n"
        "CRITICAL -- get the direction right:\n"
        "- 'works great but the bottle is tiny' = endorses performance, faults packaging\n"
        "- a high rating with a listed drawback is NOT unqualified endorsement\n"
        "- state WHAT the complaint is about, not just that there was one\n\n"
        "HARD CONSTRAINT: never write a brand, product, model, seller, or retailer "
        "name. Refer to items only by generic type ('a hair tool', 'a skincare "
        "product'). Output containing a proper noun will be discarded.\n\n"
        f"Items:\n{corpus}\n\n"
        "Return JSON only:\n"
        '{"quote_signals": ["<generic item type>: <stance, e.g. endorses X but faults Y>"]}'
    )
    resp = client.chat.completions.create(
        model="gpt-oss", messages=[{"role": "user", "content": prompt}]
    )
    result = extract_json(resp.choices[0].message.content)
    if not result:
        return []
    return result.get("quote_signals", [])


def scrub_signals(signals, blocklist):
    """Belt-and-braces: drop any signal that still contains a blocked term."""
    if not signals:
        return []
    lowered = {t.lower() for t in blocklist}
    kept = []
    for sig in signals:
        toks = {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9'\-]{2,}", str(sig))}
        if toks & lowered:
            continue
        kept.append(sig)
    return kept


# ---------------------------------------------------------------------------
# Stage 5: PVQ-40 on review prose
# ---------------------------------------------------------------------------

@retry_with_backoff()
def administer_pvq_to_review_text(user_id, user_text, quote_signals=None):
    """Administer PVQ-40 conditioned on the user's own review prose.

    Register differs from the Reddit arm (review prose vs conversational turns).
    That difference is the treatment under test on Axis A, not a defect -- but it
    is recorded in the output row so it can be reported rather than assumed away.
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
            "\nSecondary context (low weight): items the user chose to engage with "
            "and their stance toward them. The user's own words above are the "
            "primary signal.\n" + "\n".join(f"- {t}" for t in quote_signals)
        )

    prompt = (
        "You are participating in a psychology study as the user described below.\n"
        "The text is written by the user about things they bought and used. Judge "
        "the person from how they evaluate, complain, qualify, and decide -- not "
        "from which product categories happen to appear.\n\n"
        f"USER'S OWN WRITING FOR {user_id}:\n{user_text[:6000]}"
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
# Main loop
# ---------------------------------------------------------------------------

def enrich_amazon():
    print(f"Starting Amazon enrichment: {INPUT_FILE} -> {OUTPUT_FILE} (MOCK_LLM={MOCK_LLM})")
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
    skipped_floor = 0
    skipped_noheld = 0

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
            heldout = data.get("heldout")

            try:
                # 1. Pre-parse (training history only -- heldout never enters here).
                segments = preparse_reviews(history)
                if not segments:
                    print(f"  -> {user_id}: no segments after pre-parse, skipping.")
                    continue

                # 2. Attribution is deterministic for review bodies (see module docstring).

                # 3. Holdout: use the dataset's own held-out review. Fail closed
                #    rather than silently promoting a training item -- an arm with
                #    no holdout cannot carry the utility signal.
                if not heldout:
                    skipped_noheld += 1
                    print(f"  -> {user_id}: no heldout review, skipping.")
                    continue
                heldout_text = _clean_text(heldout.get("review_full") or "").strip()
                if not heldout_text:
                    skipped_noheld += 1
                    continue

                stats = compute_depth_stats(segments)
                if not passes_quality_floor(stats):
                    skipped_floor += 1
                    print(
                        f"  -> {user_id}: below floor "
                        f"({stats['user_words']}w, {stats['stance_sentences']} stance sents), skipping."
                    )
                    continue

                audit_terms, scrub_terms = build_proper_noun_blocklist(history, heldout)

                # 4. Quote signals from training history only, then scrubbed.
                quote_signals = scrub_signals(
                    extract_quote_signals_amazon(user_id, history) or [], scrub_terms
                )

                # 5. PVQ on non-scaffold USER prose.
                user_text = "\n".join(
                    s["text"] for s in segments
                    if s["label"] == "USER" and not s["is_scaffold"]
                )
                pvq_assignment = administer_pvq_to_review_text(user_id, user_text, quote_signals)
                if not pvq_assignment:
                    print(f"  -> {user_id}: PVQ administration failed, skipping.")
                    continue

                record = {
                    "user_id": user_id,
                    "source": "amazon_reviews_2023",
                    "category": data.get("category"),
                    "pvq_attribution_register": "review_prose",
                    "attribution_pass": "deterministic_review_body",
                    "clean_corpus": segments,          # heldout excluded by construction
                    "quote_signals": quote_signals,
                    "heldout_post": heldout_text,      # parity with Reddit `heldout_post`
                    # Target-side only. NEVER feed heldout_meta to the generator;
                    # it exists so mock_targets can be derived post-split.
                    "heldout_meta": {
                        "parent_asin": heldout.get("parent_asin"),
                        "rating": heldout.get("rating"),
                        "product_title": (heldout.get("product") or {}).get("title"),
                        "category": (heldout.get("product") or {}).get("main_category"),
                    },
                    "proper_noun_blocklist": audit_terms,   # post-generation assertion
                    "scrub_terms": scrub_terms,             # auto-discard (brand-ish only)
                    "depth_stats": stats,
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

    print(
        f"Amazon enrichment done. enriched={enriched_count} "
        f"skipped_quality_floor={skipped_floor} skipped_no_heldout={skipped_noheld}"
    )


def depth_report():
    """Print the cross-arm depth distribution. Run this on BOTH arms' outputs.

    If Amazon corpora are systematically thinner than Reddit, personas will be
    thinner for reasons that have nothing to do with source domain -- a confound
    masquerading as a source effect. Report these two distributions side by side.
    """
    import statistics

    rows = []
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        print("No rows.")
        return

    words = sorted(r["depth_stats"]["user_words"] for r in rows if "depth_stats" in r)
    dens = [r["depth_stats"]["stance_density"] for r in rows if "depth_stats" in r]
    print(f"n={len(rows)}")
    print(
        f"user_words: min={words[0]} p25={words[len(words)//4]} "
        f"median={statistics.median(words):.0f} p75={words[3*len(words)//4]} max={words[-1]}"
    )
    print(f"stance_density: mean={statistics.mean(dens):.3f} median={statistics.median(dens):.3f}")

    by_cat = {}
    for r in rows:
        by_cat[r.get("category")] = by_cat.get(r.get("category"), 0) + 1
    print("category composition:")
    for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
        print(f"  {cat}: {n}")


if __name__ == "__main__":
    if os.getenv("DEPTH_REPORT"):
        depth_report()
    else:
        enrich_amazon()