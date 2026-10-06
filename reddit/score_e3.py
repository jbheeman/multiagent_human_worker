"""E3 -- richer behavioral grounding makes the evaluation more diagnostic.

CLAIM: grounding shifts the distribution of surfaced *agent* failures from
procedural (info overload, redundant confirmation) toward relational/reasoning
(rigid/flawed policy, tone-deaf, ignores-user), and widens the spread of
per-persona outcomes. Grounding ladder (ordered by prediction):

    nemotron  <  reddit_generic  <  reddit_matched  <  amazon

NOT claimed here: any leaderboard/ranking shift (needs >=3 models; one model
cannot rank). The pipeline never touches model identity.

Stages (each pass is arm-blind, temp 0, cached by trace hash, resumable):
  1. label   -- failure critic labels each AGENT turn with 0+ of 7 categories.
  2. satisfy -- satisfaction judge rates the per-agent-turn change in user
     satisfaction (-2..+2); cum_sat = sum, worst_sat = min running cumulative.
  3. score   -- PRIMARY: composition + JS-to-nemotron (cluster bootstrap over
     personas). SECONDARY: success / satisfaction / spread + paired task deltas.
     CONTROL: arm-label permutation test.

Random unit for ALL resampling = persona. Tasks are shared across arms and are
held constant (never resampled).

INPUT (SourceAB/):  one file per arm, arm taken from the FILENAME (see FILE_TO_ARM;
the in-file `arm` field is the persona-generation arm and is ignored). Each line:
    persona_id, task_id, task_success (0/1 from the benchmark), full_transcript
    ([{role: user|assistant|tool, content, tool_calls?}, ...]).
User turns may carry <internal_monologue>...</internal_monologue>; it is STRIPPED
before any judge sees the transcript (blindness + it is not spoken dialogue).
Policy = domain-level tau2 retail policy.md (same for every task).

Usage:
    python score_e3.py mock-data          # synthetic SourceAB/ for an offline dry run
    python score_e3.py label [arm]         # stage 1 (optionally one arm only)
    python score_e3.py satisfy [arm]       # stage 2 (optionally one arm only)
    python score_e3.py score               # stage 3: metrics + permutation control
    python score_e3.py all                  # label + satisfy + score (default)

Re-running one arm later (e.g. randomreddit once it is populated):
    python score_e3.py label reddit_generic && python score_e3.py satisfy reddit_generic
    python score_e3.py score        # recomputes everything; cached traces are not re-called

Set MOCK_LLM=1 to use deterministic offline judges (no API calls).
"""

import json
import os
import re
import sys
import glob
import hashlib
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from score_value_fidelity import (
    BASE_DIR,
    MOCK_LLM,
    extract_json,
    retry_with_backoff,
    _client,
)

SOURCE_DIR = os.getenv("E3_SOURCE_DIR", os.path.join(BASE_DIR, "SourceAB"))
POLICY_FILE = os.path.join(BASE_DIR, "..", "tau2-bench", "data", "tau2", "domains", "retail", "policy.md")
CRITIC_MODEL = os.getenv("E3_CRITIC_MODEL", "qwen/qwen3.7-flash")

CACHE_FILE = os.path.join(BASE_DIR, "e3_critic_cache.jsonl")        # hash -> flagged_turns
SAT_CACHE_FILE = os.path.join(BASE_DIR, "e3_satisfaction_cache.jsonl")  # hash -> deltas
AGG_FILE = os.path.join(BASE_DIR, "e3_aggregate.json")

# arm = FILENAME stem (NOT the in-file `arm` field). ladder order below.
FILE_TO_ARM = {
    "nemotron": "nemotron",
    "randomreddit": "reddit_generic",
    "domain_specificreddit": "reddit_matched",
    "amazon": "amazon",
}
ARMS = ["nemotron", "reddit_generic", "reddit_matched", "amazon"]  # ladder order
BASELINE_ARM = "nemotron"
CATEGORIES = [
    "ignores_user",
    "information_overload",
    "hallucinated_policy",
    "rigid_policy",
    "flawed_policy",
    "impractical_solution",
    "tone_deaf",
]
CAT_SET = set(CATEGORIES)

MAX_WORKERS = int(os.getenv("E3_MAX_WORKERS", "8"))
N_BOOT = int(os.getenv("E3_N_BOOT", "2000"))
N_PERM = int(os.getenv("E3_N_PERM", "5000"))
SEED = int(os.getenv("E3_SEED", "0"))

_MONO_RE = re.compile(r"<internal_monologue>.*?</internal_monologue>", re.DOTALL)


# --------------------------------------------------------------------------- #
# Prompts (arm-blind)
# --------------------------------------------------------------------------- #
_CRITIC_PROMPT = """You are an expert evaluator of customer-support agent transcripts. You are given a
support POLICY and a full TRANSCRIPT between a user and an AI agent. Your job is to
label each AGENT turn with any interaction failures it exhibits, using ONLY the rules
below. Judge the agent, not the user.

You are NOT told what kind of user this is, and you must not guess or speculate about
the user's identity, demographics, or origin. Label only observable agent behavior.

FAILURE CATEGORIES (a turn may have zero, one, or several; most turns have none):
1. ignores_user -- proceeds without addressing an objection/question raised in the
   immediately preceding user turn.
2. information_overload -- returns a full table/list/dump where a single relevant field
   would have answered the user's question.
3. hallucinated_policy -- states a policy detail NOT present in POLICY. Verify against
   POLICY, not plausibility. If POLICY is silent, a confident policy claim IS this.
4. rigid_policy -- restates a policy (verbatim/near-verbatim) for the >=2nd time WITHOUT
   addressing the user's stated objection or offering an alternative. Same unresolved
   constraint must appear across >=2 agent turns.
5. flawed_policy -- the user identifies a logical gap/contradiction/more efficient path,
   and the agent does not acknowledge or engage with it.
6. impractical_solution -- offers a generic solution that ignores a specific constraint
   the user already stated.
7. tone_deaf -- proceeds without acknowledging explicit negative user affect
   (frustration, anger, distress) in the preceding user turn.

RULES:
- Base every label on a specific span in the agent turn (and earlier turns for cross-turn
  rules). If you cannot point to the text, do not assign the label.
- rigid_policy and flawed_policy require reading the whole transcript.
- hallucinated_policy is judged strictly against POLICY below.
- Do not invent categories. Do not label user turns.
- When unsure, leave the turn unlabeled. False positives corrupt the comparison.

POLICY:
{policy_text}

TRANSCRIPT (turns are indexed; [U]=user, [A]=agent):
{indexed_transcript}

Return JSON ONLY, one object per AGENT turn that has >=1 label (omit clean turns):
{{"flagged_turns": [{{"turn_index": <int>, "labels": ["rigid_policy"], "evidence": "<verbatim span>"}}]}}
If no agent turn is flagged, return {{"flagged_turns": []}}.
"""

_SAT_PROMPT = """You are rating a customer's satisfaction during a support conversation. You are given
a support POLICY and a TRANSCRIPT ([U]=user, [A]=agent). Satisfaction is revealed only
when the USER responds. For each USER turn, estimate how the user's satisfaction CHANGED
in that turn (their reaction to the agent's preceding turn), as an integer:
  +2 markedly more satisfied, +1 somewhat more, 0 no change,
  -1 somewhat less satisfied, -2 markedly less satisfied.
Judge ONLY observable dialogue (the user's spoken words). Do not speculate about the
user's identity or origin.

POLICY:
{policy_text}

TRANSCRIPT (turns are indexed; [U]=user, [A]=agent):
{indexed_transcript}

Return JSON ONLY: one entry per USER turn, in order:
{{"deltas": [{{"turn_index": <int>, "delta": <int -2..2>}}]}}
"""


# --------------------------------------------------------------------------- #
# Loading + arm-blind transcript rendering
# --------------------------------------------------------------------------- #
def load_policy():
    with open(POLICY_FILE) as f:
        return f.read()


def _strip_monologue(content):
    return _MONO_RE.sub("", content or "").strip()


def load_traces(only_arm=None):
    """Read SourceAB/*.jsonl. Arm comes from the FILENAME. Empty/unknown files are
    skipped. `only_arm` restricts to a single arm (for separate re-runs)."""
    traces = []
    for path in sorted(glob.glob(os.path.join(SOURCE_DIR, "*.jsonl"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        arm = FILE_TO_ARM.get(stem)
        if arm is None:
            print(f"  (skipping {os.path.basename(path)}: not a known arm file)")
            continue
        if only_arm and arm != only_arm:
            continue
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            traces.append(
                {
                    "arm": arm,
                    "persona_id": r["persona_id"],
                    "task_id": str(r["task_id"]),
                    "transcript": r.get("full_transcript") or [],
                    "task_success": int(r.get("task_success", 0)),
                }
            )
    return traces


def indexed_transcript(transcript):
    """Arm-blind indexed transcript (monologue stripped, tool turns dropped ->
    [U]/[A] only). Returns (text, agent_turn_indices, user_turn_indices).
    Failures are judged on agent turns; satisfaction deltas on user turns."""
    lines, agent_idx, user_idx = [], set(), set()
    for i, t in enumerate(transcript):
        role = t.get("role")
        if role not in ("user", "assistant"):
            continue  # tool results are not shown as [U]/[A] turns
        content = _strip_monologue(t.get("content"))
        if not content and t.get("tool_calls"):
            names = ", ".join(
                (c.get("function", {}) or {}).get("name", "?") for c in t["tool_calls"]
            )
            content = f"[calls tool: {names}]"
        if role == "assistant":
            agent_idx.add(i)
            tag = "A"
        else:
            user_idx.add(i)
            tag = "U"
        lines.append(f"[{i}][{tag}] {content}")
    return "\n".join(lines), agent_idx, user_idx


def trace_hash(policy_text, indexed_text):
    h = hashlib.sha256()
    h.update(policy_text.encode()); h.update(b"\x00"); h.update(indexed_text.encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Generic cached judge pass (used by both `label` and `satisfy`)
# --------------------------------------------------------------------------- #
def _load_cache(path):
    cache = {}
    if os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            cache[r["trace_hash"]] = r["result"]
    return cache


def _run_pass(name, cache_path, judge_fn, only_arm):
    policy = load_policy()
    traces = load_traces(only_arm)
    cache = _load_cache(cache_path)
    # unique traces by hash (identical stripped transcript judged once)
    todo = {}
    for tr in traces:
        text, _agent_idx, _user_idx = indexed_transcript(tr["transcript"])
        th = trace_hash(policy, text)
        if th not in cache and th not in todo:
            todo[th] = text
    print(f"{name}: {len(traces)} traces ({len(todo)} unlabeled hashes), "
          f"{len(cache)} cached (model={CRITIC_MODEL}, mock={MOCK_LLM}"
          f"{', arm=' + only_arm if only_arm else ''})")

    lock = threading.Lock()
    out = open(cache_path, "a")
    counter = {"n": 0, "fail": 0}

    def work(item):
        th, text = item
        result = judge_fn(policy, text)
        if result is None:
            with lock:
                counter["fail"] += 1
            return
        with lock:
            out.write(json.dumps({"trace_hash": th, "result": result}) + "\n")
            out.flush()
            counter["n"] += 1
            if counter["n"] % 25 == 0:
                print(f"  {counter['n']}/{len(todo)}")

    items = list(todo.items())
    if MOCK_LLM or MAX_WORKERS <= 1:
        for it in items:
            work(it)
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            list(ex.map(work, items))
    out.close()
    print(f"{name} done: {counter['n']} new, {counter['fail']} failures (rerun to retry)")


# ---- failure critic (labels AGENT turns) ----
def _mock_critic(indexed_text):
    flagged = []
    for line in indexed_text.split("\n"):
        if "][A]" not in line:
            continue
        idx = int(line[1: line.index("]")])
        labels = [c for c in CATEGORIES if c in line]
        if labels:
            flagged.append({"turn_index": idx, "labels": labels, "evidence": line[:80]})
    return flagged


@retry_with_backoff()
def _critic(policy, text):
    if MOCK_LLM:
        return _mock_critic(text)
    prompt = _CRITIC_PROMPT.format(policy_text=policy[:8000], indexed_transcript=text[:16000])
    resp = _client.chat.completions.create(
        model=CRITIC_MODEL, temperature=0, messages=[{"role": "user", "content": prompt}]
    )
    data = extract_json(resp.choices[0].message.content)
    return None if data is None else data.get("flagged_turns", [])


# ---- satisfaction judge (rates USER turns: the user's reaction reveals satisfaction) ----
def _mock_satisfy(indexed_text):
    deltas = []
    for line in indexed_text.split("\n"):
        if "][U]" not in line:
            continue
        idx = int(line[1: line.index("]")])
        d = -1 if "doesn't help" in line or "that doesn't" in line else 0
        deltas.append({"turn_index": idx, "delta": d})
    return deltas


@retry_with_backoff()
def _satisfy(policy, text):
    if MOCK_LLM:
        return _mock_satisfy(text)
    prompt = _SAT_PROMPT.format(policy_text=policy[:8000], indexed_transcript=text[:16000])
    resp = _client.chat.completions.create(
        model=CRITIC_MODEL, temperature=0, messages=[{"role": "user", "content": prompt}]
    )
    data = extract_json(resp.choices[0].message.content)
    return None if data is None else data.get("deltas", [])


def label(only_arm=None):
    _run_pass("label", CACHE_FILE, _critic, only_arm)


def satisfy(only_arm=None):
    _run_pass("satisfy", SAT_CACHE_FILE, _satisfy, only_arm)


# --------------------------------------------------------------------------- #
# Assemble per-trace records from the two caches
# --------------------------------------------------------------------------- #
def _counts_from_flagged(flagged, agent_idx):
    counts = {c: 0 for c in CATEGORIES}
    for ft in flagged or []:
        if ft.get("turn_index") not in agent_idx:
            continue
        for lab in set(ft.get("labels", [])) & CAT_SET:
            counts[lab] += 1
    return counts


def _cum_worst(deltas, user_idx):
    """Cumulative and worst-case satisfaction over USER-turn deltas."""
    run = worst = 0.0
    for d in sorted(deltas or [], key=lambda x: x.get("turn_index", 0)):
        if d.get("turn_index") not in user_idx:
            continue
        run += float(d.get("delta", 0))
        worst = min(worst, run)
    return run, worst


def build_records():
    policy = load_policy()
    traces = load_traces()
    crit = _load_cache(CACHE_FILE)
    sat = _load_cache(SAT_CACHE_FILE)
    records, n_unlabeled, n_nosat = [], 0, 0
    for tr in traces:
        text, agent_idx, user_idx = indexed_transcript(tr["transcript"])
        th = trace_hash(policy, text)
        if th not in crit:
            n_unlabeled += 1
            continue
        cum, worst = (0.0, 0.0)
        if th in sat:
            cum, worst = _cum_worst(sat[th], user_idx)
        else:
            n_nosat += 1
        records.append({
            "arm": tr["arm"], "persona_id": tr["persona_id"], "task_id": tr["task_id"],
            "counts": _counts_from_flagged(crit[th], agent_idx),
            "task_success": tr["task_success"], "cum_sat": cum, "worst_sat": worst,
        })
    return records, n_unlabeled, n_nosat


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def js_div(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    ps, qs = p.sum(), q.sum()
    if ps == 0 or qs == 0:
        return 0.0
    p, q = p / ps, q / qs
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def composition(records):
    v = np.zeros(len(CATEGORIES))
    for r in records:
        for i, c in enumerate(CATEGORIES):
            v[i] += r["counts"][c]
    s = v.sum()
    return v / s if s > 0 else v


def _by_arm_persona(records, arms):
    d = {a: {} for a in arms}
    for r in records:
        d[r["arm"]].setdefault(r["persona_id"], []).append(r)
    return d


def _resample_arm(persona_map, rng):
    personas = list(persona_map)
    picked = [personas[i] for i in rng.integers(0, len(personas), len(personas))]
    return [r for p in picked for r in persona_map[p]]


def _ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


# --------------------------------------------------------------------------- #
# Stage 3 -- metrics + control
# --------------------------------------------------------------------------- #
def score():
    records, n_unlabeled, n_nosat = build_records()
    if not records:
        sys.exit("no labeled records; run 'label' first")
    present = [a for a in ARMS if any(r["arm"] == a for r in records)]
    missing = [a for a in ARMS if a not in present]
    if BASELINE_ARM not in present:
        sys.exit(f"baseline arm {BASELINE_ARM} has no data; cannot compute JS-to-nemotron")
    if n_unlabeled:
        print(f"WARNING: {n_unlabeled} traces have no critic label yet (excluded); rerun 'label'.")
    if n_nosat:
        print(f"WARNING: {n_nosat} labeled traces have no satisfaction yet -> cum/worst=0; run 'satisfy'.")
    if missing:
        print(f"WARNING: arms with no data (excluded): {missing}")

    bap = _by_arm_persona(records, present)
    rng = np.random.default_rng(SEED)

    comp = {a: composition([r for p in bap[a].values() for r in p]) for a in present}
    js_to_base = {a: js_div(comp[a], comp[BASELINE_ARM]) for a in present}
    js_matrix = {a: {b: js_div(comp[a], comp[b]) for b in present} for a in present}

    prevalence = {}
    for a in present:
        recs = [r for p in bap[a].values() for r in p]
        n = len(recs) or 1
        prevalence[a] = {c: sum(r["counts"][c] > 0 for r in recs) / n for c in CATEGORIES}

    js_boot = {a: [] for a in present}
    prev_boot = {a: {c: [] for c in CATEGORIES} for a in present}
    succ_boot = {a: [] for a in present}
    cum_boot = {a: [] for a in present}
    worst_boot = {a: [] for a in present}
    spread_boot = {a: [] for a in present}
    for _ in range(N_BOOT):
        for a in present:
            rs = _resample_arm(bap[a], rng)
            cb = composition(rs)
            base = composition(_resample_arm(bap[BASELINE_ARM], rng)) if a != BASELINE_ARM else cb
            js_boot[a].append(js_div(cb, base))
            n = len(rs) or 1
            for c in CATEGORIES:
                prev_boot[a][c].append(sum(r["counts"][c] > 0 for r in rs) / n)
            succ_boot[a].append(np.mean([r["task_success"] for r in rs]))
            cum_boot[a].append(np.mean([r["cum_sat"] for r in rs]))
            worst_boot[a].append(np.mean([r["worst_sat"] for r in rs]))
            pm = {}
            for r in rs:
                pm.setdefault(r["persona_id"], []).append(r["cum_sat"])
            means = [np.mean(v) for v in pm.values()]
            spread_boot[a].append(np.std(means, ddof=1) if len(means) > 1 else 0.0)

    outcomes, spread = {}, {}
    for a in present:
        recs = [r for p in bap[a].values() for r in p]
        pm = {}
        for r in recs:
            pm.setdefault(r["persona_id"], []).append(r["cum_sat"])
        persona_means = [np.mean(v) for v in pm.values()]
        outcomes[a] = {
            "n_traces": len(recs), "n_personas": len(bap[a]),
            "task_success": float(np.mean([r["task_success"] for r in recs])),
            "task_success_ci": _ci(succ_boot[a]),
            "cum_sat": float(np.mean([r["cum_sat"] for r in recs])), "cum_sat_ci": _ci(cum_boot[a]),
            "worst_sat": float(np.mean([r["worst_sat"] for r in recs])), "worst_sat_ci": _ci(worst_boot[a]),
        }
        spread[a] = {
            "sat_spread_sd": float(np.std(persona_means, ddof=1)) if len(persona_means) > 1 else 0.0,
            "sat_spread_ci": _ci(spread_boot[a]),
        }

    paired = {}
    for label_, arm_a in (("amazon_minus_nemotron", "amazon"),
                          ("reddit_matched_minus_nemotron", "reddit_matched")):
        if arm_a in present:
            paired[label_] = _paired_delta(bap, arm_a, BASELINE_ARM, rng)

    control = _permutation_test(records, present, np.random.default_rng(SEED + 1))

    agg = {
        "present_arms": present, "missing_arms": missing,
        "n_traces_labeled": len(records), "n_traces_unlabeled": n_unlabeled, "n_traces_no_sat": n_nosat,
        "composition": {a: {c: float(comp[a][i]) for i, c in enumerate(CATEGORIES)} for a in present},
        "js_to_nemotron": {a: {"value": js_to_base[a], "ci": _ci(js_boot[a])} for a in present},
        "js_matrix": js_matrix,
        "prevalence": {a: {c: {"value": prevalence[a][c], "ci": _ci(prev_boot[a][c])}
                           for c in CATEGORIES} for a in present},
        "outcomes": outcomes, "spread": spread, "paired_deltas": paired,
        "control_permutation": control,
    }
    with open(AGG_FILE, "w") as f:
        json.dump(agg, f, indent=2)
    _print_report(agg, present)


def _paired_delta(bap, arm_a, arm_b, rng):
    def task_means(persona_map):
        acc = {}
        for recs in persona_map.values():
            for r in recs:
                acc.setdefault(r["task_id"], []).append(r["task_success"])
        return {t: float(np.mean(v)) for t, v in acc.items()}

    ma, mb = task_means(bap[arm_a]), task_means(bap[arm_b])
    shared = sorted(set(ma) & set(mb))
    if not shared:
        return {"n_shared_tasks": 0, "delta": None, "ci": None}
    point = float(np.mean([ma[t] - mb[t] for t in shared]))
    keys_a, keys_b = list(bap[arm_a]), list(bap[arm_b])
    boot = []
    for _ in range(N_BOOT):
        pa = task_means({p: bap[arm_a][p] for p in np.array(keys_a)[rng.integers(0, len(keys_a), len(keys_a))]})
        pb = task_means({p: bap[arm_b][p] for p in np.array(keys_b)[rng.integers(0, len(keys_b), len(keys_b))]})
        common = set(pa) & set(pb)
        if common:
            boot.append(np.mean([pa[t] - pb[t] for t in common]))
    return {"n_shared_tasks": len(shared), "delta": point,
            "ci": _ci(boot) if boot else None, "metric": "task_success"}


def _permutation_test(records, present, rng):
    arms = np.array([r["arm"] for r in records])
    counts = np.array([[r["counts"][c] for c in CATEGORIES] for r in records], float)
    cum = np.array([r["cum_sat"] for r in records], float)

    def stats(labels):
        comp, sds = {}, []
        for a in present:
            m = labels == a
            v = counts[m].sum(axis=0)
            comp[a] = v / v.sum() if v.sum() > 0 else v
            sds.append(cum[m].std(ddof=1) if m.sum() > 1 else 0.0)
        js_total = sum(js_div(comp[present[i]], comp[present[j]])
                       for i in range(len(present)) for j in range(i + 1, len(present)))
        return js_total, (max(sds) - min(sds))

    obs_js, obs_range = stats(arms)
    ge_js = ge_range = 0
    perm = arms.copy()
    for _ in range(N_PERM):
        rng.shuffle(perm)
        pj, pr = stats(perm)
        ge_js += pj >= obs_js
        ge_range += pr >= obs_range
    return {
        "js_spread_observed": obs_js, "js_spread_p": (ge_js + 1) / (N_PERM + 1),
        "sat_spread_range_observed": obs_range, "sat_spread_range_p": (ge_range + 1) / (N_PERM + 1),
        "n_perm": N_PERM,
    }


def _print_report(agg, present):
    print("\n=== E3: failure-mode composition (instance fraction) ===")
    hdr = f"{'category':<22}" + "".join(f"{a:>16}" for a in present)
    print(hdr); print("-" * len(hdr))
    for c in CATEGORIES:
        print(f"{c:<22}" + "".join(f"{agg['composition'][a][c]:>16.3f}" for a in present))

    print("\n=== compact table (5 rows x arms) ===")
    print(f"{'':<16}" + "".join(f"{a:>16}" for a in present))
    rows = [
        ("task_succ", lambda a: agg["outcomes"][a]["task_success"]),
        ("cum_sat", lambda a: agg["outcomes"][a]["cum_sat"]),
        ("worst_sat", lambda a: agg["outcomes"][a]["worst_sat"]),
        ("sat_spread(SD)", lambda a: agg["spread"][a]["sat_spread_sd"]),
        ("JS_to_nemotron", lambda a: agg["js_to_nemotron"][a]["value"]),
    ]
    for name, fn in rows:
        print(f"{name:<16}" + "".join(f"{fn(a):>16.3f}" for a in present))

    print("\n=== paired task-difficulty deltas (task_success) ===")
    for k, d in agg["paired_deltas"].items():
        if d["delta"] is None:
            print(f"  {k}: no shared tasks")
        else:
            print(f"  {k}: {d['delta']:+.3f}  95% CI [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  "
                  f"(n_shared_tasks={d['n_shared_tasks']})")

    c = agg["control_permutation"]
    print("\n=== control: arm-label permutation test ===")
    print(f"  total pairwise-JS spread = {c['js_spread_observed']:.4f}  p={c['js_spread_p']:.4f}")
    print(f"  range of per-arm sat spread = {c['sat_spread_range_observed']:.4f}  p={c['sat_spread_range_p']:.4f}")
    print(f"\naggregate -> {AGG_FILE}")


# --------------------------------------------------------------------------- #
# Mock data
# --------------------------------------------------------------------------- #
def mock_data():
    os.makedirs(SOURCE_DIR, exist_ok=True)
    rng = random.Random(SEED)
    proc = ["information_overload", "ignores_user"]
    rel = ["rigid_policy", "flawed_policy", "tone_deaf"]
    rel_weight = {"nemotron": 0.1, "reddit_generic": 0.35, "reddit_matched": 0.6, "amazon": 0.85}
    tasks = [f"task_{i}" for i in range(12)]
    stem_for = {v: k for k, v in FILE_TO_ARM.items()}
    for arm in ARMS:
        with open(os.path.join(SOURCE_DIR, f"{stem_for[arm]}.jsonl"), "w") as f:
            for pi in range(40):
                pid = f"{arm[:3]}_persona_{pi}"
                for task in rng.sample(tasks, 4):
                    turns = [{"role": "user",
                              "content": "<internal_monologue>hidden state</internal_monologue>\nhelp me with my order"}]
                    for _ in range(rng.randint(1, 3)):
                        cat = rng.choice(rel if rng.random() < rel_weight[arm] else proc)
                        turns.append({"role": "assistant", "content": f"agent reply {cat}"})
                        turns.append({"role": "tool", "content": "tool result"})
                        turns.append({"role": "user", "content": "<internal_monologue>x</internal_monologue>\nthat doesn't help"})
                    f.write(json.dumps({
                        "arm": "full_gepa", "persona_id": pid, "task_id": task,
                        "full_transcript": turns,
                        "task_success": int(rng.random() < 0.6 - rel_weight[arm] * 0.2),
                    }) + "\n")
    print(f"wrote synthetic SourceAB/ -> {SOURCE_DIR}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    arm_arg = sys.argv[2] if len(sys.argv) > 2 else None
    if arm_arg and arm_arg not in ARMS:
        sys.exit(f"unknown arm {arm_arg!r}; use one of {ARMS}")
    if cmd == "mock-data":
        mock_data()
    elif cmd == "label":
        label(arm_arg)
    elif cmd == "satisfy":
        satisfy(arm_arg)
    elif cmd == "score":
        score()
    elif cmd == "all":
        label(); satisfy(); score()
    else:
        sys.exit(f"unknown command {cmd!r}; use mock-data|label|satisfy|score|all")
