"""Spec B (minimal) -- behavior-fidelity scoring across the four persona arms.

For each of the ~800 tau2 traces, judge whether the user simulator's OBSERVABLE
behavior (its user turns) is consistent with the persona it was conditioned on:

    b_matched = anchor_score( judge(profile_text[persona], clean_user_turns) )   [1-5]

Two passes of judge calls (fully parallel, resumable):
  Pass 1 -- matched: one judge call per non-empty trace (~800). Empty transcripts
           (0 user turns after stripping [TASK_DONE]) are logged as `no_surface`
           and excluded from the mean -- no call spent.
  Pass 2 -- pooled shuffle: ~150 trace x foreign-persona pairs, ONE baseline band
           for the whole table (not a per-arm column).

Aggregation: trace -> persona (mean) -> arm mean +/- sd (cluster bootstrap over
personas), because trace counts per persona are uneven.

Reuses the shared OpenRouter/gpt-4o-mini plumbing and persona loader from
score_value_fidelity, and the established TAU behavioral-judge rubric.

Usage:
    python score_behavior_fidelity.py matched   # pass 1 (~800 calls, resumable)
    python score_behavior_fidelity.py shuffle    # pass 2 (~150 calls, resumable)
    python score_behavior_fidelity.py score       # offline aggregate + table
    python score_behavior_fidelity.py all          # all three (default)

Set MOCK_LLM=1 to exercise the pipeline offline (deterministic fake judge).
"""

import json
import os
import sys
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from score_value_fidelity import (
    BASE_DIR,
    MOCK_LLM,
    MODEL,
    ARM_FILES,
    load_personas,
    extract_json,
    retry_with_backoff,
    _client,
)

TRACE_FILE = os.path.join(BASE_DIR, "Eval/mcts_phase1/AB_rollouts_core.jsonl")
ROWS_FILE = os.path.join(BASE_DIR, "behavior_fidelity_rows.jsonl")
SHUFFLE_FILE = os.path.join(BASE_DIR, "shuffle_baseline.jsonl")
AGG_FILE = os.path.join(BASE_DIR, "behavior_fidelity_aggregate.json")

N_SHUFFLE = int(os.getenv("BF_N_SHUFFLE", "150"))
MAX_WORKERS = int(os.getenv("BF_MAX_WORKERS", "8"))
N_BOOTSTRAP = int(os.getenv("BF_N_BOOTSTRAP", "2000"))
SEED = int(os.getenv("BF_SEED", "0"))


# --------------------------------------------------------------------------- #
# Judge (reuses the production TAU behavioral rubric: 1-5 anchor_score, temp 0)
# --------------------------------------------------------------------------- #
_JUDGE_PROMPT = """You are an expert Evaluator of simulated customer-support interactions.

Your Goal: Judge whether the User Simulator's OBSERVABLE BEHAVIOR (its requests,
refusals, escalations, persistence, and tone in the dialogue) is consistent with
the priorities implied by the Persona Description.

Judge ONLY the dialogue acts. Do NOT reward the user for naming psychological
values; a persona that merely announces its values but behaves inconsistently
should score LOW.

### METHOD (do this before scoring)
1. From the PERSONA, extract 3-4 concrete, checkable behavioral predictions.
2. For each, find whether the TRANSCRIPT confirms or violates it, quoting the line.
3. Only then assign an overall anchor_score (1-5).

### ANCHOR SCALE (1-5) -- for anchor_score only
- 5 (Perfect): Every prediction confirmed.
- 4 (Strong): Most confirmed; at most one weakly supported; none contradicted.
- 3 (Passable): Mix of confirmed and unsupported; broadly plausible.
- 2 (Weak): At most one prediction confirmed, or wrong register.
- 1 (Fail): Random, breaks character, or contradicts the persona.

### INPUT DATA
**PERSONA:**
{persona}

**TRANSCRIPT (user turns only):**
{transcript}

### OUTPUT FORMAT
Return a valid JSON object with:
1. "predictions": array of 3-4 objects: {{"prediction","verdict","cite"}}
   where verdict is one of "confirmed"|"partial"|"violated"|"no_evidence".
2. "anchor_score": integer 1-5 from the scale above.
3. "critique": brief justification.
"""


@retry_with_backoff()
def judge_behavior(persona_text, clean_transcript, key=""):
    """Return the 1-5 anchor_score (float) or None on failure."""
    if MOCK_LLM:
        return 1 + ((len(persona_text) + len(clean_transcript)) % 5)
    prompt = _JUDGE_PROMPT.format(persona=persona_text[:6000], transcript=clean_transcript[:8000])
    resp = _client.chat.completions.create(
        model=MODEL, temperature=0, messages=[{"role": "user", "content": prompt}]
    )
    data = extract_json(resp.choices[0].message.content)
    if not data or data.get("anchor_score") is None:
        print(f"  -> {key}: judge returned no anchor_score")
        return None
    try:
        return max(1.0, min(5.0, float(data["anchor_score"])))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Trace loading + user-turn extraction
# --------------------------------------------------------------------------- #
def clean_user_turns(trace):
    """User turns only, [TASK_DONE] stripped, empties dropped (per spec)."""
    user_turns = [
        t["content"].replace("[TASK_DONE]", "").strip()
        for t in trace["full_transcript"]
        if t.get("role") == "user"
    ]
    return "\n".join(u for u in user_turns if u)


def load_traces():
    traces = []
    for line in open(TRACE_FILE):
        r = json.loads(line)
        traces.append(
            {
                "run_id": r["run_id"],
                "persona_id": r["persona_id"],
                "arm": r["arm"],
                "task_id": r["task_id"],
                "n_turns": r.get("n_turns", 0),
                "clean": clean_user_turns(r),
            }
        )
    return traces


def _profile_map():
    personas, _ = load_personas()
    return {(p["arm"], p["persona_id"]): p["profile_text"] for p in personas}


# --------------------------------------------------------------------------- #
# Pass 1 -- matched
# --------------------------------------------------------------------------- #
def _load_rows():
    done = {}
    if os.path.exists(ROWS_FILE):
        for line in open(ROWS_FILE):
            r = json.loads(line)
            done[r["run_id"]] = r
    return done


def matched():
    traces = load_traces()
    profiles = _profile_map()
    done = _load_rows()

    todo = [t for t in traces if t["run_id"] not in done]
    n_empty = sum(1 for t in traces if not t["clean"])
    print(f"matched: {len(traces)} traces ({n_empty} no_surface), "
          f"{len(done)} cached, {len(todo)} to do (model={MODEL}, mock={MOCK_LLM})")

    lock = threading.Lock()
    out = open(ROWS_FILE, "a")
    counter = {"n": 0}

    def work(t):
        row = {
            "persona_id": t["persona_id"],
            "arm": t["arm"],
            "task_id": t["task_id"],
            "run_id": t["run_id"],
            "n_turns": t["n_turns"],
            "b_matched": None,
            "no_surface": False,
        }
        if not t["clean"]:
            row["no_surface"] = True
        else:
            profile = profiles.get((t["arm"], t["persona_id"]))
            if profile is None:
                print(f"  -> {t['persona_id']}/{t['arm']}: no profile_text; skipping")
                return False
            score = judge_behavior(profile, t["clean"], key=t["run_id"])
            if score is None:
                return False  # transient judge failure -> retry next run (not written)
            row["b_matched"] = score
        with lock:
            out.write(json.dumps(row) + "\n")
            out.flush()
            counter["n"] += 1
            if counter["n"] % 25 == 0:
                print(f"  {counter['n']}/{len(todo)}")
        return True

    if MOCK_LLM or MAX_WORKERS <= 1:
        oks = [work(t) for t in todo]
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            oks = list(ex.map(work, todo))
    out.close()
    print(f"matched done: {sum(oks)}/{len(todo)} new written "
          f"({len(todo) - sum(oks)} failed -- rerun to retry)")


# --------------------------------------------------------------------------- #
# Pass 2 -- pooled shuffle (one baseline for the whole table)
# --------------------------------------------------------------------------- #
def _load_shuffle():
    done = {}
    if os.path.exists(SHUFFLE_FILE):
        for line in open(SHUFFLE_FILE):
            r = json.loads(line)
            done[r["pair_id"]] = r
    return done


def shuffle():
    traces = [t for t in load_traces() if t["clean"]]  # judgeable only
    profiles = _profile_map()
    personas = sorted({p for (_, p) in profiles})
    arms = list(ARM_FILES)
    rng = random.Random(SEED)

    # Deterministic set of ~N_SHUFFLE trace x foreign-persona pairs.
    pairs = []
    for t in rng.sample(traces, min(N_SHUFFLE, len(traces))):
        other = rng.choice([p for p in personas if p != t["persona_id"]])
        arm = rng.choice(arms)
        pairs.append((t, other, arm))

    done = _load_shuffle()
    todo = [(t, o, a) for (t, o, a) in pairs if f"{t['run_id']}|{o}|{a}" not in done]
    print(f"shuffle: {len(pairs)} pairs, {len(done)} cached, {len(todo)} to do")

    lock = threading.Lock()
    out = open(SHUFFLE_FILE, "a")

    def work(item):
        t, other, arm = item
        pair_id = f"{t['run_id']}|{other}|{arm}"
        profile = profiles.get((arm, other))
        score = judge_behavior(profile, t["clean"], key=pair_id)
        if score is None:
            return False
        with lock:
            out.write(json.dumps({"pair_id": pair_id, "b_shuffled": score}) + "\n")
            out.flush()
        return True

    if MOCK_LLM or MAX_WORKERS <= 1:
        oks = [work(i) for i in todo]
    else:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            oks = list(ex.map(work, todo))
    out.close()
    print(f"shuffle done: {sum(oks)}/{len(todo)} new written")


# --------------------------------------------------------------------------- #
# Aggregation (offline)
# --------------------------------------------------------------------------- #
def score():
    rows = list(_load_rows().values())
    # per (arm, persona) mean over judged traces (exclude no_surface / unscored)
    by_arm = {}  # arm -> persona -> {"scores":[...], "turns":[...]}
    n_surface = 0
    for r in rows:
        if r["no_surface"] or r["b_matched"] is None:
            continue
        n_surface += 1
        d = by_arm.setdefault(r["arm"], {}).setdefault(r["persona_id"], {"scores": [], "turns": []})
        d["scores"].append(r["b_matched"])
        d["turns"].append(r["n_turns"])

    np_rng = np.random.default_rng(SEED)
    agg = {}
    for arm in ARM_FILES:
        per_persona = by_arm.get(arm, {})
        if not per_persona:
            continue
        persona_means = np.array([np.mean(v["scores"]) for v in per_persona.values()])
        persona_turns = np.array([np.mean(v["turns"]) for v in per_persona.values()])
        n = len(persona_means)
        boot = [persona_means[np_rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
        agg[arm] = {
            "n_personas": int(n),
            "n_traces_scored": int(sum(len(v["scores"]) for v in per_persona.values())),
            "b_matched_mean": float(persona_means.mean()),
            "b_matched_sd": float(np.std(boot, ddof=1)),
            "b_matched_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "mean_n_turns": float(persona_turns.mean()),
        }

    shuf = np.array([r["b_shuffled"] for r in _load_shuffle().values()], dtype=float)
    baseline = None
    if shuf.size:
        boot = [shuf[np_rng.integers(0, shuf.size, shuf.size)].mean() for _ in range(N_BOOTSTRAP)]
        baseline = {
            "n_pairs": int(shuf.size),
            "b_shuffled_mean": float(shuf.mean()),
            "b_shuffled_sd": float(shuf.std(ddof=1)) if shuf.size > 1 else 0.0,
            "b_shuffled_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        }

    out = {"arms": agg, "shuffle_baseline": baseline,
           "n_traces_total": len(rows), "n_no_surface": sum(r["no_surface"] for r in rows)}
    with open(AGG_FILE, "w") as f:
        json.dump(out, f, indent=2)
    _print_summary(out)


def _print_summary(out):
    print(f"\n{out['n_traces_total']} traces "
          f"({out['n_no_surface']} no_surface, excluded)  scale: anchor_score 1-5\n")
    hdr = f"{'arm':<14}{'personas':>9}{'traces':>8}{'b_matched':>18}{'n_turns':>9}"
    print(hdr)
    print("-" * len(hdr))
    for arm in ARM_FILES:
        a = out["arms"].get(arm)
        if not a:
            continue
        print(f"{arm:<14}{a['n_personas']:>9}{a['n_traces_scored']:>8}"
              f"{a['b_matched_mean']:>10.3f}±{a['b_matched_sd']:<7.3f}"
              f"{a['mean_n_turns']:>9.2f}")
    b = out["shuffle_baseline"]
    if b:
        print("-" * len(hdr))
        print(f"pooled shuffle baseline (n={b['n_pairs']}): "
              f"b_shuffled = {b['b_shuffled_mean']:.3f} "
              f"[95% CI {b['b_shuffled_ci95'][0]:.3f}, {b['b_shuffled_ci95'][1]:.3f}]")
    print(f"\nrows -> {ROWS_FILE}\nbaseline -> {SHUFFLE_FILE}\naggregate -> {AGG_FILE}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd not in ("matched", "shuffle", "score", "all"):
        sys.exit(f"unknown command {cmd!r}; use matched|shuffle|score|all")
    if cmd in ("matched", "all"):
        matched()
    if cmd in ("shuffle", "all"):
        shuffle()
    if cmd in ("score", "all"):
        score()
