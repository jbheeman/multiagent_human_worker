"""Single-condition deep-dive analysis tool.

Reads one result JSON file (or merges a directory of per-persona JSON files)
and produces five sections:

  1. Score summary        — headline stats
  2. Score distributions  — histograms with automatic calibration warnings
  3. Category comparison  — easy / moderate / challenging breakdown
  4. Per-persona table    — one row per persona, sorted by outcome
  5. Paper tables         — Table 2, Robustness, Trait Sensitivity (reused from run_comparison.py)

Plus optional CSV export (--csv path).

Usage:
    python -m eval.analyze eval/results/multidomain_parallel/retail/layer.json
    python -m eval.analyze eval/results/multidomain_parallel/retail/   # merges all .json
    python -m eval.analyze eval/results/multidomain_parallel/retail/layer.json --csv out.csv
    python -m eval.analyze eval/results/multidomain_parallel/retail/layer.json --domain retail
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Optional

from eval.metrics import (
    ConditionReport,
    TaskMetrics,
    evaluate_condition,
    extract_task_metrics,
    bootstrap_ci,
    _mean,
)
from eval.run_comparison import (
    print_table2,
    print_robustness_table,
    print_trait_sensitivity_table,
)
from persona.categories import categorize_persona, PersonaCategory


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_from_path(path: str) -> tuple[list[dict], str]:
    """Load simulation records from a file or directory of per-persona JSON files.

    Directory mode deduplicates by sim id, so it's safe even if a consolidated
    file happens to coexist with the individual per-persona files.

    Returns:
        (list of simulation dicts, human-readable source description)
    """
    p = Path(path)

    if p.is_file():
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict) and "simulations" in data:
            data = data["simulations"]
        return data, p.name

    if p.is_dir():
        files = sorted(p.glob("*.json"))
        if not files:
            print(f"No .json files found in {path}", file=sys.stderr)
            sys.exit(1)
        seen: set[str] = set()
        sims: list[dict] = []
        for fp in files:
            with open(fp) as f:
                d = json.load(f)
            if isinstance(d, dict) and "simulations" in d:
                d = d["simulations"]
            if isinstance(d, list):
                for sim in d:
                    sid = sim.get("id", "")
                    if sid and sid in seen:
                        continue
                    if sid:
                        seen.add(sid)
                    sims.append(sim)
        return sims, f"{p.name}/  ({len(files)} files, {len(sims)} sims)"

    print(f"Path not found: {path}", file=sys.stderr)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: SCORE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def print_score_summary(sims: list[dict], report: ConditionReport, domain: str, source: str):
    """Print headline stats."""
    rewards = [s["reward_info"]["reward"] for s in sims]
    personas = set(s.get("_persona_id", "") for s in sims)
    tasks = set(s.get("task_id", "") for s in sims)
    msgs = [len(s.get("messages", [])) for s in sims]
    durations = [s.get("duration", 0) for s in sims]

    critics = [
        s["reward_info"].get("info", {}).get("persona_critic", {})
        for s in sims
    ]
    trust_scores = [c.get("trust") for c in critics if c and c.get("trust")]
    use_again = [c.get("use_again") for c in critics if c and c.get("use_again") is not None]
    trust_agreements = [c.get("trust_agreement", 1.0) for c in critics if c and c.get("trust")]

    cond = sims[0].get("_condition", "unknown") if sims else "unknown"

    print()
    print("═" * 64)
    print(f"  RESULTS: {domain} / {cond}  [{len(sims)} sims, {len(personas)} personas, {len(tasks)} tasks]")
    print(f"  Source: {source}")
    print("═" * 64)
    n_suc = sum(1 for r in rewards if r >= 1.0)
    print(f"  Task Success  : {_mean(rewards):.3f}  ({n_suc}/{len(rewards)})")
    if trust_scores:
        mean_agr = _mean(trust_agreements)
        print(f"  Mean Trust    : {_mean(trust_scores):.2f} / 7.0   (judge k=3, mean agreement {mean_agr:.2f})")
    if use_again:
        n_yes = sum(1 for u in use_again if u >= 1)
        print(f"  Use Again     : {_mean(use_again):.3f}  ({n_yes}/{len(use_again)} yes)")
    print(f"  Mean Messages : {_mean(msgs):.1f}   Mean Duration: {_mean(durations):.0f}s")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: SCORE DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════

def _bar(count: int, total: int, width: int = 30) -> str:
    """Render an inline bar: filled portion + empty portion."""
    filled = round(count / total * width) if total > 0 else 0
    return "▓" * filled + "░" * (width - filled)


def print_score_distributions(task_metrics: list[TaskMetrics]):
    """Print score histograms for reward, trust, and use_again."""
    n = len(task_metrics)
    if n == 0:
        return

    print("SCORE DISTRIBUTIONS")
    print("─" * 64)

    # Reward
    n_fail = sum(1 for m in task_metrics if m.reward < 0.5)
    n_succ = sum(1 for m in task_metrics if m.reward >= 0.5)
    print(f"  Reward   0.0  {_bar(n_fail, n)}  {n_fail:>3}/{n} ({100*n_fail/n:.1f}%)")
    print(f"           1.0  {_bar(n_succ, n)}  {n_succ:>3}/{n} ({100*n_succ/n:.1f}%)")
    print()

    # Trust
    trust_vals = [m.trust for m in task_metrics if m.trust > 0]
    if trust_vals:
        print(f"  Trust (1-7)   [n={len(trust_vals)}]")
        from collections import Counter
        counts = Counter(int(round(t)) for t in trust_vals)
        nt = len(trust_vals)
        for score in range(1, 8):
            c = counts.get(score, 0)
            pct = 100 * c / nt if nt > 0 else 0
            line = f"    {score}  {_bar(c, nt)}  {c:>3}/{nt} ({pct:.1f}%)"
            # Warning: >80% clustering at min or max
            if (score == min(counts) or score == max(counts)) and pct > 80:
                line += f"  ← WARNING: {pct:.0f}% at {'min' if score == 1 else 'max'} value"
            if c > 0:
                print(line)
        print()

    # Use Again
    use_vals = [m.use_again for m in task_metrics if m.trust > 0]  # only where judge ran
    if use_vals:
        n_no = sum(1 for u in use_vals if u < 0.5)
        n_yes = sum(1 for u in use_vals if u >= 0.5)
        nu = len(use_vals)
        print(f"  Use Again     [n={nu}]")
        print(f"     No   {_bar(n_no, nu)}  {n_no:>3}/{nu} ({100*n_no/nu:.1f}%)")
        print(f"    Yes   {_bar(n_yes, nu)}  {n_yes:>3}/{nu} ({100*n_yes/nu:.1f}%)")

    # Judge agreement
    critics = [m for m in task_metrics if m.trust > 0]
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: CATEGORY COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

def print_category_comparison(task_metrics: list[TaskMetrics]):
    """Group personas by easy/moderate/challenging, compare stats per group."""
    # Assign category to each simulation
    cat_sims: dict[str, list[TaskMetrics]] = defaultdict(list)
    uncategorized = 0
    for tm in task_metrics:
        if tm.persona_codes:
            cat = categorize_persona(tm.persona_codes)
            cat_sims[cat].append(tm)
        else:
            uncategorized += 1

    if not cat_sims:
        print("CATEGORY COMPARISON: no persona_codes found in results — skipping")
        return

    # Unique personas per category (avg across trials)
    cat_personas: dict[str, dict[str, list[float]]] = {}
    for cat, tms in cat_sims.items():
        by_persona: dict[str, list[TaskMetrics]] = defaultdict(list)
        for tm in tms:
            by_persona[tm.persona_id].append(tm)
        cat_personas[cat] = {
            pid: [t.reward for t in pts]
            for pid, pts in by_persona.items()
        }

    print("CATEGORY COMPARISON")
    print("  (scored dims: emotional_entry_state, bandwidth, goal_clarity, relational_stance,")
    print("   stress_response, stakes, interaction_friction, communicative_repertoire, domain_familiarity)")
    print("─" * 84)
    hdr = f"  {'Category':<14} {'N pers':>7} {'N sims':>7} {'Reward':>14} {'Trust':>14} {'Use Again':>14} {'Msgs':>8}"
    print(hdr)
    print("─" * 84)

    for cat in ["easy", "moderate", "challenging"]:
        tms = cat_sims.get(cat, [])
        if not tms:
            print(f"  {cat:<14}  {'—':>7}  {'—':>7}")
            continue

        n_personas = len(cat_personas.get(cat, {}))
        n_sims = len(tms)

        rewards = [m.reward for m in tms]
        trust_vals = [m.trust for m in tms if m.trust > 0]
        use_vals = [m.use_again for m in tms if m.trust > 0]
        msg_vals = [m.num_messages for m in tms]

        r_mean = _mean(rewards)
        r_std = stdev(rewards) if len(rewards) > 1 else 0.0
        t_mean = _mean(trust_vals) if trust_vals else None
        t_std = stdev(trust_vals) if len(trust_vals) > 1 else 0.0
        u_mean = _mean(use_vals) if use_vals else None
        u_std = stdev(use_vals) if len(use_vals) > 1 else 0.0
        m_mean = _mean(msg_vals)

        reward_str = f"{r_mean:.2f}±{r_std:.2f}"
        trust_str = f"{t_mean:.2f}±{t_std:.2f}" if t_mean is not None else "—"
        use_str = f"{u_mean:.2f}±{u_std:.2f}" if u_mean is not None else "—"

        print(f"  {cat:<14} {n_personas:>7} {n_sims:>7} {reward_str:>14} {trust_str:>14} {use_str:>14} {m_mean:>8.1f}")

    print("─" * 84)
    if uncategorized:
        print(f"  ({uncategorized} sims had no persona_codes and were excluded)")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: PER-PERSONA TABLE
# ═══════════════════════════════════════════════════════════════════════════

def print_per_persona_table(task_metrics: list[TaskMetrics]):
    """One row per persona, sorted by reward desc then use_again desc."""
    # Group by persona_id, aggregate trials
    by_persona: dict[str, list[TaskMetrics]] = defaultdict(list)
    for tm in task_metrics:
        pid = tm.persona_id or "unknown"
        by_persona[pid].append(tm)

    # Build rows
    rows = []
    for pid, tms in by_persona.items():
        rewards = [m.reward for m in tms]
        trust_vals = [m.trust for m in tms if m.trust > 0]
        use_vals = [m.use_again for m in tms if m.trust > 0]
        msgs = [m.num_messages for m in tms]
        codes = tms[0].persona_codes if tms[0].persona_codes else {}
        cat = categorize_persona(codes)[0:4] if codes else "?"  # first 4 chars: easy/mode/chal

        n_tasks = len(tms)
        n_success = sum(1 for r in rewards if r >= 1.0)
        mean_reward = _mean(rewards)
        mean_trust = _mean(trust_vals) if trust_vals else 0.0
        mean_use = _mean(use_vals) if use_vals else 0.0

        rows.append({
            "pid": pid,
            "cat": cat,
            "n_tasks": n_tasks,
            "n_success": n_success,
            "mean_reward": mean_reward,
            "mean_trust": mean_trust,
            "mean_use": mean_use,
            "msgs": msgs,
        })

    # Sort: reward desc, use_again desc, trust desc
    rows.sort(key=lambda r: (-r["mean_reward"], -r["mean_use"], -r["mean_trust"]))

    print(f"PER-PERSONA RESULTS ({len(rows)} personas)")
    print("─" * 90)
    print(f"  {'Persona ID':<42} {'Cat':<5} {'Tasks':>6} {'Reward':>7} {'Trust':>6} {'Use?':>5}  Messages")
    print("─" * 90)

    for r in rows:
        pid_short = r["pid"]
        if len(pid_short) > 41:
            pid_short = pid_short[:38] + "..."
        msgs_str = ", ".join(str(m) for m in r["msgs"])
        if len(msgs_str) > 16:
            msgs_str = msgs_str[:14] + "…"
        tasks_str = f"{r['n_success']}/{r['n_tasks']}"
        trust_str = f"{r['mean_trust']:.1f}" if r["mean_trust"] > 0 else "—"
        use_str = str(int(r["mean_use"])) if r["mean_trust"] > 0 else "—"
        print(
            f"  {pid_short:<42} {r['cat']:<5} {tasks_str:>6}"
            f" {r['mean_reward']:>7.2f} {trust_str:>6} {use_str:>5}  {msgs_str}"
        )

    print("─" * 90)
    print()


# ═══════════════════════════════════════════════════════════════════════════
# CSV EXPORT
# ═══════════════════════════════════════════════════════════════════════════

# All 12 persona dimensions in fixed order
_DIMENSION_COLS = [
    "situation_construal", "relational_stance", "agency", "epistemic", "stress_response",
    "communicative_repertoire", "domain_familiarity", "stakes", "interaction_friction",
    "emotional_entry_state", "bandwidth", "goal_clarity",
]

def export_csv(task_metrics: list[TaskMetrics], sims: list[dict], path: str):
    """Export one row per simulation to a flat CSV.

    Columns: sim_id, persona_id, category, condition, task_id, trial, reward, trust,
             use_again, trust_agreement, use_again_agreement, primary_dimension_violated,
             num_messages, duration, <12 dimension codes>, goal_achievement,
             cognitive_effort, intent_alignment
    """
    # Build a lookup from sim index to raw sim dict (for extra fields from reward_info.info)
    sim_lookup: dict[str, dict] = {}
    for s in sims:
        sim_id = s.get("id", "")
        sim_lookup[sim_id] = s

    fieldnames = [
        "sim_id", "persona_id", "category", "condition", "task_id", "trial",
        "reward", "trust", "use_again", "trust_agreement", "use_again_agreement",
        "primary_dimension_violated", "num_messages", "duration",
        *_DIMENSION_COLS,
        "goal_achievement", "cognitive_effort", "intent_alignment",
    ]

    # Map persona_id → sim for additional raw fields
    # We need to match task_metrics to sims by task_id + persona_id
    # Build a list paired by index (evaluate_condition preserves order)
    rows = []
    for i, tm in enumerate(task_metrics):
        # Find matching raw sim (best effort by persona_id + task_id)
        raw_sim = None
        for s in sims:
            if (s.get("_persona_id", "") == tm.persona_id
                    and s.get("task_id", "") == tm.task_id):
                raw_sim = s
                break

        critic = {}
        trial = 0
        sim_id = ""
        duration = tm.duration
        if raw_sim:
            sim_id = raw_sim.get("id", "")
            trial = raw_sim.get("trial", 0)
            duration = raw_sim.get("duration", tm.duration)
            info = raw_sim.get("reward_info", {}).get("info", {}) or {}
            critic = info.get("persona_critic", {}) or {}

        codes = tm.persona_codes or {}
        cat = categorize_persona(codes) if codes else ""

        row = {
            "sim_id": sim_id,
            "persona_id": tm.persona_id,
            "category": cat,
            "condition": tm.condition,
            "task_id": tm.task_id,
            "trial": trial,
            "reward": tm.reward,
            "trust": tm.trust if tm.trust > 0 else "",
            "use_again": tm.use_again if tm.trust > 0 else "",
            "trust_agreement": critic.get("trust_agreement", ""),
            "use_again_agreement": critic.get("use_again_agreement", ""),
            "primary_dimension_violated": critic.get("primary_dimension_violated", ""),
            "num_messages": tm.num_messages,
            "duration": f"{duration:.1f}",
            **{dim: codes.get(dim, "") for dim in _DIMENSION_COLS},
            "goal_achievement": f"{tm.goal_achievement:.2f}",
            "cognitive_effort": f"{tm.cognitive_effort:.2f}",
            "intent_alignment": f"{tm.intent_alignment:.2f}",
        }
        rows.append(row)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV exported: {path}  ({len(rows)} rows, {len(fieldnames)} columns)")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Single-condition persona result analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m eval.analyze eval/results/multidomain_parallel/retail/layer.json
  python -m eval.analyze eval/results/multidomain_parallel/retail/
  python -m eval.analyze eval/results/multidomain_parallel/retail/layer.json --csv out.csv
        """,
    )
    parser.add_argument("path", help="Path to result .json file or directory of .json files")
    parser.add_argument("--domain", default="retail", help="Domain name for table headers")
    parser.add_argument("--csv", metavar="PATH", help="Export flat CSV to this path")
    parser.add_argument("--no-paper-tables", action="store_true",
                        help="Skip Table 2, Robustness, and Trait Sensitivity tables")
    args = parser.parse_args()

    sims, source = load_from_path(args.path)
    if not sims:
        print("No simulations loaded.", file=sys.stderr)
        sys.exit(1)

    # Infer condition from data
    condition = sims[0].get("_condition", "unknown") if sims else "unknown"

    # Run full evaluation pipeline (computes all paper metrics)
    print(f"Evaluating {len(sims)} simulations…", end=" ", flush=True)
    report = evaluate_condition(sims, condition=condition)
    print("done.")

    # ── Section 1: Summary
    print_score_summary(sims, report, args.domain, source)

    # ── Section 2: Distributions
    print_score_distributions(report.task_metrics_list)

    # ── Section 3: Category comparison
    print_category_comparison(report.task_metrics_list)

    # ── Section 4: Per-persona table
    print_per_persona_table(report.task_metrics_list)

    # ── Section 5: Paper tables
    if not args.no_paper_tables:
        print_table2([report], domain=args.domain)
        print_robustness_table([report])
        print_trait_sensitivity_table([report])

    # ── CSV export
    if args.csv:
        export_csv(report.task_metrics_list, sims, args.csv)


if __name__ == "__main__":
    main()
