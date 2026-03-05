"""Run comparison evaluation across persona conditions — paper table output.

Produces the 4 tables from "Persona-Conditioned Evaluation of Agentic Systems":

  Table 1: Task success rate per agent/condition
  Table 2: 5-dimension persona-conditioned metrics (3 proxy + 2 LLM judge)
  Robustness table: variance, tail risk, persona gap (Section 2.3)
  Trait sensitivity table: E[M(A,p) | t_j = v] per dimension (Section 2.3)

Usage:
    # From pre-existing result files:
    python -m eval.run_comparison \
        --layer eval/results/layer.json \
        --none eval/results/none.json \
        --output eval/results/comparison.json

    # Legacy (Schwartz) condition:
    python -m eval.run_comparison \
        --layer eval/results/layer.json \
        --legacy eval/results/legacy.json \
        --none eval/results/none.json
"""

import argparse
import json
import sys
from typing import Optional

from eval.metrics import (
    ConditionReport,
    compare_conditions,
    evaluate_condition,
    bootstrap_ci,
)


def load_results(path: str) -> list[dict]:
    """Load simulation results from a JSON file or directory of JSON files.

    Directory mode merges all *.json files and deduplicates by sim id,
    so it's safe whether or not a consolidated file also exists.
    """
    from pathlib import Path
    p = Path(path)

    if p.is_dir():
        seen: set[str] = set()
        sims: list[dict] = []
        for fp in sorted(p.glob("*.json")):
            with open(fp) as f:
                d = json.load(f)
            if isinstance(d, dict) and "simulations" in d:
                d = d["simulations"]
            if not isinstance(d, list):
                continue
            for sim in d:
                sid = sim.get("id", "")
                if sid and sid in seen:
                    continue
                if sid:
                    seen.add(sid)
                sims.append(sim)
        return sims

    with open(p) as f:
        data = json.load(f)
    if isinstance(data, dict) and "simulations" in data:
        return data["simulations"]
    if isinstance(data, list):
        return data
    return []


# ═══════════════════════════════════════════════════════════════════════════
# TABLE PRINTERS — Paper-aligned output
# ═══════════════════════════════════════════════════════════════════════════

def print_table1(reports: list[ConditionReport], domain: str = "retail"):
    """Print Table 1: Task Success Rate per condition.

    Format:
        Domain    | condition_1 | condition_2 | ...
        ----------|-------------|-------------|----
        Retail    | 0.70        | 0.66        | ...
    """
    conditions = [r.condition for r in reports]
    col_width = max(14, max(len(c) for c in conditions) + 4)

    print("\n" + "=" * 70)
    print("TABLE 1: Task Success Rate")
    print("=" * 70)

    # Header
    header = f"{'Domain':<14}" + "".join(f"{c:>{col_width}}" for c in conditions)
    print(header)
    print("-" * len(header))

    # Data row
    values = []
    for r in reports:
        rate = r.task_metrics.get("task_success_rate", 0.0)
        n = r.n_simulations
        values.append(f"{rate:.2f} (n={n})")
    row = f"{domain:<14}" + "".join(f"{v:>{col_width}}" for v in values)
    print(row)
    print()


def print_table2(reports: list[ConditionReport], domain: str = "retail"):
    """Print Table 2: Persona-conditioned quality metrics.

    5 metrics:
      - Goal Achievement (1-7, proxy)
      - Trust (1-7, LLM judge*)
      - Cognitive Effort (1-7, proxy, lower=better)
      - Intent Alignment (1-7, proxy)
      - Use Again (0-1, LLM judge*)
    """
    conditions = [r.condition for r in reports]
    # Use wider columns to accommodate CI brackets
    col_width = max(22, max(len(c) for c in conditions) + 4)

    print("\n" + "=" * 80)
    print("TABLE 2: Persona-Conditioned Quality Metrics")
    print("(* = LLM judge with k=3 agreement; rest are transcript-derived)")
    print("=" * 80)

    # Header
    header = f"{'Domain':<10}{'Metric':<25}" + "".join(
        f"{c:>{col_width}}" for c in conditions
    )
    print(header)
    print("-" * len(header))

    metrics = [
        ("Goal Achievement", "goal_achievement", ".2f"),
        ("Trust*", "trust", ".2f"),
        ("Cognitive Effort", "cognitive_effort", ".2f"),
        ("Intent Alignment", "intent_alignment", ".2f"),
        ("Use Again*", "use_again", ".2f"),
    ]

    for i, (label, key, fmt) in enumerate(metrics):
        values = []
        for r in reports:
            val = r.table2.get(key)
            if val is not None:
                ci = r.table2.get(f"{key}_ci")
                if ci:
                    values.append(f"{val:{fmt}} [{ci[0]:{fmt}}, {ci[1]:{fmt}}]")
                else:
                    values.append(f"{val:{fmt}}")
            else:
                values.append("—")

        domain_col = domain if i == 0 else ""
        row = f"{domain_col:<10}{label:<25}" + "".join(
            f"{v:>{col_width}}" for v in values
        )
        print(row)

    print()


def print_robustness_table(reports: list[ConditionReport]):
    """Print robustness metrics table (Section 2.3 / 3.3).

    Format:
        Agent   | Mean  | Var   | Min   | Max   | Gap   | Q_10  | Q_25  | N
        --------|-------|-------|-------|-------|-------|-------|-------|---
        layer   | 0.70  | 0.032 | 0.45  | 0.90  | 0.45  | 0.50  | 0.58  | 20
    """
    print("\n" + "=" * 70)
    print("ROBUSTNESS METRICS (Section 2.3)")
    print("=" * 70)

    columns = [
        ("Mean", "mean_performance", ".3f"),
        ("Var", "cross_persona_variance", ".4f"),
        ("Min", "worst_case", ".3f"),
        ("Max", "best_case", ".3f"),
        ("Gap", "persona_gap", ".3f"),
        ("Q_10", "tail_risk_10", ".3f"),
        ("Q_25", "tail_risk_25", ".3f"),
        ("N_pers", "n_personas", "d"),
    ]

    col_width = 10
    header = f"{'Condition':<14}" + "".join(
        f"{name:>{col_width}}" for name, _, _ in columns
    )
    print(header)
    print("-" * len(header))

    for r in reports:
        rob = r.robustness
        if not rob:
            print(f"{r.condition:<14}{'(no robustness data — needs persona variation)':>40}")
            continue

        values = []
        for _, key, fmt in columns:
            val = rob.get(key)
            if val is not None:
                values.append(f"{val:{fmt}}")
            else:
                values.append("—")
        row = f"{r.condition:<14}" + "".join(f"{v:>{col_width}}" for v in values)
        print(row)

    print()


def print_trait_sensitivity_table(reports: list[ConditionReport]):
    """Print trait sensitivity table (Section 2.3 — core paper contribution).

    Shows E[M(A,p) | t_j = v] for each dimension and value.
    Only prints for conditions that have trait sensitivity data.

    Format:
        Condition | Dimension               | Value | Mean  | CI_lo | CI_hi | n
        ----------|-------------------------|-------|-------|-------|-------|---
        layer     | agency                  | 3A    | 0.85  | 0.72  | 0.94  | 12
    """
    # Collect all reports that have trait sensitivity data
    reports_with_ts = [r for r in reports if r.trait_sensitivity]
    if not reports_with_ts:
        print("\nTRAIT SENSITIVITY: No data (requires persona_codes in simulation dicts)")
        return

    print("\n" + "=" * 90)
    print("TRAIT SENSITIVITY: E[M(A,p) | t_j = v]  (Section 2.3)")
    print("=" * 90)

    header = f"{'Condition':<12}{'Dimension':<28}{'Value':<8}{'Mean':>8}{'CI_lo':>8}{'CI_hi':>8}{'n':>6}"
    print(header)
    print("-" * len(header))

    for r in reports_with_ts:
        first_cond = True
        for dim in sorted(r.trait_sensitivity.keys()):
            dim_data = r.trait_sensitivity[dim]
            first_dim = True
            for code in sorted(dim_data.keys(), key=_trait_sort_key):
                entry = dim_data[code]
                cond_col = r.condition if first_cond else ""
                dim_col = dim if first_dim else ""
                row = (
                    f"{cond_col:<12}{dim_col:<28}{code:<8}"
                    f"{entry['mean']:>8.3f}"
                    f"{entry['ci_lo']:>8.3f}"
                    f"{entry['ci_hi']:>8.3f}"
                    f"{entry['n']:>6d}"
                )
                print(row)
                first_cond = False
                first_dim = False

        # Separator between conditions
        if r != reports_with_ts[-1]:
            print("-" * len(header))

    print()


def print_full_report(
    reports: list[ConditionReport],
    domain: str = "retail",
):
    """Print all paper tables."""
    print("\n" + "█" * 70)
    print("  PERSONA-CONDITIONED EVALUATION RESULTS")
    print("  Paper: 'Persona-Conditioned Evaluation of Agentic Systems'")
    print("█" * 70)

    print_table1(reports, domain=domain)
    print_table2(reports, domain=domain)
    print_robustness_table(reports)
    print_trait_sensitivity_table(reports)

    # Summary comparison
    summary = compare_conditions(reports)
    _print_comparison_summary(summary)


def _print_comparison_summary(summary: dict):
    """Print a compact comparison overview."""
    conditions = list(summary.keys())
    if len(conditions) < 2:
        return

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    col_width = max(16, max(len(c) for c in conditions) + 4)

    rows = [
        ("Task Success", "task_success_rate", ".3f"),
        ("N Simulations", "n_simulations", "d"),
        ("Mean Messages", "mean_messages", ".1f"),
        ("Goal Achiev.", "table2_goal_achievement", ".2f"),
        ("Trust*", "table2_trust", ".2f"),
        ("Cog. Effort", "table2_cognitive_effort", ".2f"),
        ("Intent Align.", "table2_intent_alignment", ".2f"),
        ("Use Again*", "table2_use_again", ".2f"),
        ("Rob: Variance", "robustness_cross_persona_variance", ".4f"),
        ("Rob: Gap", "robustness_persona_gap", ".3f"),
        ("Rob: Worst", "robustness_worst_case", ".3f"),
        ("Rob: Q_10", "robustness_tail_risk_10", ".3f"),
    ]

    header = f"{'Metric':<22}" + "".join(f"{c:>{col_width}}" for c in conditions)
    print(header)
    print("-" * len(header))

    for label, key, fmt in rows:
        values = []
        for c in conditions:
            val = summary[c].get(key)
            if val is not None:
                values.append(f"{val:{fmt}}")
            else:
                values.append("—")
        row = f"{label:<22}" + "".join(f"{v:>{col_width}}" for v in values)
        print(row)

    print("=" * 70)
    print("(* = LLM judge with k=3 agreement)")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_from_files(
    results_by_condition: dict[str, str],
    output_path: Optional[str] = None,
    domain: str = "retail",
):
    """Evaluate from pre-existing tau-bench result files.

    Args:
        results_by_condition: {"layer": "path.json", "none": "path.json", ...}
        output_path: Where to save the comparison JSON.
        domain: Domain name for table headers.
    """
    reports = []
    for condition, path in results_by_condition.items():
        print(f"Loading {condition} results from {path}...")
        sims = load_results(path)
        print(f"  {len(sims)} simulations loaded")
        report = evaluate_condition(sims, condition=condition)
        reports.append(report)

    # Print paper tables
    print_full_report(reports, domain=domain)

    # Build serializable summary
    summary = compare_conditions(reports)

    # Add trait sensitivity to summary (compare_conditions doesn't include it
    # because it's too nested for the flat comparison dict)
    for report in reports:
        if report.trait_sensitivity:
            summary[report.condition]["trait_sensitivity"] = _serialize_trait_sensitivity(
                report.trait_sensitivity
            )

    if output_path:
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return summary


def _serialize_trait_sensitivity(trait_sens: dict) -> dict:
    """Make trait sensitivity JSON-serializable (remove raw reward lists)."""
    result = {}
    for dim, codes in trait_sens.items():
        result[dim] = {}
        for code, entry in codes.items():
            result[dim][code] = {
                "mean": entry["mean"],
                "ci_lo": entry["ci_lo"],
                "ci_hi": entry["ci_hi"],
                "n": entry["n"],
            }
    return result


def _trait_sort_key(code: str) -> tuple:
    """Sort trait codes naturally: 1A < 1B, C1 < C2 < C12."""
    if not code:
        return ("", 0)
    if code[0].isdigit():
        return (int(code[0]), code[1:])
    else:
        try:
            return (code[0], int(code[1:]))
        except ValueError:
            return (code[0], 0)


def main():
    parser = argparse.ArgumentParser(
        description="Persona comparison evaluation — paper table output"
    )
    parser.add_argument("--layer", help="Path to layer-persona results JSON")
    parser.add_argument("--legacy", help="Path to legacy (Schwartz) results JSON")
    parser.add_argument("--none", help="Path to no-persona baseline results JSON")
    parser.add_argument("--domain", default="retail", help="Domain name for tables")
    parser.add_argument(
        "--output", default="comparison_results.json",
        help="Output path for comparison results JSON"
    )

    args = parser.parse_args()

    conditions = {}
    if args.layer:
        conditions["layer"] = args.layer
    if args.legacy:
        conditions["legacy"] = args.legacy
    if getattr(args, "none", None):
        conditions["none"] = args.none

    if not conditions:
        print("Provide at least one condition: --layer, --legacy, or --none")
        sys.exit(1)

    evaluate_from_files(conditions, output_path=args.output, domain=args.domain)


if __name__ == "__main__":
    main()
