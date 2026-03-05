"""Multi-domain evaluation runner.

Runs layer-based persona simulations across multiple domains sequentially,
with per-persona resume support (skips already-completed personas).

Supports all τ²-bench domains: airline, retail, telecom, telecom-workflow.

Usage:
    # Run all domains with defaults (2 tasks each):
    python -m eval.run_multidomain

    # Run specific domains:
    python -m eval.run_multidomain --domains airline retail

    # Use task splits instead of specific task IDs:
    python -m eval.run_multidomain --task-split small

    # Custom number of tasks per domain:
    python -m eval.run_multidomain --num-tasks 5

    # Full test across all domains:
    python -m eval.run_multidomain --domains airline retail telecom --num-tasks 10 --concurrency 10
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path when run as a module
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from eval.run_experiment import run_condition, load_layer_personas


# ── Defaults ──────────────────────────────────────────────────────────────

ALL_DOMAINS = ["airline", "retail", "telecom"]

# Default task IDs per domain (used when no --task-split or --num-tasks given).
# Chosen to be quick representative tasks for each domain.
DEFAULT_TASK_IDS = {
    "airline": ["0", "1"],
    "retail": ["0", "1"],
    "telecom": [
        "[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]",
        "[service_issue]airplane_mode_on[PERSONA:None]",
    ],
    "telecom-workflow": [
        "[mobile_data_issue]user_abroad_roaming_enabled_off[PERSONA:None]",
        "[service_issue]airplane_mode_on[PERSONA:None]",
    ],
}

DEFAULT_PERSONA_DIR = "reddit/eval_personas_all"
DEFAULT_SPECS_DIR = "specs"
DEFAULT_OUTPUT_BASE = "eval/results/multidomain"
DEFAULT_MODEL = "openai/llama3-sdsc"
DEFAULT_AGENT_TYPE = "llm_agent_gt"
DEFAULT_CONCURRENCY = 20


def main():
    parser = argparse.ArgumentParser(
        description="Run persona simulations across multiple τ²-bench domains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick run on all domains (2 tasks each):
  python -m eval.run_multidomain

  # Airline + retail only:
  python -m eval.run_multidomain --domains airline retail

  # Use the "small" task split (20 tasks) for telecom:
  python -m eval.run_multidomain --domains telecom --task-split small

  # 5 random tasks per domain, higher concurrency:
  python -m eval.run_multidomain --num-tasks 5 --concurrency 30

Available domains: airline (50 tasks), retail (114 tasks),
                   telecom (2285 tasks), telecom-workflow (2285 tasks)
        """,
    )

    parser.add_argument(
        "--domains", nargs="+", default=ALL_DOMAINS,
        choices=["airline", "retail", "telecom", "telecom-workflow"],
        help=f"Domains to evaluate (default: {ALL_DOMAINS})",
    )
    parser.add_argument(
        "--task-ids", nargs="+", default=None,
        help="Specific task IDs (applied to ALL domains). Overrides defaults.",
    )
    parser.add_argument(
        "--task-split", default=None,
        choices=["base", "small", "train", "test"],
        help="Use a named task split instead of specific IDs (e.g., 'small' for telecom).",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=None,
        help="Limit to N tasks per domain (random subset). Overrides --task-ids.",
    )
    parser.add_argument(
        "--personas", default=DEFAULT_PERSONA_DIR,
        help=f"Path to persona YAML directory (default: {DEFAULT_PERSONA_DIR})",
    )
    parser.add_argument(
        "--specs", default=DEFAULT_SPECS_DIR,
        help=f"Path to layer specs directory (default: {DEFAULT_SPECS_DIR})",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_BASE,
        help=f"Output base directory (default: {DEFAULT_OUTPUT_BASE})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"LLM model for agent and user simulator (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--agent-type", default=DEFAULT_AGENT_TYPE,
        choices=["llm_agent", "llm_agent_gt"],
        help=f"Agent implementation (default: {DEFAULT_AGENT_TYPE})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=DEFAULT_CONCURRENCY,
        help=f"Max parallel simulations (default: {DEFAULT_CONCURRENCY})",
    )
    parser.add_argument(
        "--num-trials", type=int, default=1,
        help="Number of trials per task (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=300,
        help="Random seed (default: 300)",
    )
    parser.add_argument(
        "--task-persona-filter", default=None,
        choices=["None", "Easy", "Hard"],
        help="Filter tasks by built-in tau2 persona tag (None/Easy/Hard)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Multi-Domain Persona Evaluation")
    print("=" * 60)
    print(f"  Domains:     {args.domains}")
    print(f"  Personas:    {args.personas}")
    print(f"  Output:      {args.output}")
    print(f"  Model:       {args.model}")
    print(f"  Agent:       {args.agent_type}")
    print(f"  Concurrency: {args.concurrency}")
    if args.task_split:
        print(f"  Task split:  {args.task_split}")
    elif args.num_tasks:
        print(f"  Tasks/domain: {args.num_tasks}")
    elif args.task_ids:
        print(f"  Task IDs:    {args.task_ids}")
    else:
        print(f"  Task IDs:    per-domain defaults (2 each)")
    print()

    # Load personas once
    all_personas = load_layer_personas(args.personas, args.specs)
    print(f"Loaded {len(all_personas)} personas")
    if not all_personas:
        print("ERROR: No personas loaded. Check --personas path.")
        sys.exit(1)

    # Run each domain
    for domain in args.domains:
        domain_output = os.path.join(args.output, domain)
        os.makedirs(domain_output, exist_ok=True)

        # Determine task selection: --num-tasks > --task-split > --task-ids > defaults
        task_ids = None
        num_tasks = None

        if args.num_tasks:
            num_tasks = args.num_tasks
        elif args.task_split:
            # task_split is handled by get_tasks() in run_condition
            # We pass it as task_ids=None and let the split loader handle it
            pass
        elif args.task_ids:
            task_ids = args.task_ids
        else:
            task_ids = DEFAULT_TASK_IDS.get(domain)

        # Filter personas that haven't been run for THIS domain yet
        personas_to_run = []
        for p in all_personas:
            safe_pid = p['persona_id'].replace("/", "_")
            p_file = os.path.join(domain_output, f"{safe_pid}.json")
            if not os.path.exists(p_file):
                personas_to_run.append(p)

        if not personas_to_run:
            print(f"\n[{domain.upper()}] All {len(all_personas)} personas already completed. Skipping.")
            continue

        skipped = len(all_personas) - len(personas_to_run)
        skip_msg = f" ({skipped} already done)" if skipped > 0 else ""
        pf = f" [PERSONA:{args.task_persona_filter}]" if args.task_persona_filter else ""
        task_desc = (
            f"{num_tasks} tasks" if num_tasks
            else f"split '{args.task_split}'" if args.task_split
            else f"tasks {task_ids}" if task_ids
            else "all tasks"
        ) + pf
        print(f"\n[{domain.upper()}] Running {len(personas_to_run)} personas on {task_desc}{skip_msg}")

        try:
            run_condition(
                condition="layer",
                domain=domain,
                personas=personas_to_run,
                output_dir=domain_output,
                task_ids=task_ids,
                num_tasks=num_tasks,
                task_split=args.task_split if not num_tasks and not task_ids else None,
                task_persona_filter=args.task_persona_filter,
                num_trials=args.num_trials,
                seed=args.seed,
                llm_agent=args.model,
                llm_user=args.model,
                agent_type=args.agent_type,
                max_concurrency=args.concurrency,
            )
        except Exception as e:
            print(f"ERROR in domain {domain}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print(f"Done. Results in: {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
