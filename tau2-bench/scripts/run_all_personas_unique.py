#!/usr/bin/env python3
"""
Run tau2 for every (model x persona) combination with no persona reuse per model.

This is designed for a 200x1 setup:
  - many distinct personas
  - one task per persona (set --num-tasks 1)

Usage:
  # From tau2-bench repo root (with venv active):
  python scripts/run_all_personas_unique.py --models gpt-4o --personas-dir /path/to/200_personas --num-tasks 1 --expected-personas 200

  # Optional: resume safely by skipping outputs that already exist
  python scripts/run_all_personas_unique.py --models gpt-4o --personas-dir /path/to/200_personas --num-tasks 1 --skip-existing
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

from persona_jsonl import materialize_personas_from_jsonl


load_dotenv()
NAUT_API_KEY = os.getenv("NAUT_API_KEY")

def run_with_retry(
    cmd: list,
    env: dict,
    cwd: Path,
    max_retries: int = 9,
    initial_delay: float = 2.0,
    backoff_factor: float = 2.0,
) -> int:
    """Run command with exponential backoff on non-zero exit. Returns exit code (0 on success)."""
    delay = initial_delay
    # Auto-answer "y" to tau2's resume prompts so runs stay non-interactive
    stdin_input = "y\ny\n"
    for attempt in range(max_retries):
        result = subprocess.run(cmd, env=env, cwd=cwd, input=stdin_input, text=True)
        if result.returncode == 0:
            return 0
        if attempt == max_retries - 1:
            return result.returncode
        print(
            f"  Attempt {attempt + 1}/{max_retries} failed (exit {result.returncode}). Retrying in {delay:.1f}s...",
            file=sys.stderr,
        )
        time.sleep(delay)
        delay = min(delay * backoff_factor, 120.0)
    return result.returncode


def to_fs_label(value: str) -> str:
    """Map model/provider strings to a stable filesystem-safe label."""
    return value.replace("/", "__")


def load_task_ids(task_set_name: str, task_split_name: str) -> list[str]:
    """Load ordered task IDs from tau2 task registry."""
    from tau2.run import load_tasks

    tasks = load_tasks(task_set_name=task_set_name, task_split_name=task_split_name)
    return [task.id for task in tasks]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tau2 for each model and persona once, then copy results to Eval dir."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Agent model names (e.g. gpt-4o kimi). Passed as openai/<model> to --agent-llm.",
    )
    parser.add_argument(
        "--personas-dir",
        type=Path,
        default=None,
        help="Directory of persona YAML files (your 200-persona folder).",
    )
    parser.add_argument(
        "--personas-jsonl",
        type=Path,
        default=None,
        help="JSONL with persona_yaml per row. Mutually exclusive with --personas-dir.",
    )
    parser.add_argument(
        "--personas-limit",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N rows from --personas-jsonl.",
    )
    parser.add_argument(
        "--personas-cache-dir",
        type=Path,
        default=None,
        help="Where to write YAML extracted from --personas-jsonl.",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Base dir for outputs (default: ../multiagent_human_worker/reddit/Eval).",
    )
    parser.add_argument(
        "--domain",
        default="retail",
        help="Domain for tau2 run (default: retail).",
    )
    parser.add_argument(
        "--task-set-name",
        default=None,
        help="Task set name for loading IDs when cycling tasks (default: same as --domain).",
    )
    parser.add_argument(
        "--task-split-name",
        default="base",
        help="Task split for loading IDs when cycling tasks (default: base).",
    )
    parser.add_argument(
        "--user-llm",
        default="openai/gemma3",
        help="User simulator LLM (default: openai/gemma3).",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=1,
        help="Number of trials per task (default: 1).",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=1,
        help="Number of tasks per persona (default: 1 for 200x1 setup).",
    )
    parser.add_argument(
        "--cycle-tasks",
        action="store_true",
        help="Cycle through task IDs so consecutive personas get different tasks (uses --task-ids).",
    )
    parser.add_argument(
        "--no-cycle-tasks",
        action="store_true",
        help="Disable task cycling even if --cycle-tasks is set.",
    )
    parser.add_argument(
        "--expected-personas",
        type=int,
        default=200,
        help="Fail fast unless this many persona files are present (default: 200). Set 0 to disable.",
    )
    parser.add_argument(
        "--simulations-dir",
        type=Path,
        default=None,
        help="Where tau2 writes JSON (default: <repo>/data/simulations).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and copy targets only, do not run.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip (model, persona) pairs that already have output in <eval_dir>/<model>/<persona>_<model>.json.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=9,
        metavar="N",
        help="Retry each tau2 run up to N times on failure with exponential backoff (default: 9).",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="Extra args passed to 'tau2 run' (e.g. --max-steps 50).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.personas_dir and args.personas_jsonl:
        print("Use only one of --personas-dir or --personas-jsonl.", file=sys.stderr)
        sys.exit(1)
    if args.personas_dir is None and args.personas_jsonl is None:
        print("One of --personas-dir or --personas-jsonl is required.", file=sys.stderr)
        sys.exit(1)
    if args.eval_dir is None:
        args.eval_dir = repo_root.parent / "multiagent_human_worker" / "reddit" / "Eval"
    if args.simulations_dir is None:
        args.simulations_dir = repo_root / "data" / "simulations"

    args.personas_dir = args.personas_dir.resolve() if args.personas_dir else None
    if str(args.eval_dir).strip() in ("", "="):
        print(
            "Invalid --eval-dir (did you use spaces? Use: --eval-dir /path or --eval-dir=/path)",
            file=sys.stderr,
        )
        sys.exit(1)
    args.eval_dir = args.eval_dir.resolve()
    args.simulations_dir = args.simulations_dir.resolve()

    if args.eval_dir.exists() and not args.eval_dir.is_dir():
        print(f"Eval dir is not a directory: {args.eval_dir}", file=sys.stderr)
        sys.exit(1)
    if args.personas_jsonl:
        args.personas_jsonl = args.personas_jsonl.resolve()
        if not args.personas_jsonl.is_file():
            print(f"Personas JSONL not found: {args.personas_jsonl}", file=sys.stderr)
            sys.exit(1)
        try:
            persona_files = materialize_personas_from_jsonl(
                args.personas_jsonl,
                cache_dir=args.personas_cache_dir,
                limit=args.personas_limit,
            )
        except ValueError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)
        cache_dir = persona_files[0].parent
        print(
            f"Loaded {len(persona_files)} persona(s) from {args.personas_jsonl} "
            f"(cache: {cache_dir})"
        )
    else:
        if not args.personas_dir.is_dir():
            print(f"Personas dir not found: {args.personas_dir}", file=sys.stderr)
            sys.exit(1)
        persona_files = sorted(args.personas_dir.glob("*.yaml"))
        if not persona_files:
            print(f"No .yaml files in {args.personas_dir}", file=sys.stderr)
            sys.exit(1)

    if args.expected_personas > 0 and len(persona_files) != args.expected_personas:
        print(
            f"Expected {args.expected_personas} personas, found {len(persona_files)} in {args.personas_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    base_cmd = [
        "tau2",
        "run",
        "--domain",
        args.domain,
        "--user-llm",
        args.user_llm,
        "--num-trials",
        str(args.num_trials),
        "--task-split-name",
        args.task_split_name,
        *args.extra,
    ]
    if args.task_set_name:
        base_cmd += ["--task-set-name", args.task_set_name]

    cycle_tasks = args.cycle_tasks and not args.no_cycle_tasks
    task_ids = []
    if cycle_tasks:
        task_set_name = args.task_set_name or args.domain
        task_ids = load_task_ids(task_set_name=task_set_name, task_split_name=args.task_split_name)
        if not task_ids:
            print(
                f"No task IDs found for task_set={task_set_name} split={args.task_split_name}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Loaded {len(task_ids)} tasks for cycling from {task_set_name}/{args.task_split_name}.")

    for model in args.models:
        agent_llm = f"openai/{model}" if "/" not in model else model
        model_label = to_fs_label(model)
        model_dir = args.eval_dir / model_label
        if not args.dry_run:
            model_dir.mkdir(parents=True, exist_ok=True)

        used_personas = set()
        for idx, persona_path in enumerate(persona_files):
            persona_stem = persona_path.stem

            # Explicit non-reuse guard (requested): each persona is run at most once per model.
            if persona_stem in used_personas:
                print(f"Skipping duplicate persona in run list: {persona_stem}", file=sys.stderr)
                continue
            used_personas.add(persona_stem)

            save_to = f"{model_label}_{persona_stem}_{args.domain}"
            run_save_path = args.simulations_dir / f"{save_to}.json"
            dest_path = model_dir / f"{persona_stem}_{model_label}_{args.domain}.json"

            env = os.environ.copy()
            env["TAU2_PERSONA_FILE"] = str(persona_path.resolve())
            cmd = base_cmd + ["--agent-llm", agent_llm, "--save-to", save_to]
            if cycle_tasks:
                chosen_task_id = task_ids[idx % len(task_ids)]
                cmd += ["--task-ids", chosen_task_id, "--num-tasks", "1"]
            else:
                cmd += ["--num-tasks", str(args.num_tasks)]

            if args.dry_run:
                print(f"TAU2_PERSONA_FILE={env['TAU2_PERSONA_FILE']}")
                print(" ".join(cmd))
                print(f"  -> copy to {dest_path}\n")
                continue

            if args.skip_existing and dest_path.exists():
                print(f"Skipping model={model} persona={persona_stem} (already exists: {dest_path})")
                continue

            print(f"Running model={model} persona={persona_stem} ...")
            exit_code = run_with_retry(cmd, env, repo_root, max_retries=args.max_retries)
            if exit_code != 0:
                print(
                    f"tau2 run failed for {model} / {persona_stem} after {args.max_retries} attempt(s)",
                    file=sys.stderr,
                )
                sys.exit(exit_code)

            if run_save_path.exists():
                shutil.copy2(run_save_path, dest_path)
                print(f"  -> {dest_path}")
            else:
                print(f"  Warning: run output not found at {run_save_path}", file=sys.stderr)

        print(f"Model {model}: executed {len(used_personas)} unique personas.")

    print("Done.")


if __name__ == "__main__":
    main()
