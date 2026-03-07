#!/usr/bin/env python3
"""
Run tau2 for every (model × persona) combination and copy results into Eval folders.

Usage:
  # From tau2-bench repo root (with venv active):
  python scripts/run_all_personas.py --models gpt-4o kimi --eval-dir /path/to/reddit/Eval

  # Or with defaults: personas from src/tau2/user/eval_personas, copy to ../multiagent_human_worker/reddit/Eval
  python scripts/run_all_personas.py --models gpt-4o

Each run uses TAU2_PERSONA_FILE so the saved JSON includes persona_name/persona_file in user_info.
Output is saved to data/simulations/<model>_<persona>.json then copied to <eval_dir>/<model>/<persona>_<model>.json.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run tau2 for each model and persona, then copy results to Eval dir."
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
        help="Directory of persona YAML files. Default: tau2-bench src/tau2/user/eval_personas",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Base dir for outputs, e.g. .../reddit/Eval. Each model gets a subdir: <eval_dir>/<model>/.",
    )
    parser.add_argument(
        "--domain",
        default="retail",
        help="Domain for tau2 run (default: retail).",
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
        default=20,
        help="Number of tasks per run (default: 20).",
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
        "extra",
        nargs="*",
        help="Extra args passed to 'tau2 run' (e.g. --max-steps 50).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if args.personas_dir is None:
        args.personas_dir = repo_root / "src" / "tau2" / "user" / "eval_personas"
    if args.eval_dir is None:
        args.eval_dir = repo_root.parent / "multiagent_human_worker" / "reddit" / "Eval"
    if args.simulations_dir is None:
        args.simulations_dir = repo_root / "data" / "simulations"

    args.personas_dir = args.personas_dir.resolve()
    args.eval_dir = args.eval_dir.resolve()
    args.simulations_dir = args.simulations_dir.resolve()

    if not args.personas_dir.is_dir():
        print(f"Personas dir not found: {args.personas_dir}", file=sys.stderr)
        sys.exit(1)

    persona_files = sorted(args.personas_dir.glob("*.yaml"))
    if not persona_files:
        print(f"No .yaml files in {args.personas_dir}", file=sys.stderr)
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
        "--num-tasks",
        str(args.num_tasks),
        *args.extra,
    ]

    for model in args.models:
        agent_llm = f"openai/{model}" if "/" not in model else model
        model_dir = args.eval_dir / model
        if not args.dry_run:
            model_dir.mkdir(parents=True, exist_ok=True)

        for persona_path in persona_files:
            persona_stem = persona_path.stem
            save_to = f"{model}_{persona_stem}"
            run_save_path = args.simulations_dir / f"{save_to}.json"
            dest_path = model_dir / f"{persona_stem}_{model}.json"

            env = os.environ.copy()
            env["TAU2_PERSONA_FILE"] = str(persona_path.resolve())

            cmd = base_cmd + ["--agent-llm", agent_llm, "--save-to", save_to]

            if args.dry_run:
                print(f"TAU2_PERSONA_FILE={env['TAU2_PERSONA_FILE']}")
                print(" ".join(cmd))
                print(f"  -> copy to {dest_path}\n")
                continue

            print(f"Running model={model} persona={persona_stem} ...")
            result = subprocess.run(cmd, env=env, cwd=repo_root)
            if result.returncode != 0:
                print(f"tau2 run failed for {model} / {persona_stem}", file=sys.stderr)
                sys.exit(result.returncode)

            if run_save_path.exists():
                shutil.copy2(run_save_path, dest_path)
                print(f"  -> {dest_path}")
            else:
                print(f"  Warning: run output not found at {run_save_path}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()
