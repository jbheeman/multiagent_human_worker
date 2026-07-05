"""Run STATE-Bench for every (model x persona) and save reddit/tau2-style aggregates.

Mirrors tau2-bench/scripts/run_all_personas.py: one JSON per (persona, model) holding
ALL tasks, written to <eval-dir>/<model>/<persona>_<model>_<domain>.json. Personas are
keyed by the JSONL row's user_id (the reddit username), so 100 users stay distinct.

Unlike tau2 this runs in-process (no `tau2 run` subprocess): it calls the shared
run_one_trajectory() used by run_unofficial, then packs the per-task Trajectories into a
{timestamp, info, tasks, simulations} envelope. The `simulations` entries carry
STATE-Bench's real metrics (terminal_state, state_requirements_met, satisfaction_*),
NOT tau2's reward_info -- the container shape matches reddit, the payload is STATE-Bench.

Examples:
    # dry run: print every planned (model, persona) and its output path
    uv run python -m state_bench.scripts.run_all_personas \
        --models gpt-oss \
        --personas-jsonl ../reddit/personas_axis_a_reddit_schwartz_unopt_k100.jsonl \
        --eval-dir ../reddit/Eval/statebench_schwartz_unopt --dry-run

    # real run: gemma user-sim, gpt-oss agent, 20 travel tasks, resume-safe
    uv run python -m state_bench.scripts.run_all_personas \
        --models gpt-oss --domain travel --num-tasks 20 \
        --personas-jsonl ../reddit/personas_axis_a_reddit_schwartz_unopt_k100.jsonl \
        --eval-dir ../reddit/Eval/statebench_schwartz_unopt --skip-existing
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from state_bench.agents.loader import load_root_agent_class, load_root_client_class
from state_bench.paths import domain_tasks_dir
from state_bench.schemas import TaskDefinition
from state_bench.scripts.persona_injection import load_persona_yaml, persona_id_from_yaml
from state_bench.scripts.persona_jsonl import materialize_personas_from_jsonl
from state_bench.scripts.run_unofficial import run_one_trajectory


def to_fs_label(value: str) -> str:
    """Map a model/provider slug to a stable filesystem-safe label (matches tau2)."""
    return value.replace("/", "__")


def _leading_num(task_id: str) -> int:
    match = re.match(r"(\d+)", task_id)
    return int(match.group(1)) if match else 10**9


def _resolve_tasks(domain: str, task_args: list[str] | None, num_tasks: int) -> list[TaskDefinition]:
    tasks_dir = domain_tasks_dir(domain)
    if task_args:
        ids: list[str] = []
        for value in task_args:
            ids.extend(part.strip() for part in value.split(",") if part.strip())
        tasks, missing = [], []
        for task_id in ids:
            path = tasks_dir / f"{task_id}.json"
            (tasks.append(TaskDefinition.load(path)) if path.exists() else missing.append(task_id))
        if missing:
            raise SystemExit(f"Task(s) not found in {tasks_dir}: {', '.join(missing)}")
        return tasks
    # Default: first num_tasks by leading number (== reddit's `--num-tasks 20`).
    all_paths = sorted(tasks_dir.glob("*.json"), key=lambda p: _leading_num(p.stem))
    return [TaskDefinition.load(p) for p in all_paths[:num_tasks]]


def _resolve_personas(args: argparse.Namespace) -> list[Path]:
    if args.personas_dir and args.personas_jsonl:
        raise SystemExit("Use only one of --personas-dir or --personas-jsonl.")
    if args.personas_jsonl:
        jsonl = args.personas_jsonl.resolve()
        if not jsonl.is_file():
            raise SystemExit(f"Personas JSONL not found: {jsonl}")
        files = materialize_personas_from_jsonl(jsonl, cache_dir=args.personas_cache_dir, limit=args.personas_limit)
        print(f"Loaded {len(files)} persona(s) from {jsonl} (cache: {files[0].parent})")
        return files
    if args.personas_dir:
        pdir = args.personas_dir.resolve()
        files = sorted(pdir.glob("*.yaml"))
        if not files:
            raise SystemExit(f"No *.yaml personas found in {pdir}")
        if args.personas_limit:
            files = files[: args.personas_limit]
        return files
    raise SystemExit("Provide --personas-jsonl or --personas-dir")


def _git_commit() -> str | None:
    try:
        here = Path(__file__).resolve().parent
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=False
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - metadata only
        return None


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run STATE-Bench per (model, persona), reddit-style aggregates")
    parser.add_argument("--models", nargs="+", required=True, help="Agent model slugs (e.g. gpt-oss qwen3).")
    parser.add_argument("--personas-jsonl", type=Path, default=None, help="JSONL with persona_yaml per row.")
    parser.add_argument("--personas-dir", type=Path, default=None, help="Directory of *.yaml personas.")
    parser.add_argument("--personas-limit", type=int, default=None, help="Use only the first N personas.")
    parser.add_argument("--personas-cache-dir", type=Path, default=None, help="Where to write YAML from JSONL.")
    parser.add_argument("--eval-dir", type=Path, required=True, help="Base output dir (e.g. ../reddit/Eval/statebench_x).")
    parser.add_argument("--domain", default="travel", help="Domain (default: travel).")
    parser.add_argument("--num-tasks", type=int, default=20, help="First N tasks by number (default: 20).")
    parser.add_argument("--task", type=str, nargs="+", default=None, help="Explicit task IDs (overrides --num-tasks).")
    parser.add_argument("--sim-model", default=os.environ.get("NAUT_SIM_MODEL", "gemma"), help="User-sim model (default: gemma).")
    parser.add_argument(
        "--satisfaction-model",
        default=os.environ.get("NAUT_SATISFACTION_MODEL", "qwen3-small"),
        help="Turn-level satisfaction critic model (default: qwen3-small).",
    )
    parser.add_argument("--no-satisfaction", action="store_true", help="Skip the satisfaction critic.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip (model, persona) whose aggregate JSON exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs and output paths only.")
    args = parser.parse_args()

    if str(args.eval_dir).strip() in ("", "="):
        raise SystemExit("Invalid --eval-dir (did you leave spaces? use --eval-dir /path or --eval-dir=/path)")

    persona_files = _resolve_personas(args)
    tasks = _resolve_tasks(args.domain, args.task, args.num_tasks)
    eval_dir = args.eval_dir.resolve()
    commit = _git_commit()

    print(
        f"Domain={args.domain} | models={args.models} | sim={args.sim_model} | "
        f"personas={len(persona_files)} | tasks={len(tasks)} | "
        f"satisfaction={'OFF' if args.no_satisfaction else args.satisfaction_model} | judge=OFF"
    )

    # Clients are built lazily (dry-run needs no API key).
    agent_clients: dict[str, object] = {}
    sim_client = satisfaction_client = agent_class = None
    if not args.dry_run:
        client_class = load_root_client_class("NautilusClient")
        sim_client = client_class.from_env(model=args.sim_model)
        satisfaction_client = None if args.no_satisfaction else client_class.from_env(model=args.satisfaction_model)
        agent_class = load_root_agent_class("NautilusAgent")
        for model in args.models:
            agent_clients[model] = client_class.from_env(model=model)

    planned = done = 0
    for model in args.models:
        model_label = to_fs_label(model)
        model_dir = eval_dir / model_label
        for persona_path in persona_files:
            persona_key = persona_path.stem
            dest = model_dir / f"{persona_key}_{model_label}_{args.domain}.json"

            if args.dry_run:
                print(f"[plan] model={model} persona={persona_key} -> {dest}")
                planned += 1
                continue
            if args.skip_existing and dest.exists():
                print(f"[skip] model={model} persona={persona_key} (exists: {dest})")
                continue

            persona_yaml = load_persona_yaml(persona_path)
            persona_id = persona_id_from_yaml(persona_yaml, fallback=persona_key)
            print(f"[run] model={model} persona={persona_key} (id={persona_id}) ...")

            simulations: list[dict] = []
            task_meta: list[dict] = []
            for task in tasks:
                if not task.user_id:
                    print(f"    [skip] {task.task_id}: task has no user_id")
                    continue
                try:
                    traj = run_one_trajectory(
                        domain_name=args.domain,
                        task=task,
                        persona_yaml=persona_yaml,
                        persona_key=persona_key,
                        persona_id=persona_id,
                        persona_path=persona_path,
                        agent_client=agent_clients[model],
                        sim_client=sim_client,
                        agent_class=agent_class,
                        satisfaction_client=satisfaction_client,
                        sim_model=args.sim_model,
                        agent_model=model,
                    )
                except Exception as exc:  # noqa: BLE001 - fail loud per task, keep going
                    print(f"    [ERR] {task.task_id}: {type(exc).__name__}: {exc}")
                    continue
                sim = traj.to_dict()
                sim["task_id"] = task.task_id
                simulations.append(sim)
                task_meta.append({"task_id": task.task_id, "task_summary": getattr(task, "task_summary", None)})
                term = traj.metadata.get("terminal_state")
                met = traj.state_requirements_score.score if traj.state_requirements_score else None
                print(f"    [ok] {task.task_id}: terminal={term} | state_requirements_met={met}")

            aggregate = {
                "timestamp": _dt.datetime.now().isoformat(),
                "info": {
                    "git_commit": commit,
                    "num_tasks": len(simulations),
                    "domain": args.domain,
                    "scoring": "unofficial persona run (no judge)",
                    "user_info": {
                        "implementation": "user_simulator",
                        "llm": args.sim_model,
                        "persona_key": persona_key,
                        "persona_id": persona_id,
                        "persona_file": str(persona_path),
                    },
                    "agent_info": {"llm": model},
                    "satisfaction_model": None if args.no_satisfaction else args.satisfaction_model,
                },
                "tasks": task_meta,
                "simulations": simulations,
            }
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"  -> {dest} ({len(simulations)} tasks)")
            done += 1

    if args.dry_run:
        print(f"\nDry run: {planned} (model, persona) aggregates would be written under {eval_dir}/")
    else:
        print(f"\nDone: wrote {done} aggregate file(s) under {eval_dir}/")


if __name__ == "__main__":
    main()
