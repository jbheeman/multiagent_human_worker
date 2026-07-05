"""Fold per-task STATE-Bench transcripts into reddit/tau2-style aggregates.

run_unofficial writes one JSON per (persona, task) under
<results-dir>/<user_id>/<task_id>.json. This script packs each persona's task files
into a single {timestamp, info, tasks, simulations} envelope at
<eval-dir>/<model>/<user_id>_<model>_<domain>.json -- the exact layout run_all_personas
produces and the reddit pipeline uses.

Idempotent: re-run it any time (e.g. after resuming run_unofficial with --skip-existing)
to refresh the aggregates from whatever per-task files currently exist.

Example:
    uv run python -m state_bench.scripts.merge_to_eval \
        --results-dir outputs/travel_persona_k100_bystem \
        --eval-dir ../reddit/Eval/statebench_schwartz_unopt \
        --domain travel
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path


def _leading_num(task_id: str) -> int:
    match = re.match(r"(\d+)", task_id)
    return int(match.group(1)) if match else 10**9


def _fs_label(value: str) -> str:
    return value.replace("/", "__")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-task transcripts into reddit-style aggregates")
    parser.add_argument("--results-dir", type=Path, required=True, help="Per-task dir (<user>/<task>.json).")
    parser.add_argument("--eval-dir", type=Path, required=True, help="Aggregate output base dir.")
    parser.add_argument("--domain", default="travel", help="Domain label for the filename (default: travel).")
    parser.add_argument("--model", default=None, help="Override agent-model label (default: read from files).")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    eval_dir = args.eval_dir.resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"Results dir not found: {results_dir}")

    persona_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()])
    if not persona_dirs:
        raise SystemExit(f"No persona subdirs under {results_dir}")

    written = 0
    total_sims = 0
    for pdir in persona_dirs:
        task_files = sorted(pdir.glob("*.json"), key=lambda p: _leading_num(p.stem))
        if not task_files:
            continue
        sims = [json.loads(f.read_text()) for f in task_files]

        first = sims[0]
        agent_model = args.model or first.get("agent_model") or "unknown"
        sim_model = first.get("sim_model")
        persona_key = first.get("persona_key") or pdir.name
        persona_id = first.get("persona_id")
        persona_file = first.get("persona_file")
        model_label = _fs_label(agent_model)

        aggregate = {
            "timestamp": _dt.datetime.now().isoformat(),
            "info": {
                "num_tasks": len(sims),
                "domain": args.domain,
                "scoring": "unofficial persona run (no judge)",
                "merged_from": str(pdir),
                "user_info": {
                    "implementation": "user_simulator",
                    "llm": sim_model,
                    "persona_key": persona_key,
                    "persona_id": persona_id,
                    "persona_file": persona_file,
                },
                "agent_info": {"llm": agent_model},
                "satisfaction_model": "present" if first.get("satisfaction_cumulative") is not None else None,
            },
            "tasks": [{"task_id": s.get("task_id"), "task_summary": s.get("task_summary")} for s in sims],
            "simulations": sims,
        }

        dest = eval_dir / model_label / f"{persona_key}_{model_label}_{args.domain}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"  -> {dest} ({len(sims)} tasks)")
        written += 1
        total_sims += len(sims)

    print(f"\nMerged {written} persona aggregate(s), {total_sims} task run(s) total, under {eval_dir}/")


if __name__ == "__main__":
    main()
