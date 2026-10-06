"""Score persona-run transcripts with an UNOFFICIAL Nautilus judge (no Azure GPT-5.4).

Reuses the core scoring path (state_bench.scripts.score.score_one): deterministic state
requirements + LLM task-requirements judge + LLM UX judge, writing scores back into each
trajectory JSON in place. The only substitution is the judge client: instead of the locked
GPT-5.4 (build_locked_judge_client), we use a NautilusClient on a model of your choice.

This is NOT protocol-official. Use it for internal / relative analysis (e.g. E6 ranking
shift). To avoid self-evaluation bias, use a judge model DIFFERENT from the agent under
test — the script warns if they match (read from each trajectory's agent_model metadata).

Operates on the run_unofficial layout: <results-dir>/<persona_id>/<task_id>.json (globs
**/*.json), not the official run1/ layout.

Env: NAUT_API_KEY (required); NAUT_JUDGE_MODEL (default: kimi) or --judge-model.

Example:
    uv run python -m state_bench.scripts.score_unofficial \
        --domain travel --results-dir outputs/travel_persona --judge-model kimi
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from state_bench.agents.loader import load_root_client_class
from state_bench.domain import get_domain_config
from state_bench.paths import domain_tasks_dir
from state_bench.scoring import TaskRequirementsJudge, UXQualityJudge
from state_bench.scripts.score import score_one


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Score persona transcripts with an unofficial Nautilus judge")
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--results-dir", type=str, required=True, help="run_unofficial output dir (globs **/*.json)")
    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.environ.get("NAUT_JUDGE_MODEL", "kimi"),
        help="Nautilus model for the judge (default: NAUT_JUDGE_MODEL or kimi)",
    )
    parser.add_argument("--no-ux-score", action="store_true", help="Score task/state only; skip the UX judge.")
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=None,
        help="Only score these task IDs (space/comma separated). Default: all found.",
    )
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel judge workers (default: 4)")
    args = parser.parse_args()

    domain = get_domain_config(args.domain)
    tasks_dir = domain_tasks_dir(args.domain)
    results_dir = Path(args.results_dir)
    traj_files = sorted(results_dir.rglob("*.json"))
    if args.task:
        wanted = {part.strip() for value in args.task for part in value.split(",") if part.strip()}
        traj_files = [tf for tf in traj_files if tf.stem in wanted]
    if not traj_files:
        raise SystemExit(f"No trajectory JSONs found under {results_dir}")

    judge_client = load_root_client_class("NautilusClient").from_env(model=args.judge_model)
    task_requirements_judge = TaskRequirementsJudge(
        client=judge_client,
        prompts_dir=domain.prompts_dir,
        system_prompt=domain.judge_system_prompt,
    )
    ux_judge = None
    if not args.no_ux_score:
        ux_judge = UXQualityJudge(
            client=judge_client,
            prompts_dir=domain.prompts_dir,
            system_prompt=domain.judge_system_prompt,
        )

    print(f"Unofficial judge: {args.judge_model} | domain: {args.domain} | files: {len(traj_files)} | workers: {args.num_workers}")
    print("NOTE: results are NOT protocol-official (judge is not the locked GPT-5.4).")

    # Bias guard: warn once if the judge model equals the agent model in these transcripts.
    try:
        agent_model = json.loads(traj_files[0].read_text()).get("agent_model")
    except Exception:
        agent_model = None
    if isinstance(agent_model, str) and agent_model == args.judge_model:
        print(
            f"  WARNING: judge model ({args.judge_model}) == agent model ({agent_model}); "
            "self-evaluation bias. Use a different --judge-model."
        )

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(score_one, tf, tasks_dir, task_requirements_judge, ux_judge, tf, None, domain.name): tf
            for tf in traj_files
        }
        for future in as_completed(futures):
            tf = futures[future]
            r = future.result()
            results.append(r)
            extra = f" ux={r['ux_score']}" if "ux_score" in r else ""
            print(f"  [{len(results)}/{len(traj_files)}] {tf.parent.name}/{tf.stem}: {r['status']}{extra}")

    ok = [r for r in results if r["status"] == "OK"]
    errs = [r for r in results if r["status"] != "OK"]
    print(f"\nScored {len(ok)}/{len(results)} OK ({len(errs)} errors). Trajectories updated in place.")
    if ok:
        # Re-read to report the two axes side by side.
        rows = [json.loads(tf.read_text()) for tf in traj_files]
        state_pass = sum(1 for d in rows if d.get("state_requirements_met") == 1)
        task_pass = sum(1 for d in rows if d.get("task_requirements_met") == 1)
        completion_pass = sum(1 for d in rows if d.get("task_completion_pass") == 1)
        print(
            f"state_requirements_met: {state_pass}/{len(rows)} | "
            f"task_requirements_met: {task_pass}/{len(rows)} | "
            f"task_completion_pass: {completion_pass}/{len(rows)}"
        )


if __name__ == "__main__":
    main()
