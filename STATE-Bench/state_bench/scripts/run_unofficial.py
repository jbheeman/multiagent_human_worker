"""Unofficial tau2-style persona run for STATE-Bench (no judge during collection).

Drives the user simulator with a YAML persona on a cheap Nautilus model, runs an arbitrary
agent under test on Nautilus, collects one transcript per (persona, task), and classifies
the terminal state (success / transfer / abandoned / incomplete) into trajectory metadata.
No judge is ever called — score transcripts later with state_bench.scripts.score.

This bypasses the official CLI (which hardcodes the locked GPT-5.4 user-sim/judge) by
calling orchestrator.run_task() directly with a separate simulator_client.

Env vars (see STATE-Bench/.env): NAUT_API_KEY (required), NAUT_API_BASE, NAUT_VERIFY_SSL.
Model selection: --agent-model / NAUT_AGENT_MODEL (agent under test) and
--sim-model / NAUT_SIM_MODEL (user simulator), each falling back to NAUT_MODEL.

Examples:
    uv run python -m state_bench.scripts.run_unofficial \
        --domain travel --task 1-cancel_economy_domestic \
        --persona-file ../reddit/.personas_yaml_cache_personas_axis_a_reddit_schwartz_unopt/bitparity.yaml \
        --sim-model gemma3 --agent-model gpt-oss

    uv run python -m state_bench.scripts.run_unofficial \
        --domain travel --persona-dir ../reddit/.personas_yaml_cache_personas_axis_a_reddit_schwartz_unopt
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from state_bench.agents.loader import load_root_agent_class, load_root_client_class
from state_bench.client import BaseLLMClient
from state_bench.domain import get_domain_config
from state_bench.env_loader import load_task_environment
from state_bench.orchestrator import run_task
from state_bench.paths import domain_tasks_dir
from state_bench.schemas import StateDiff, TaskDefinition
from state_bench.scoring import evaluate_state_requirements
from state_bench.scripts.persona_injection import (
    classify_terminal,
    load_persona_yaml,
    persona_id_from_yaml,
    wrap_build_simulator_prompt,
)
from state_bench.scripts.satisfaction_critic import score_transcript


def _resolve_persona_files(args: argparse.Namespace) -> list[Path]:
    if args.persona_file:
        return [Path(args.persona_file)]
    if args.persona_dir:
        files = sorted(Path(args.persona_dir).glob("*.yaml"))
        if not files:
            raise SystemExit(f"No *.yaml personas found in {args.persona_dir}")
        return files
    env_file = os.environ.get("STATE_BENCH_PERSONA_FILE")
    if env_file:
        return [Path(env_file)]
    raise SystemExit("Provide --persona-file, --persona-dir, or set STATE_BENCH_PERSONA_FILE")


def _resolve_tasks(domain_name: str, task_args: list[str] | None) -> list[TaskDefinition]:
    tasks_dir = domain_tasks_dir(domain_name)
    if task_args:
        task_ids: list[str] = []
        for value in task_args:
            task_ids.extend(part.strip() for part in value.split(",") if part.strip())
        tasks: list[TaskDefinition] = []
        missing: list[str] = []
        for task_id in task_ids:
            path = tasks_dir / f"{task_id}.json"
            if not path.exists():
                missing.append(task_id)
                continue
            tasks.append(TaskDefinition.load(path))
        if missing:
            raise SystemExit(f"Task(s) not found in {tasks_dir}: {', '.join(missing)}")
        return tasks
    return [TaskDefinition.load(path) for path in sorted(tasks_dir.glob("*.json"))]


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Unofficial persona-conditioned STATE-Bench run (no judge)")
    parser.add_argument("--domain", type=str, default="travel", help="Domain name (default: travel)")
    parser.add_argument("--task", type=str, nargs="+", default=None, help="Task IDs (default: all tasks in domain)")
    parser.add_argument("--persona-file", type=str, default=None, help="Single persona YAML file")
    parser.add_argument("--persona-dir", type=str, default=None, help="Directory of *.yaml personas")
    parser.add_argument("--output-dir", type=str, default=None, help="Output dir (default: outputs/<domain>_persona)")
    parser.add_argument(
        "--sim-model",
        type=str,
        default=os.environ.get("NAUT_SIM_MODEL") or os.environ.get("NAUT_MODEL"),
        help="User-simulator model (default: NAUT_SIM_MODEL or NAUT_MODEL)",
    )
    parser.add_argument(
        "--agent-model",
        type=str,
        default=os.environ.get("NAUT_AGENT_MODEL") or os.environ.get("NAUT_MODEL"),
        help="Agent-under-test model (default: NAUT_AGENT_MODEL or NAUT_MODEL)",
    )
    parser.add_argument(
        "--satisfaction-model",
        type=str,
        default=os.environ.get("NAUT_SATISFACTION_MODEL", "qwen3-small"),
        help="Critic model for turn-level satisfaction scoring (default: qwen3-small)",
    )
    parser.add_argument(
        "--no-satisfaction",
        action="store_true",
        help="Skip the turn-level satisfaction critic (transcript only).",
    )
    args = parser.parse_args()

    if not args.sim_model:
        parser.error("--sim-model is required (or set NAUT_SIM_MODEL / NAUT_MODEL)")
    if not args.agent_model:
        parser.error("--agent-model is required (or set NAUT_AGENT_MODEL / NAUT_MODEL)")

    persona_files = _resolve_persona_files(args)
    tasks = _resolve_tasks(args.domain, args.task)

    client_class = load_root_client_class("NautilusClient")
    agent_client = client_class.from_env(model=args.agent_model)
    sim_client = client_class.from_env(model=args.sim_model)
    if not isinstance(agent_client, BaseLLMClient):
        raise TypeError("NautilusClient.from_env() must return a BaseLLMClient")
    agent_class = load_root_agent_class("NautilusAgent")
    satisfaction_client = None if args.no_satisfaction else client_class.from_env(model=args.satisfaction_model)

    base_output = Path(args.output_dir) if args.output_dir else Path(f"outputs/{args.domain}_persona")

    print(f"Domain: {args.domain} | sim={args.sim_model} | agent={args.agent_model}")
    print(f"Personas: {len(persona_files)} | Tasks: {len(tasks)} | Judge: OFF")

    summary: list[dict] = []
    for persona_path in persona_files:
        persona_yaml = load_persona_yaml(persona_path)
        persona_id = persona_id_from_yaml(persona_yaml, fallback=persona_path.stem)
        print(f"\n=== Persona: {persona_id} ({persona_path}) ===")

        for task in tasks:
            user_id = task.user_id
            if not user_id:
                print(f"  [skip] {task.task_id}: task has no user_id")
                continue
            # Fresh domain per (persona, task) so we wrap a clean build_simulator_prompt.
            domain = get_domain_config(args.domain)
            domain.build_simulator_prompt = wrap_build_simulator_prompt(
                domain.build_simulator_prompt, persona_yaml
            )
            metadata = {
                "sim_model": args.sim_model,
                "agent_model": args.agent_model,
                "persona_id": persona_id,
                "persona_file": str(persona_path),
                "scoring": "none (unofficial persona run)",
            }
            try:
                env_data, _ = load_task_environment(domain, task)
                trajectory = run_task(
                    task,
                    env_data,
                    user_id,
                    agent_client,
                    domain=domain,
                    agent=None,
                    env=None,
                    trajectory_metadata=metadata,
                    simulator_client=sim_client,
                    agent_class=agent_class,
                )
            except Exception as exc:  # noqa: BLE001 - fail loud per task, keep the batch going
                print(f"  [ERR] {task.task_id}: {type(exc).__name__}: {exc}")
                summary.append({"task_id": task.task_id, "persona_id": persona_id, "status": "ERR"})
                continue

            trajectory.metadata.update(classify_terminal(trajectory.conversation))
            # Deterministic objective axis (no judge): state requirements vs saved state_diff.
            # Orthogonal to terminal_state, which only reflects persona satisfaction.
            state_score = evaluate_state_requirements(
                task, trajectory.state_diff or StateDiff(created={}, modified={}, deleted={})
            )
            trajectory.state_requirements_score = state_score
            # Turn-level satisfaction critic (transcript-only, at end of task).
            if satisfaction_client is not None:
                trajectory.metadata.update(score_transcript(satisfaction_client, trajectory.conversation))
            output_path = base_output / persona_id / f"{task.task_id}.json"
            trajectory.save(output_path)
            term = trajectory.metadata["terminal_state"]
            trig = trajectory.metadata.get("terminal_trigger")
            state_met = state_score.score if state_score else None
            sat = trajectory.metadata.get("satisfaction_cumulative")
            print(
                f"  [ok] {task.task_id}: terminal={term} | state_requirements_met={state_met}"
                + (f" | satisfaction(cum={sat}, worst={trajectory.metadata.get('satisfaction_worst_case')})" if sat is not None else "")
                + (f" | trigger={trig!r}" if trig else "")
            )
            summary.append(
                {
                    "task_id": task.task_id,
                    "persona_id": persona_id,
                    "status": "OK",
                    "terminal_state": term,
                    "state_requirements_met": state_met,
                }
            )

    ok = [s for s in summary if s["status"] == "OK"]
    print(f"\nDone: {len(ok)}/{len(summary)} runs OK. Output under {base_output}/")
    if ok:
        from collections import Counter

        counts = Counter(s["terminal_state"] for s in ok)
        print("Terminal states (persona satisfaction): " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        scored = [s for s in ok if s.get("state_requirements_met") is not None]
        if scored:
            passed = sum(1 for s in scored if s["state_requirements_met"] == 1)
            print(f"State requirements (deterministic, objective): {passed}/{len(scored)} passed")
    print(
        f"\nScore later with: uv run python -m state_bench.scripts.score "
        f"--domain {args.domain} --results-dir {base_output}"
    )


if __name__ == "__main__":
    main()
