#!/usr/bin/env python3
"""MCTS Phase-1 orchestrator: shared assignment × arms × models × benches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.mcts.api_errors import is_external_api_error  # noqa: E402
from eval.mcts.assignment import AssignmentRow, read_assignment_csv  # noqa: E402
from eval.mcts.backends.statebench import StateBenchBackend  # noqa: E402
from eval.mcts.backends.tau2 import Tau2Backend  # noqa: E402
from eval.mcts.backends import RolloutRequest  # noqa: E402
from eval.mcts.ids import assert_arm_covers_ids  # noqa: E402
from eval.mcts.persona_cache import load_arm_or_none, materialize_persona_yaml  # noqa: E402
from eval.mcts.schema import (  # noqa: E402
    build_core_record,
    hash_persona_text,
    make_config_hash,
    make_run_id,
    validate_core_record,
)


def _load_yaml_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"Config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit("Config must be a mapping")
    return data


def _resolve_path(base: Path, value: str | Path | None) -> Path | None:
    """Resolve a path. Absolute paths stay absolute; relatives are under base."""
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def _resolve_cli_or_cfg(
    *,
    cli_value: Path | str | None,
    cfg_value: str | Path | None,
    default: Path,
    cfg_dir: Path,
    cwd: Path,
) -> Path:
    """CLI paths resolve from cwd; config-relative from cfg_dir; else default."""
    if cli_value is not None:
        return _resolve_path(cwd, cli_value)  # type: ignore[arg-type]
    if cfg_value is not None:
        return _resolve_path(cfg_dir, cfg_value)  # type: ignore[return-value]
    return default.resolve()


def assert_sim_not_in_models(sim_model: str, models: list[str]) -> None:
    """Hard-fail if the user-sim model is also under evaluation."""
    sim_norm = sim_model.split("/")[-1].lower()
    for model in models:
        model_norm = model.split("/")[-1].lower()
        if model_norm == sim_norm or model.lower() == sim_model.lower():
            raise SystemExit(
                f"SIM_MODEL contamination: sim={sim_model!r} appears in MODELS={models!r}"
            )


def load_completed_run_ids(rollouts_path: Path) -> dict[str, str]:
    """Map run_id -> config_hash for resume."""
    done: dict[str, str] = {}
    if not rollouts_path.is_file():
        return done
    with rollouts_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec.get("run_id")
            ch = rec.get("config_hash")
            if rid and ch:
                done[rid] = ch
    return done


def append_rollout(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _leading_num(task_id: str) -> int:
    import re

    match = re.match(r"(\d+)", str(task_id))
    return int(match.group(1)) if match else 10**9


def filter_assignment_rows(
    rows: list[AssignmentRow],
    *,
    domain: str,
    task_ids: set[str] | None,
    persona_ids: set[str] | None,
    limit_tasks: int | None,
    limit_personas_per_task: int | None,
) -> list[AssignmentRow]:
    filtered = [r for r in rows if r.domain == domain]
    if task_ids is not None:
        filtered = [r for r in filtered if r.task_id in task_ids]
    if persona_ids is not None:
        filtered = [r for r in filtered if r.persona_id in persona_ids]

    # Stable numeric-ish task order (tau2 "0","1",... and statebench "1-foo","2-bar").
    filtered.sort(key=lambda r: (_leading_num(r.task_id), r.task_id, r.slot))

    if limit_tasks is not None:
        seen: list[str] = []
        for r in filtered:
            if r.task_id not in seen:
                seen.append(r.task_id)
            if len(seen) >= limit_tasks:
                break
        keep = set(seen[:limit_tasks])
        filtered = [r for r in filtered if r.task_id in keep]

    if limit_personas_per_task is not None:
        per_task: dict[str, int] = {}
        kept: list[AssignmentRow] = []
        for r in filtered:
            n = per_task.get(r.task_id, 0)
            if n < limit_personas_per_task:
                kept.append(r)
                per_task[r.task_id] = n + 1
        filtered = kept

    return filtered


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parents[1]

    parser = argparse.ArgumentParser(description="Run MCTS Phase-1 persona-sampled eval")
    parser.add_argument("--config", type=Path, default=None, help="YAML config (optional)")
    parser.add_argument("--bench", choices=("tau2", "statebench"), default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--sim-model", default=None, help="User sim (statebench / short name)")
    parser.add_argument("--user-llm", default=None, help="Tau2 user LLM (e.g. openai/gemma)")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help="Subset of arm names from config (default: all configured)",
    )
    parser.add_argument("--assignment-csv", type=Path, default=None)
    parser.add_argument("--schwartz-jsonl", type=Path, default=None)
    parser.add_argument("--eval-dir", type=Path, default=None)
    parser.add_argument("--rollouts-jsonl", type=Path, default=None)
    parser.add_argument("--block", type=int, default=None)
    parser.add_argument(
        "--arm-jsonl",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override/add arm JSONL (repeatable). Use NAME= to clear / fixed_prompt.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required for real runs (after cost print). Not needed for --dry-run.",
    )
    parser.add_argument("--limit-tasks", type=int, default=None, help="Dry-run / smoke: first N tasks")
    parser.add_argument(
        "--limit-personas-per-task",
        type=int,
        default=None,
        help="Dry-run / smoke: first N personas per task",
    )
    parser.add_argument("--task-ids", nargs="+", default=None)
    parser.add_argument("--persona-ids", nargs="+", default=None)
    parser.add_argument("--tau2-root", type=Path, default=repo / "tau2-bench")
    parser.add_argument("--statebench-root", type=Path, default=repo / "STATE-Bench")
    parser.add_argument("--max-retries", type=int, default=9)
    parser.add_argument(
        "--no-satisfaction",
        action="store_true",
        default=True,
        help="STATE-Bench: skip turn-level satisfaction critic (default). Tau2 ignores this.",
    )
    parser.add_argument(
        "--with-satisfaction",
        action="store_true",
        help="STATE-Bench: enable satisfaction critic (overrides --no-satisfaction).",
    )
    parser.add_argument(
        "--satisfaction-model",
        default=None,
        help="STATE-Bench critic model when --with-satisfaction (default: qwen3-small).",
    )
    parser.add_argument(
        "--cost-per-conversation-usd",
        type=float,
        default=None,
        help="For cost estimate print",
    )
    args = parser.parse_args()

    cfg = _load_yaml_config(args.config.resolve() if args.config else None)
    cfg_dir = args.config.resolve().parent if args.config else here

    bench = args.bench or cfg.get("bench") or "tau2"
    domain = args.domain or cfg.get("domain") or ("retail" if bench == "tau2" else "travel")
    models = args.models or cfg.get("MODELS") or cfg.get("models")
    if not models:
        raise SystemExit("--models is required (or set MODELS in config)")
    sim_model = (
        args.sim_model
        or args.user_llm
        or cfg.get("SIM_MODEL")
        or cfg.get("sim_model")
        or ("openai/gemma" if bench == "tau2" else "gemma")
    )
    # Normalize for contamination check vs agent models.
    assert_sim_not_in_models(sim_model, models)

    arms_cfg: dict[str, Any] = dict(cfg.get("ARMS") or cfg.get("arms") or {})
    # Defaults if no config: fixed_prompt only.
    if not arms_cfg:
        arms_cfg = {"fixed_prompt": None}
    for item in args.arm_jsonl:
        if "=" not in item:
            raise SystemExit(f"--arm-jsonl must be NAME=PATH, got {item!r}")
        name, _, path_s = item.partition("=")
        arms_cfg[name] = path_s if path_s else None
    if args.arms:
        missing = [a for a in args.arms if a not in arms_cfg]
        if missing:
            raise SystemExit(f"--arms not in config/overrides: {missing}")
        arms_cfg = {a: arms_cfg[a] for a in args.arms}

    cwd = Path.cwd()
    assignment_csv = _resolve_cli_or_cfg(
        cli_value=args.assignment_csv,
        cfg_value=cfg.get("assignment_csv"),
        default=here / "assignments" / "assignment_block1.csv",
        cfg_dir=cfg_dir,
        cwd=cwd,
    )
    eval_dir = _resolve_cli_or_cfg(
        cli_value=args.eval_dir,
        cfg_value=cfg.get("eval_dir"),
        default=repo / "reddit" / "Eval" / "mcts_phase1",
        cfg_dir=cfg_dir,
        cwd=cwd,
    )
    if args.rollouts_jsonl is not None:
        rollouts_jsonl = _resolve_path(cwd, args.rollouts_jsonl)
    elif cfg.get("rollouts_jsonl") is not None:
        rollouts_jsonl = _resolve_path(cfg_dir, cfg.get("rollouts_jsonl"))
    else:
        rollouts_jsonl = (eval_dir / "rollouts_core.jsonl").resolve()

    block = args.block if args.block is not None else int(cfg.get("B") or cfg.get("block") or 1)

    if assignment_csv is None or not assignment_csv.is_file():
        raise SystemExit(f"Assignment CSV not found: {assignment_csv}")

    meta, all_rows = read_assignment_csv(assignment_csv, block=block)
    rows = filter_assignment_rows(
        all_rows,
        domain=domain,
        task_ids=set(args.task_ids) if args.task_ids else None,
        persona_ids=set(args.persona_ids) if args.persona_ids else None,
        limit_tasks=args.limit_tasks,
        limit_personas_per_task=args.limit_personas_per_task,
    )
    if not rows:
        raise SystemExit(f"No assignment rows for domain={domain} after filters")

    required_ids = {r.persona_id for r in rows}
    arm_indexes: dict[str, dict[str, Any] | None] = {}
    for arm_name, raw_path in arms_cfg.items():
        if arm_name == "fixed_prompt" or raw_path in (None, "null", ""):
            arm_indexes[arm_name] = None
            continue
        # Arm paths from --arm-jsonl are cwd-relative; from config are cfg_dir-relative.
        arm_base = cwd if any(a.startswith(f"{arm_name}=") for a in args.arm_jsonl) else cfg_dir
        path = _resolve_path(arm_base, raw_path)
        idx = load_arm_or_none(arm_name, path)
        assert_arm_covers_ids(arm_name, idx or {}, required_ids)
        arm_indexes[arm_name] = idx

    config_for_hash = {
        "bench": bench,
        "domain": domain,
        "models": models,
        "sim_model": sim_model,
        "arms": sorted(arm_indexes),
        "assignment_csv": str(assignment_csv),
        "assignment_master_seed": meta.get("master_seed"),
        "block": block,
        "strata_mode": meta.get("strata_mode"),
        "S": meta.get("S"),
    }
    config_hash = make_config_hash(config_for_hash)

    n_rollouts = len(rows) * len(models) * len(arm_indexes)
    cost_each = (
        args.cost_per_conversation_usd
        if args.cost_per_conversation_usd is not None
        else float(cfg.get("cost_per_conversation_usd") or 0.05)
    )
    print(
        f"MCTS schedule: bench={bench} domain={domain} block={block} "
        f"rows={len(rows)} models={models} arms={list(arm_indexes)} "
        f"=> {n_rollouts} conversations"
    )
    print(f"Estimated cost ≈ ${n_rollouts * cost_each:.2f} ({cost_each}/conv)")
    print(f"config_hash={config_hash} strata_mode={meta.get('strata_mode')} S={meta.get('S')}")
    print(f"rollouts_jsonl={rollouts_jsonl}")

    if not args.dry_run and not args.confirm:
        raise SystemExit("Refusing real run without --confirm (or pass --dry-run)")

    # Backend
    if bench == "tau2":
        backend = Tau2Backend(
            repo_root=args.tau2_root.resolve(),
            max_retries=args.max_retries,
        )
        # Prefer full user-llm string for tau2.
        if args.user_llm:
            sim_for_backend = args.user_llm
        elif sim_model.startswith("openai/"):
            sim_for_backend = sim_model
        else:
            sim_for_backend = f"openai/{sim_model}"
    else:
        sat_model = args.satisfaction_model or cfg.get("satisfaction_model") or "qwen3-small"
        backend = StateBenchBackend(
            repo_root=args.statebench_root.resolve(),
            satisfaction_model=sat_model,
        )
        sim_for_backend = sim_model.split("/")[-1]

    no_satisfaction = not args.with_satisfaction
    if bench == "statebench":
        sat_label = "OFF" if no_satisfaction else f"ON ({sat_model})"
        print(f"satisfaction={sat_label}")

    completed = load_completed_run_ids(rollouts_jsonl) if args.skip_existing else {}
    cache_root = eval_dir / ".persona_yaml_cache"
    planned = skipped = ran = errors = api_errors = 0

    for model in models:
        for arm_name, arm_index in arm_indexes.items():
            for row in rows:
                run_id = make_run_id(
                    bench=bench,
                    block=block,
                    domain=domain,
                    task_id=row.task_id,
                    persona_id=row.persona_id,
                    arm=arm_name,
                    model=model,
                    sim_seed=row.rollout_seed,
                )
                if args.skip_existing and run_id in completed:
                    if completed[run_id] == config_hash:
                        skipped += 1
                        continue
                    # Different config_hash: re-run (do not skip).
                planned += 1

                persona_path = None
                persona_text = None
                if arm_index is not None:
                    cache_dir = cache_root / arm_name
                    persona_path, persona_text = materialize_persona_yaml(
                        arm_index=arm_index,
                        persona_id=row.persona_id,
                        cache_dir=cache_dir,
                    )

                req = RolloutRequest(
                    bench=bench,
                    domain=domain,
                    task_id=row.task_id,
                    persona_id=row.persona_id,
                    arm=arm_name,
                    model=model,
                    sim_model=sim_for_backend,
                    sim_seed=row.rollout_seed,
                    block=block,
                    persona_yaml_path=persona_path,
                    persona_yaml_text=persona_text,
                    eval_dir=eval_dir,
                    dry_run=args.dry_run,
                    no_satisfaction=no_satisfaction,
                )
                result = backend.run(req)

                if args.dry_run:
                    # Emit a schema-valid placeholder so dry-run acceptance can check JSONL.
                    record = build_core_record(
                        bench=bench,
                        config_hash=config_hash,
                        block=block,
                        domain=domain,
                        task_id=row.task_id,
                        persona_id=row.persona_id,
                        arm=arm_name,
                        model=model,
                        sim_model=sim_for_backend,
                        sim_seed=row.rollout_seed,
                        judge_model=None,
                        judge_prompt_hash=None,
                        terminal_state="failure_no_transfer",
                        task_success=False,
                        transfer=False,
                        n_turns=0,
                        full_transcript=[],
                        persona_variant_hash=hash_persona_text(persona_text),
                        extra={"dry_run": True, "artifact_path": str(result.artifact_path)},
                    )
                    append_rollout(rollouts_jsonl, record)
                    ran += 1
                    continue

                if is_external_api_error(result.error):
                    # Log + count only; do not write JSONL so --skip-existing can retry.
                    api_errors += 1
                    errors += 1
                    print(f"[api-err] {run_id}: {result.error}", file=sys.stderr)
                    continue

                record = build_core_record(
                    bench=bench,
                    config_hash=config_hash,
                    block=block,
                    domain=domain,
                    task_id=row.task_id,
                    persona_id=row.persona_id,
                    arm=arm_name,
                    model=model,
                    sim_model=sim_for_backend,
                    sim_seed=row.rollout_seed,
                    judge_model=None,
                    judge_prompt_hash=None,
                    terminal_state=result.terminal_state,
                    task_success=result.task_success,
                    transfer=result.transfer,
                    n_turns=result.n_turns,
                    full_transcript=result.full_transcript,
                    persona_variant_hash=hash_persona_text(persona_text),
                    extra={
                        "artifact_path": str(result.artifact_path) if result.artifact_path else None,
                        "backend_error": result.error,
                        "backend_raw": result.raw,
                    },
                )
                errs = validate_core_record(record)
                if errs:
                    print(f"SCHEMA ERROR {run_id}: {errs}", file=sys.stderr)
                    errors += 1
                append_rollout(rollouts_jsonl, record)
                ran += 1
                if result.error:
                    errors += 1
                    print(f"[err] {run_id}: {result.error}", file=sys.stderr)
                else:
                    print(
                        f"[ok] {run_id} terminal={result.terminal_state} "
                        f"turns={result.n_turns}"
                    )

    print(
        f"Done. planned_or_ran={ran} skipped={skipped} errors={errors} "
        f"api_errors={api_errors} jsonl={rollouts_jsonl}"
    )
    if errors and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
