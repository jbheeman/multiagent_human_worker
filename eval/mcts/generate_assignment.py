#!/usr/bin/env python3
"""Generate frozen assignment_block{b}.csv for MCTS eval."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Allow running as `python eval/mcts/generate_assignment.py` from repo root.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.mcts.assignment import (  # noqa: E402
    generate_block_assignment,
    validate_assignment,
    write_assignment_csv,
)


def _leading_num(task_id: str) -> int:
    match = re.match(r"(\d+)", task_id)
    return int(match.group(1)) if match else 10**9


def load_tau2_task_ids(domain: str, n: int, tau2_root: Path) -> list[str]:
    tasks_path = tau2_root / "data" / "tau2" / "domains" / domain / "tasks.json"
    if not tasks_path.is_file():
        raise SystemExit(f"tau2 tasks not found: {tasks_path}")
    data = json.loads(tasks_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Unexpected tasks.json shape in {tasks_path}")
    ids = [str(t["id"]) for t in data]
    if len(ids) < n:
        raise SystemExit(f"Domain {domain}: need {n} tasks, found {len(ids)}")
    return ids[:n]


def load_statebench_task_ids(domain: str, n: int, state_root: Path) -> list[str]:
    tasks_dir = state_root / "state_bench" / "domains" / domain / "tasks"
    if not tasks_dir.is_dir():
        raise SystemExit(f"STATE-Bench tasks dir not found: {tasks_dir}")
    paths = sorted(tasks_dir.glob("*.json"), key=lambda p: _leading_num(p.stem))
    if len(paths) < n:
        raise SystemExit(f"Domain {domain}: need {n} tasks, found {len(paths)}")
    return [p.stem for p in paths[:n]]


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Generate MCTS assignment CSV")
    parser.add_argument("--master-seed", type=int, default=42)
    parser.add_argument("--block", type=int, default=1)
    parser.add_argument("--k", type=int, default=10, help="Rollouts per task (default 10)")
    parser.add_argument("--t", type=int, default=20, help="Tasks per domain (default 20)")
    parser.add_argument(
        "--schwartz-jsonl",
        type=Path,
        default=repo / "reddit" / "AblationPersonas" / "seeded_200_personas.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: eval/mcts/assignments/assignment_block{b}.csv)",
    )
    parser.add_argument(
        "--bench",
        choices=("tau2", "statebench", "both"),
        default="both",
        help="Which bench domains to include (default: both)",
    )
    parser.add_argument(
        "--tau2-domains",
        nargs="+",
        default=["retail", "telecom", "airline"],
    )
    parser.add_argument(
        "--statebench-domains",
        nargs="+",
        default=["travel"],
    )
    parser.add_argument(
        "--tau2-root",
        type=Path,
        default=repo / "tau2-bench",
    )
    parser.add_argument(
        "--statebench-root",
        type=Path,
        default=repo / "STATE-Bench",
    )
    args = parser.parse_args()

    out = args.out or (
        Path(__file__).resolve().parent / "assignments" / f"assignment_block{args.block}.csv"
    )
    if out.exists():
        out.unlink()

    specs: list[tuple[str, list[str]]] = []
    if args.bench in ("tau2", "both"):
        for domain in args.tau2_domains:
            specs.append((domain, load_tau2_task_ids(domain, args.t, args.tau2_root)))
    if args.bench in ("statebench", "both"):
        for domain in args.statebench_domains:
            specs.append((domain, load_statebench_task_ids(domain, args.t, args.statebench_root)))

    all_rows = []
    last_meta = None
    for i, (domain, task_ids) in enumerate(specs):
        meta, rows = generate_block_assignment(
            schwartz_jsonl=args.schwartz_jsonl.resolve(),
            domain=domain,
            task_ids=task_ids,
            master_seed=args.master_seed,
            block=args.block,
            k=args.k,
        )
        errors = validate_assignment(rows, k=args.k, expected_s=meta.s, strata_mode=meta.strata_mode)
        if errors:
            print("VALIDATION FAILED:", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            sys.exit(1)
        write_assignment_csv(out, meta, rows, append_domain=(i > 0))
        all_rows.extend(rows)
        last_meta = meta
        print(
            f"Wrote domain={domain} rows={len(rows)} S={meta.s} strata_mode={meta.strata_mode}"
        )

    print(f"Assignment written: {out} ({len(all_rows)} rows, block={args.block})")
    if last_meta:
        print(f"strata_mode={last_meta.strata_mode} S={last_meta.s} master_seed={args.master_seed}")


if __name__ == "__main__":
    main()
