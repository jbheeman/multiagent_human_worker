#!/usr/bin/env python3
"""Validate MCTS assignment CSV invariants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.mcts.assignment import read_assignment_csv, validate_assignment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MCTS assignment CSV")
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parent / "assignments" / "assignment_block1.csv",
    )
    parser.add_argument("--k", type=int, default=None, help="Override K (default: from header)")
    args = parser.parse_args()

    if not args.csv.is_file():
        print(f"Missing assignment file: {args.csv}", file=sys.stderr)
        sys.exit(1)

    meta, rows = read_assignment_csv(args.csv)
    k = args.k if args.k is not None else int(meta.get("K", "10"))
    s = int(meta["S"]) if "S" in meta else None
    mode = meta.get("strata_mode")
    errors = validate_assignment(rows, k=k, expected_s=s, strata_mode=mode)
    if errors:
        print(f"FAIL: {len(errors)} invariant violation(s) in {args.csv}")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    domains = sorted({r.domain for r in rows})
    print(
        f"OK: {args.csv} rows={len(rows)} domains={domains} "
        f"K={k} S={s} strata_mode={mode} master_seed={meta.get('master_seed')}"
    )


if __name__ == "__main__":
    main()
