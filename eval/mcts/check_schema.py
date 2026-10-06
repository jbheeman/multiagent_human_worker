#!/usr/bin/env python3
"""Validate core rollouts JSONL against the Phase-1 schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.mcts.schema import validate_core_record  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate MCTS core rollouts JSONL")
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.jsonl.is_file():
        print(f"Missing: {args.jsonl}", file=sys.stderr)
        sys.exit(1)

    n = bad = 0
    with args.jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            n += 1
            rec = json.loads(line)
            errs = validate_core_record(rec)
            if errs:
                bad += 1
                print(f"line {n} run_id={rec.get('run_id')}: {errs}")
            if args.limit is not None and n >= args.limit:
                break

    if bad:
        print(f"FAIL: {bad}/{n} records invalid")
        sys.exit(1)
    print(f"OK: {n} records valid in {args.jsonl}")


if __name__ == "__main__":
    main()
