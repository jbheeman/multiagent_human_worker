"""Add turn-level satisfaction scores to existing persona transcripts (in place).

Runs the tau2-style satisfaction critic (state_bench.scripts.satisfaction_critic) over saved
transcripts and writes the per-turn scores + aggregates (cumulative / worst-case / mean /
final) into each JSON. Needs only the transcript — no STATE-Bench judge, no task files.

Use this to (re)score transcripts you already collected, or after swapping in the first
author's calibrated critic prompt. Default critic model: qwen3-small.

Example:
    uv run python -m state_bench.scripts.score_satisfaction \
        --results-dir outputs/travel_persona --satisfaction-model qwen3-small \
        --task 1-cancel_economy_domestic --num-workers 4
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

from state_bench.agents.loader import load_root_client_class
from state_bench.scripts.satisfaction_critic import score_transcript


def _score_file(path: Path, client) -> dict:
    try:
        data = json.loads(path.read_text())
        conversation = data.get("conversation") or []
        summary = score_transcript(client, conversation)
        data.update(summary)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return {"path": str(path), "status": "OK", "cumulative": summary["satisfaction_cumulative"]}
    except Exception as exc:  # noqa: BLE001 - report per-file, keep the batch going
        return {"path": str(path), "status": "ERR", "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Add turn-level satisfaction scores to transcripts (in place)")
    parser.add_argument("--results-dir", type=str, required=True, help="Dir of transcripts (globs **/*.json)")
    parser.add_argument(
        "--satisfaction-model",
        type=str,
        default=os.environ.get("NAUT_SATISFACTION_MODEL", "qwen3-small"),
        help="Critic model (default: qwen3-small)",
    )
    parser.add_argument("--task", type=str, nargs="+", default=None, help="Only score these task IDs.")
    parser.add_argument("--num-workers", type=int, default=4, help="Parallel critic workers (default: 4)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    traj_files = sorted(results_dir.rglob("*.json"))
    if args.task:
        wanted = {part.strip() for value in args.task for part in value.split(",") if part.strip()}
        traj_files = [tf for tf in traj_files if tf.stem in wanted]
    if not traj_files:
        raise SystemExit(f"No trajectory JSONs found under {results_dir}")

    client = load_root_client_class("NautilusClient").from_env(model=args.satisfaction_model)
    print(f"Satisfaction critic: {args.satisfaction_model} | files: {len(traj_files)} | workers: {args.num_workers}")

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(_score_file, tf, client): tf for tf in traj_files}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            tag = f"cum={r['cumulative']}" if r["status"] == "OK" else r.get("error", "")
            print(f"  [{len(results)}/{len(traj_files)}] {Path(r['path']).stem}: {r['status']} {tag}")

    ok = [r for r in results if r["status"] == "OK"]
    print(f"\nScored {len(ok)}/{len(results)} OK. Transcripts updated in place under {results_dir}/")


if __name__ == "__main__":
    main()
