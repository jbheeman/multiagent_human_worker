import argparse
import random
from pathlib import Path

import os
import sys
# Ensure project root is on sys.path so absolute imports work when running as a script
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

# Reuse the benchmark utilities to run orchestrator with timeouts and capture logs

from benchmark_runner import (
    load_tasks_from_metadata,
    create_output_directory,
    run_benchmark,
    save_benchmark_summary,
)


def sample_tasks_by_level(tasks, num_level1=10, num_level2=10, num_level3=5, seed=42):
    random.seed(seed)
    by_level = {1: [], 2: [], 3: []}
    for t in tasks:
        level = t.get("level")
        if isinstance(level, str):
            try:
                level = int(level)
            except Exception:
                level = None
        if level in by_level:
            by_level[level].append(t)

    sampled = []
    if by_level[1]:
        sampled += random.sample(by_level[1], min(num_level1, len(by_level[1])))
    if by_level[2]:
        sampled += random.sample(by_level[2], min(num_level2, len(by_level[2])))
    if by_level[3]:
        sampled += random.sample(by_level[3], min(num_level3, len(by_level[3])))

    random.shuffle(sampled)
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Run HumanLLM Orchestrator on sampled GAIA tasks")
    parser.add_argument("--metadata-file", default="gaia/metadata.jsonl", help="Path to GAIA metadata file (.jsonl or .parquet)")
    parser.add_argument("--output-dir", default="benchmark_results", help="Base output directory")
    parser.add_argument("--n1", type=int, default=10, help="Number of Level 1 tasks")
    parser.add_argument("--n2", type=int, default=10, help="Number of Level 2 tasks")
    parser.add_argument("--n3", type=int, default=5, help="Number of Level 3 tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--timeout", type=int, default=420, help="Timeout per task in seconds (default: 420 = 7 minutes)")
    parser.add_argument("--skip-attachments", action="store_true", help="Skip tasks that have associated files (non-empty file_path)")
    args = parser.parse_args()

    # Load all tasks from metadata
    tasks = load_tasks_from_metadata(args.metadata_file, skip_attachments=args.skip_attachments)
    if not tasks:
        print("No tasks loaded from metadata file.")
        return

    # Sample tasks per level
    sampled_tasks = sample_tasks_by_level(tasks, args.n1, args.n2, args.n3, args.seed)
    if not sampled_tasks:
        print("No sampled tasks available for the requested levels.")
        return

    print(f"Selected {len(sampled_tasks)} tasks (L1={args.n1}, L2={args.n2}, L3={args.n3} requested)")

    # Create timestamped output directory
    output_dir = create_output_directory(args.output_dir)

    # Run benchmark on the sampled tasks
    results = run_benchmark(sampled_tasks, output_dir, max_tasks=len(sampled_tasks), timeout_seconds=args.timeout)

    # Save summary
    save_benchmark_summary(output_dir, results)
    print(f"Finished. Results saved under: {output_dir}")


if __name__ == "__main__":
    main()
