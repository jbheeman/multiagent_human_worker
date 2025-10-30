#!/usr/bin/env python3
"""
Benchmark Runner Script for HumanLLM Orchestrator

This script runs the Orchestrator on tasks from metadata.jsonl and captures:
1. Console output for each task
2. Final answers
3. Execution results

Usage:
    python benchmark_runner.py [--task-id TASK_ID] [--output-dir OUTPUT_DIR]
"""

import os
import sys
import json
import argparse
import time
import signal
import threading
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
import traceback

# Add the current directory to Python path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Orchestrator import run_orchestrator_task
from common_tools.sideinformation import get_side_info


def load_tasks_from_metadata(file_path="gaia/metadata.jsonl", skip_attachments: bool = False):
    """Load tasks from GAIA metadata. Supports JSONL and Parquet.

    Args:
        file_path: Path to metadata (.jsonl or .parquet)
        skip_attachments: If True, skip tasks with a non-empty file_path
    """
    tasks = []

    lower = file_path.lower()
    if lower.endswith(".parquet"):
        # Lazy import to avoid hard dependency if user sticks to jsonl
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Reading Parquet requires pyarrow. Please install it: pip install pyarrow"
            ) from e

        table = pq.read_table(file_path)
        rows = table.to_pylist()
        for idx, row in enumerate(rows, 1):
            task_id = row.get("task_id") or f"row_{idx}"
            question = row.get("Question", "")
            expected_answer = row.get("Final answer", "")
            level = row.get("Level", "")
            file_path_col = row.get("file_path", "") or row.get("file_name", "")
            annotator = row.get("Annotator Metadata", {}) or {}

            if skip_attachments and file_path_col:
                continue

            tasks.append({
                'line_number': idx,
                'task_id': task_id,
                'question': question,
                'expected_answer': expected_answer,
                'level': level,
                'annotator_metadata': annotator,
                'raw_data': row,
            })
        return tasks

    # Fallback: JSONL
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue

            try:
                task_data = json.loads(line)
                file_path_col = task_data.get('file_path', '') or task_data.get('file_name', '')
                if skip_attachments and file_path_col:
                    continue
                tasks.append({
                    'line_number': line_num,
                    'task_id': task_data.get('task_id', f'line_{line_num}'),
                    'question': task_data.get('Question', ''),
                    'expected_answer': task_data.get('Final answer', ''),
                    'level': task_data.get('Level', ''),
                    'annotator_metadata': task_data.get('Annotator Metadata', {}),
                    'raw_data': task_data
                })
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue

    return tasks


def create_output_directory(base_dir="benchmark_results"):
    """Create timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Task execution timed out")

def capture_console_output_with_timeout(func, timeout_seconds=600, *args, **kwargs):
    """Capture stdout and stderr from a function execution with timeout."""
    stdout_capture = StringIO()
    stderr_capture = StringIO()
    result = None
    error = None
    
    def run_task():
        nonlocal result, error
        try:
            # The function now returns a tuple (result, log_dir_path)
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                func_result = func(*args, **kwargs)
            if isinstance(func_result, tuple) and len(func_result) == 2:
                result = func_result # Keep the tuple (result, log_dir)
            else:
                result = (func_result, None) # Wrap in tuple if log_dir is not returned
        except Exception as e:
            error = traceback.format_exc()
    
    # Start the task in a separate thread
    task_thread = threading.Thread(target=run_task)
    task_thread.daemon = True
    task_thread.start()
    
    # Wait for completion or timeout
    task_thread.join(timeout=timeout_seconds)
    # The line above is replaced with the one below to disable the timeout.
    #task_thread.join()
    
    if task_thread.is_alive():
        # Task is still running, it timed out
        error = f"Task execution timed out after {timeout_seconds} seconds"
        return (None, None), stdout_capture.getvalue(), stderr_capture.getvalue(), error
    
    # Task completed (either successfully or with error)
    if error:
        return (None, None), stdout_capture.getvalue(), stderr_capture.getvalue(), error
    else:
        return result, stdout_capture.getvalue(), stderr_capture.getvalue(), None


def save_task_output(output_dir, task, console_output, stderr_output, final_answer, execution_time, log_dir_path=None, error=None):
    """Save the output for a specific task."""
    task_id = task['task_id']
    task_dir = output_dir / f"task_{task_id}"
    task_dir.mkdir(exist_ok=True)
    
    # Save console output
    with open(task_dir / "console_output.txt", "w", encoding="utf-8") as f:
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Question: {task['question']}\n")
        f.write(f"Expected Answer: {task['expected_answer']}\n")
        f.write(f"Level: {task['level']}\n")
        f.write(f"Execution Time: {execution_time:.2f} seconds\n")
        if log_dir_path:
            f.write(f"Detailed Log Directory: {log_dir_path}\n")
        f.write("=" * 80 + "\n\n")
        f.write("STDOUT:\n")
        f.write(console_output)
        f.write("\n\nSTDERR:\n")
        f.write(stderr_output)
        
        if error:
            f.write("\n\nERROR:\n")
            f.write(error)
    
    # Save final answer
    with open(task_dir / "final_answer.txt", "w", encoding="utf-8") as f:
        f.write(f"Task ID: {task_id}\n")
        f.write(f"Expected Answer: {task['expected_answer']}\n")
        f.write(f"System Answer: {final_answer}\n")
        f.write(f"Match: {str(final_answer).strip().lower() == str(task['expected_answer']).strip().lower()}\n")
    
    # Save task metadata
    with open(task_dir / "task_metadata.json", "w", encoding="utf-8") as f:
        json.dump(task['raw_data'], f, indent=2, ensure_ascii=False)
    
    # Save annotator steps for reference
    if task['annotator_metadata'].get('Steps'):
        with open(task_dir / "annotator_steps.txt", "w", encoding="utf-8") as f:
            f.write("Annotator Steps:\n")
            f.write(task['annotator_metadata']['Steps'])


def run_benchmark(tasks, output_dir, max_tasks=None, timeout_seconds=300):
    """Run the benchmark on all tasks with timeout and error handling."""
    results = []
    total_tasks = len(tasks) if max_tasks is None else min(max_tasks, len(tasks))
    
    print(f"Starting benchmark with {total_tasks} tasks...")
    print(f"Output directory: {output_dir}")
    print(f"Timeout per task: {timeout_seconds} seconds (5 minutes)")
    
    for i, task in enumerate(tasks[:total_tasks], 1):
        print(f"\n{'='*60}")
        print(f"Running Task {i}/{total_tasks}: {task['task_id']}")
        print(f"Question: {task['question'][:100]}...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Prepare side information from annotator metadata
            side_info = None
            if task['annotator_metadata']:
                try:
                    # Convert annotator metadata to side info format
                    annotator_str = json.dumps(task['annotator_metadata'])
                    side_info = get_side_info(annotator_str)
                except Exception as e:
                    print(f"Warning: Could not process side info for task {task['task_id']}: {e}")
                    side_info = None
            
            # Run the orchestrator task with timeout
            (result, log_dir_path), console_output, stderr_output, error = capture_console_output_with_timeout(
                run_orchestrator_task,
                timeout_seconds=timeout_seconds,
                user_goal=task['question'],
                side_info=side_info
            )
            
            execution_time = time.time() - start_time
            
            # Extract final answer from result
            final_answer = "No answer generated"
            if result:
                # Try to extract answer from the result
                if isinstance(result, str):
                    final_answer = result
                elif hasattr(result, 'model_dump_json'):
                    final_answer = result.model_dump_json()
                else:
                    final_answer = str(result)
            
            # Save outputs
            save_task_output(
                output_dir, task, console_output, stderr_output, 
                final_answer, execution_time, log_dir_path, error
            )
            
            # Store result summary
            result_summary = {
                'task_id': task['task_id'],
                'question': task['question'],
                'expected_answer': task['expected_answer'],
                'system_answer': final_answer,
                'execution_time': execution_time,
                'success': error is None,
                'answer_match': str(final_answer).strip().lower() == str(task['expected_answer']).strip().lower(),
                'error': error,
                'timed_out': 'timed out' in str(error).lower() if error else False
            }
            results.append(result_summary)
            
            print(f"Task completed in {execution_time:.2f} seconds")
            print(f"Success: {result_summary['success']}")
            print(f"Answer Match: {result_summary['answer_match']}")
            
            if error:
                if result_summary['timed_out']:
                    print(f"⏰ Task timed out after {timeout_seconds} seconds")
                else:
                    print(f"❌ Error occurred: {error[:200]}...")
            else:
                print("✅ Task completed successfully")
                
        except Exception as e:
            # Catch any unexpected errors and continue with next task
            execution_time = time.time() - start_time
            error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
            
            print(f"💥 Unexpected error in task {task['task_id']}: {str(e)}")
            print("Continuing with next task...")
            
            # Save error output
            save_task_output(
                output_dir, task, "", "", "Error occurred", execution_time, None, error_msg
            )
            
            # Store error result
            result_summary = {
                'task_id': task['task_id'],
                'question': task['question'],
                'expected_answer': task['expected_answer'],
                'system_answer': "Error occurred",
                'execution_time': execution_time,
                'success': False,
                'answer_match': False,
                'error': error_msg,
                'timed_out': False
            }
            results.append(result_summary)
    
    return results


def save_benchmark_summary(output_dir, results):
    """Save a summary of all benchmark results."""
    summary_file = output_dir / "benchmark_summary.json"
    
    timed_out_tasks = sum(1 for r in results if r.get('timed_out', False))
    
    summary = {
        'total_tasks': len(results),
        'successful_tasks': sum(1 for r in results if r['success']),
        'failed_tasks': sum(1 for r in results if not r['success']),
        'timed_out_tasks': timed_out_tasks,
        'correct_answers': sum(1 for r in results if r['answer_match']),
        'total_execution_time': sum(r['execution_time'] for r in results),
        'average_execution_time': sum(r['execution_time'] for r in results) / len(results) if results else 0,
        'results': results
    }
    
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Also create a human-readable summary
    summary_txt = output_dir / "benchmark_summary.txt"
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("BENCHMARK SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Tasks: {summary['total_tasks']}\n")
        f.write(f"Successful Tasks: {summary['successful_tasks']}\n")
        f.write(f"Failed Tasks: {summary['failed_tasks']}\n")
        f.write(f"Timed Out Tasks: {summary['timed_out_tasks']}\n")
        f.write(f"Correct Answers: {summary['correct_answers']}\n")
        f.write(f"Accuracy: {summary['correct_answers']/summary['total_tasks']*100:.1f}%\n")
        f.write(f"Success Rate: {summary['successful_tasks']/summary['total_tasks']*100:.1f}%\n")
        f.write(f"Total Execution Time: {summary['total_execution_time']:.2f} seconds\n")
        f.write(f"Average Execution Time: {summary['average_execution_time']:.2f} seconds\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 50 + "\n")
        for result in results:
            f.write(f"Task {result['task_id']}: ")
            f.write(f"Success={result['success']}, ")
            f.write(f"Match={result['answer_match']}, ")
            f.write(f"Time={result['execution_time']:.2f}s")
            if result.get('timed_out', False):
                f.write(", TIMED OUT")
            f.write("\n")
            if result['error']:
                f.write(f"  Error: {result['error'][:100]}...\n")
    
    print(f"\nBenchmark summary saved to: {summary_file}")
    print(f"Human-readable summary saved to: {summary_txt}")


def main():
    parser = argparse.ArgumentParser(description="Run HumanLLM Orchestrator benchmark")
    parser.add_argument("--task-id", help="Run only a specific task ID")
    parser.add_argument("--output-dir", default="benchmark_results", help="Base output directory")
    parser.add_argument("--max-tasks", type=int, help="Maximum number of tasks to run")
    parser.add_argument("--metadata-file", default="gaia/metadata.jsonl", help="Path to metadata file")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout per task in seconds (default: 300 = 5 minutes)")
    
    args = parser.parse_args()
    
    # Load tasks
    print("Loading tasks from metadata...")
    tasks = load_tasks_from_metadata(args.metadata_file)
    
    if not tasks:
        print("No valid tasks found in metadata file!")
        return
    
    # Filter by task ID if specified
    if args.task_id:
        tasks = [t for t in tasks if t['task_id'] == args.task_id]
        if not tasks:
            print(f"Task ID {args.task_id} not found!")
            return
    
    # Create output directory
    output_dir = create_output_directory(args.output_dir)
    
    # Run benchmark
    results = run_benchmark(tasks, output_dir, args.max_tasks, args.timeout)
    
    # Save summary
    save_benchmark_summary(output_dir, results)
    
    print(f"\nBenchmark completed! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
