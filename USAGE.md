# SimRAG Driver Usage Guide

This guide explains how to use the `simrag_driver.py` script, which is the main entry point for running the multi-agent orchestration system.

## Overview

`simrag_driver.py` allows you to run the system in several modes:
- **Parallel Inference:** Run multiple research goals concurrently to generate product recommendations.
- **Training Mode:** Run a single goal with a known ground truth to evaluate the agent's performance (SimRAG).
- **Goal Generation:** Generate a set of new research goals based on a user's purchase history and inferred persona.

## Installation

Before running the script, ensure you have the required dependencies installed. You can install them using `pip`:

```bash
pip install -r requirements.txt
```

## Configuration

The system requires an API key for the language model. This key should be placed in a `.env` file in the root of the project directory.

Create a file named `.env` and add the following line:

```
NAUT_API_KEY="your_api_key_here"
```

Replace `"your_api_key_here"` with your actual API key.

## Usage

The script is run from the command line using the following structure:

```bash
python simrag_driver.py [options]
```

### Arguments

The behavior of the script is controlled by the following command-line arguments:

-   `--mode`: Sets the operational mode of the driver.
    -   `parallel_inference`: (Default) Runs multiple research goals concurrently.
    -   `train_orchestrator`: Runs the agent in training mode against a single goal with a known ground truth.
    -   `generate_goals`: Generates a new set of research goals based on a mock user's purchase history.

-   `--output`: Specifies the output file path where results (e.g., training data, inference results) will be saved. Defaults to `simrag_training_data.jsonl`.

-   `--n_goals`: Defines the number of research goals to generate or run in parallel inference mode. Defaults to 5.

-   `--max_loops`: Sets the maximum number of refinement loops the orchestrator will perform for each goal before making a final decision. Defaults to 3.

-   `--top_k`: Specifies the number of top entries to choose as the final answer. Defaults to 3.

### Examples

**Run parallel inference with 5 goals, 2 refinement loops, and get the top 3 recommendations:**
```bash
python simrag_driver.py --mode parallel_inference --n_goals 5 --max_loops 2 --top_k 3
```

**Generate 10 new research goals:**
```bash
python simrag_driver.py --mode generate_goals --n_goals 10
```

**Run in training mode:**
```bash
python simrag_driver.py --mode train_orchestrator
```

## Output

### Log Structure

The script generates detailed logs for each run in the `logs/` directory. Each run is saved in a directory named `run_<timestamp>`.

```
logs/
└── run_2025-11-19_10-00-00/
    ├── stdout.log
    ├── goal_1/
    │   ├── overview.log
    │   ├── ManagerAgent_Loop1.log
    │   └── PersonaAgent_Refine_Loop1.log
    ├── goal_2/
    │   └── ...
    └── ...
```

-   `stdout.log`: Contains a copy of the console output.
-   `goal_x/`: A subdirectory for each goal that was run.
    -   `overview.log`: A detailed log of all activity for that specific goal.
    -   `ManagerAgent_LoopX.log`: The output of the ManagerAgent for each refinement loop.
    -   `PersonaAgent_Refine_LoopX.log`: The output of the PersonaAgent's refinement process for each loop.

### Output File

The results of the run (e.g., final recommendations, training data) are saved in the file specified by the `--output` argument (defaults to `simrag_training_data.jsonl`). This is a JSONL file where each line is a JSON object representing the result for a single goal.
