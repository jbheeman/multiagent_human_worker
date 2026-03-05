# Layered Personas for Agent Evaluation

A multi-agent system that generates psychologically-grounded personas to evaluate how well LLM agents handle realistic, non-cooperative human behavior. Built on [τ²-Bench](https://github.com/tau-bench).

## Core Ideas

- **3-Layer Persona Architecture**: Deep orientations (L0), stable background (L1), and situational state (L2) combine to produce ~393 billion unique persona configurations.
- **5-Level Valence**: Every dimension code is tagged positive / slightly positive / neutral / slightly negative / negative, enabling controlled sampling and difficulty categorization.
- **Reddit Grounding**: Personas can be extracted from real Reddit user histories for ecological validity.
- **Anti-Imitation Rules**: System prompts are hardened to prevent the LLM simulator from "helping" the agent or behaving like an AI.
- **All τ²-Bench Domains**: Personas work across airline (50 tasks), retail (114 tasks), telecom (2285 tasks), and telecom-workflow.

---

## Quick Start

### 1. Install

```bash
make install
```

### 2. Configure

Create a `.env` file:

```env
NAUTILUS_API_KEY=your_api_key_here
NAUTILUS_API_BASE=https://ellm.nrp-nautilus.io/v1
```

### 3. Generate Personas

```bash
# Synthetic personas biased toward difficult interactions:
python reddit/pipeline.py --mode synthetic --tier negative --num-synthetic 10 \
    --output reddit/eval_personas/

# Reddit-grounded personas (requires personasforpaper.jsonl):
python reddit/pipeline.py --mode fully_grounded --output reddit/eval_personas/
```

See [Generating Personas](#generating-personas) for all options.

### 4. Run Benchmark

```bash
# Single domain (retail, 2 tasks):
make run-benchmark DOMAIN=retail

# All domains (airline + retail + telecom), 2 tasks each:
make run-multidomain

# Airline only with the test split:
python -m eval.run_experiment \
    --condition layer \
    --personas reddit/eval_personas_all/ \
    --domain airline \
    --task-split test \
    --max-concurrency 20
```

See [Running Simulations](#running-simulations) for all options.

### 5. Analyze Results

```bash
# Deep-dive analysis for one domain:
make analyze DOMAIN=retail

# Export to CSV:
python -m eval.analyze eval/results/retail/ --csv results.csv

# Cross-condition comparison (layer vs. none baseline):
python -m eval.run_comparison \
    --layer eval/results/retail/ \
    --none eval/results/baseline/
```

See [Analyzing Results](#analyzing-results) for all options.

---

## Generating Personas

**Script:** `reddit/pipeline.py`

```bash
python reddit/pipeline.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `fully_grounded` | `fully_grounded` (L0+L1 from Reddit, L2 sampled), `partially_grounded` (L0 from Reddit), `synthetic` (all sampled) |
| `--tier` | unbiased | `positive` (easy personas), `neutral`, `negative` (difficult personas) |
| `--num-synthetic` | `0` | Additional purely synthetic personas to append |
| `--input` | `reddit/personasforpaper.jsonl` | Reddit user data |
| `--output` | `reddit/eval_personas` | Output directory for YAML files |
| `--seed` | `None` | Random seed for reproducibility |
| `--model` | `qwen3` | LLM for extraction, compilation, conformance checking |

**Examples:**

```bash
# 20 purely synthetic personas, hard difficulty:
python reddit/pipeline.py --mode synthetic --tier negative --num-synthetic 20 \
    --output reddit/eval_personas/

# Reddit-grounded with only L0 extracted, rest sampled neutral:
python reddit/pipeline.py --mode partially_grounded --tier neutral \
    --output reddit/eval_personas/

# Reproducible synthetic set:
python reddit/pipeline.py --mode synthetic --seed 42 --num-synthetic 50
```

**Output:** One `.yaml` file per persona containing `persona_profile`, `state_transition_rules`, and `termination_conditions`.

---

## Persona Architecture

Each persona is a combination of **12 dimensions** across 3 layers:

### Layer 0 — Deep Substrate (orientation)
Stable psychological dispositions. Not scored for difficulty (these are what we measure empirically).

| Dimension | Codes | Description |
|-----------|-------|-------------|
| `situation_construal` | 1A–1H | How the user interprets what's happening |
| `relational_stance` | 2A–2J | How the user relates to the agent |
| `agency` | 3A–3G | The user's sense of control |
| `epistemic` | 4A–4H | How the user forms beliefs |
| `stress_response` | 5A–5J | How the user behaves under pressure |

### Layer 1 — Stable Background
Persistent individual characteristics. Scored for difficulty.

| Dimension | Codes | Description |
|-----------|-------|-------------|
| `communicative_repertoire` | C1–C9 | Communication style and register |
| `domain_familiarity` | D1–D9 | Prior knowledge of the domain |
| `stakes` | S1–S9 | How much this interaction matters |
| `interaction_friction` | F1–F9 | External barriers present |

### Layer 2 — Situational State
Current episode state. Scored for difficulty.

| Dimension | Codes | Description |
|-----------|-------|-------------|
| `emotional_entry_state` | E1–E10 | Mood when starting the call |
| `bandwidth` | B1–B9 | Cognitive/attention capacity |
| `goal_clarity` | G1–G8 | How clearly the user knows what they want |

### Valence (5-level)

Every code has a valence in `persona/valence.py`:

```
POSITIVE          → easy for the agent (e.g., collaborative, clear)
SLIGHTLY_POSITIVE → mildly easy
NEUTRAL           → neither helps nor hinders
SLIGHTLY_NEGATIVE → mildly hard
NEGATIVE          → hard for the agent (e.g., hostile, vague, high-stakes)
```

**Difficulty scoring** combines the 9 scored dimensions (L1 + L2 stress_response + L1 relational_stance) into a `[0, 1]` score. Personas below 0.33 are `easy`, above 0.67 are `challenging`.

---

## Running Simulations

### Three Experimental Conditions

| Condition | Description | Use when |
|-----------|-------------|----------|
| `layer` | 3-layer persona system (this project) | Primary experiments |
| `legacy` | Old Schwartz-values YAML personas | Comparison to prior approach |
| `none` | No persona (stock τ²-bench user simulator) | Baseline |

### Supported Domains

| Domain | Tasks | Splits | Notes |
|--------|-------|--------|-------|
| `airline` | 50 | base, train, test | Flight booking/modification/cancellation |
| `retail` | 114 | base, train, test | E-commerce customer service |
| `telecom` | 2285 | base, small (20), train, test, full | Telecom support, manual policy |
| `telecom-workflow` | 2285 | same as telecom | Telecom support, workflow policy |

### Single-Domain Runner

```bash
python -m eval.run_experiment \
    --condition layer \
    --personas reddit/eval_personas_all/ \
    --domain airline \
    --task-split test \
    --max-concurrency 20 \
    --llm-agent openai/llama3-sdsc \
    --llm-user openai/llama3-sdsc \
    --output eval/results/airline/
```

**Key options:**

| Option | Description |
|--------|-------------|
| `--condition` | `layer`, `legacy`, `none` (multiple allowed) |
| `--domain` | `airline`, `retail`, `telecom`, `telecom-workflow` |
| `--task-ids 0 1 2` | Run specific tasks by ID |
| `--task-split test` | Use a named split (`base`, `small`, `train`, `test`) |
| `--num-tasks 5` | Run first N tasks |
| `--max-concurrency 20` | Parallel simulations (higher = faster, needs more API headroom) |
| `--agent-type llm_agent_gt` | Use ground-truth agent (recommended for eval) |
| `--num-trials 3` | Multiple trials per task for variance estimation |

### Multi-Domain Runner

Runs all domains sequentially with per-persona resume support (skips already-completed personas):

```bash
# All domains, default tasks (2 each):
python -m eval.run_multidomain

# Airline + retail only:
python -m eval.run_multidomain --domains airline retail

# All domains with the "test" split (where available):
python -m eval.run_multidomain --task-split test --concurrency 20

# 5 tasks per domain, custom output:
python -m eval.run_multidomain --num-tasks 5 --output eval/results/quicktest/
```

**Makefile shortcut:**

```bash
make run-multidomain
make run-multidomain ARGS="--domains airline retail --num-tasks 5"
```

---

## Analyzing Results

Results are saved as one JSON file per persona in `output/domain/`. All analysis tools accept either a single JSON file or a directory.

### Single-Condition Deep Dive

```bash
python -m eval.analyze eval/results/retail/
# or: make analyze DOMAIN=retail
```

**Output sections:**
1. **Score summary** — task success rate, trust (1–7), use-again rate, message count
2. **Score distributions** — histograms with calibration warnings
3. **Category comparison** — easy / moderate / challenging breakdown
4. **Per-persona table** — one row per persona sorted by reward
5. **Paper tables** — Table 2 metrics, robustness stats, trait sensitivity

```bash
# Export CSV for external analysis:
python -m eval.analyze eval/results/retail/ --csv results.csv

# Show only summary (skip paper tables):
python -m eval.analyze eval/results/retail/ --no-paper-tables

# Specify domain name for table headers:
python -m eval.analyze eval/results/airline/ --domain airline
```

### Cross-Condition Comparison

Compares layer personas vs. legacy vs. no-persona baseline:

```bash
python -m eval.run_comparison \
    --layer eval/results/layer/ \
    --none  eval/results/baseline/ \
    --legacy eval/results/legacy/ \
    --domain retail

# or: make compare LAYER_DIR=eval/results/retail NONE_DIR=eval/results/baseline
```

**Output tables:**
- **Table 1**: Task success rate per condition
- **Table 2**: Goal achievement, trust\*, cognitive effort, intent alignment, use-again\*
- **Robustness**: Mean, variance, min, max, persona gap, tail risk Q₁₀/Q₂₅
- **Trait sensitivity**: E[M(A,p) | t_j = v] — how each dimension code predicts performance

(\* = LLM judge with k=3 agreement)

### Metrics Reference

| Metric | Source | Scale | Description |
|--------|--------|-------|-------------|
| Task success | τ²-bench evaluator | 0 / 1 | Did the agent complete the task correctly? |
| Trust | LLM judge (k=3) | 1–7 | How much does this persona trust the agent? |
| Use again | LLM judge (k=3) | 0 / 1 | Would this persona use the service again? |
| Goal achievement | Transcript proxy | 1–7 | Did the user's goal get met? |
| Cognitive effort | Transcript proxy | 1–7 | How much work did the user have to do? (lower = better) |
| Intent alignment | Transcript proxy | 1–7 | Did the agent understand what the user wanted? |

---

## Project Structure

```
.
├── Makefile                  # CLI commands
├── persona/                  # Core persona engine
│   ├── valence.py            # Canonical 5-level valence map (single source of truth)
│   ├── categories.py         # Difficulty scoring (easy / moderate / challenging)
│   ├── sampler.py            # Weighted sampling with valence tiers
│   ├── assembler.py          # System prompt assembly from layer codes
│   ├── compiler.py           # IF/THEN rule generation (LLM)
│   ├── conformance.py        # Conformance checking with retry loop
│   ├── extractor.py          # Layer code extraction from Reddit data
│   ├── registry.py           # Spec file parser + PersonaConfig builder
│   └── schema.py             # Data models (PersonaConfig, LayerSelection, etc.)
├── eval/                     # Evaluation suite
│   ├── run_experiment.py     # Single-domain runner (all conditions + domains)
│   ├── run_multidomain.py    # Multi-domain orchestrator with resume support
│   ├── analyze.py            # Single-condition deep-dive analysis
│   ├── run_comparison.py     # Cross-condition comparison + paper tables
│   └── metrics.py            # Metric computation (trust, use_again, robustness, etc.)
├── reddit/                   # Persona generation pipeline
│   └── pipeline.py           # Reddit data → YAML persona spec (4-step pipeline)
├── specs/                    # Layer specifications (markdown)
│   ├── layer0_persona_spec.md  # L0: 5 orientation dimensions
│   ├── layer1_persona_spec.md  # L1: 4 background dimensions
│   └── layer2_persona_spec.md  # L2: 3 situational dimensions
└── tau2-bench/               # Forked τ²-bench evaluation environment
```

---

## License

MIT
