# Layered Personas for Agent Evaluation

.PHONY: install generate-personas run-benchmark run-multidomain analyze compare view clean help

# ── Variables (override on command line: make run-benchmark DOMAIN=airline) ──
PYTHON = python3
VENV = .venv
BIN = $(VENV)/bin/python
OUTPUT_DIR = eval/results
PERSONA_DIR = reddit/eval_personas_all
SPECS_DIR = specs
DOMAIN = retail
TASK_IDS = 0 1
MODEL = openai/llama3-sdsc
AGENT_TYPE = llm_agent_gt
CONCURRENCY = 20

help:
	@echo "Layered Personas — Research CLI"
	@echo "──────────────────────────────────────────────────"
	@echo ""
	@echo "  make install             Create venv and install dependencies"
	@echo "  make generate-personas   Run the persona generation pipeline"
	@echo ""
	@echo "  make run-benchmark       Run simulations (single domain)"
	@echo "  make run-multidomain     Run all domains (airline + retail + telecom)"
	@echo ""
	@echo "  make analyze             Deep-dive analysis for one domain"
	@echo "  make compare             Cross-condition comparison tables"
	@echo "  make view                View a simulation transcript"
	@echo ""
	@echo "  make clean               Remove caches and temp files"
	@echo ""
	@echo "Variables (override with VAR=value):"
	@echo "  DOMAIN       = $(DOMAIN)    (airline | retail | telecom | telecom-workflow)"
	@echo "  PERSONA_DIR  = $(PERSONA_DIR)"
	@echo "  MODEL        = $(MODEL)"
	@echo "  CONCURRENCY  = $(CONCURRENCY)"
	@echo "  TASK_IDS     = $(TASK_IDS)"

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --no-cache-dir -r requirements.txt
	$(VENV)/bin/pip install --no-deps -e tau2-bench/
	$(VENV)/bin/pip install --no-cache-dir litellm requests pydantic loguru pyyaml

generate-personas:
	@echo "Generating personas from reddit/personasforpaper.jsonl..."
	source $(VENV)/bin/activate && $(BIN) reddit/pipeline.py

# ── Single-domain benchmark ──────────────────────────────────────────────
# Usage: make run-benchmark DOMAIN=airline TASK_IDS="0 1 2"
run-benchmark:
	@echo "Running tau2-bench (Domain: $(DOMAIN), Concurrency: $(CONCURRENCY))..."
	source $(VENV)/bin/activate && $(BIN) -m eval.run_experiment \
		--condition layer \
		--personas $(PERSONA_DIR) \
		--specs $(SPECS_DIR) \
		--domain $(DOMAIN) \
		--task-ids $(TASK_IDS) \
		--num-trials 1 \
		--llm-agent $(MODEL) \
		--llm-user $(MODEL) \
		--agent-type $(AGENT_TYPE) \
		--max-concurrency $(CONCURRENCY) \
		--output $(OUTPUT_DIR)

# ── Multi-domain benchmark ───────────────────────────────────────────────
# Runs airline + retail + telecom with per-persona resume support.
# Pass additional args: make run-multidomain ARGS="--num-tasks 5"
run-multidomain:
	@echo "Running multi-domain evaluation..."
	source $(VENV)/bin/activate && $(BIN) -m eval.run_multidomain $(ARGS)

# ── Analysis ─────────────────────────────────────────────────────────────
# Usage: make analyze DOMAIN=retail
#        make analyze DOMAIN=airline OUTPUT_DIR=eval/results/multidomain
analyze:
	@echo "Analyzing $(OUTPUT_DIR)/$(DOMAIN)/..."
	source $(VENV)/bin/activate && $(BIN) -m eval.analyze $(OUTPUT_DIR)/$(DOMAIN)/ --domain $(DOMAIN)

# Usage: make compare LAYER_DIR=eval/results/retail NONE_DIR=eval/results/baseline
compare:
	source $(VENV)/bin/activate && $(BIN) -m eval.run_comparison \
		$(if $(LAYER_DIR),--layer $(LAYER_DIR)) \
		$(if $(NONE_DIR),--none $(NONE_DIR)) \
		$(if $(LEGACY_DIR),--legacy $(LEGACY_DIR)) \
		--domain $(DOMAIN)

view:
	@echo "Usage: make view FILE=path/to/results [PERSONA=id] [TASK=id]"
	source $(VENV)/bin/activate && $(PYTHON) view_transcripts.py $(FILE) $(if $(PERSONA),--persona $(PERSONA)) $(if $(TASK),--task $(TASK))

clean:
	rm -rf __pycache__ persona/__pycache__ eval/__pycache__ reddit/__pycache__
	rm -f output*.txt
	@echo "Cleaned."
