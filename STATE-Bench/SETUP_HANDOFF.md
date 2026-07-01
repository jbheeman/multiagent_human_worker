# STATE-Bench Setup Handoff

Summary for continuing work on STATE-Bench in this repo. Written after initial exploration for **Main Track** setup with **Nautilus** as the agent provider (similar to tau-bench persona eval workflow).

---

## Goal (user intent)

1. Run STATE-Bench **Main Track** before integrating reddit personas.
2. Use **Nautilus** (`https://ellm.nrp-nautilus.io/v1`) for the **agent under test** — same pattern as `reddit/personaAdapter.py` (`OpenAIServerModel` / OpenAI-compatible endpoint).
3. Ideally replicate a **tau-bench-style** unofficial workflow:
   - Cheap user simulator on Nautilus (e.g. Gemma)
   - Agent under test on Nautilus (e.g. gpt-oss, kimi)
   - **No judge** during collection; score/judge transcripts independently later
   - Output similar in spirit to `reddit/Eval/ClusteredPersonas/airline/gpt-oss/_tpyo_gpt-oss_airline.json`

---

## Repo layout (relevant paths)

| Path | Purpose |
|------|---------|
| `STATE-Bench/` | Benchmark package (use **uv** here only) |
| `STATE-Bench/pyproject.toml` | Requires Python **3.12+** |
| `STATE-Bench/.venv/` | Created by `uv sync` (CPython 3.14.6 installed by uv) |
| `STATE-Bench/.env` | Local env (created; includes `NAUT_*` from workspace root) |
| `STATE-Bench/clients/nautilus_client.py` | Custom `BaseLLMClient` for Nautilus agent |
| `STATE-Bench/agents/nautilus_agent.py` | Custom `BaseAgent` (chat completions + tool calling) |
| `reddit/personaAdapter.py` | Reference Nautilus wiring (lines ~115–120) |
| `tau2-bench/` | Separate benchmark; persona eval already works there |
| `/.env` (workspace root) | Has `NAUT_API_KEY` |

**Do not** set up uv for the whole monorepo unless desired — only STATE-Bench (and optionally tau2-bench) have `pyproject.toml`.

---

## Environment setup (done)

```bash
# Install uv (if missing)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

cd STATE-Bench
uv sync
cp .env.example .env   # already done; NAUT_* appended from ../.env
```

System Python was **3.10.12** — too old for STATE-Bench. `uv sync` pulls **3.12+** automatically.

Verify CLI:

```bash
cd STATE-Bench
uv run python -m state_bench.scripts.run_task --help
```

---

## Three LLM roles in STATE-Bench

| Role | Locked? | Default wiring | Nautilus? |
|------|---------|----------------|-----------|
| **User simulator** | **Yes** for official runs (GPT-5.4) | `build_user_sim_client()` → `STATE_BENCH_EVAL_*` | Not via CLI; possible unofficially via `orchestrator.run_task(simulator_client=...)` |
| **Judge** | **Yes** for official runs (GPT-5.4) | `build_locked_judge_client()` | Skippable with `--no-score` |
| **Agent under test** | **No** (user choice) | Builtin `StateBenchAgent` + Azure/OpenAI | **Requires custom client** (see below) |

Official docs:
- Main track: `docs/RUN_BENCHMARK.md`
- Eval client (sim + judge): `docs/setup/eval-client.md`
- Custom agent: `docs/USE_CUSTOM_CLIENT.md`
- Builtin agent: `docs/agents/builtin.md`

---

## Why builtin agent does NOT work with Nautilus

1. **Third-party base URLs rejected** in `state_bench/client.py`:
   ```python
   if os.environ.get("STATE_BENCH_AGENT_BASE_URL") or os.environ.get("OPENAI_BASE_URL"):
       raise ValueError("Third-party OpenAI-compatible base URLs are not supported for StateBenchAgent.")
   ```

2. **Builtin uses Responses API** (`responses.create` with `previous_response_id` tool loop), not chat completions. See `state_bench/agents/state_bench.py` and `LLMClient.complete_with_tools()`.

3. Nautilus / smolagents pattern is **OpenAI-compatible chat completions** at `https://ellm.nrp-nautilus.io/v1`.

**Conclusion:** Use `--agent-class NautilusAgent --agent-client-class NautilusClient` (already scaffolded).

---

## Custom Nautilus integration (done)

### `clients/nautilus_client.py`

- Subclasses `BaseLLMClient`
- Env vars:
  - `NAUT_API_KEY` (required)
  - `NAUT_MODEL` (default: `kimi`)
  - `NAUT_API_BASE` (default: `https://ellm.nrp-nautilus.io/v1`)
  - `NAUT_VERIFY_SSL` (default: `false` — matches personaAdapter SSL skip)
- Implements `generate()` via `chat.completions.create`
- **Does not** implement `complete_chat()` — needed if this client is used for the **user simulator** (see unofficial path below)

### `agents/nautilus_agent.py`

- Subclasses `BaseAgent`, implements `generate_next_turn()`
- Converts STATE-Bench tool schemas → OpenAI chat `function` format
- Converts canonical conversation → chat messages (including tool result replay)
- Loaded by class name via harness (`state_bench/agents/loader.py`)

Verified loading:

```bash
cd STATE-Bench && source ../.env
uv run python -c "
from state_bench.agents.loader import load_root_agent_class, load_root_client_class
print(load_root_client_class('NautilusClient').from_env().model_name)
print(load_root_agent_class('NautilusAgent'))
"
```

---

## Official run command (when Azure GPT-5.4 eval is available)

Fill `STATE_BENCH_EVAL_*` in `.env` per `docs/setup/eval-client.md`.

```bash
cd STATE-Bench
uv run python -m state_bench.scripts.run_task \
  --task 1-cancel_economy_domestic \
  --domain travel \
  --agent-class NautilusAgent \
  --agent-client-class NautilusClient \
  --agent-model-name kimi \
  --no-score \
  --num-workers 1
```

Batch:

```bash
uv run python -m state_bench.scripts.run_batch \
  --domain travel \
  --agent-class NautilusAgent \
  --agent-client-class NautilusClient \
  --agent-model-name kimi \
  --num-runs 1 \
  --num-workers 1 \
  --no-score \
  --output-dir outputs/travel/
```

Outputs: `outputs/<domain>/run<N>/<task_id>.json` per task.

---

## Current blocker (official / default CLI path)

**Smoke test failed** because `run_task` always calls `build_user_sim_client()`, which requires:

```bash
STATE_BENCH_EVAL_ENDPOINT="https://your-gpt54-resource.openai.azure.com"
STATE_BENCH_EVAL_DEPLOYMENTS="<gpt-5.4 deployment>"
# STATE_BENCH_EVAL_API_KEY="..."  # or Azure CLI / DefaultAzureCredential
```

Error observed:
```
ValueError: Azure OpenAI endpoint required. Set STATE_BENCH_EVAL_ENDPOINT environment variable...
```

`--no-score` skips the **judge** but **not** the user simulator. Every task needs a simulated user.

**User does not currently have Azure GPT-5.4 configured** (as of this handoff).

---

## Unofficial tau-bench-style path (not implemented yet)

### What the user wants

- User sim: e.g. Gemma on Nautilus
- Agent: e.g. gpt-oss / kimi on Nautilus
- No judge; collect transcripts for external eval
- Eventually: inject reddit YAML personas (like tau-bench `UserSimulator` with `yaml_content`)

### What STATE-Bench already supports in code

`orchestrator.run_task()` accepts **separate** `client` (agent) and `simulator_client`:

```python
resolved_simulator_client = simulator_client or client
simulator = UserSimulator(resolved_simulator_client, sim_prompt)
```

CLI scripts **do not** expose `--simulator-client-class` or `--simulator-model`. They hardcode `build_user_sim_client()`.

### Gaps to implement unofficial runner

1. **Small script** (e.g. `scripts/run_unofficial.py`) that:
   - Loads `NautilusClient` for agent and sim (different `NAUT_MODEL` / env vars)
   - Calls `orchestrator.run_task()` directly
   - Saves trajectory JSON
   - Never calls judge

2. **`complete_chat()` on Nautilus adapter** for user simulator:
   - `UserSimulator.respond()` calls `client.complete_chat(messages)` 
   - Builtin `LLMClient.complete_chat` uses Responses API
   - Nautilus needs simple chat completion (no tools)

3. **Persona injection** (later, for reddit work):
   - STATE-Bench tasks use per-task `user_simulator` JSON (`user_sim_context`, `known_info`, `task_rules`)
   - Prompt built in `state_bench/domains/<domain>/simulator.py` → `build_simulator_prompt()`
   - **Not** compatible with reddit ClusteredPersonas YAML out of the box
   - Would need custom `UserSimulator` subclass or custom `build_simulator_prompt()` to inject YAML like tau-bench

### Output format vs tau-bench

| | tau-bench | STATE-Bench |
|---|-----------|-------------|
| File layout | One big JSON with `tasks[]`, `info`, timestamps | One JSON **per task** |
| Transcript | In task simulation messages | `conversation` array in trajectory |
| Termination token | `###STOP###` | `[TASK_DONE]` |
| State | Task-specific | `state_diff` (DB before/after) |
| Persona | YAML in user sim system prompt | Task `user_simulator` fields + domain prompts |

---

## Protocol / metadata notes

- Locked protocol: `state_bench/configs/eval_protocols/gpt54.json`
- Simulator model recorded as `gpt-5.4` in protocol metadata stamped on trajectories
- Unofficial runs still load default protocol; metadata may say `simulator_model: gpt-5.4` even if you used a different client — misleading for unofficial runs; consider custom metadata in unofficial script

---

## Key architecture references

| Component | File |
|-----------|------|
| Task loop | `state_bench/orchestrator.py` |
| User simulator | `state_bench/simulator.py` |
| Builtin agent (Responses API) | `state_bench/agents/state_bench.py` |
| Custom agent contract | `state_bench/agents/base.py` → `generate_next_turn()` |
| Trajectory output | `state_bench/schemas.py` → `Trajectory.to_dict()` / `.save()` |
| Run single task CLI | `state_bench/scripts/run_task.py` |
| Run batch CLI | `state_bench/scripts/run_batch.py` |
| Extension loading | `state_bench/agents/loader.py` (repo-root `clients/`, `agents/`) |

Harness tool execution: custom agents return `AgentToolCallRequest`; STATE-Bench executes domain tools, not the agent.

---

## Recommended next steps (priority order)

1. **If user gets Azure GPT-5.4:** Fill `STATE_BENCH_EVAL_*`, run smoke test with Nautilus agent + `--no-score`.

2. **If staying Nautilus-only (no Azure):** Implement unofficial runner:
   - Add `complete_chat()` to `NautilusClient` (or `NautilusSimulatorClient`)
   - Env: `NAUT_SIM_MODEL`, `NAUT_AGENT_MODEL` (or two client classes)
   - Script calling `orchestrator.run_task()` with separate clients
   - `--no-score` or no scoring calls at all

3. **Persona eval on STATE-Bench tasks:** Design how reddit YAML maps into `build_simulator_prompt()` or custom `UserSimulator` — separate from model wiring.

4. **Keep tau-bench for persona airline eval** if STATE-Bench persona injection is out of scope short-term.

---

## Commands cheat sheet

```bash
# Setup
cd STATE-Bench && uv sync && source $HOME/.local/bin/env

# Load env (STATE-Bench .env has NAUT_*; eval vars still missing)
set -a && source .env && set +a

# Verify custom classes load
uv run python -c "from state_bench.agents.loader import load_root_client_class; load_root_client_class('NautilusClient').from_env()"

# Official-path smoke (needs STATE_BENCH_EVAL_*)
uv run python -m state_bench.scripts.run_task \
  --task 1-cancel_economy_domestic \
  --domain travel \
  --agent-class NautilusAgent \
  --agent-client-class NautilusClient \
  --agent-model-name kimi \
  --no-score

# Tests
uv run pytest tests/ -q
```

---

## Open questions for the user / next agent

1. Does the user have **Azure GPT-5.4** for official sim+judge, or is everything **Nautilus-only**?
2. Is the immediate goal **smoke test one task** or **batch transcript collection** for persona research?
3. Should persona YAML from `reddit/ClusteredPersonas/` be injected into STATE-Bench sim, or is tau-bench sufficient for persona work while STATE-Bench is used for procedural-memory benchmark only?
4. Which Nautilus models for sim vs agent? (tau setup used `gemma3` sim + `gpt-oss` agent)

---

## What was verified vs not

| Item | Status |
|------|--------|
| `uv sync` in STATE-Bench | ✅ Done |
| `NautilusClient` / `NautilusAgent` load from env | ✅ Verified |
| End-to-end task run | ❌ Blocked on `STATE_BENCH_EVAL_*` (default CLI) |
| Unofficial dual-Nautilus runner | ❌ Not built |
| Persona YAML injection | ❌ Not started |
| Official submission-ready run | ❌ Needs GPT-5.4 sim+judge + 5 runs + metrics |
